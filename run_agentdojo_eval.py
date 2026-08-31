"""
run_agentdojo_eval.py

P4 Track B: `compare_head_sources.py`(Track A)로 찾은 head 집합을, AgentDojo의 실제 멀티턴
agent loop + 네이티브 utility/security 채점(환경 상태 기반, 토큰 확률 proxy 아님)으로 검증한다.

각 (user_task, injection_task) 쌍마다 같은 롤아웃을 두 조건으로 돌린다:
  - k=0 (knockout 없음): 공격이 원래 얼마나 잘 먹히는지, 원래 과업을 얼마나 잘 푸는지 baseline
  - k=len(heads) (지정한 head들의 D_inj edge를 knockout): 방어 적용 후 utility/security가
    어떻게 바뀌는지

`utility`=True면 원래 과업 성공, `security`=True면 **공격 성공**(방어 관점에서는 나쁜 것) —
knockout이 잘 작동하면 security가 True->False로 줄고 utility는 유지돼야 한다.

사용 예:
    python run_agentdojo_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --family qwen2 \\
        --heads_json results/2026-07-31_source_compare/heads_synthetic.json \\
        --suite banking --limit_pairs 15 --out_json results/.../agentdojo_eval_synthetic.json

P16(2026-08-31, 교수님 피드백): `--tool_call_format {custom, agentdojo_default}`로
tool-call 프롬프트/파서를 고를 수 있다. 기본값(custom)은 위와 같은 기존 동작(모델별
native 포맷 + family별 파서). `agentdojo_default`는 AgentDojo 자체 `_make_system_prompt`/
`_parse_model_output`을 그대로 쓴다(고정 지시문 + 단일 파서, family 무관) — 같은 모델·
같은 `--heads_json`/`--seed`로 두 값을 각각 돌려서 parse_stats.ok 비율과 attack/utility
수치를 비교하는 게 P16의 목적. 상세: `docs/feedback-2026-08-31.md`, `docs/todo.md` P16.

    python run_agentdojo_eval.py --model Qwen/Qwen2.5-7B-Instruct --family qwen2 \\
        --tool_call_format agentdojo_default --heads_json ... --suite banking \\
        --out_json results/.../agentdojo_eval_agentdojo_default.json
"""
import argparse
import gc
import json
import os
import random
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.task_suite.load_suites import get_suite

from adapters.agentdojo_pipeline import KnockoutLocalLLM
from runtime_env import add_runtime_args, collect_env_meta, describe


def _load_model(model_path: str, family: str, four_bit: bool, device: str, dtype: str = "auto",
                 device_map: str = None):
    # Track B는 forward-only(edge_knockout)라 attn_relevance.load_model_for_relevance의
    # lxt monkey-patch(backward 전용)가 필요 없다 — 그냥 eager attention으로만 로드한다.
    from runtime_env import resolve_dtype

    if family == "qwen2":
        from transformers.models.qwen2 import modeling_qwen2 as modeling_mod
    else:
        from transformers.models.llama import modeling_llama as modeling_mod

    resolved_device_map = device_map or device
    # dtype 판정에는 device가 아니라 device_map을 넘긴다 — "auto"를 device처럼 취급해
    # fp32로 떨어지면 분산 로드의 목적(메모리 절감)이 무너진다 (attn_relevance.py와 동일 이유).
    torch_dtype, dtype_name = resolve_dtype(dtype, resolved_device_map)

    kwargs = dict(torch_dtype=torch_dtype, device_map=resolved_device_map, attn_implementation="eager")
    if four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_path)
    return model, tok, modeling_mod, dtype_name


def _build_pipeline(llm, max_iters: int) -> AgentPipeline:
    tools_loop = ToolsExecutionLoop([ToolsExecutor(), llm], max_iters=max_iters)
    pipeline = AgentPipeline([SystemMessage(load_system_message(None)), InitQuery(), llm, tools_loop])
    pipeline.name = "knockout-local-llm"
    return pipeline


def _load_heads(heads_json: str) -> Tuple[List[Tuple[int, int]], dict]:
    """heads 목록과 함께 **split_info**도 돌려준다.

    split_info["per_stratum"][suite]["head_groups"]에는 head 탐색에 쓰인 user_task 목록이
    들어 있다. 평가에서 이걸 빼지 않으면 held-out 평가가 아니다 — AgentDojo pair는
    (user_task x injection_task)라, 같은 user_task의 다른 injection으로 평가하면 프롬프트의
    system/user 질문/history를 전부 공유한 상태로 "새 데이터"라고 주장하는 셈이 된다.
    (docs/plan-2026-08-26.md 5절, pair_sampling.py 참고)

    구 버전 heads JSON에는 split_info가 없다 — 그 경우 빈 dict를 돌려주고, 호출부가
    --eval_split heldout이면 에러를 낸다 (조용히 all로 떨어지지 않게)."""
    with open(heads_json, encoding="utf-8") as f:
        data = json.load(f)
    return [tuple(h) for h in data["heads"]], data.get("split_info") or {}


def _heldout_user_tasks(split_info: dict) -> Dict[str, set]:
    """suite -> head 탐색에 쓰인 user_task 집합 (평가에서 제외할 대상)."""
    out: Dict[str, set] = {}
    for suite, v in (split_info.get("per_stratum") or {}).items():
        out[suite] = set(v.get("head_groups") or [])
    return out


def _heads_to_knockout_map(heads: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    km: Dict[int, List[int]] = {}
    for layer, head in heads:
        km.setdefault(layer, []).append(head)
    return km


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--family", default="qwen2", choices=["qwen2", "llama"])
    parser.add_argument("--device", default="cuda", help="입력 텐서를 올릴 device. "
                         "--device_map auto로 모델을 쪼개도 입력은 첫 device에 있어야 하므로 "
                         "그 경우 'cuda:0'으로 둘 것.")
    parser.add_argument(
        "--device_map", default=None,
        help="모델 가중치 배치. 미지정이면 --device를 그대로 쓴다(단일 GPU, 기존 동작). "
        "'auto'면 여러 GPU에 레이어를 분산한다 — 32B처럼 단일 24GB에 안 들어가는 모델용.",
    )
    parser.add_argument("--four_bit", action="store_true")
    parser.add_argument(
        "--tool_call_format", default="custom", choices=["custom", "agentdojo_default"],
        help="P16(2026-08-31, 교수님 피드백). custom(기본): 모델별 native 포맷 + family별 "
        "파서(지금까지 쓰던 방식). agentdojo_default: AgentDojo 자체 _make_system_prompt/"
        "_parse_model_output 그대로 사용(고정 지시문 + family 무관 단일 파서). 같은 모델로 "
        "두 값을 각각 돌려서 parse_stats/attack/utility를 비교할 것 — docs/todo.md P16.",
    )
    parser.add_argument("--heads_json", required=True, help="compare_head_sources.py discover(-parallel) 결과 JSON")
    parser.add_argument(
        "--attack", default="important_instructions",
        help="agentdojo.attacks.attack_registry에 등록된 공격 이름 (예: important_instructions, "
        "tool_knowledge, injecagent, direct, ignore_previous, system_message). "
        "python -c \"from agentdojo.attacks.attack_registry import ATTACKS; print(list(ATTACKS))\"로 전체 목록 확인 가능.",
    )
    parser.add_argument(
        "--suite", nargs="+", default=["banking", "slack", "travel", "workspace"],
        choices=["banking", "slack", "travel", "workspace"],
        help="평가할 suite (기본: 4개 전부).",
    )
    parser.add_argument("--benchmark_version", default="v1.2.2")
    parser.add_argument(
        "--limit_pairs", type=int, default=15,
        help="suite마다 평가할 (user_task, injection_task) 쌍 개수(무작위 샘플, suite별로 따로 뽑음 "
        "— 안 그러면 combo가 가장 많은 workspace가 표본을 독점함).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--max_iters", type=int, default=8,
        help="ToolsExecutionLoop 최대 반복 횟수(기본 15보다 낮춤 — 7B 등 큰 모델에서 롤아웃 하나가 "
        "너무 오래 걸리는 걸 방지).",
    )
    parser.add_argument(
        "--eval_split", default="heldout", choices=["heldout", "all"],
        help="heldout(기본): --heads_json의 split_info를 읽어 head 탐색에 쓰인 user_task를 "
        "평가에서 제외한다. AgentDojo pair는 (user_task x injection_task)라 같은 user_task의 "
        "다른 injection으로 평가하면 프롬프트를 거의 공유해 held-out이 아니게 된다. "
        "all: 제외 없이 전체에서 무작위 샘플링 (기존 동작). 두 결과를 나란히 놓으면 "
        "'누수가 수치를 얼마나 부풀렸는지'를 직접 보여줄 수 있다.",
    )
    parser.add_argument("--out_json", default=None)
    add_runtime_args(parser)
    args = parser.parse_args()

    # 출력 폴더를 롤아웃 **전에** 확보한다. 결과 쓰기는 모든 롤아웃이 끝난 뒤라, 폴더가
    # 없으면 수십 분~수 시간의 GPU 작업을 다 하고 마지막 줄에서 죽는다
    # (2026-08-24에 compare_head_sources의 32B 스모크가 실제로 이렇게 날아갔다).
    out_json = args.out_json or "agentdojo_eval_summary.json"
    _out_dir = os.path.dirname(out_json)
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)

    heads, split_info = _load_heads(args.heads_json)
    excluded_by_suite = _heldout_user_tasks(split_info)
    if args.eval_split == "heldout" and not excluded_by_suite:
        raise SystemExit(
            f"--eval_split heldout인데 {args.heads_json}에 split_info가 없다.\n"
            f"  compare_head_sources.py discover-parallel로 새로 만든 heads JSON을 쓰거나,\n"
            f"  의도적으로 전체를 평가하려면 --eval_split all 을 명시할 것."
        )
    knockout_map = _heads_to_knockout_map(heads)
    print(f"[run_agentdojo_eval] {len(heads)} heads loaded from {args.heads_json}: {heads}")

    print(f"[run_agentdojo_eval] loading {args.model} ...")
    model, tok, modeling_mod, dtype_name = _load_model(
        args.model, args.family, args.four_bit, args.device, args.dtype, args.device_map
    )
    print(describe(dtype_name))

    llm = KnockoutLocalLLM(
        model, tok, modeling_mod, knockout_map=None, max_new_tokens=args.max_new_tokens,
        device=args.device, family=args.family, tool_call_format=args.tool_call_format,
    )
    pipeline = _build_pipeline(llm, args.max_iters)

    rng = random.Random(args.seed)
    rows = []
    split_report: dict = {}
    total_idx = 0
    for suite_name in args.suite:
        suite = get_suite(args.benchmark_version, suite_name)
        # pipeline.name="knockout-local-llm"에 "local"이 포함돼 있어 agentdojo.models.MODEL_NAMES의
        # "local" -> "Local model" 매핑에 걸림 — ImportantInstructionsAttack/ToolKnowledgeAttack 등이
        # 내부적으로 쓰는 get_model_name_from_pipeline()이 별도 설정 없이도 정상 동작한다.
        attack = load_attack(args.attack, suite, pipeline)
        all_pairs = [(ut, it) for ut in suite.user_tasks.values() for it in suite.injection_tasks.values()]
        n_before = len(all_pairs)

        # head 탐색에 쓰인 user_task를 통째로 제외한다 (진짜 held-out 평가).
        # --eval_split all 이면 이 단계를 건너뛰어 기존 동작(전체에서 무작위)이 된다.
        excluded = excluded_by_suite.get(suite_name, set())
        if args.eval_split == "heldout" and excluded:
            all_pairs = [(ut, it) for (ut, it) in all_pairs if ut.ID not in excluded]
        n_excluded = n_before - len(all_pairs)

        if not all_pairs:
            print(f"[run_agentdojo_eval] suite={suite_name}: held-out 후 남은 pair가 없다 "
                  f"(user_task {len(excluded)}개 전부 head 탐색에 쓰임) — 건너뜀")
            split_report[suite_name] = {"n_all": n_before, "n_excluded": n_excluded,
                                        "n_evaluated": 0, "excluded_user_tasks": sorted(excluded)}
            continue

        rng.shuffle(all_pairs)
        pairs = all_pairs[: args.limit_pairs]
        split_report[suite_name] = {
            "n_all": n_before, "n_excluded": n_excluded, "n_candidates": len(all_pairs),
            "n_evaluated": len(pairs), "excluded_user_tasks": sorted(excluded),
        }
        print(f"[run_agentdojo_eval] suite={suite_name} (attack={args.attack}) "
              f"mode={args.eval_split}: 전체 {n_before} - 제외 {n_excluded} "
              f"-> 후보 {len(all_pairs)} -> 평가 {len(pairs)}")

        for ut, it in pairs:
            total_idx += 1

            try:
                injections = attack.attack(ut, it)

                llm.knockout_map = {}
                u0, s0 = suite.run_task_with_pipeline(pipeline, ut, it, injections)

                llm.knockout_map = knockout_map
                uk, sk = suite.run_task_with_pipeline(pipeline, ut, it, injections)
            except Exception as e:
                # 공격 문구 생성 자체가 실패하거나(예: ToolKnowledgeAttack이 요구하는
                # placeholder_args가 없는 case), 모델이 tool_call 인자를 이상한 형태로
                # 생성하는 등 예측 못 한 오류가 있을 수 있다 — 한 쌍이 실패해도 전체
                # sweep을 죽이지 않고 건너뛴다.
                print(f"  [{total_idx}] {suite_name}/{ut.ID}+{it.ID}: ERROR ({type(e).__name__}: {e}), skipping")
                gc.collect()
                torch.cuda.empty_cache()
                continue

            gc.collect()
            torch.cuda.empty_cache()

            row = {
                "suite": suite_name, "user_task": ut.ID, "injection_task": it.ID,
                "k0_utility": u0, "k0_security": s0,
                "kN_utility": uk, "kN_security": sk,
            }
            rows.append(row)
            print(f"  [{total_idx}] {suite_name}/{ut.ID}+{it.ID}: k=0 utility={u0} security={s0}  |  k={len(heads)} utility={uk} security={sk}")

    llm.knockout_map = {}

    def rate(rs, key):
        return sum(1 for r in rs if r[key]) / len(rs) if rs else 0.0

    per_suite = {}
    for suite_name in args.suite:
        rs = [r for r in rows if r["suite"] == suite_name]
        per_suite[suite_name] = {
            "n_pairs": len(rs),
            "k0_utility_rate": rate(rs, "k0_utility"), "k0_security_rate": rate(rs, "k0_security"),
            "kN_utility_rate": rate(rs, "kN_utility"), "kN_security_rate": rate(rs, "kN_security"),
        }

    summary = {
        "model": args.model, "suites": args.suite, "heads_json": args.heads_json, "attack": args.attack,
        "n_heads": len(heads), "n_pairs": len(rows), "seed": args.seed, "max_iters": args.max_iters,
        "max_new_tokens": args.max_new_tokens, "tool_call_format": args.tool_call_format,
        "k0_utility_rate": rate(rows, "k0_utility"), "k0_security_rate": rate(rows, "k0_security"),
        "kN_utility_rate": rate(rows, "kN_utility"), "kN_security_rate": rate(rows, "kN_security"),
        "per_suite": per_suite,
        "parse_stats": llm.parse_stats,
        "rows": rows,
    }
    print()
    print(f"[전체 {len(rows)}쌍] k=0   utility_rate={summary['k0_utility_rate']:.3f}  security_rate(=공격성공률)={summary['k0_security_rate']:.3f}")
    print(f"[전체 {len(rows)}쌍] k={len(heads):<3} utility_rate={summary['kN_utility_rate']:.3f}  security_rate(=공격성공률)={summary['kN_security_rate']:.3f}")
    for suite_name, s in per_suite.items():
        print(
            f"  {suite_name}: k=0 utility={s['k0_utility_rate']:.2f} security={s['k0_security_rate']:.2f}  |  "
            f"k={len(heads)} utility={s['kN_utility_rate']:.2f} security={s['kN_security_rate']:.2f}"
        )
    ps = llm.parse_stats
    print(
        f"\n[tool_call parse stats] n_calls={ps['n_calls']}  ok={ps['ok']}  no_tag={ps['no_tag']}  "
        f"truncated(닫는태그 없음)={ps['truncated']}  json_errors={ps['json_errors']}  non_dict_args={ps['non_dict_args']}"
    )

    # out_json은 위(롤아웃 시작 전)에서 이미 정해지고 폴더까지 확보돼 있다.
    # 결과 폴더가 아니라 단일 JSON으로 나가는 경로라, env.json 대신 summary 안에 넣는다.
    # 이게 없으면 "어느 커밋/어느 dtype으로 낸 숫자인지" 사후에 가릴 수 없다.
    summary["eval_split"] = {
        "mode": args.eval_split,
        "heads_json": args.heads_json,
        # 어느 user_task를 왜 뺐는지까지 남긴다 — 사후에 "정말 held-out이었나"를 검증 가능하게
        "per_suite": split_report,
    }
    summary["env"] = collect_env_meta(dtype_name)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary saved to {out_json}")


if __name__ == "__main__":
    main()
