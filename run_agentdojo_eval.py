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
"""
import argparse
import gc
import json
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


def _load_model(model_path: str, family: str, four_bit: bool, device: str):
    # Track B는 forward-only(edge_knockout)라 attn_relevance.load_model_for_relevance의
    # lxt monkey-patch(backward 전용)가 필요 없다 — 그냥 eager attention으로만 로드한다.
    if family == "qwen2":
        from transformers.models.qwen2 import modeling_qwen2 as modeling_mod
    else:
        from transformers.models.llama import modeling_llama as modeling_mod

    kwargs = dict(torch_dtype=torch.bfloat16, device_map=device, attn_implementation="eager")
    if four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_path)
    return model, tok, modeling_mod


def _build_pipeline(llm, max_iters: int) -> AgentPipeline:
    tools_loop = ToolsExecutionLoop([ToolsExecutor(), llm], max_iters=max_iters)
    pipeline = AgentPipeline([SystemMessage(load_system_message(None)), InitQuery(), llm, tools_loop])
    pipeline.name = "knockout-local-llm"
    return pipeline


def _load_heads(heads_json: str) -> List[Tuple[int, int]]:
    with open(heads_json, encoding="utf-8") as f:
        data = json.load(f)
    return [tuple(h) for h in data["heads"]]


def _heads_to_knockout_map(heads: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    km: Dict[int, List[int]] = {}
    for layer, head in heads:
        km.setdefault(layer, []).append(head)
    return km


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--family", default="qwen2", choices=["qwen2", "llama"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--four_bit", action="store_true")
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
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    heads = _load_heads(args.heads_json)
    knockout_map = _heads_to_knockout_map(heads)
    print(f"[run_agentdojo_eval] {len(heads)} heads loaded from {args.heads_json}: {heads}")

    print(f"[run_agentdojo_eval] loading {args.model} ...")
    model, tok, modeling_mod = _load_model(args.model, args.family, args.four_bit, args.device)

    llm = KnockoutLocalLLM(model, tok, modeling_mod, knockout_map=None, max_new_tokens=args.max_new_tokens, device=args.device)
    pipeline = _build_pipeline(llm, args.max_iters)

    rng = random.Random(args.seed)
    rows = []
    total_idx = 0
    for suite_name in args.suite:
        suite = get_suite(args.benchmark_version, suite_name)
        # pipeline.name="knockout-local-llm"에 "local"이 포함돼 있어 agentdojo.models.MODEL_NAMES의
        # "local" -> "Local model" 매핑에 걸림 — ImportantInstructionsAttack/ToolKnowledgeAttack 등이
        # 내부적으로 쓰는 get_model_name_from_pipeline()이 별도 설정 없이도 정상 동작한다.
        attack = load_attack(args.attack, suite, pipeline)
        all_pairs = [(ut, it) for ut in suite.user_tasks.values() for it in suite.injection_tasks.values()]
        rng.shuffle(all_pairs)
        pairs = all_pairs[: args.limit_pairs]
        print(f"[run_agentdojo_eval] evaluating {len(pairs)}/{len(all_pairs)} pairs from suite={suite_name} (attack={args.attack})")

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
        "k0_utility_rate": rate(rows, "k0_utility"), "k0_security_rate": rate(rows, "k0_security"),
        "kN_utility_rate": rate(rows, "kN_utility"), "kN_security_rate": rate(rows, "kN_security"),
        "per_suite": per_suite,
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

    out_json = args.out_json or "agentdojo_eval_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary saved to {out_json}")


if __name__ == "__main__":
    main()
