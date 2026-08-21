"""
run_pipeline.py

Phase 0~1 smoke test 진입점. 5070 Ti에서:

    python run_pipeline.py --model Qwen/Qwen2.5-0.5B-Instruct --family qwen2

로 먼저 극소형 모델로 shape/버그를 잡은 뒤, --model을 Qwen2.5-1.5B-Instruct 등으로
올려서 본 실험을 돌린다.

이 스크립트가 하는 일:
  1. dataset.py로 (internal, external) 예시 쌍 생성
  2. 각 예시에 대해 attn_relevance.py로 read / internal / external 세 종류의
     head relevance를 뽑아 데이터셋 전체에서 평균
  3. head_ranking.py로 top-K head 랭킹 + read/internal/external 겹침(Jaccard) 출력
     + functional map 이미지 저장
  4. edge_ablation.py로 (internal ∩ external) control head 후보를 knockout하며
     ASR proxy / read 정확도 proxy sweep 실행 및 출력
"""
import argparse
import datetime
import gc
import json
import os
import re
from typing import List

import torch

from dataset import build_phase0_batch, build_unseen_style_batch
from attn_relevance import load_model_for_relevance, compute_head_relevance
from head_ranking import (
    aggregate_scores,
    summarize_overlap,
    plot_functional_map,
    topk_heads,
    normalize_score,
    jaccard,
    random_heads,
    expected_jaccard_by_chance,
)
from edge_ablation import sweep_knockout
from runtime_env import add_runtime_args, describe, resolve_dtype, write_env_json


def _clean_ks(full_ks: List[int], n_heads: int) -> List[int]:
    """sweep_knockout에 넘길 ks 목록을 head 개수에 맞게 정리한다. n_heads보다 큰
    k는 어차피 전체 head를 다 쓰는 것과 같은 결과라 (review-2026-07-29.md 3-1의
    k=80==k=40 문제와 동일 패턴), n_heads로 캡하고 0/n_heads는 항상 포함시킨다."""
    return sorted({0, n_heads} | {k for k in full_ks if 0 < k < n_heads})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--family", default="qwen2", choices=["qwen2", "llama"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--four_bit", action="store_true")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--dataset_limit",
        type=int,
        default=None,
        help="템플릿 개수를 앞에서부터 제한 (Phase 0 smoke test용, 예: 2). 기본값은 전체(30개).",
    )
    parser.add_argument(
        "--heldout_style_idx",
        type=int,
        default=None,
        choices=range(5),
        help="P2-a held-out split: _INJECTION_STYLES(5종) 중 이 인덱스는 head 선정("
        "[2/4],[3/4])에서 완전히 배제하고, [4/4] knockout sweep에서 held-out 전용으로만 "
        "평가한다. 기본값 None이면 held-out 없이 5종 전부 사용(기존 동작).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="functional_map.png / summary.txt를 저장할 디렉토리. "
        "기본값은 results/<날짜>_<모델명>[_4bit]/ (실행마다 자동 생성, 덮어쓰기 방지).",
    )
    parser.add_argument(
        "--injecagent",
        action="store_true",
        help="P2-c: [2/4]/[3/4]에서 우리 데이터셋으로 찾은 control_heads_both를 그대로 써서, "
        "InjecAgent 외부 벤치마크(adapters/injecagent.py)에도 knockout sweep을 추가로 돌린다. "
        "사전에 `git clone https://github.com/uiuc-kang-lab/InjecAgent.git external_injecagent` 필요.",
    )
    parser.add_argument(
        "--injecagent_repo_dir", default="external_injecagent",
        help="InjecAgent 저장소를 clone한 경로 (--injecagent 사용 시).",
    )
    parser.add_argument(
        "--injecagent_limit", type=int, default=None,
        help="InjecAgent test case 개수를 앞에서부터 제한 (--injecagent 사용 시, 기본 전체).",
    )
    parser.add_argument(
        "--unseen_styles",
        action="store_true",
        help="P2-b: _INJECTION_STYLES_UNSEEN(다국어 혼용/코드블록 위장/유니코드 난독화/"
        "짧고 우회적인 표현, 4종 x 도메인 6종 = 24개)에도 control_heads_both로 knockout "
        "sweep을 추가로 돌린다. head 선정에는 절대 섞이지 않는 순수 평가 전용 데이터셋.",
    )
    parser.add_argument(
        "--injecagent_headsplit",
        action="store_true",
        help="P2-d: InjecAgent 일부(--injecagent_head_n개, 고정 seed로 무작위 분리)로 직접 "
        "control head를 찾아, 우리 합성 데이터셋의 control_heads_both와 교집합한 뒤, 나머지 "
        "InjecAgent case로 (a) 합성 단독 (b) InjecAgent 단독 (c) 교집합 knockout을 비교한다.",
    )
    parser.add_argument(
        "--injecagent_head_n", type=int, default=60,
        help="P2-d: head 선정에 쓸 InjecAgent case 개수 (나머지는 전부 평가용). 기본 60 — "
        "compute_head_relevance에 프로세스 내내 누적되는 GPU 메모리 버그(모델 재로드로도 "
        "안 없어짐, docs/todo.md 참고) 때문에 80쌍(=160회 호출) 근처에서 OOM 나는 걸 확인했으니 "
        "그보다 크게 잡지 말 것 (필요하면 배치별 서브프로세스 분리가 근본 해결책).",
    )
    parser.add_argument(
        "--injecagent_seed", type=int, default=42,
        help="P2-d: head/eval 분리에 쓰는 고정 seed (재현성).",
    )
    parser.add_argument(
        "--injecagent_headsplit_from",
        default=None,
        help="P2-d: 이전 실행이 저장한 p2d_heads.json 경로. 지정하면 (비용이 큰 backward-pass "
        "head 탐색을 다시 안 하고) 그 안의 head 목록 그대로 eval sweep만 새로 돌린다 — "
        "compute_head_relevance의 GPU 메모리 누수 때문에 head 탐색 직후 eval sweep이 물리 "
        "VRAM 거의 꽉 찬 상태에서 극도로 느려지는 문제를 우회하는 용도(새 프로세스로 재실행).",
    )
    add_runtime_args(parser)
    args = parser.parse_args()

    # dtype을 run_dir 이름보다 먼저 확정한다. 폴더명에 dtype을 넣어야 하기 때문 —
    # 안 넣으면 같은 날 같은 모델을 fp16/fp32로 각각 돌렸을 때 뒤엣것이 앞엣것을
    # 조용히 덮어쓴다 (P2-a의 --heldout_style_idx 덮어쓰기 버그와 같은 유형).
    # dtype이 다르면 수치가 다르므로, 이건 서로 다른 실행으로 보존돼야 한다.
    _torch_dtype, dtype_name = resolve_dtype(args.dtype, args.device)

    if args.out_dir:
        run_dir = args.out_dir
    else:
        model_slug = re.sub(r"[^A-Za-z0-9]+", "-", args.model).strip("-")
        suffix = "_4bit" if args.four_bit else ""
        suffix += f"_{dtype_name}"
        if args.heldout_style_idx is not None:
            suffix += f"_heldout{args.heldout_style_idx}"
        if args.unseen_styles:
            suffix += "_unseen"
        if args.injecagent_headsplit or args.injecagent_headsplit_from:
            suffix += "_headsplit"
        date_str = datetime.date.today().isoformat()
        run_dir = os.path.join("results", f"{date_str}_{model_slug}{suffix}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"[1/4] loading {args.model} (family={args.family}, four_bit={args.four_bit}) ...")
    # 위에서 이미 해결한 이름을 그대로 넘긴다 (idempotent) — "auto"를 다시 넘기면
    # 폴더명과 실제 dtype이 어긋날 여지가 생긴다.
    model, tok, dtype_name = load_model_for_relevance(
        model_path=args.model, four_bit=args.four_bit, device=args.device,
        model_family=args.family, dtype=dtype_name,
    )
    print(describe(dtype_name))
    # 실행 환경을 결과 폴더에 남긴다 — 모델 로드 직후에 써서, 뒤에서 OOM으로 죽어도
    # "무슨 환경에서 어디까지 갔는지"는 남게 한다 (docs/plan-2026-08-26.md 1.3절)
    write_env_json(run_dir, dtype_name, extra={"stage": "run_pipeline", "run_dir": run_dir})

    head_style_indices = None
    if args.heldout_style_idx is not None:
        head_style_indices = [i for i in range(5) if i != args.heldout_style_idx]
        print(
            f"[P2-a] held-out split: style_idx={args.heldout_style_idx} excluded from "
            f"head selection, styles={head_style_indices} used instead."
        )

    print("[2/4] building synthetic IPI dataset ...")
    pairs = build_phase0_batch(
        tok, device=args.device, limit=args.dataset_limit, style_indices=head_style_indices
    )

    read_scores_list, internal_scores_list, external_scores_list = [], [], []

    for ex in pairs:
        # read: 정상 컨텐츠(D_benign)를 읽는지. 주입문이 없는 clean 프롬프트에서 재서
        # read head의 baseline이 공격에 오염되지 않게 한다.
        read_ex = ex["read_clean"]
        read_groups = compute_head_relevance(
            model, read_ex.input_ids, read_ex.read_target,
            key_spans={"data_benign": read_ex.spans["data_benign"]},
        )
        read_scores_list.append(read_groups)

        # internal: 자유 텍스트 답변이 D_inj의 영향을 받는지 -> target=exec_target
        internal_ex = ex["internal"]
        internal_groups = compute_head_relevance(
            model, internal_ex.input_ids, internal_ex.exec_target,
            key_spans={"data_inj": internal_ex.spans["data_inj"]},
        )
        internal_scores_list.append(internal_groups)

        # external: tool_call 인자가 D_inj의 영향을 받는지 -> target=exec_target
        external_ex = ex["external"]
        external_groups = compute_head_relevance(
            model, external_ex.input_ids, external_ex.exec_target,
            key_spans={"data_inj": external_ex.spans["data_inj"]},
        )
        external_scores_list.append(external_groups)

    read_score = aggregate_scores(read_scores_list, "data_benign")
    internal_score = aggregate_scores(internal_scores_list, "data_inj")
    external_score = aggregate_scores(external_scores_list, "data_inj")

    print("[3/4] ranking heads & computing overlap ...")
    summary = summarize_overlap(read_score, internal_score, external_score, k=args.topk)
    print(f"  top-{args.topk} Jaccard(read, internal)   = {summary['jaccard_read_internal']:.3f}")
    print(f"  top-{args.topk} Jaccard(read, external)   = {summary['jaccard_read_external']:.3f}")
    print(f"  top-{args.topk} Jaccard(internal,external)= {summary['jaccard_internal_external']:.3f}")
    print(f"  control_heads_both (internal ∩ external)  = {summary['control_heads_both']}")
    print(
        f"  jaccard_chance_at_k={args.topk} (우연 기대값)     = {summary['jaccard_chance_at_k']:.3f}"
        f"  (internal,external 관측값 대비 {summary['jaccard_internal_external'] / summary['jaccard_chance_at_k']:.1f}배)"
        if summary["jaccard_chance_at_k"] > 0 else ""
    )
    print(f"  dual_use_candidates (read와도 겹치는 control head) = {summary['dual_use_candidates']}")

    plot_path = plot_functional_map(
        read_score, internal_score, external_score, top_k=args.topk,
        save_path=os.path.join(run_dir, "functional_map.png"),
    )
    print(f"  functional map saved to {plot_path}")

    print("[4/4] edge-knockout sweep on control head candidates ...")
    if args.family == "qwen2":
        from transformers.models.qwen2 import modeling_qwen2 as modeling_mod
    else:
        from transformers.models.llama import modeling_llama as modeling_mod

    # internal/external은 프롬프트 길이·타깃이 달라 relevance 절대 스케일이 다르다.
    # 각각 정규화한 뒤 더해야 한쪽이 랭킹을 독점하지 않는다.
    combined = normalize_score(internal_score) + normalize_score(external_score)
    control_ranking: List = topk_heads(combined, args.topk * 2)

    # 예시 하나가 아니라 데이터셋 전체 평균으로 knockout 효과를 측정.
    # utility(read) 축은 끊을 D_inj 엣지가 있어야 하므로 read_injected를 쓴다.
    exec_examples = [ex["external"] for ex in pairs]
    read_examples = [ex["read_injected"] for ex in pairs]
    sweep_results = sweep_knockout(
        model, modeling_mod, exec_examples, read_examples, control_ranking,
    )
    for row in sweep_results:
        print(
            f"  k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
            f"  read_token_prob={row['read_token_prob']:.4f}"
            f"  (n={row['n_examples']})"
        )

    # [P7] 랜덤 head 기준선: control_ranking(top-40, union 기반)과 같은 개수의 무작위
    # head를 knockout해서 같은 sweep을 돌린다. review-2026-07-29.md 3-4/3-5 —
    # 똑같이 떨어지면 head 선정이 기여하는 바가 없다는 뜻이고, 안 떨어지면 지금까지의
    # 결론이 튼튼해진다. forward-only라 비용이 거의 없음.
    num_layers, num_heads_per_layer = internal_score.shape
    print(f"[P7] random-head baseline sweep (control_ranking과 동일 개수={len(control_ranking)}) ...")
    random_ranking = random_heads(num_layers, num_heads_per_layer, len(control_ranking), seed=42)
    random_sweep_results = sweep_knockout(
        model, modeling_mod, exec_examples, read_examples, random_ranking,
    )
    for row in random_sweep_results:
        print(
            f"  [random] k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
            f"  read_token_prob={row['read_token_prob']:.4f}"
            f"  (n={row['n_examples']})"
        )

    # [P7] control_heads_both(internal ∩ external, 교집합) sweep. review-2026-07-29.md
    # 3-1 — methodology.md/발표자료는 "control head = 교집합"이라 서술하지만 위
    # sweep_results는 실제로 top-40 union 랭킹을 쓴다. 서술과 실제 개입 대상을
    # 코드는 그대로 두고(사용자 결정) 교집합 sweep을 별도로 돌려 나란히 비교한다.
    intersection_ranking = summary["control_heads_both"]
    intersection_ks = _clean_ks([5, 10, 20, 40], len(intersection_ranking))
    print(
        f"[P7] control_heads_both(교집합, n={len(intersection_ranking)}) sweep "
        f"(union top-40 sweep과 나란히 비교용) ..."
    )
    intersection_sweep_results = sweep_knockout(
        model, modeling_mod, exec_examples, read_examples, intersection_ranking, ks=intersection_ks,
    )
    for row in intersection_sweep_results:
        print(
            f"  [intersection] k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
            f"  read_token_prob={row['read_token_prob']:.4f}"
            f"  (n={row['n_examples']})"
        )
    intersection_random_ranking = random_heads(
        num_layers, num_heads_per_layer, len(intersection_ranking), seed=42
    )
    intersection_random_sweep_results = sweep_knockout(
        model, modeling_mod, exec_examples, read_examples, intersection_random_ranking, ks=intersection_ks,
    )
    for row in intersection_random_sweep_results:
        print(
            f"  [intersection-random] k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
            f"  read_token_prob={row['read_token_prob']:.4f}"
            f"  (n={row['n_examples']})"
        )

    # [P7] top-K sweep (K=5/10/20/40): K를 바꿔가며 internal/external을 다시 top-K로
    # 뽑고 그 교집합 크기·jaccard·knockout 효과가 어떻게 변하는지 본다 — "왜 K=20인가"
    # 에 대한 근거(review-2026-07-29.md 3-2, todo.md P7 항목2). 각 K의 교집합 크기만큼
    # 무작위 head를 뽑은 기준선도 같이 비교한다.
    print("[P7] top-K sensitivity sweep (K=5/10/20/40) ...")
    topk_sweep_results = []
    for K in (5, 10, 20, 40):
        internal_heads_K = topk_heads(internal_score, K)
        external_heads_K = topk_heads(external_score, K)
        heads_K = sorted(set(internal_heads_K) & set(external_heads_K))
        jacc_K = jaccard(internal_heads_K, external_heads_K)
        chance_K = expected_jaccard_by_chance(K, num_layers, num_heads_per_layer)

        real_res = sweep_knockout(
            model, modeling_mod, exec_examples, read_examples, heads_K,
            ks=[0, len(heads_K)],
        )
        random_heads_K = random_heads(num_layers, num_heads_per_layer, len(heads_K), seed=42)
        random_res = sweep_knockout(
            model, modeling_mod, exec_examples, read_examples, random_heads_K,
            ks=[0, len(heads_K)],
        )
        row = {
            "K": K, "n_intersection": len(heads_K), "jaccard": jacc_K, "jaccard_chance": chance_K,
            "heads": heads_K, "real": real_res, "random": random_res,
        }
        topk_sweep_results.append(row)
        print(
            f"  K={K:>3}  |intersection|={len(heads_K):>3}  jaccard={jacc_K:.3f} (chance={chance_K:.3f})  "
            f"real: k=0 {real_res[0]['malicious_token_prob']:.4f} -> k={real_res[-1]['k']} {real_res[-1]['malicious_token_prob']:.4f}  "
            f"random: k=0 {random_res[0]['malicious_token_prob']:.4f} -> k={random_res[-1]['k']} {random_res[-1]['malicious_token_prob']:.4f}"
        )

    heldout_sweep_results = None
    if args.heldout_style_idx is not None:
        print(
            f"[P2-a] building held-out dataset (style_idx={args.heldout_style_idx} only) "
            "for out-of-distribution knockout eval ..."
        )
        heldout_pairs = build_phase0_batch(
            tok, device=args.device, style_indices=[args.heldout_style_idx]
        )
        heldout_exec_examples = [ex["external"] for ex in heldout_pairs]
        heldout_read_examples = [ex["read_injected"] for ex in heldout_pairs]
        heldout_sweep_results = sweep_knockout(
            model, modeling_mod, heldout_exec_examples, heldout_read_examples, control_ranking,
        )
        for row in heldout_sweep_results:
            print(
                f"  [held-out] k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
                f"  read_token_prob={row['read_token_prob']:.4f}"
                f"  (n={row['n_examples']})"
            )

    injecagent_sweep_results = None
    if args.injecagent:
        from adapters.injecagent import build_injecagent_batch

        print(f"[P2-c] building InjecAgent dataset from {args.injecagent_repo_dir} ...")
        injecagent_examples = build_injecagent_batch(
            tok, device=args.device, repo_dir=args.injecagent_repo_dir, limit=args.injecagent_limit,
        )
        print(f"  {len(injecagent_examples)} InjecAgent examples loaded.")
        injecagent_sweep_results = sweep_knockout(
            model, modeling_mod, injecagent_examples, injecagent_examples, control_ranking,
        )
        for row in injecagent_sweep_results:
            print(
                f"  [InjecAgent] k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
                f"  read_token_prob={row['read_token_prob']:.4f}"
                f"  (n={row['n_examples']})"
            )

    unseen_sweep_results = None
    if args.unseen_styles:
        print("[P2-b] building unseen-style dataset (_INJECTION_STYLES_UNSEEN) ...")
        unseen_pairs = build_unseen_style_batch(tok, device=args.device)
        unseen_exec_examples = [ex["external"] for ex in unseen_pairs]
        unseen_read_examples = [ex["read_injected"] for ex in unseen_pairs]
        unseen_sweep_results = sweep_knockout(
            model, modeling_mod, unseen_exec_examples, unseen_read_examples, control_ranking,
        )
        for row in unseen_sweep_results:
            print(
                f"  [unseen-style] k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
                f"  read_token_prob={row['read_token_prob']:.4f}"
                f"  (n={row['n_examples']})"
            )

    headsplit_results = None
    if args.injecagent_headsplit or args.injecagent_headsplit_from:
        from adapters.injecagent import build_injecagent_pairs, split_pairs

        if args.injecagent_headsplit_from:
            # 비싼 backward-pass head 탐색은 건너뛰고, 이전 실행이 저장해둔 head 목록만
            # 새 프로세스에서 불러와 eval sweep만 돌린다 (아래 메모리 누수 문제 우회).
            print(f"[P2-d] loading pre-computed heads from {args.injecagent_headsplit_from} ...")
            with open(args.injecagent_headsplit_from, encoding="utf-8") as f:
                loaded = json.load(f)
            head_n, seed = loaded["head_n"], loaded["seed"]
            synthetic_heads = [tuple(h) for h in loaded["synthetic_heads"]]
            ia_heads = [tuple(h) for h in loaded["ia_heads"]]
            intersection_heads = [tuple(h) for h in loaded["intersection_heads"]]

            print(f"[P2-d] building InjecAgent pairs from {args.injecagent_repo_dir} ...")
            ia_pairs = build_injecagent_pairs(tok, device=args.device, repo_dir=args.injecagent_repo_dir)
            _, eval_pairs = split_pairs(ia_pairs, head_n=head_n, seed=seed)
            print(
                f"  {len(ia_pairs)} total pairs -> {head_n} were used for head discovery "
                f"(seed={seed}, loaded from file), {len(eval_pairs)} for eval."
            )
            print(f"  synthetic_heads (loaded) = {synthetic_heads}")
            print(f"  ia_heads (loaded) = {ia_heads}")
            print(f"  intersection_heads (loaded) = {intersection_heads}")
        else:
            print(f"[P2-d] building InjecAgent pairs from {args.injecagent_repo_dir} ...")
            ia_pairs = build_injecagent_pairs(tok, device=args.device, repo_dir=args.injecagent_repo_dir)
            head_n, seed = args.injecagent_head_n, args.injecagent_seed
            head_pairs, eval_pairs = split_pairs(ia_pairs, head_n=head_n, seed=seed)
            print(
                f"  {len(ia_pairs)} total pairs -> {len(head_pairs)} for head discovery "
                f"(seed={seed}), {len(eval_pairs)} for eval."
            )

            # head_pairs로만 relevance 계산 (eval_pairs는 이 단계에서 전혀 안 봄 — 진짜 held-out).
            # ⚠️ 알려진 한계: compute_head_relevance 호출 1번당 ~0.12GB씩 GPU 메모리가 계속
            # 누적되는 버그가 있다 (gc.collect()/empty_cache()는 물론, 모델을 통째로 다시 로드해도
            # 전혀 줄지 않음 — 실험으로 확인함, [2/4]의 90번 호출에서도 같은 비율로 이미 새고 있었음).
            # 모델 인스턴스 문제가 아니라 프로세스 전역/CUDA 드라이버 레벨로 보이므로, 이 프로세스
            # 안에서는 고칠 방법이 없다 — head_n을 총 호출 수(이 프로세스에서 이미 쓴 [2/4]의 90번+
            # head_n*2)가 대략 300~350을 넘지 않는 선으로 잡아야 안전하다 (실측상 약 80쌍=160번
            # 부근에서 OOM). 완전히 고치려면 배치마다 별도 프로세스(새 CUDA 컨텍스트)로 나눠 돌려야
            # 하는데, docs/todo.md/메모리에 향후 작업으로 기록해둠 — 지금은 head_n을 안전 범위로 제한.
            print("  [P2-d] starting head-discovery relevance loop ...")
            ia_read_scores_list, ia_agent_scores_list = [], []
            for i, pair in enumerate(head_pairs):
                clean_ex = pair["clean"]
                ia_read_scores_list.append(
                    compute_head_relevance(
                        model, clean_ex.input_ids, clean_ex.read_target,
                        key_spans={"data_benign": clean_ex.spans["data_benign"]},
                    )
                )
                inj_ex = pair["injected"]
                ia_agent_scores_list.append(
                    compute_head_relevance(
                        model, inj_ex.input_ids, inj_ex.exec_target,
                        key_spans={"data_inj": inj_ex.spans["data_inj"]},
                    )
                )
                gc.collect()
                torch.cuda.empty_cache()

                if (i + 1) % 20 == 0:
                    alloc = torch.cuda.memory_allocated() / (1024**3)
                    reserv = torch.cuda.memory_reserved() / (1024**3)
                    print(
                        f"  [P2-d] head relevance {i + 1}/{len(head_pairs)} "
                        f"(cuda allocated={alloc:.2f}GiB reserved={reserv:.2f}GiB) ..."
                    )

            # relevance loop가 끝나면 더 이상 새로운 backward call이 없으므로, 여기서 한 번 더
            # 세게 정리해본다 — 하지만 실측 결과 이 정리도 전혀 효과가 없었다 (정리 전/후 GPU
            # 메모리가 완전히 동일). 즉 이 메모리는 프로세스가 죽을 때까지 회수가 안 되므로,
            # eval sweep을 이어서 같은 프로세스에서 돌리면 물리 VRAM이 거의 꽉 찬 상태로 시작해
            # 스와핑/스래싱으로 극도로 느려진다. 그래서 head 목록을 json으로 저장해두고,
            # `--injecagent_headsplit_from`으로 새 프로세스에서 eval sweep만 깨끗하게 다시
            # 돌리는 걸 권장한다 (아래 저장 로직).
            print("  [P2-d] head-discovery loop done, attempting cleanup (known to be ineffective, see comment above) ...")
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserv = torch.cuda.memory_reserved() / (1024**3)
            print(f"  [P2-d] post-cleanup GPU memory: allocated={alloc:.2f}GiB reserved={reserv:.2f}GiB")

            ia_read_score = aggregate_scores(ia_read_scores_list, "data_benign")
            ia_agent_score = aggregate_scores(ia_agent_scores_list, "data_inj")
            # InjecAgent는 우리 internal/external 같은 두 채널이 없고 tool-call 한 채널뿐이라,
            # control head 후보는 이 채널 하나에서 바로 top-k를 뽑는다 (교집합 불필요).
            ia_heads = topk_heads(ia_agent_score, args.topk)
            synthetic_heads = summary["control_heads_both"]
            intersection_heads = sorted(set(synthetic_heads) & set(ia_heads))

            print(f"  InjecAgent-derived top-{args.topk} heads = {ia_heads}")
            print(
                f"  jaccard(read_ia, agent_ia) = "
                f"{jaccard(topk_heads(ia_read_score, args.topk), ia_heads):.3f}"
            )
            print(f"  intersection(synthetic control_heads_both, InjecAgent heads) = {intersection_heads}")

            heads_json_path = os.path.join(run_dir, "p2d_heads.json")
            with open(heads_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "head_n": head_n, "seed": seed, "topk": args.topk,
                        "synthetic_heads": synthetic_heads, "ia_heads": ia_heads,
                        "intersection_heads": intersection_heads,
                    },
                    f, indent=2,
                )
            print(
                f"  [P2-d] heads saved to {heads_json_path} "
                "(rerun with --injecagent_headsplit_from <이 경로>로 eval sweep만 새 프로세스에서 재실행 가능)"
            )

        eval_examples = [p["injected"] for p in eval_pairs]

        def _headsplit_point(name, heads):
            print(f"  [P2-d] running '{name}' sweep over {len(eval_examples)} eval examples ({len(heads)} heads) ...")
            res = sweep_knockout(
                model, modeling_mod, eval_examples, eval_examples, heads, ks=[0, len(heads)],
            )
            for row in res:
                print(
                    f"  [P2-d:{name}] k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
                    f"  read_token_prob={row['read_token_prob']:.4f}"
                    f"  (n={row['n_examples']})"
                )
            return res

        headsplit_results = {
            "n_head": head_n,
            "seed": seed,
            "n_eval": len(eval_pairs),
            "ia_heads": ia_heads,
            "synthetic_heads": synthetic_heads,
            "intersection_heads": intersection_heads,
            "synthetic_only": _headsplit_point("synthetic-only", synthetic_heads),
            "injecagent_only": _headsplit_point("injecagent-only", ia_heads),
            "intersection": _headsplit_point("intersection", intersection_heads),
        }

    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"model={args.model} family={args.family} four_bit={args.four_bit} "
                 f"dtype={dtype_name} "
                 f"topk={args.topk} dataset_limit={args.dataset_limit} "
                 f"heldout_style_idx={args.heldout_style_idx}\n")
        # dtype이 다르면 수치를 직접 비교하면 안 된다. 자세한 환경은 같은 폴더의 env.json.
        f.write("(전체 실행 환경: 같은 폴더의 env.json — commit/dtype/GPU/패키지 버전)\n\n")
        f.write(f"jaccard(read,internal)   = {summary['jaccard_read_internal']:.3f}\n")
        f.write(f"jaccard(read,external)   = {summary['jaccard_read_external']:.3f}\n")
        f.write(f"jaccard(internal,external)= {summary['jaccard_internal_external']:.3f}\n")
        f.write(f"jaccard_chance_at_k={args.topk}  = {summary['jaccard_chance_at_k']:.3f}\n")
        f.write(f"control_heads_both = {summary['control_heads_both']}\n")
        f.write(f"dual_use_candidates = {summary['dual_use_candidates']}\n\n")
        if args.heldout_style_idx is not None:
            f.write(f"-- in-distribution (styles={head_style_indices}, used for head selection) --\n")
        for row in sweep_results:
            f.write(
                f"k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
                f"  read_token_prob={row['read_token_prob']:.4f}"
                f"  (n={row['n_examples']})\n"
            )

        f.write(f"\n-- [P7] random-head baseline (n={len(control_ranking)}, same size as control_ranking) --\n")
        for row in random_sweep_results:
            f.write(
                f"k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
                f"  read_token_prob={row['read_token_prob']:.4f}"
                f"  (n={row['n_examples']})\n"
            )

        f.write(f"\n-- [P7] control_heads_both intersection sweep (n={len(intersection_ranking)}) --\n")
        f.write(f"heads = {intersection_ranking}\n")
        for row in intersection_sweep_results:
            f.write(
                f"k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
                f"  read_token_prob={row['read_token_prob']:.4f}"
                f"  (n={row['n_examples']})\n"
            )
        f.write(f"\n-- [P7] control_heads_both intersection sweep - random baseline (n={len(intersection_ranking)}) --\n")
        for row in intersection_random_sweep_results:
            f.write(
                f"k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
                f"  read_token_prob={row['read_token_prob']:.4f}"
                f"  (n={row['n_examples']})\n"
            )

        f.write("\n-- [P7] top-K sensitivity sweep (K=5/10/20/40) --\n")
        for row in topk_sweep_results:
            f.write(
                f"K={row['K']:>3}  |intersection|={row['n_intersection']:>3}  "
                f"jaccard={row['jaccard']:.3f}  jaccard_chance={row['jaccard_chance']:.3f}\n"
            )
            f.write(f"  heads = {row['heads']}\n")
            f.write(
                f"  real:   k=0 malicious={row['real'][0]['malicious_token_prob']:.4f} read={row['real'][0]['read_token_prob']:.4f}"
                f"  ->  k={row['real'][-1]['k']} malicious={row['real'][-1]['malicious_token_prob']:.4f} read={row['real'][-1]['read_token_prob']:.4f}\n"
            )
            f.write(
                f"  random: k=0 malicious={row['random'][0]['malicious_token_prob']:.4f} read={row['random'][0]['read_token_prob']:.4f}"
                f"  ->  k={row['random'][-1]['k']} malicious={row['random'][-1]['malicious_token_prob']:.4f} read={row['random'][-1]['read_token_prob']:.4f}\n"
            )

        if heldout_sweep_results is not None:
            f.write(f"\n-- held-out (style_idx={args.heldout_style_idx}, excluded from head selection) --\n")
            for row in heldout_sweep_results:
                f.write(
                    f"k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
                    f"  read_token_prob={row['read_token_prob']:.4f}"
                    f"  (n={row['n_examples']})\n"
                )
        if injecagent_sweep_results is not None:
            f.write(f"\n-- InjecAgent external benchmark (n={len(injecagent_examples)} test cases) --\n")
            for row in injecagent_sweep_results:
                f.write(
                    f"k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
                    f"  read_token_prob={row['read_token_prob']:.4f}"
                    f"  (n={row['n_examples']})\n"
                )
        if unseen_sweep_results is not None:
            f.write(f"\n-- unseen-style (P2-b, _INJECTION_STYLES_UNSEEN, excluded from head selection) --\n")
            for row in unseen_sweep_results:
                f.write(
                    f"k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
                    f"  read_token_prob={row['read_token_prob']:.4f}"
                    f"  (n={row['n_examples']})\n"
                )
        if headsplit_results is not None:
            f.write(
                f"\n-- P2-d InjecAgent head-split (n_head={headsplit_results['n_head']}, "
                f"n_eval={headsplit_results['n_eval']}, seed={headsplit_results['seed']}) --\n"
            )
            f.write(f"synthetic_heads (from our dataset) = {headsplit_results['synthetic_heads']}\n")
            f.write(f"injecagent_heads (from {headsplit_results['n_head']} InjecAgent cases) = {headsplit_results['ia_heads']}\n")
            f.write(f"intersection = {headsplit_results['intersection_heads']}\n\n")
            for label in ("synthetic_only", "injecagent_only", "intersection"):
                f.write(f"[{label}]\n")
                for row in headsplit_results[label]:
                    f.write(
                        f"k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
                        f"  read_token_prob={row['read_token_prob']:.4f}"
                        f"  (n={row['n_examples']})\n"
                    )
    print(f"  summary saved to {summary_path}")


if __name__ == "__main__":
    main()
