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
)
from edge_ablation import sweep_knockout


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
    args = parser.parse_args()

    if args.out_dir:
        run_dir = args.out_dir
    else:
        model_slug = re.sub(r"[^A-Za-z0-9]+", "-", args.model).strip("-")
        suffix = "_4bit" if args.four_bit else ""
        if args.heldout_style_idx is not None:
            suffix += f"_heldout{args.heldout_style_idx}"
        if args.unseen_styles:
            suffix += "_unseen"
        date_str = datetime.date.today().isoformat()
        run_dir = os.path.join("results", f"{date_str}_{model_slug}{suffix}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"[1/4] loading {args.model} (family={args.family}, four_bit={args.four_bit}) ...")
    model, tok = load_model_for_relevance(
        model_path=args.model, four_bit=args.four_bit, device=args.device, model_family=args.family
    )

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

    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"model={args.model} family={args.family} four_bit={args.four_bit} "
                 f"topk={args.topk} dataset_limit={args.dataset_limit} "
                 f"heldout_style_idx={args.heldout_style_idx}\n\n")
        f.write(f"jaccard(read,internal)   = {summary['jaccard_read_internal']:.3f}\n")
        f.write(f"jaccard(read,external)   = {summary['jaccard_read_external']:.3f}\n")
        f.write(f"jaccard(internal,external)= {summary['jaccard_internal_external']:.3f}\n")
        f.write(f"control_heads_both = {summary['control_heads_both']}\n\n")
        if args.heldout_style_idx is not None:
            f.write(f"-- in-distribution (styles={head_style_indices}, used for head selection) --\n")
        for row in sweep_results:
            f.write(
                f"k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
                f"  read_token_prob={row['read_token_prob']:.4f}"
                f"  (n={row['n_examples']})\n"
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
    print(f"  summary saved to {summary_path}")


if __name__ == "__main__":
    main()
