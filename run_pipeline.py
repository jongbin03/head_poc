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

from dataset import build_phase0_batch
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
        "--out_dir",
        default=None,
        help="functional_map.png / summary.txt를 저장할 디렉토리. "
        "기본값은 results/<날짜>_<모델명>[_4bit]/ (실행마다 자동 생성, 덮어쓰기 방지).",
    )
    args = parser.parse_args()

    if args.out_dir:
        run_dir = args.out_dir
    else:
        model_slug = re.sub(r"[^A-Za-z0-9]+", "-", args.model).strip("-")
        suffix = "_4bit" if args.four_bit else ""
        date_str = datetime.date.today().isoformat()
        run_dir = os.path.join("results", f"{date_str}_{model_slug}{suffix}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"[1/4] loading {args.model} (family={args.family}, four_bit={args.four_bit}) ...")
    model, tok = load_model_for_relevance(
        model_path=args.model, four_bit=args.four_bit, device=args.device, model_family=args.family
    )

    print("[2/4] building synthetic IPI dataset ...")
    pairs = build_phase0_batch(tok, device=args.device, limit=args.dataset_limit)

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

    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"model={args.model} family={args.family} four_bit={args.four_bit} "
                 f"topk={args.topk} dataset_limit={args.dataset_limit}\n\n")
        f.write(f"jaccard(read,internal)   = {summary['jaccard_read_internal']:.3f}\n")
        f.write(f"jaccard(read,external)   = {summary['jaccard_read_external']:.3f}\n")
        f.write(f"jaccard(internal,external)= {summary['jaccard_internal_external']:.3f}\n")
        f.write(f"control_heads_both = {summary['control_heads_both']}\n\n")
        for row in sweep_results:
            f.write(
                f"k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
                f"  read_token_prob={row['read_token_prob']:.4f}"
                f"  (n={row['n_examples']})\n"
            )
    print(f"  summary saved to {summary_path}")


if __name__ == "__main__":
    main()
