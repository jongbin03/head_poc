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
from typing import List

import torch

from dataset import build_phase0_batch
from attn_relevance import load_model_for_relevance, compute_head_relevance
from head_ranking import aggregate_scores, summarize_overlap, plot_functional_map, topk_heads
from edge_ablation import sweep_knockout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--family", default="qwen2", choices=["qwen2", "llama"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--four_bit", action="store_true")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    print(f"[1/4] loading {args.model} (family={args.family}, four_bit={args.four_bit}) ...")
    model, tok = load_model_for_relevance(
        model_path=args.model, four_bit=args.four_bit, device=args.device, model_family=args.family
    )

    print("[2/4] building synthetic IPI dataset ...")
    pairs = build_phase0_batch(tok, device=args.device)

    read_scores_list, internal_scores_list, external_scores_list = [], [], []

    for internal_ex, external_ex in pairs:
        # read: 정상 컨텐츠(D_benign)를 읽는지 -> target=read_target, 관심 group='data_benign'
        read_groups = compute_head_relevance(
            model, internal_ex.input_ids, internal_ex.read_target,
            key_spans={"data_benign": internal_ex.spans["data_benign"]},
        )
        read_scores_list.append(read_groups)

        # internal: 자유 텍스트 답변이 D_inj의 영향을 받는지 -> target=exec_target
        internal_groups = compute_head_relevance(
            model, internal_ex.input_ids, internal_ex.exec_target,
            key_spans={"data_inj": internal_ex.spans["data_inj"]},
        )
        internal_scores_list.append(internal_groups)

        # external: tool_call 인자가 D_inj의 영향을 받는지 -> target=exec_target
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

    plot_path = plot_functional_map(read_score, internal_score, external_score, top_k=args.topk)
    print(f"  functional map saved to {plot_path}")

    print("[4/4] edge-knockout sweep on control head candidates ...")
    if args.family == "qwen2":
        from transformers.models.qwen2 import modeling_qwen2 as modeling_mod
    else:
        from transformers.models.llama import modeling_llama as modeling_mod

    control_ranking: List = topk_heads(internal_score + external_score, args.topk * 2)
    internal_ex0, external_ex0 = pairs[0]
    sweep_results = sweep_knockout(
        model, modeling_mod, external_ex0, internal_ex0, control_ranking,
    )
    for row in sweep_results:
        print(
            f"  k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
            f"  read_token_prob={row['read_token_prob']:.4f}"
        )


if __name__ == "__main__":
    main()
