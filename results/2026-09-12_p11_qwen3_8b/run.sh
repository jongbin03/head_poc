#!/usr/bin/env bash
# Qwen3-8B 세대 축 교차검증 — P11 / feedback-2026-09-06.md §1 (다음주 태스크)
# 진단(tools/diag_qwen3_relevance.py, 2026-09-12) 통과 후 착수:
#   position 0 비중 qwen3 16.71% vs qwen2 대조군 0.49% (~34배, 실재하는 쏠림)
#   data_inj span 비중 qwen3 32.78% vs qwen2 37.71% (비슷한 수준, 여전히 최상위 신호)
#   -> group-sum 방식(span 단위)이라 position 0은 head 탐색 합산에 안 들어감 -> 진행.
# 대조군(Qwen2.5-7B-Instruct bf16, results/2026-08-31_p16_aisecking/qwen7b_agentdojo_default.json류)과
# 나란히 비교. 단일 GPU로 충분 (8B bf16) — device_map_plan 불필요.
set -uo pipefail
cd ~/head_poc
source env.sh >/dev/null 2>&1

MODEL=Qwen/Qwen3-8B
OUT=results/2026-09-12_p11_qwen3_8b
GPU=0   # gpu_free로 매번 재확인! (70B trackA 작업과 겹치지 않는 카드로)

case "${1:-}" in

smoke)
  CUDA_VISIBLE_DEVICES=$GPU python compare_head_sources.py discover-parallel \
    --source agentdojo --model "$MODEL" --family qwen3 \
    --dtype bf16 \
    --head_n 20 --max_seq_len 1200 --batch_size 5 \
    --out_json "$OUT/heads_agentdojo_SMOKE.json" 2>&1 | tee "$OUT/console_smoke.log"
  ;;

trackA)
  # bf16 (8B는 4bit 불필요, Llama-8B 탐색과 동일 조건). head_n 200 = 8B 탐색과 동일.
  CUDA_VISIBLE_DEVICES=$GPU python compare_head_sources.py discover-parallel \
    --source agentdojo --model "$MODEL" --family qwen3 \
    --dtype bf16 \
    --head_n 200 --max_seq_len 1200 --batch_size 5 \
    --out_json "$OUT/heads_agentdojo.json" 2>&1 | tee "$OUT/console_trackA.log"
  ;;

eval-heldout)
  CUDA_VISIBLE_DEVICES=$GPU python run_agentdojo_eval.py \
    --model "$MODEL" --family qwen3 \
    --tool_call_format agentdojo_default \
    --heads_json "$OUT/heads_agentdojo.json" \
    --suite slack --eval_split heldout --limit_pairs 60 \
    --out_json "$OUT/eval_slack_heldout.json" 2>&1 | tee "$OUT/console_eval_heldout.log"
  ;;

eval-heldout-tk)
  CUDA_VISIBLE_DEVICES=$GPU python run_agentdojo_eval.py \
    --model "$MODEL" --family qwen3 --attack tool_knowledge \
    --tool_call_format agentdojo_default \
    --heads_json "$OUT/heads_agentdojo.json" \
    --suite slack --eval_split heldout --limit_pairs 60 \
    --out_json "$OUT/eval_slack_heldout_tk.json" 2>&1 | tee "$OUT/console_eval_heldout_tk.log"
  ;;

eval-all)
  CUDA_VISIBLE_DEVICES=$GPU python run_agentdojo_eval.py \
    --model "$MODEL" --family qwen3 \
    --tool_call_format agentdojo_default \
    --heads_json "$OUT/heads_agentdojo.json" \
    --suite slack --eval_split all --limit_pairs 200 \
    --out_json "$OUT/eval_slack_all.json" 2>&1 | tee "$OUT/console_eval_all.log"
  ;;

*)
  echo "usage: $0 {smoke|trackA|eval-heldout|eval-heldout-tk|eval-all}"
  exit 1
  ;;
esac
