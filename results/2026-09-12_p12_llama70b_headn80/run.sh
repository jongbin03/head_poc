#!/usr/bin/env bash
# Llama-3.1-70B Track A 재탐색(head_n 80) — 4-3 / status-2026-09-09.md §4-3
# 목적: head_n=200(기존)이 slack heldout을 15쌍으로 깎았음 -> head_n 낮춰 heldout 확대.
# 실측(2026-09-12): quota=26/suite, slack head_groups=7/20 -> slack eval(heldout pool)=48쌍
#   (head_n=200 대비 15->48, 3.2배).
# set -e는 쓰지 않는다(source env.sh + tee pipefail 조합 이력, 2026-09-08).
set -uo pipefail
cd ~/head_poc
source env.sh >/dev/null 2>&1

MODEL=meta-llama/Llama-3.1-70B-Instruct
OUT=results/2026-09-12_p12_llama70b_headn80
A6000=1   # gpu_free로 매번 재확인!

case "${1:-}" in

trackA)
  # 본 탐색. head_n=80(기존 200의 40%), 나머지 배선은 2026-09-08/09와 동일.
  CUDA_VISIBLE_DEVICES=1,0,2 python compare_head_sources.py discover-parallel \
    --source agentdojo --model "$MODEL" --family llama \
    --four_bit --dtype bf16 \
    --device cuda:0 --device_map_plan 0:32,1:30,2:18 \
    --head_n 80 --max_seq_len 1000 --batch_size 4 \
    --out_json "$OUT/heads_agentdojo.json" 2>&1 | tee "$OUT/console_trackA.log"
  ;;

eval-heldout)
  # important_instructions, 확대된 heldout(48쌍 풀, run_agentdojo_eval 자체 후보 계산은
  # suite 전체 - head_groups 방식이라 실제 n은 다를 수 있음 -> limit_pairs 여유있게).
  CUDA_VISIBLE_DEVICES=$A6000 python run_agentdojo_eval.py \
    --model "$MODEL" --family llama --four_bit \
    --tool_call_format agentdojo_default \
    --heads_json "$OUT/heads_agentdojo.json" \
    --suite slack --eval_split heldout --limit_pairs 60 \
    --out_json "$OUT/eval_slack_heldout.json" 2>&1 | tee "$OUT/console_eval_heldout.log"
  ;;

eval-heldout-tk)
  # tool_knowledge 공격축 교차확인.
  CUDA_VISIBLE_DEVICES=$A6000 python run_agentdojo_eval.py \
    --model "$MODEL" --family llama --four_bit --attack tool_knowledge \
    --tool_call_format agentdojo_default \
    --heads_json "$OUT/heads_agentdojo.json" \
    --suite slack --eval_split heldout --limit_pairs 60 \
    --out_json "$OUT/eval_slack_heldout_tk.json" 2>&1 | tee "$OUT/console_eval_heldout_tk.log"
  ;;

*)
  echo "usage: $0 {trackA|eval-heldout|eval-heldout-tk}"
  exit 1
  ;;
esac
