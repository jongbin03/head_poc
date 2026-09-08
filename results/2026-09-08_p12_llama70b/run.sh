#!/usr/bin/env bash
# Llama-3.1-70B Track B (knockout 전이 평가) — P12 / feedback-2026-09-06 §1
# 사용: 각 단계를 필요할 때 하나씩. 긴 건 tmux 안에서.
set -euo pipefail
cd ~/head_poc
source env.sh >/dev/null 2>&1

MODEL=meta-llama/Llama-3.1-70B-Instruct
HEADS=results/2026-08-25_s6_llama8b/heads_agentdojo.json
OUT=results/2026-09-08_p12_llama70b
A6000=1   # gpu_free로 매번 재확인! (PCI_BUS_ID 순서: 0=PRO4500, 1=A6000, 2=4090)

case "${1:-}" in

download)
  # ~140GB fp16. original/*(consolidated pth)는 제외. 재개 가능.
  hf download "$MODEL" --exclude "original/*"
  ;;

gpucheck)
  CUDA_VISIBLE_DEVICES=$A6000 python -c "import torch; print(torch.cuda.get_device_name(0))"
  ;;

smoke)
  # slack 3쌍, A6000 단독. 로드/OOM/완주 확인용.
  CUDA_VISIBLE_DEVICES=$A6000 python run_agentdojo_eval.py \
    --model "$MODEL" --family llama --four_bit \
    --tool_call_format agentdojo_default \
    --heads_json "$HEADS" \
    --suite slack --eval_split heldout --limit_pairs 3 \
    --out_json "$OUT/smoke_slack.json" 2>&1 | tee "$OUT/console_smoke.log"
  ;;

slack-heldout)
  # 본 실행. A6000 단독. tmux 권장.
  CUDA_VISIBLE_DEVICES=$A6000 python run_agentdojo_eval.py \
    --model "$MODEL" --family llama --four_bit \
    --tool_call_format agentdojo_default \
    --heads_json "$HEADS" \
    --suite slack --eval_split heldout --limit_pairs 50 \
    --out_json "$OUT/eval_slack_heldout.json" 2>&1 | tee "$OUT/console_slack_heldout.log"
  ;;

slack-all)
  # 누수 영향 비교용 (선택).
  CUDA_VISIBLE_DEVICES=$A6000 python run_agentdojo_eval.py \
    --model "$MODEL" --family llama --four_bit \
    --tool_call_format agentdojo_default \
    --heads_json "$HEADS" \
    --suite slack --eval_split all --limit_pairs 50 \
    --out_json "$OUT/eval_slack_all.json" 2>&1 | tee "$OUT/console_slack_all.log"
  ;;

slack-heldout-split)
  # A6000 단독 OOM 시: A6000(0)+4090(1) 분산. CUDA_VISIBLE_DEVICES 순서에 맞춰 max_memory 인덱스.
  CUDA_VISIBLE_DEVICES=1,2 python run_agentdojo_eval.py \
    --model "$MODEL" --family llama --four_bit \
    --tool_call_format agentdojo_default \
    --device cuda:0 --device_map auto --max_memory 0:44GiB 1:22GiB \
    --heads_json "$HEADS" \
    --suite slack --eval_split heldout --limit_pairs 50 \
    --out_json "$OUT/eval_slack_heldout.json" 2>&1 | tee "$OUT/console_slack_heldout.log"
  ;;

*)
  echo "usage: $0 {download|gpucheck|smoke|slack-heldout|slack-all|slack-heldout-split}"
  echo "  먼저 gpu_free로 A6000 인덱스 확인하고 이 파일 상단 A6000= 값 맞출 것"
  exit 1
  ;;
esac
