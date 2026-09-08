#!/usr/bin/env bash
# Llama-3.1-70B Track B (knockout 전이 평가) — P12 / feedback-2026-09-06 §1
# 사용: 각 단계를 필요할 때 하나씩. 긴 건 tmux 안에서.
# set -e는 쓰지 않는다 — `source env.sh`(venv activate) + `| tee` pipefail 조합에서
# 조용히 죽는 사례 있었음(2026-09-08). 각 단계는 tee 로그로 결과 확인.
# source는 파이프에 물리면 서브셸에서 돌아 venv 활성화가 안 됨 — 리다이렉트만.
set -uo pipefail
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

## ── Track A: 70B 자체 헤드 탐색 (discover-parallel) ──────────────────
## nf4+dq(2026-09-08 배선). Blackwell(PRO4500) 제외 → A6000(idx0)+4090(idx1) 분산.
## CUDA_VISIBLE_DEVICES=1,2 → 프로세스 안에서 0=A6000, 1=4090.

trackA-smoke)
  # head_n 20, max_seq_len 1000, batch_size 2. OOM율/수율/시간 확인용.
  CUDA_VISIBLE_DEVICES=1,2 python compare_head_sources.py discover-parallel \
    --source agentdojo --model "$MODEL" --family llama \
    --four_bit --dtype bf16 \
    --device cuda:0 --device_map auto --max_memory 0:32GiB 1:15GiB \
    --head_n 20 --max_seq_len 1000 --batch_size 2 \
    --out_json "$OUT/heads_agentdojo_SMOKE.json" 2>&1 | tee "$OUT/console_trackA_smoke.log"
  ;;

trackA)
  # 본 탐색. head_n/max_seq_len/batch_size는 스모크 결과 보고 조정. tmux 필수.
  CUDA_VISIBLE_DEVICES=1,2 python compare_head_sources.py discover-parallel \
    --source agentdojo --model "$MODEL" --family llama \
    --four_bit --dtype bf16 \
    --device cuda:0 --device_map auto --max_memory 0:32GiB 1:15GiB \
    --head_n 150 --max_seq_len 1400 --batch_size 2 \
    --out_json "$OUT/heads_agentdojo.json" 2>&1 | tee "$OUT/console_trackA.log"
  ;;

*)
  echo "usage: $0 {download|gpucheck|smoke|slack-heldout|slack-all|slack-heldout-split|trackA-smoke|trackA}"
  echo "  먼저 gpu_free로 A6000 인덱스 확인하고 이 파일 상단 A6000= 값 맞출 것"
  exit 1
  ;;
esac
