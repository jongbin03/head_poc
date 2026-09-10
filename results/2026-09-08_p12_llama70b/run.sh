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

## ── Track A eval: "70B 자체 헤드" vs "8B 전이 헤드" knockout 비교 (2026-09-09) ──
## 같은 모델·같은 slack 105쌍(--eval_split all), knockout 헤드 집합만 교체.
## 09-08 모호성 판정: 70B가 저항하는가 vs 8B의 (틀린) 헤드를 껐던 것인가.
## ⚠️ 70B 자체 헤드는 slack user_task로 탐색됨 → --eval_split all은 eval B에 누수.
##    heldout(7쌍)도 같이 보되 표본이 얇음. A6000 단독, tmux, 각 ~2시간.

trackA-eval-8bheads)
  CUDA_VISIBLE_DEVICES=$A6000 python run_agentdojo_eval.py \
    --model "$MODEL" --family llama --four_bit \
    --tool_call_format agentdojo_default \
    --heads_json results/2026-08-25_s6_llama8b/heads_agentdojo.json \
    --suite slack --eval_split all --limit_pairs 200 \
    --out_json "$OUT/eval_slack_all_8bheads.json" 2>&1 | tee "$OUT/console_slack_all_8bheads.log"
  ;;

trackA-eval-70bheads)
  CUDA_VISIBLE_DEVICES=$A6000 python run_agentdojo_eval.py \
    --model "$MODEL" --family llama --four_bit \
    --tool_call_format agentdojo_default \
    --heads_json "$OUT/heads_agentdojo.json" \
    --suite slack --eval_split all --limit_pairs 200 \
    --out_json "$OUT/eval_slack_all_70bheads.json" 2>&1 | tee "$OUT/console_slack_all_70bheads.log"
  ;;

trackA-eval-70bheads-heldout)
  CUDA_VISIBLE_DEVICES=$A6000 python run_agentdojo_eval.py \
    --model "$MODEL" --family llama --four_bit \
    --tool_call_format agentdojo_default \
    --heads_json "$OUT/heads_agentdojo.json" \
    --suite slack --eval_split heldout --limit_pairs 50 \
    --out_json "$OUT/eval_slack_heldout_70bheads.json" 2>&1 | tee "$OUT/console_slack_heldout_70bheads.log"
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
## nf4+dq. 2026-09-09: --device_map_plan(수동 device_map) 신설 — auto 배치는 층 편중+
## root 집중으로 backward OOM (09-08 확인). 09-08에 T=1000 완주했던 수동 배치를 재현:
## A6000 32L+embed+norm+lm_head / Blackwell 30L / 4090 18L.
## CUDA_VISIBLE_DEVICES=1,0,2 → 프로세스 안 0=A6000(48G,root), 1=Blackwell(32G), 2=4090(24G).
## ⚠️ Blackwell nf4 커널 손상 이력(2.1.20) — 단 09-08에 32B greedy byte-identical.
## 스모크가 실제 프롬프트에서 relevance finite 확인하는 것이 목적.

trackA-smoke)
  # head_n 20, max_seq_len 1000, batch_size 4. OOM율/수율/시간 + finite 확인용.
  # batch_size=1 예비 확인: 첫 3배치 clean(1ok/0oom/0nan) — plumbing·finite OK, 재로드가
  # 배치당 ~8분이라 느림. batch_size=4로 본 탐색 배치값도 함께 검증(A6000 headroom 확인됨).
  CUDA_VISIBLE_DEVICES=1,0,2 python compare_head_sources.py discover-parallel \
    --source agentdojo --model "$MODEL" --family llama \
    --four_bit --dtype bf16 \
    --device cuda:0 --device_map_plan 0:32,1:30,2:18 \
    --head_n 20 --max_seq_len 1000 --batch_size 4 \
    --out_json "$OUT/heads_agentdojo_SMOKE.json" 2>&1 | tee "$OUT/console_trackA_smoke.log"
  ;;

trackA)
  # 본 탐색. 스모크(2026-09-09): 16/16 ok, 0 oom, 0 nan, heads layer 26-44
  # (8B는 layer 11-22/32 ≈ 같은 상대 깊이 40% — sanity OK). batch_size=4 leak 없음 확인.
  # head_n 200 = 8B 탐색과 동일 (8B: head_n 200 → 실제 149쌍, eval 31쌍). T=1000 → 137 all_pairs.
  # 예상 ~6시간 (배치당 ~12분, 모델 재로드가 병목). tmux 필수.
  CUDA_VISIBLE_DEVICES=1,0,2 python compare_head_sources.py discover-parallel \
    --source agentdojo --model "$MODEL" --family llama \
    --four_bit --dtype bf16 \
    --device cuda:0 --device_map_plan 0:32,1:30,2:18 \
    --head_n 200 --max_seq_len 1000 --batch_size 4 \
    --out_json "$OUT/heads_agentdojo.json" 2>&1 | tee "$OUT/console_trackA.log"
  ;;

*)
  echo "usage: $0 {download|gpucheck|smoke|slack-heldout|slack-all|slack-heldout-split|trackA-smoke|trackA}"
  echo "  먼저 gpu_free로 A6000 인덱스 확인하고 이 파일 상단 A6000= 값 맞출 것"
  exit 1
  ;;
esac
