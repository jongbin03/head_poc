#!/usr/bin/env bash
# 다운로드 완료 대기 → smoke → (smoke OK면) slack-heldout. tmux 안에서 돌린다.
# 진행: tail -f results/2026-09-08_p12_llama70b/drive.log
set -uo pipefail
cd ~/head_poc
source env.sh >/dev/null 2>&1

OUT=results/2026-09-08_p12_llama70b
CACHE=.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct
DLPID=16604
COMMON=(--model meta-llama/Llama-3.1-70B-Instruct --family llama --four_bit
        --tool_call_format agentdojo_default
        --heads_json results/2026-08-25_s6_llama8b/heads_agentdojo.json)

exec > >(tee -a "$OUT/drive.log") 2>&1
echo "════ drive start $(date -u +%FT%TZ) ════"

# ── 1. 다운로드 완료 대기 ────────────────────────────────────────────
echo "[drive] wait for download (PID $DLPID) ..."
while kill -0 "$DLPID" 2>/dev/null; do sleep 60; done
echo "[drive] download process exited @ $(date -u +%FT%TZ)"

inc=$(find "$CACHE" -name '*.incomplete' 2>/dev/null | wc -l)
snap_dir=$(find "$CACHE/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
shards=$(ls "$snap_dir"/model-*-of-00030.safetensors 2>/dev/null | wc -l)
echo "[drive] incomplete=$inc  shards_in_snapshot=$shards/30  snap=$snap_dir"
if [ "$inc" -ne 0 ] || [ "$shards" -ne 30 ]; then
  echo "[drive] ✗ FAIL: 다운로드 미완. 재개: bash $OUT/run.sh download  그다음 이 스크립트 재실행"
  exit 1
fi
echo "[drive] ✓ download complete"

# ── 2. GPU (A6000) 확인 ──────────────────────────────────────────────
GPU=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', *' '/A6000/{print $1; exit}')
if [ -z "${GPU:-}" ]; then echo "[drive] ✗ FAIL: A6000 없음"; exit 1; fi
free_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
echo "[drive] A6000 = index $GPU (used ${free_mib} MiB)"

run_eval () {
  local tag=$1 out=$2 lim=$3
  echo "[drive] ▶ $tag start @ $(date -u +%FT%TZ)  (limit_pairs=$lim)"
  CUDA_VISIBLE_DEVICES=$GPU python run_agentdojo_eval.py "${COMMON[@]}" \
    --suite slack --eval_split heldout --limit_pairs "$lim" --out_json "$out"
  local rc=$?
  echo "[drive] ◀ $tag exit rc=$rc @ $(date -u +%FT%TZ)"
  return $rc
}

summarize () {
  python - "$1" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
r=d.get("rows",[])
print(f"[drive]   rows={len(r)}  k0_sec={d.get('k0_security_rate')}  kN_sec={d.get('kN_security_rate')}"
      f"  k0_util={d.get('k0_utility_rate')}  kN_util={d.get('kN_utility_rate')}")
ps=d.get("per_suite",{}).get("slack",{})
print(f"[drive]   slack: {ps}")
sys.exit(0 if r else 2)
PY
}

# ── 3. smoke (slack 3쌍) ─────────────────────────────────────────────
run_eval smoke "$OUT/smoke_slack.json" 3 || { echo "[drive] ✗ smoke 실행 실패 — 본 실행 중단"; exit 1; }
if [ ! -s "$OUT/smoke_slack.json" ]; then echo "[drive] ✗ smoke 출력 없음 — 중단"; exit 1; fi
summarize "$OUT/smoke_slack.json" || { echo "[drive] ✗ smoke rows=0 (롤아웃이 안 돎) — 중단"; exit 1; }
echo "[drive] ✓ smoke OK"

# ── 4. 본 실행 slack held-out 전수 ──────────────────────────────────
run_eval slack_heldout "$OUT/eval_slack_heldout.json" 50 || { echo "[drive] ✗ slack_heldout 실패"; exit 1; }
summarize "$OUT/eval_slack_heldout.json" || echo "[drive] ⚠ slack_heldout rows=0"

echo "════ ALL DONE $(date -u +%FT%TZ) ════"
