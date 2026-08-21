# env.sh — 공용 SSH 서버(연구실 Titan RTX x3)에서 매 세션 source 할 것.
#
#     source ~/jbwon/atlas_poc/env.sh
#
# 목적: 파이썬 생태계가 기본으로 홈(~)에 쏟아붓는 캐시를 전부 ~/jbwon 안으로 접는다.
# 교수님 규칙이 "~/jbwon 에서만 작업, 다른 곳 절대 수정 금지"인데, 기본값을 그대로 두면
# HF 모델 가중치(수십 GB)가 ~/.cache/huggingface에, pip 패키지가 ~/.local에 깔린다.
# 홈이 공용이면 그건 곧 남의 환경 오염이다.
#
# 자세한 배경: docs/plan-2026-08-26.md 1절.

# ---- 작업 루트 -------------------------------------------------------------
# 저장소를 다른 곳에 뒀다면 JB만 고치면 된다.
export JB="$HOME/jbwon"

# ---- 캐시 리다이렉트 -------------------------------------------------------
# XDG_CACHE_HOME이 ~/.cache/* 대부분을 한 번에 포섭하지만, 아래 것들은 자체 기본값을
# 따로 갖고 있어 개별 지정이 필요하다.
export XDG_CACHE_HOME="$JB/.cache"
export HF_HOME="$JB/.cache/huggingface"
export TORCH_HOME="$JB/.cache/torch"
export PIP_CACHE_DIR="$JB/.cache/pip"
export MPLCONFIGDIR="$JB/.cache/matplotlib"
export TORCH_EXTENSIONS_DIR="$JB/.cache/torch_extensions"
export TRITON_CACHE_DIR="$JB/.cache/triton"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" \
         "$MPLCONFIGDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"

# ---- GPU -------------------------------------------------------------------
# 이걸 안 켜면 CUDA_VISIBLE_DEVICES 번호가 nvidia-smi 번호와 달라질 수 있다.
# 공용 서버에서 "빈 GPU 골라 잡기"를 하려면 두 번호가 일치해야 한다.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ---- 파이썬 환경 -----------------------------------------------------------
# 시스템 python은 3.8.19라 transformers 4.51.3(>=3.9)/agentdojo(>=3.10)가 안 깔린다.
# pyenv는 소스 빌드라 시스템 패키지(libssl-dev 등)를 요구해 공용 서버에서 쓸 수 없다.
# → Miniforge를 ~/jbwon 안에 넣는다 (설치 절차는 docs/run-guide.md "공용 서버" 절).
if [ -f "$JB/miniforge3/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$JB/miniforge3/bin/activate" "$JB/envs/atlas"
else
    echo "[env.sh] 경고: $JB/miniforge3 가 없다. docs/run-guide.md의 Miniforge 설치를 먼저 할 것." >&2
fi

# ---- 확인 -----------------------------------------------------------------
gpu_free() {
    # 지금 비어 있는 GPU를 보여준다. 공용이므로 실행 전에 항상 확인할 것.
    # 다른 사람 프로세스가 보이면 그 GPU는 건드리지 않는다 (kill 절대 금지).
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader
    echo "--- 실행 중인 프로세스 ---"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
}

echo "[env.sh] JB=$JB"
echo "[env.sh] python=$(command -v python)  ($(python --version 2>&1))"
echo "[env.sh] HF_HOME=$HF_HOME"
echo "[env.sh] 실행 전 'gpu_free' 로 빈 GPU 확인 후 CUDA_VISIBLE_DEVICES=N 으로 지정할 것"
