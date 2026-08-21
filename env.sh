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
# conda를 쓰게 될 경우를 대비 (아래 기본 경로는 venv라 보통 안 쓰임)
export CONDA_PKGS_DIRS="$JB/.conda/pkgs"
export CONDA_ENVS_DIRS="$JB/envs"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" \
         "$MPLCONFIGDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"

# ---- 공용 user-site 차단 (중요) --------------------------------------------
# conda env는 venv와 달리 ~/.local/lib/pythonX.Y/site-packages를 그대로 읽는다.
# 이 서버의 ~/.local에는 다른 사용자가 깐 패키지가 많아(torch-geometric, openai,
# scikit-learn, typing-extensions 등) 그대로 두면 두 가지 문제가 난다:
#   1) `pip install X==핀버전`이 "Requirement already satisfied"로 건너뛰고
#      남의 버전을 쓰게 된다 → 핀이 무력화되고 재현성이 깨진다
#   2) torch 버전이 안 맞는 확장(torch_cluster 1.6.3+pt24cu124 등)이 import될 수 있다
# PYTHONNOUSERSITE=1이면 파이썬이 user-site를 아예 무시한다.
export PYTHONNOUSERSITE=1

# ---- GPU -------------------------------------------------------------------
# 이걸 안 켜면 CUDA_VISIBLE_DEVICES 번호가 nvidia-smi 번호와 달라질 수 있다.
# 공용 서버에서 "빈 GPU 골라 잡기"를 하려면 두 번호가 일치해야 한다.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ---- 파이썬 환경 -----------------------------------------------------------
# 시스템 python은 3.8.19라 transformers 4.51.3(>=3.9)/agentdojo(>=3.10)가 안 깔린다.
# pyenv는 소스 빌드라 시스템 패키지(libssl-dev 등)를 요구해 공용 서버에서 쓸 수 없다.
# → Miniforge를 ~/jbwon 안에 넣고, 그 python으로 **평범한 venv**를 만든다.
#
# 왜 `conda create`가 아니라 venv인가: conda는 환경을 만들 때 ~/.conda/environments.txt를
# 홈에 기록한다. "~/jbwon 밖 수정 금지" 규칙에 걸리므로 conda 명령 자체를 쓰지 않는다.
# (설치 절차는 docs/run-guide.md 부록 A)
# 환경은 venv / conda env 둘 다 지원한다. Miniforge base가 3.13이라 핀과 안 맞는 경우
# `conda create -p $JB/envs/atlas python=3.11`로 만들게 되는데, conda env에는 venv와 달리
# `bin/activate`가 없어서 분기가 필요하다 (conda env는 `conda-meta/`로 식별).
if [ -f "$JB/envs/atlas/bin/activate" ]; then
    # venv
    # shellcheck disable=SC1091
    source "$JB/envs/atlas/bin/activate"
elif [ -d "$JB/envs/atlas/conda-meta" ] && [ -f "$JB/miniforge3/bin/activate" ]; then
    # conda env — miniforge base를 먼저 켜야 `conda activate`가 셸 함수로 존재한다
    # shellcheck disable=SC1091
    source "$JB/miniforge3/bin/activate"
    conda activate "$JB/envs/atlas"
elif [ -f "$JB/miniforge3/bin/activate" ]; then
    # 환경이 아직 없고 miniforge만 있는 경우 — base만 활성화 (환경 생성 직전 상태)
    # shellcheck disable=SC1091
    source "$JB/miniforge3/bin/activate"
    echo "[env.sh] 경고: $JB/envs/atlas 환경이 없다. miniforge base만 활성화됨." >&2
else
    echo "[env.sh] 경고: $JB/miniforge3 가 없다. docs/run-guide.md 부록 A를 먼저 진행할 것." >&2
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
