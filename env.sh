# env.sh — 공용 SSH 서버에서 매 세션 source 할 것.
#
#     source ~/head_poc/env.sh   (또는 이 저장소를 어디에 clone했든 그 경로)
#
# 목적: 파이썬 생태계가 기본으로 홈(~)에 쏟아붓는 캐시를 전부 저장소 디렉토리 안으로 접는다.
# 공용 서버에서 기본값을 그대로 두면 HF 모델 가중치(수십 GB)가 ~/.cache/huggingface에,
# pip 패키지가 ~/.local에 깔려 남의 환경(같은 홈을 쓰는 다른 계정, 또는 이 계정의 다른 용도
# ~/.local)을 오염시킬 수 있다.
#
# 자세한 배경: docs/plan-2026-08-26.md 1절 (Titan RTX 서버 기준 원안).
#
# ⚠️ 2026-08-31, 서버 이전(aisec-king, RTX PRO 4500 32GB + RTX A6000 48GB + RTX 4090 24GB —
# 4090은 같은 날 중으로 추가됨)으로 $JB를 하드코딩된 "$HOME/jbwon"에서 **"이 env.sh가 있는
# 디렉토리"(=저장소 루트)를 자동 감지**하는
# 방식으로 바꿨다 — clone 위치가 서버마다 달라져도(이번엔 ~/head_poc, 예전엔 ~/jbwon/atlas_poc)
# 이 파일을 고칠 필요가 없다. 캐시/venv를 저장소 안(.cache/, envs/, miniforge3/ — 전부
# .gitignore 처리됨)에 가두면 "내 작업 공간 밖은 안 건드린다"는 원 취지도 그대로 지켜진다.

# ---- 작업 루트 -------------------------------------------------------------
# bash/zsh 겸용 가드. "지금 source되고 있는 이 파일의 경로"를 얻는 방법이 셸마다 다르다:
#   bash: ${BASH_SOURCE[0]}
#   zsh : ${(%):-%x}   ((%) 플래그로 %x(현재 실행 중인 파일명)를 prompt-expansion)
# 가드 없이 $0 같은 걸 쓰면 dirname이 조용히 "."로 떨어져 $JB가 "지금 cd해 있던 아무
# 디렉토리"가 된다 — 이 파일이 막으려는 사고(HF 가중치 수십 GB가 workspace 밖에 떨어짐)가
# 경고 없이 그대로 일어난다(2026-08-31 코드리뷰로 발견, 이후 zsh 지원 추가). 아래 zsh
# 분기는 bash에서는 절대 실행되지 않는 죽은 코드라 bash 파서가 `${(%):-%x}` 문법 자체를
# 검증하지 않는다(실측 확인) — bash 쪽에서 이 파일이 깨질 걱정은 없다.
# dash/sh 등 그 외 셸은 여전히 미지원 — 명시적으로 중단.
if [ -n "${ZSH_VERSION:-}" ]; then
    _env_sh_self="${(%):-%x}"
elif [ -n "${BASH_SOURCE:-}" ]; then
    _env_sh_self="${BASH_SOURCE[0]}"
else
    echo "[env.sh] 에러: bash 또는 zsh에서 source해야 한다 (\$BASH_SOURCE/\$ZSH_VERSION 둘 다" >&2
    echo "  비어있음 — dash/sh 등에서 실행 중일 수 있다). bash나 zsh로 먼저 들어간 뒤 다시 source할 것." >&2
    return 1 2>/dev/null || exit 1
fi
export JB="$(cd "$(dirname "$_env_sh_self")" && pwd)"
unset _env_sh_self

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

# ⚠️ AgentDojo head 탐색의 OOM은 "누수"가 아니라 **할당자 단편화**였다.
# todo.md P2-d에 "모델을 통째로 재로드해도 안 없어진다"고 기록된 증상이 그 단서다 —
# 파이썬 레벨 누수라면 재로드로 풀렸어야 한다. 실제 원인은 예시마다 attention 텐서
# 크기(H x T^2)가 제각각이라 캐싱 할당자에 조각이 남아 큰 블록을 못 잡는 것이었다.
# expandable_segments는 정확히 이 패턴(가변 크기 대형 할당 반복)을 위한 옵션이다.
#
# 실측 (Qwen2.5-7B 4bit, bf16, head_n=150, max_seq_len=2000, batch_size=5, Titan RTX 24GB):
#   미적용: 81/150 성공 (배치당 OOM 약 2.3개)
#   적용  : 배치당 OOM 0~1개로 감소
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- 파이썬 환경 -----------------------------------------------------------
# 시스템 python이 너무 오래돼(예: 3.8.19) transformers 4.51.3(>=3.9)/agentdojo(>=3.10)가
# 안 깔리는 서버가 있다. pyenv는 소스 빌드라 시스템 패키지(libssl-dev 등)를 요구해 공용
# 서버에서 쓸 수 없다. → Miniforge를 $JB(=저장소 루트) 안에 넣고, 그 python으로
# **평범한 venv**를 만든다.
#
# 왜 `conda create`가 아니라 venv인가: conda는 환경을 만들 때 ~/.conda/environments.txt를
# 홈에 기록한다. "내 작업 공간(저장소 디렉토리) 밖 수정 금지" 원칙에 걸리므로 conda 명령
# 자체를 쓰지 않는다. (설치 절차는 docs/run-guide.md 부록 A/B)
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
    echo "[env.sh] 경고: $JB/miniforge3 가 없다. docs/run-guide.md 부록 A/B를 먼저 진행할 것." >&2
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
