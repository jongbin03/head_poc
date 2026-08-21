"""
runtime_env.py

실행 환경(dtype / GPU / 코드 버전 / 패키지 버전)을 **해결하고 기록**하는 공용 모듈.

두 가지 문제를 동시에 다룬다:

1. **dtype** — 서버(Titan RTX, Turing sm_75)에는 bf16 하드웨어 지원이 없다(Ampere sm_80부터).
   기존 코드가 `torch.bfloat16`을 하드코딩하고 있어 Turing에서는 커널에 따라
   `CUBLAS_STATUS_NOT_SUPPORTED`로 죽거나 에뮬레이션으로 매우 느려진다.
   `--dtype`으로 빼되 **기본값 "auto"는 감지 결과를 반드시 기록**한다.

2. **재현성** — 지금까지 결과 폴더에 `seed`만 남고 코드 버전·dtype·패키지 버전이 안 남아서,
   8/19에 7B 실행 3건이 서로 안 맞았을 때 원인을 가릴 수 없었다
   (docs/todo.md "측정 재현성 문제", docs/plan-2026-08-26.md 1.3절).
   `write_env_json()`으로 실행마다 env.json을 남긴다.

⚠️ **bf16과 fp16 결과는 수치가 같지 않다.** fp16은 가수부가 더 길고(10비트 vs 7비트)
지수 범위는 좁다. knockout 붕괴 지점이 밀릴 수 있으므로 **dtype이 다른 실행을 같은
비교표에 섞지 않는다.** env.json의 `dtype` 필드로 사후 확인할 것.
"""
import datetime
import json
import os
import platform
import subprocess
import sys
from typing import Dict, Optional

import torch

DTYPE_CHOICES = ("auto", "bf16", "fp16", "fp32")

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def add_runtime_args(parser):
    """모델을 로드하는 모든 엔트리포인트에서 호출할 것."""
    parser.add_argument(
        "--dtype", default="auto", choices=list(DTYPE_CHOICES),
        help="모델 가중치/연산 dtype. 기본 auto는 bf16 지원 GPU면 bf16, 아니면 fp16을 고른다 "
        "(Titan RTX 등 Turing은 bf16이 없어 fp16으로 내려감). 해결된 값은 결과 env.json에 "
        "기록되므로, 서로 다른 dtype으로 나온 결과를 같은 비교표에 섞지 말 것.",
    )
    return parser


def resolve_dtype(name: str = "auto", device: str = "cuda") -> tuple:
    """
    반환: (torch.dtype, 해결된 이름)

    "auto"는 bf16 지원 여부를 실제로 조회해서 고른다. 감지 결과를 이름으로 같이 돌려주는
    이유는 호출자가 그걸 env.json에 기록해야 하기 때문 — 자동 감지만 하고 기록을 안 하면
    "조용히 다른 숫자가 나오는" 최악의 경우가 된다.
    """
    if name != "auto":
        return _DTYPE_MAP[name], name

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return torch.float32, "fp32"

    # ⚠️ `torch.cuda.is_bf16_supported()`를 쓰면 안 된다.
    #    최신 PyTorch에서 이 함수는 기본값이 including_emulation=True라, bf16 하드웨어가
    #    없어도 "에뮬레이션으로 돌릴 수 있으면" True를 반환한다.
    #    실측(2026-08-21, 서버 Titan RTX + torch 2.13.0+cu126): capability (7,5)인데
    #    is_bf16_supported()가 True를 돌려줬다 — auto가 bf16을 골라버려 정확히 피하려던
    #    상황(에뮬레이션 bf16 = 매우 느림)이 된다.
    #    bf16 하드웨어 지원은 Ampere(sm_80)부터이므로 compute capability를 직접 본다.
    major, _minor = torch.cuda.get_device_capability()
    if major >= 8:
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def has_nonfinite(group_scores: Dict[str, torch.Tensor]) -> bool:
    """
    relevance 결과에 NaN/inf가 있는지 검사.

    fp16에는 loss scaling이 없어 lxt의 backward pass에서 relevance가 NaN/inf로 죽을 수
    있다(bf16이 원래 이걸 막아주던 부분). 호출자는 이 경우를 **OOM 스킵과 구분해서**
    카운트해야 한다 — 섞이면 또 원인 불명의 재현성 문제가 된다.
    """
    for t in group_scores.values():
        if not torch.isfinite(t).all():
            return True
    return False


def git_commit() -> dict:
    """현재 체크아웃된 커밋과 working tree 오염 여부."""
    repo_dir = os.path.dirname(os.path.abspath(__file__))

    def _run(*cmd):
        try:
            return subprocess.check_output(
                cmd, cwd=repo_dir, stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return None

    head = _run("git", "rev-parse", "HEAD")
    # --untracked-files=no가 중요하다. 이게 없으면 실행이 방금 만든 results/<run_dir>가
    # untracked로 잡혀 **항상** dirty=True가 된다 (실측 2026-08-21 서버 첫 실행).
    # 여기서 알고 싶은 건 "추적 중인 코드가 커밋과 다른가"이지 "새 산출물이 있는가"가 아니다.
    status = _run("git", "status", "--porcelain", "--untracked-files=no")
    return {
        "commit": head,
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        # dirty=True면 커밋되지 않은 수정 위에서 돌린 결과라 그 커밋으로 재현되지 않는다
        "dirty": bool(status) if status is not None else None,
        # 무엇이 수정됐는지도 남긴다 — dirty=True일 때 사후 추적이 가능해야 한다
        "dirty_files": status.splitlines() if status else [],
    }


def gpu_info() -> list:
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        out.append({
            "index": i,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gib": round(props.total_memory / (1024 ** 3), 2),
            # Turing(7.5)에는 bf16 하드웨어 지원이 없다 — Ampere(8.0)부터
            "bf16_supported": props.major >= 8,
        })
    return out


def installed_packages() -> dict:
    """`pip freeze` 대신 importlib.metadata로 — 서브프로세스 없이 빠르게."""
    try:
        from importlib import metadata
    except ImportError:
        return {}
    pkgs = {}
    for dist in metadata.distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:
            continue
        if name:
            pkgs[name] = dist.version
    return dict(sorted(pkgs.items(), key=lambda kv: kv[0].lower()))


# 재현에 직접 영향을 주는 것들 — env.json 상단에 따로 뽑아 눈으로 바로 보이게 한다
KEY_PACKAGES = ("torch", "transformers", "lxt", "accelerate", "bitsandbytes", "agentdojo")


def collect_env_meta(dtype_name: str, extra: Optional[dict] = None) -> dict:
    pkgs = installed_packages()
    meta = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "dtype": dtype_name,
        "git": git_commit(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda": getattr(torch.version, "cuda", None),
        "gpus": gpu_info(),
        "key_packages": {k: pkgs.get(k) for k in KEY_PACKAGES},
        "packages": pkgs,
    }
    if extra:
        meta.update(extra)
    return meta


def write_env_json(run_dir: str, dtype_name: str, extra: Optional[dict] = None) -> str:
    """
    결과 폴더에 env.json을 쓴다. 같은 폴더에 두 번 쓰이면(예: 같은 run_dir로 재실행)
    이전 것을 env.json.<n>.bak으로 밀어두고 새로 쓴다 — P2-a에서 run_dir 충돌로 결과가
    덮어써진 전례가 있어(todo.md P2-a "작업 중 발견한 버그") 환경 기록만은 잃지 않게 한다.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "env.json")
    if os.path.exists(path):
        n = 1
        while os.path.exists(f"{path}.{n}.bak"):
            n += 1
        os.replace(path, f"{path}.{n}.bak")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(collect_env_meta(dtype_name, extra), f, indent=2, ensure_ascii=False)
    return path


def describe(dtype_name: str) -> str:
    """콘솔 한 줄 요약 — 실행 시작 시 찍어서 로그만 봐도 환경을 알 수 있게."""
    g = git_commit()
    commit = (g["commit"] or "?")[:8] + ("+dirty" if g["dirty"] else "")
    gpus = gpu_info()
    gpu = gpus[0]["name"] if gpus else "cpu"
    return f"[env] commit={commit} dtype={dtype_name} gpu={gpu} torch={torch.__version__}"
