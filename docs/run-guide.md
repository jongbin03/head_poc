# 실행 가이드 — 5070 Ti (16GB)에서 IPI read/control head PoC 돌리기

전제: Windows, Git Bash(또는 PowerShell) 사용. 아래 명령은 Git Bash 기준이며,
PowerShell을 쓸 경우 `source .venv/Scripts/activate` 대신
`.venv\Scripts\Activate.ps1`을 쓴다.

모든 명령은 이 디렉토리(`atlas_poc/`)에서 실행한다.

> 🖥️ **연구실 공용 SSH 서버에서 돌리는 경우 아래 0~2단계를 그대로 따르면 안 된다.**
> pyenv/sudo/pip --user 등은 공용 서버에서 못 쓰고, 서버마다 python 버전·GPU 세대가
> 달라 절차가 갈린다. 어느 서버인지에 따라 아래로 갈 것:
> - Titan RTX 24GB × 3(Turing, bf16 하드웨어 없음, 시스템 python 3.8.19) →
>   **[부록 A](#부록-a-공용-ssh-서버-titan-rtx-24gb--3-절차)**
> - aisec-king(RTX PRO 4500 32GB + RTX A6000 48GB, 둘 다 bf16 네이티브 지원) →
>   **[부록 B](#부록-b-공용-서버-aisec-king-rtx-pro-4500-32gb--rtx-a6000-48gb-절차-신설-2026-08-31)**

---

## 0단계 — Python 버전 준비

pyenv가 구버전(3.7.4 등)으로 잡혀 있으면 `transformers`/`lxt`가 안 돈다.

```bash
pyenv install 3.11.9        # 이미 설치돼 있으면 자동으로 skip됨
pyenv local 3.11.9          # 이 폴더 전용 .python-version 생성
python --version             # Python 3.11.9 확인
```

---

## 1단계 — 가상환경 + PyTorch (CUDA 12.8 / sm_120, 5070 Ti 필수 조건)

5070 Ti는 Blackwell(sm_120)이라 구버전 CUDA 빌드 PyTorch(cu121/cu124 등)는
커널이 없어서 실패하거나 조용히 CPU로 폴백한다. **반드시 cu128 이상** 빌드를 쓴다.
(정확한 wheel URL/버전은 시점에 따라 바뀌므로 https://pytorch.org 에서
"Stable/Preview, CUDA 12.8+"로 안내하는 명령을 그대로 쓰는 게 안전 — 아래는
그 형태의 예시.)

```bash
python -m venv .venv
source .venv/Scripts/activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**검증 (반드시 통과하고 다음 단계로 갈 것):**

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

기대 출력 예:
```
2.x.x+cu128
True
NVIDIA GeForce RTX 5070 Ti
```

`is_available()`이 `False`거나 `get_device_name`이 에러나면 여기서 멈추고
드라이버/CUDA 버전부터 다시 맞춘다 (`nvidia-smi`로 드라이버가 CUDA 12.8+를
지원하는지 먼저 확인).

---

## 2단계 — 나머지 의존성 설치

```bash
pip install lxt transformers accelerate bitsandbytes matplotlib
```

버전 확인 (문제 생기면 여기부터 공유):

```bash
pip show torch transformers lxt accelerate bitsandbytes
```

> `bitsandbytes`는 Phase 4(4bit, 8B급)에서만 쓰인다. Windows에서 설치가
> 말썽이면 지금은 건너뛰고 Phase 0~3까지 먼저 진행해도 된다.

---

## 3단계 — Phase 0: 극소형 모델로 파이프라인 자체 검증 (smoke test)

GPU 메모리 걱정 없이 코드 버그(shape, span 오프셋, grad None 등)를 먼저 잡는다.

`dataset.py`는 기본으로 30개 템플릿(콘텐츠 도메인 6개 x 공격 문구 스타일 5개)을
만들고, 템플릿마다 4가지 mode의 프롬프트를 생성한다:

| mode | 주입문 | assistant_prefix | 쓰이는 곳 |
|---|---|---|---|
| `read_clean` | 없음 | `The answer is` | read head relevance baseline (오염 없이) |
| `read_injected` | 있음 | `The answer is` | knockout sweep의 utility 축 (끊을 D_inj 엣지 필요) |
| `internal` | 있음 | `Sure, I will forward this to` | y_internal relevance |
| `external` | 있음 | tool_call JSON | y_external relevance + ASR 측정 |

템플릿 순서는 style-major라 `--dataset_limit 6`이면 도메인 6개가 전부 한 번씩
들어간다(스타일 1종). Phase 0에서는 이걸로 잘라 빠르게 돈다.

```bash
python run_pipeline.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --family qwen2 \
  --device cuda \
  --topk 20 \
  --dataset_limit 2
```

**체크리스트 (하나라도 안 되면 다음 단계로 넘어가지 말 것):**

1. 콘솔에 `[1/4] loading ...` ~ `[4/4] edge-knockout sweep ...` 네 단계가
   에러 없이 순서대로 다 찍힌다.
2. `top-20 Jaccard(...)` 세 줄이 0~1 사이 숫자로 찍힌다 (NaN이면 span이
   비어있다는 뜻 — `dataset.py` 쪽 문제).
3. `functional_map.png`가 생성되고, 열어보면 layer×head 산점도가 그려져 있다.
4. **가장 중요**: `[4/4]` 출력에서 `k=0`부터 `k=80`까지
   `malicious_token_prob` 값이 **유의미하게 움직인다** (전부 똑같은 숫자면
   knockout이 실제로 적용 안 되고 있다는 뜻 — `edge_ablation.py`의
   `eager_attention_forward` 패치가 깨진 회귀. lxt의 monkey_patch가
   `Qwen2Attention.forward`를 통째로 갈아끼웠을 가능성도 여기서 드러난다).
5. `k=0`에서 `malicious_token_prob`과 `read_token_prob`이 **둘 다 0이 아니다**.
   0에 가까우면 타깃 토큰이 그 자리에 올 수 없는 조합이라는 뜻이므로
   (prefix 끝 공백 vs 타깃 토큰 앞 공백) `dataset.py`의 prefix/타깃 짝을 재점검.
6. `IndexError: key_positions가 입력 길이를 벗어남`이 뜨면 서로 다른 mode의
   span을 섞어 쓴 것 — 어떤 예시의 span을 어떤 입력에 넣는지 확인.

---

## 4단계 — Phase 1~3: 본 실험 (1.5B → 3B), VRAM 실측

별도 터미널에서 VRAM을 관찰하며 돌린다.

```bash
# 터미널 A: 관찰용
nvidia-smi -l 1
```

```bash
# 터미널 B: 본 실행
python run_pipeline.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --family qwen2 \
  --device cuda \
  --topk 20
```

문제없이 끝나고 VRAM에 여유가 있으면 3B로 올린다:

```bash
python run_pipeline.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --family qwen2 \
  --device cuda \
  --topk 20
```

**결과 해석:**

| 출력 | 의미 |
|---|---|
| `jaccard(read, internal)`, `jaccard(read, external)` 낮음 | read와 instruction-following이 head 레벨에서 분리됨 (교수님 질문에 긍정적 답) |
| `jaccard(internal, external)` 높음 | 내부 답변 오염과 tool-call 오염이 같은 control head 회로 → idea1("control head 한 번 찾아 knockout하면 둘 다 막힘") 전제 지지 |
| `jaccard(internal, external)` 낮음 | tool-call 포맷팅이 별도 head를 추가 동원 → 포맷별로 head 탐색을 따로 해야 함 (negative result지만 보고 가치 있음) |
| `control_heads_both` | Phase 1 knockout 후보 목록. `[4/4]` sweep이 여기서 뽑은 head들을 knock out함 |

전체 30개 템플릿 기준 비용:

- `[2/4]` relevance: 템플릿당 3회(read_clean / internal / external)
  forward+backward = **90회**. 여기가 가장 무겁고 VRAM 피크도 여기서 난다.
- `[4/4]` sweep: k 6단계 x (external 30 + read_injected 30) = **360회 forward**.
  backward가 없어 상대적으로 가볍다.

처음 한 번은 `--dataset_limit 6`(도메인 6개 전부, 스타일 1종)으로 파이프라인이
끝까지 도는지 확인한 뒤 전체(30개)로 올리는 걸 권장.

OOM이 나면 `--model`을 다시 1.5B로 낮추거나 `--dataset_limit`으로 템플릿
수를 줄인다 (attention tensor 메모리가 시퀀스 길이 제곱에 비례하므로 템플릿
문장이 긴 도메인부터 영향을 받는다).

---

## 5단계 — Phase 4: 8B급 스케일 검증 (선택)

```bash
python run_pipeline.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --family qwen2 \
  --device cuda \
  --four_bit \
  --topk 20
```

⚠️ `--four_bit`는 내부적으로 gradient checkpointing이 켜진 조합과 맞물릴 수
있는데, checkpointing이 켜지면 `attn_relevance.py`의 head-level relevance는
`.grad`가 `None`으로 채워지지 않아 조용히 실패한다 (README 경고 그대로).
8B에서 head relevance 자체가 필요하면:

```bash
# checkpointing 없이 4bit만 (VRAM이 버티는지 실측 필요)
python run_pipeline.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --family qwen2 \
  --device cuda \
  --four_bit \
  --topk 20
```
을 돌려보고, `[2/4]` 단계에서 `a.grad is None` 관련 경고/스킵이 뜨는지
콘솔을 직접 확인한다. 안전한 대안은 8B 단계에서는 **새로 head를 찾지 않고**,
1.5B/3B에서 찾은 control head가 8B에서도 knockout으로 여전히 먹히는지만
(`edge_ablation.py`, forward-only라 4bit/checkpointing 무관하게 항상 안전)
검증하는 것.

---

## 트러블슈팅 빠른 표

| 증상 | 원인 후보 | 확인 방법 |
|---|---|---|
| `torch.cuda.is_available() == False` | CUDA 버전이 sm_120 미지원 | `pip show torch` 버전에 `+cu128` 이상 있는지 |
| `a.grad is None`이 계속 뜸 | gradient checkpointing이 켜짐 | 코드에 `gradient_checkpointing_enable()` 호출 있는지 grep |
| knockout sweep 전 구간 확률 동일 | `edge_ablation.py` 패치가 eager 경로를 못 잡음 | `k=0`과 `k=80`의 `malicious_token_prob` 차이가 0인지 확인 |
| OOM | 시퀀스 길이 × attention tensor 메모리 | `nvidia-smi`로 실측, 모델 크기/템플릿 길이 축소 |
| HF 모델 다운로드 안 됨 | 네트워크/인증 | `huggingface-cli login`, 프록시 설정 확인 |
| `operator torchvision::nms does not exist` | torchvision이 torch와 다른 인덱스/빌드 | torch와 **같은 index-url**로 torchvision 재설치 (부록 A-4) |
| `Failed to import transformers.models.llama...` | 위와 같은 원인(연쇄 실패) | 위와 동일 |
| relevance가 전부 NaN / `nan_skipped`가 큼 | fp16 backward 언더플로 (loss scaling 없음) | `--dtype fp32`로 올려서 재현되는지 확인 (부록 A-5) |
| `CUBLAS_STATUS_NOT_SUPPORTED` | bf16 없는 GPU에서 bf16 요청 | `--dtype fp16` 명시. Turing(sm_75)은 bf16 미지원 |
| 같은 설정인데 지난번과 숫자가 다름 | dtype/커밋이 다름 | 두 결과의 `env.json`에서 `dtype`·`git.commit`·`git.dirty` 비교 |

---

## 부록 A. 공용 SSH 서버 (Titan RTX 24GB × 3) 절차

본문 0~2단계를 **대체**한다. 3단계 이후의 실험 명령은 그대로 쓰되 `--dtype`만 추가된다.

### A-0. 지켜야 할 제약 (교수님 규칙)

> - 누가 돌리고 있을 수도 있으니 그럴 경우 기다리고 쓸 것
> - `~/jbwon` 에서만 작업할 것. 다른 곳 절대 수정 금지. 패키지 설치도 최대한 영향 없게

홈(`~`)이 공용일 가능성이 높다는 뜻으로 해석한다. 따라서:

- **`sudo` 금지**
- **`pip install --user` 금지** — `~/.local`에 깔려 공용 파이썬의 import 경로를 바꾼다
- **pyenv 설치 금지** — 소스 빌드라 `libssl-dev` 등 시스템 패키지가 필요하고,
  `~/.pyenv`와 `.bashrc`를 오염시킨다. 본문 0단계를 서버에서 따라하지 말 것
- **`git config --global` 금지** — 저장소 로컬 설정만 사용
- 남의 GPU 프로세스 **kill 절대 금지**

### A-1. 저장소 clone + 캐시 리다이렉트

```bash
mkdir -p ~/jbwon && cd ~/jbwon
git clone http://github.com/jongbin03/head_poc atlas_poc
cd atlas_poc
```

`env.sh`가 HF/pip/torch/matplotlib 캐시를 전부 `~/jbwon` 안으로 접는다. 이걸 먼저 하지 않고
`pip install`이나 모델 다운로드를 하면 수십 GB가 `~/.cache`에 떨어진다.

```bash
source ~/jbwon/atlas_poc/env.sh   # 이 시점엔 Miniforge 경고가 뜨는 게 정상 (A-2에서 설치)
```

매 세션 이걸 `source` 하는 것을 습관화한다.

### A-2. Miniforge + venv (시스템 python 3.8.19는 못 씀)

시스템 python 3.8.19로는 `transformers==4.51.3`(≥3.9), `torch`≥2.5(≥3.9),
`agentdojo`(≥3.10)가 전부 설치되지 않는다. Miniforge는 단일 디렉토리 자기완결형이라
root도 빌드 의존성도 필요 없고, `-b` 플래그로 `.bashrc`도 건드리지 않는다.

> ⚠️ **`conda create`를 쓰지 않는다.** conda는 환경을 만들 때 `~/.conda/environments.txt`를
> **홈에** 기록한다 — "`~/jbwon` 밖 수정 금지" 규칙에 걸린다. 대신 Miniforge의 python으로
> **평범한 venv**를 만든다. conda 명령 자체를 안 쓰므로 홈이 전혀 오염되지 않는다.

```bash
cd ~/jbwon
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh -b -p ~/jbwon/miniforge3

# base python 버전 확인 — 3.10~3.12 범위여야 한다
~/jbwon/miniforge3/bin/python --version

# 그 python으로 venv 생성 (conda create 아님)
~/jbwon/miniforge3/bin/python -m venv ~/jbwon/envs/atlas

source ~/jbwon/atlas_poc/env.sh   # 이제 envs/atlas가 활성화된다
python --version                   # 3.10~3.12 확인
which python                       # ~/jbwon/envs/atlas/bin/python 이어야 함
```

base python이 3.13 이상이라 `transformers==4.51.3`이 안 맞으면, 그때만 conda로 특정 버전을
받는다 (`env.sh`가 `CONDA_PKGS_DIRS`/`CONDA_ENVS_DIRS`를 `~/jbwon`으로 돌려두므로 패키지
캐시는 안전하다. 다만 `~/.conda/environments.txt` 한 줄은 생기니 그때 교수님께 알릴 것):

```bash
~/jbwon/miniforge3/bin/conda create -y -p ~/jbwon/envs/atlas python=3.11
```

### A-3. PyTorch — Turing(sm_75)용

로컬 5070Ti와 달리 cu128이 강제되지 않는다. pytorch.org가 안내하는 stable CUDA 빌드를
쓰되, **sm_75 커널이 실제로 들어 있는지 반드시 검증한다** (없으면 조용히 CPU로 폴백하거나
`no kernel image is available` 에러가 난다):

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126   # 예시

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))   # Titan RTX면 (7, 5)
print("bf16 supported:", torch.cuda.is_bf16_supported())    # Turing이면 False가 정상
x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
print("fp16 matmul ok:", bool(torch.isfinite(x @ x).all()))
PY
```

**`arch list`에 `sm_75`가 있는지가 핵심**이다. 없으면 `capability`가 (7,5)로 잘 나와도
실제 연산에서 `no kernel image is available for execution`이 난다.

> ℹ️ **`bf16 supported: True`가 나오는데, 하드웨어 지원이라는 뜻은 아니다.**
> 최신 PyTorch의 `torch.cuda.is_bf16_supported()`는 기본값이 `including_emulation=True`라
> **bf16 하드웨어가 없어도 에뮬레이션이 가능하면 True**를 반환한다.
> Titan RTX는 `capability (7,5)`(Turing)라 bf16 하드웨어가 없다. `env.json`의
> `gpus[].bf16_supported`는 compute capability(`major >= 8`)로 판정하므로 `False`로 찍힌다 —
> 둘이 달라도 정상이다.
>
> **그래도 dtype은 bf16을 쓴다** (부록 A-5). 에뮬레이션 오버헤드가 실측 +13%에 불과했고,
> fp16은 NaN 때문에 못 쓴다.

### A-4. 나머지 의존성

> ⚠️ **먼저 `PYTHONNOUSERSITE`가 걸려 있는지 확인할 것.** conda env는 venv와 달리
> `~/.local/lib/pythonX.Y/site-packages`를 그대로 읽는다. 이 서버의 `~/.local`에는 다른
> 사용자가 깐 패키지가 많아서, 그대로 두면 **`pip install X==핀버전`이 "Requirement already
> satisfied"로 건너뛰고 남의 버전을 쓰게 된다** — 핀이 무력화되고 재현성이 깨진다.
> `env.sh`가 `PYTHONNOUSERSITE=1`을 export하므로 **source한 셸에서 설치할 것.**
>
> ```bash
> python -c "import site; print(site.ENABLE_USER_SITE)"   # False 여야 함
> pip list | wc -l    # 우리가 깐 것만 나와야 함 (수십 개, 100개 넘으면 새는 중)
> ```

```bash
pip install -r requirements.txt

# ⚠️ 필수 후처리 — torchvision을 torch와 같은 인덱스에서 다시 받는다
pip uninstall -y torchvision
pip install torchvision --index-url https://download.pytorch.org/whl/cu126   # A-3과 동일 URL
```

> ⚠️ **`torchvision` ABI 불일치 (실측 2026-08-21).** `lxt`는 import-time에 torchvision을
> 요구한다 (`lxt/efficient/models/__init__.py` → `vit_torch` → `from torchvision.models
> import vision_transformer`). 그래서 `pip install -r requirements.txt`가 torchvision을
> **기본 PyPI에서** 끌어오는데, torch를 pytorch 인덱스에서 받았다면 빌드가 달라 아래로 죽는다:
>
> ```
> RuntimeError: operator torchvision::nms does not exist
> ```
>
> transformers의 `image_utils`가 torchvision을 opportunistic import하기 때문에
> **`transformers.models.*` import 전체가 실패**한다. 증상이 lxt/transformers 쪽으로
> 보이지만 원인은 torchvision이다. 위 uninstall→재설치로 해결된다.

`requirements.txt`의 핀은 2026-08-21 서버 첫 설치에서 동작이 확인된 조합이다.
조합을 바꿀 때는 각 실행의 `env.json`에 남은 `packages`와 대조할 것.

InjecAgent는 저장소에 포함돼 있지 않으므로 따로 받는다:

```bash
git clone https://github.com/uiuc-kang-lab/InjecAgent.git external_injecagent
```

### A-5. `--dtype` — **bf16을 쓴다** (fp16 금지)

모든 실행 스크립트에 `--dtype {auto,bf16,fp16,fp32}`가 있고, **`auto`는 CUDA에서 항상
bf16**을 고른다. 해결된 값은 콘솔 첫 줄(`[env] ... dtype=bf16 ...`), 결과의 `env.json`,
`summary.txt` 헤더, 그리고 **결과 폴더 이름**에 기록된다.

```bash
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
  --model Qwen/Qwen2.5-7B-Instruct --family qwen2 --four_bit --topk 20
```

> ⚠️ **`--dtype fp16`을 쓰지 말 것 (실측 2026-08-21).** Qwen2 계열은 특정 레이어의
> activation이 fp16 최대값(65504)을 넘고, LRP 패치된 backward가 그 값을 타면 NaN이 된다.
> 0.5B 기준 **템플릿 3개 중 1개**에서 relevance 140/336개가 NaN이었다.
> 타깃 로짓 scale을 0.01/1/100으로 바꿔도 NaN 개수가 동일해 **loss scaling으로도 못 살린다**
> (scale 불변 = forward 쪽에서 이미 터졌다는 뜻). 판별 도구: `tools/diag_dtype.py`.
>
> Titan RTX에 bf16 하드웨어는 없지만 에뮬레이션 오버헤드가 **fp32 대비 +13%**에 불과했고
> (0.5B, 18.3s → 20.7s), bf16은 지수 범위가 fp32와 같아 이 문제가 원천적으로 안 난다.

> ℹ️ **기존 결과와 비교 가능하다.** Colab/5070Ti의 모든 이전 실험이 bf16이었으므로,
> 서버 bf16 결과를 같은 표에 놓을 수 있다.

> ⚠️ **NaN 가드는 dtype과 무관하게 항상 확인할 것.** 이번에 잡힌 실패 모드는 "에러 없이
> 그럴듯한 숫자가 나오는" 것이었다 — `aggregate_scores`가 NaN을 평균에 섞으면
> `topk_heads`가 점수 순서가 아니라 **인덱스 순서**((0,0),(0,1),…)를 반환해
> `jaccard=0.000`, `control_heads_both=[]`로 찍힌다. 결과를 볼 때 **먼저** 확인할 것:
>
> ```bash
> grep nan_excluded results/<run>/summary.txt        # run_pipeline
> grep -o '"n_nan_skipped": [0-9]*' <heads>.json     # compare_head_sources
> ```
>
> 0이 아니면 그만큼 줄어든 표본에서 나온 수치다. 비율이 높으면 `--dtype fp32`와 대조한다.

### A-6. GPU 점유 — 실행 전 매번 확인

`env.sh`가 `gpu_free` 함수를 정의해 둔다.

```bash
gpu_free
```

비어 있는 GPU만 골라 쓴다. **3장을 관성적으로 다 잡지 않는다** — 32B 4bit도 24GB 한 장에
들어가므로 1장으로도 진행된다. 3장이 다 비어 있으면 한 모델을 쪼개지 말고 **독립 실험 3개를
동시에** 돌리는 편이 낫다:

```bash
tmux new -s jbwon-gpu0    # 세션 이름에 식별자+GPU를 넣어 남이 오해하지 않게
CUDA_VISIBLE_DEVICES=0 python compare_head_sources.py discover-parallel --source agentdojo ...
```

SSH가 끊겨도 실행이 유지되도록 **긴 실행은 반드시 tmux 안에서** 돌린다.

### A-7. 결과 회수

서버는 **실행 전용**이다. 코드는 `git pull`만 하고, 편집은 로컬에서 한다
(docs/plan-2026-08-26.md 1.5절). 결과만 커밋해서 올린다:

```bash
git config user.name "Won"                  # --global 쓰지 말 것
git config user.email "jongbinwon@gmail.com"
git add results/ && git commit -m "..." && git push
```

`results/`는 실행마다 새 폴더라 로컬 작업과 구조적으로 충돌하지 않는다.

---

## 부록 B. 공용 서버 aisec-king (RTX PRO 4500 32GB + RTX A6000 48GB) 절차 (신설 2026-08-31)

**서버 이전.** 기존 부록 A(Titan RTX 24GB×3, Turing sm_75)는 그 서버 재현용으로 남겨둔다.
이 서버는 GPU 세대가 완전히 다르고(RTX PRO 4500=Blackwell급, A6000=Ampere sm_86 — 둘 다
`major>=8`이라 **bf16 하드웨어 네이티브 지원**, Turing처럼 에뮬레이션이 아님), 계정(`jbwon`)이
서버 홈 자체를 소유해 부록 A의 "~/jbwon 하위에서만" 강제가 그대로 적용되진 않는다. 그래도
공용 서버(다른 사용자·GPU 존재, sudo 금지)라는 제약은 동일하다.

### B-0. 지켜야 할 제약

- 다른 사용자 GPU 프로세스가 있으면 그 GPU는 건드리지 않는다 (kill 절대 금지)
- **`sudo` 금지**, **`pip install --user` 금지**, **`git config --global` 금지**(저장소 로컬만)
- 홈(`/home/jbwon`)은 이 계정 전용이라 부록 A처럼 별도 하위 폴더를 강제할 필요는 없지만,
  캐시/venv는 여전히 **저장소 디렉토리 안**(`.cache/`, `envs/`, `miniforge3/` — 전부
  `.gitignore` 처리됨)에 가둔다 — `env.sh`가 2026-08-31부로 `$JB`를 하드코딩된 경로 대신
  **자기 자신이 있는 디렉토리(=저장소 루트)를 자동 감지**하도록 바뀌어서 별도 설정 없이 됨.

### B-1. 저장소

이미 `~/head_poc`에 clone돼 있다면:

```bash
cd ~/head_poc
git pull
```

처음이라면:

```bash
git clone http://github.com/jongbin03/head_poc ~/head_poc
cd ~/head_poc
```

```bash
source ~/head_poc/env.sh   # $JB=~/head_poc로 자동 잡힘, Miniforge 없다는 경고는 B-3 전까진 정상
```

### B-2. 시스템 python 버전부터 확인

부록 A가 Miniforge를 강제했던 건 그 서버 시스템 python이 3.8.19였기 때문(transformers
>=3.9, agentdojo >=3.10 요구). 이 서버는 다를 수 있으니 먼저 확인:

```bash
python3 --version
```

- **3.10 이상이면** B-3(Miniforge)을 건너뛰고 시스템 python으로 바로 venv를 만든다:
  ```bash
  python3 -m venv "$JB/envs/atlas"
  source ~/head_poc/env.sh    # 방금 만든 venv를 활성화 분기가 자동으로 잡음
  python --version             # 3.10+ 확인
  which python                 # $JB/envs/atlas/bin/python 이어야 함
  ```
  이 경우 B-3은 건너뛰고 B-4로 간다.
- **3.10 미만이면** B-3(Miniforge)으로 간다.

### B-3. (시스템 python이 3.10 미만인 경우만) Miniforge + venv

부록 A-2와 동일한 이유(conda create 대신 venv, `~/.conda/environments.txt` 오염 방지)로
Miniforge의 python을 써서 평범한 venv를 만든다. 절차는 부록 A-2 그대로, 경로만 `$JB`
(=`~/head_poc`)로 자동 치환된다:

```bash
cd "$JB"
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh -b -p "$JB/miniforge3"

"$JB/miniforge3/bin/python" --version          # 3.10~3.12 범위여야 함
"$JB/miniforge3/bin/python" -m venv "$JB/envs/atlas"

source ~/head_poc/env.sh
python --version                                 # 3.10~3.12 확인
which python                                     # $JB/envs/atlas/bin/python 이어야 함
```

### B-4. PyTorch — 두 GPU 다 cu128 이상 하나로 충분

RTX PRO 4500(Blackwell급)은 로컬 5070Ti와 같은 이유로 **cu128 이상 필수**(구버전 CUDA
빌드는 커널이 없어 실패하거나 조용히 CPU로 폴백). RTX A6000(Ampere sm_86)은 cu128 wheel도
하위 호환 커널을 포함하므로 **같은 설치 하나로 두 GPU 다 커버된다** — 부록 A처럼 GPU마다
다른 index-url을 쓸 필요가 없다.

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**검증 — 두 GPU 다 확인할 것:**

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(f"[{i}]", torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
    # major>=8이면 bf16 하드웨어 네이티브 지원 (부록 A의 Turing 에뮬레이션 경고는 여기 해당 없음)
    print(f"    bf16 native:", torch.cuda.get_device_capability(i)[0] >= 8)
PY
```

기대 출력 예: `[0] NVIDIA RTX PRO 4500 ... (12, 0)`, `[1] NVIDIA RTX A6000 ... (8, 6)`,
둘 다 `bf16 native: True`. `cuda available`이 `False`거나 디바이스가 안 잡히면 여기서
멈추고 드라이버(`nvidia-smi` 상단 CUDA 버전, 이 서버는 13.3으로 확인됨 — cu128보다
훨씬 최신이라 호환 문제는 없어야 함)부터 재확인한다.

### B-5. 나머지 의존성 — 부록 A-4와 동일

```bash
python -c "import site; print(site.ENABLE_USER_SITE)"   # False 여야 함 (env.sh의 PYTHONNOUSERSITE)
pip install -r requirements.txt

# ⚠️ 필수 후처리 — torchvision을 torch와 같은 인덱스에서 다시 받는다 (lxt가 import-time에
# torchvision을 요구, ABI 불일치 시 "operator torchvision::nms does not exist"로 죽음.
# 원인/증상 상세는 부록 A-4 참고 — GPU 세대와 무관하게 여전히 적용됨)
pip uninstall -y torchvision
pip install torchvision --index-url https://download.pytorch.org/whl/cu128
```

외부 벤치마크 clone(전부 저장소 안, gitignore 처리됨 — 이 서버엔 아직 없으므로 새로 받아야 함):

```bash
git clone https://github.com/uiuc-kang-lab/InjecAgent.git external_injecagent
git clone https://github.com/agiresearch/ASB.git external_asb   # P15, docs/todo.md 참고
```

### B-6. `--dtype` — bf16 (부록 A와 같은 결론, 이유는 다름)

`--dtype auto`가 CUDA에서 항상 bf16을 고른다. 부록 A는 "하드웨어엔 없지만 에뮬레이션
오버헤드가 작아서" bf16을 택했지만, 이 서버는 **두 GPU 다 bf16 하드웨어를 실제로 갖고
있어**(B-4 검증) 애초에 에뮬레이션 이슈 자체가 없다. fp16을 피해야 하는 이유(Qwen2 계열
activation이 fp16 최대값을 넘어 NaN, `tools/diag_dtype.py`로 판별 가능)는 GPU 세대와
무관하게 동일하게 적용된다 — 부록 A-5 참고.

### B-7. GPU 점유 확인 — 두 장 다 매번 확인

```bash
gpu_free
```

이 서버는 GPU가 2장(PRO 4500 32GB, A6000 48GB)뿐이라 부록 A의 "3장 관성적으로 다 잡지
않기" 조언이 더 중요하다 — 필요한 크기에 맞는 GPU 한 장만 고르고, 나머지는 남겨둔다.
32B 4bit급이면 PRO 4500(32GB) 한 장으로 충분할 수 있고, 그보다 크면 A6000(48GB)을 쓴다.
SSH 끊김 대비 tmux 사용은 부록 A-6와 동일.

### B-8. 결과 회수 — 부록 A-7과 동일

```bash
git config user.name "Won"
git config user.email "jongbinwon@gmail.com"
git add results/ && git commit -m "..." && git push
```
