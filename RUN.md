# 실행 가이드 — 5070 Ti (16GB)에서 IPI read/control head PoC 돌리기

전제: Windows, Git Bash(또는 PowerShell) 사용. 아래 명령은 Git Bash 기준이며,
PowerShell을 쓸 경우 `source .venv/Scripts/activate` 대신
`.venv\Scripts\Activate.ps1`을 쓴다.

모든 명령은 이 디렉토리(`atlas_poc/`)에서 실행한다.

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

```bash
python run_pipeline.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --family qwen2 \
  --device cuda \
  --topk 20
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
   `eager_attention_forward` 패치가 깨진 회귀).

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

OOM이 나면 `--model`을 다시 1.5B로 낮추고, 그래도 나면 `dataset.py`의
템플릿 문장 길이를 줄인다 (attention tensor 메모리가 시퀀스 길이 제곱에 비례).

---

## 5단계 — (선택) `dataset.py` 템플릿 확장

`sample_templates()`가 지금 템플릿 2개뿐이라 통계적으로 얇다. 20~30개로
슬롯(이름/시간/금액/공격 문구)을 늘리는 걸 권장 — 이건 실험 설계 판단이
들어가는 부분이라 별도로 상의해서 진행.

---

## 6단계 — Phase 4: 8B급 스케일 검증 (선택)

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
