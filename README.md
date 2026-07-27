# IPI Read/Control Head 분리 PoC — 설계 노트

교수님 지시사항: **Atlas of In-Context Learning의 방법론을 이용해, 외부 데이터(툴 결과/RAG
retrieval)에 대해 (1) read에 관여하는 attention과 (2) 그 데이터 안의 명령을 따라
[내부 행동 / 외부 write-tool 호출]에 영향을 미치는 attention이 서로 구분되는지** 확인.

이 폴더는 그 실험을 위한 최소 실행 가능 스캐폴드입니다. `pkhdipraja/in-context-atlas`
(Atlas 논문 공식 코드)의 알고리즘을 그대로 계승하되, 구현은 훨씬 가볍게 갑니다 —
이유는 아래 "구현 전략" 참고.

---

## 0. 핵심 아이디어 (Atlas → IPI 번역)

Atlas 논문은 QA 세팅에서 head를 두 축으로 나눕니다.

- **§5**: in-context head (문맥을 씀) vs parametric head (내부 지식을 씀) — open-book vs
  closed-book 대비로 구분
- **§6**: in-context head를 다시 **task head**(질문=intensional frame을 해석) vs
  **retrieval head**(정답 객체를 문맥에서 복사) 로 세분화 — AttnLRP relevance를
  질문 토큰 위치(J_task)에 줄지, 정답 토큰 위치(J_ret)에 줄지로 구분 (Eq. 7-9)

우리 문제(IPI)에서 이 §6의 task/retrieval 분리가 정확히 필요한 축입니다:

| Atlas 논문 | idea1 / 우리 실험 |
|---|---|
| 질문(intensional frame)을 이해하는 task head | **주입된 명령(D_inj)을 "명령"으로 받아들여 실행에 옮기는 control head** |
| 문맥에서 정답을 복사하는 retrieval head | **정상 데이터(D_benign)를 "내용"으로 읽어들이는 read head** |
| target logit y_t = 정답 토큰 | target logit = (아래 3가지 중 하나) |

교수님이 요구한 3가지 구분을 위해 target logit을 3종류로 둡니다:

1. **y_read**: 정상 질문에 대한 올바른 답 토큰 (예: "3pm") → 이때 J_ret 자리에 D_benign,
   J_task 자리에 질문(question) 토큰을 넣고 ρ^h_read = Σ_{j∈D_benign} relevance
2. **y_internal**: 주입된 명령이 (tool 호출 문법 없이) 그냥 assistant의 **자유 텍스트 답변**을
   오염시키는 경우의 악성 토큰 (예: "attacker") → ρ^h_internal = Σ_{j∈D_inj} relevance
3. **y_external**: 동일한 악성 토큰이지만, 이번엔 tool-call 포맷 (`{"name": "forward_email",
   "arguments": {"to": "**attacker**..."}}`) 안에서 생성되는 경우 → ρ^h_external

세 head 집합(top-K)의 겹침을 비교하면 교수님 질문에 정량적으로 답할 수 있습니다:
- read ∩ internal, read ∩ external 이 작아야 함 (read와 instruction-following이 분리됨)
- internal ∩ external 이 크면 → "내부 행동"과 "외부 tool 호출"이 **같은 control head 회로**에서
  나온다는 뜻 (idea1의 "control head를 한 번만 찾아서 knockout하면 둘 다 막힌다"는 전제를 지지)
- internal ∩ external 이 작으면 → tool-call 포맷팅이 별도의 head를 추가로 동원한다는 뜻
  (idea1의 방어 범위를 tool-call 상황까지 넓히려면 head 탐색을 프롬프트 포맷별로 따로 해야 함)
  — 이 자체가 흥미로운 발견이라 negative result여도 보고할 가치가 있습니다.

---

## 1. 구현 전략: Atlas 리포를 그대로 포팅하지 않는 이유

`in-context-atlas` 리포의 `model_src/model_src/llama.py` 등은 **HF Llama 모델링 코드를
통째로 포크해서 LRP용으로 손으로 다시 짠 것**입니다 (1500줄+, `######## <--- LRP` 주석으로
표시된 부분들). 이건 논문 저자들이 "mathematically explicit" 버전을 쓴 것이고, 새 아키텍처
(Qwen2.5 등)로 확장하려면 이 포크 작업을 처음부터 다시 해야 해서 PoC 단계에는 비효율적입니다.

대신 같은 저자 그룹(Achtibat et al., AttnLRP ICML'24)이 배포하는 **`lxt` 패키지**
(`pip install lxt`, https://github.com/rachtibat/LRP-eXplains-Transformers) 의
"efficient" 구현을 씁니다:

```python
from lxt.efficient import monkey_patch
from transformers.models.qwen2 import modeling_qwen2
monkey_patch(modeling_qwen2)   # RMSNorm/gated-MLP/attention의 backward 규칙을 한 줄로 교정
```

이러면 `model_src/*.py`를 손으로 옮겨 쓸 필요 없이, `transformers`의 표준 `Qwen2ForCausalLM`을
그대로 쓰면서 AttnLRP-정합적인 relevance(= `activation * activation.grad`)를 얻습니다.
**Qwen2는 lxt 공식 지원 목록에 ✅로 있고, Llama 2/3도 ✅** 입니다 (Qwen3는 첫 토큰으로
쏠리는 버그가 있다고 README에 명시되어 있어 이번 PoC에서는 피합니다).

또한 Atlas 리포는 `preprocessing/filter_heads_composition.py`에서 relevance를 얻기 위해
자기 모델 안에 `self_attn.softmax`라는 커스텀 서브모듈을 만들어 forward hook을 겁니다.
우리는 그럴 필요 없이 **`output_attentions=True` + `attn_implementation="eager"`** 로
호출하면 HF가 layer마다 `(batch, num_heads, seq_i, seq_j)` 모양의 post-softmax attention
tensor를 그대로 돌려주므로, 여기에 `.retain_grad()`만 걸면 Atlas Eq.(7)의
`ρ^h_j = Σ_i R+(A^h_{i,j})`를 그대로 얻습니다. (`sdpa`/`flash_attention_2`는 attention
행렬을 아예 만들지 않으므로 반드시 `eager`를 명시해야 합니다.)

### ⚠️ 중요한 함정: gradient checkpointing과 `retain_grad()`는 상극
`model.gradient_checkpointing_enable()`을 켜면 체크포인팅된 구간의 중간 tensor(여기선
layer별 attention weight)는 backward 시점에 **재계산되는 별도의 tensor**가 되기 때문에,
forward 시점에 잡아둔 tensor에 `.retain_grad()`를 걸어도 `.grad`가 채워지지 않습니다
(조용히 `None`으로 남아서 디버깅이 매우 괴로운 버그입니다 — Phase 0에서 반드시 이걸로
한 번 죽어보고 넘어가는 걸 추천합니다).

→ **head-level relevance를 뽑는 단계(`attn_relevance.py`)에서는 gradient checkpointing을
끄고**, 그 대신 메모리가 부족하면 모델 크기를 줄이거나 batch=1 + 짧은 프롬프트로 갑니다.
(lxt의 공식 quantized 예제들은 checkpointing을 쓰지만, 그건 **embedding-level** relevance
`input_embeds.grad`만 뽑기 때문에 문제가 없는 것 — embedding은 체크포인팅 경계 바깥의
진짜 leaf tensor라서 안전합니다.)

4bit 양자화(`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`)
자체는 lxt 공식 예제(`examples/quantized_qwen2.py`, `quantized_llama.py`)에서 이미
검증되어 있으므로, **checkpointing만 끄면** 7B~8B급에서도 head-level relevance 추출이
될 가능성이 높습니다 (다만 아직 저희 쪽에서 실측은 안 했으니 Phase 3에서 먼저 1.5B로
파이프라인을 검증한 뒤 8B로 올릴 때 VRAM을 직접 재보시길 권합니다).

---

## 2. 5070 Ti (16GB)에서 쓸 로컬 모델 추천

| 용도 | 모델 | 정밀도 | 이유 |
|---|---|---|---|
| Phase 0 (파이프라인 뼈대 디버깅) | **Qwen2.5-0.5B-Instruct** 또는 Llama-3.2-1B-Instruct | bf16 | 거의 즉시 로드, GPU 메모리 걱정 없음. span 계산/hook 코드의 자잘한 버그를 여기서 다 잡기 |
| Phase 1~3 (본 실험: read/control head 분리 + edge knockout 검증) | **Qwen2.5-1.5B-Instruct** 또는 **Qwen2.5-3B-Instruct** | bf16, checkpointing **끔** | lxt 공식 지원, tool-calling 포맷(Hermes 스타일 `<tool_call>` JSON) 지원, backward(relevance) 포함해도 16GB에 여유 있음 |
| Phase 4 (스케일 검증/일반화) | **Qwen2.5-7B-Instruct** 또는 Llama-3.1-8B-Instruct | 4bit (bnb) + bf16 compute, checkpointing **켬** (단, embedding-level relevance만 가능) | lxt 공식 4bit 예제로 검증된 조합. head-level relevance까지 필요하면 checkpointing을 끄고 VRAM이 버티는지 먼저 확인 — 8B bf16 무양자화는 weight만 16GB라 backward에는 그대로 못 씀 |
| Edge blocking / path patching (forward-only, gradient 불필요) | 위 모든 모델 + 4bit 가능 | 4bit 무관 | 순수 forward 개입이라 양자화와 무관하게 항상 잘 됨 (기존 Phase 1 계획과 동일) |

Qwen2.5 계열을 기본으로 미는 이유: (1) lxt가 Qwen2를 공식 지원, (2) Qwen2.5-Instruct가
Hermes 스타일 `<tool_call>` JSON 함수 호출 포맷을 기본 chat template으로 지원해서
"외부 tool에 영향" 시나리오(y_external)를 별도 파싱 없이 자연스럽게 만들 수 있음.

---

## 3. 기존 4-phase 계획과의 관계 (업데이트)

이전에 정리했던 Phase 0~4 계획에서 "AttnLRP는 제일 무거우니 Phase 3까지 아껴둔다"고
했었는데, `lxt`의 efficient 구현을 쓰면 AttnLRP 한 번 뽑는 비용이 **"gradient-based
importance score(§4.3, backward 1회)"랑 사실상 같은 비용 클래스**입니다 (그냥 backward
1회 + attention tensor에 retain_grad 거는 것뿐). 즉:

- **Phase 1(edge blocking, forward-only)** 은 그대로 최저비용 스캐닝 단계로 유지
- **Phase 3(AttnLRP)** 을 굳이 마지막까지 아낄 필요 없이, **read/control head 분리 자체를
  Phase 1과 병행해서 먼저 돌려도 됩니다** — 오히려 이게 이번 교수님 질문(구분되는가?)에
  가장 직접적으로 답하는 실험이라 우선순위를 올리는 걸 추천합니다.
- Path patching(Phase 2)은 이 AttnLRP 기반 head 랭킹에서 나온 top-K 후보에 대해서만
  돌리면 되므로 순서는 그대로: **[Phase 0 디버깅] → [AttnLRP로 read/control head 분리 +
  edge knockout 1차 인과검증] → [path patching으로 정밀 검증] → [8B로 일반화 확인]**

---

## 4. 파일 구성

- `dataset.py` — synthetic IPI 템플릿 생성기. 문자열을 만들고 나중에 substring을 찾는
  대신, **세그먼트 단위로 미리 토크나이즈해서 이어붙이며 span을 구성 시점에 정확히 기록**
  합니다 (Atlas 리포가 `find_tokens_range`/offset mapping으로 씨름했던 버그 클래스를
  원천 차단).
- `attn_relevance.py` — lxt monkey-patch + `output_attentions=True`로 layer×head별
  relevance(ρ^h_read, ρ^h_internal, ρ^h_external) 추출.
- `head_ranking.py` — 데이터셋 전체에 대해 relevance를 평균 내고 top-K head 랭킹 + 세
  집합 간 Jaccard overlap 계산 + layer×head scatter plot (Atlas Fig.1/3 스타일).
- `edge_ablation.py` — Phase 1/4용 forward-only edge knockout. 지정한 (layer, head)
  목록에 대해 D_inj 쪽 attention만 pre-softmax에서 `-inf` 처리(= edge blocking, Cutting-
  off-the-Head §4.2 방식) 후 악성 토큰 확률 변화 / 정상 read 정확도 변화를 측정.
- `run_pipeline.py` — 위 네 개를 엮은 Phase 0~1 smoke test 진입점.

**주의**: 이 코드는 이 대화에서 실제 GPU/모델로 실행 검증된 것이 아닙니다 (샌드박스에
huggingface.co 접근과 GPU가 없음). Phase 0 원칙대로, 5070 Ti에서 Qwen2.5-0.5B로 먼저
돌려서 shape/버그를 잡은 뒤 본 실험으로 넘어가시길 권합니다.

```bash
pip install lxt transformers accelerate bitsandbytes matplotlib --break-system-packages
```
