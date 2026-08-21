# IPI Read/Control Head 분리 PoC — 실험 방법론

**질문**: 외부 데이터(툴 결과/RAG retrieval)를 다룰 때, 모델 안에서 (1) 그 데이터의 **내용을
읽는 데** 관여하는 attention head와 (2) 그 데이터 속 **명령을 실행에 옮기는 데** 관여하는
attention head가 서로 분리되는가? 분리된다면, (2)만 골라 무력화해도 (1)은 보존되는가?

이 문서는 그 질문에 답하기 위해 **실제로 구현되어 실행 중인 방법론**을 설명합니다
(설계 구상이 아니라 코드가 하는 일 그대로). Atlas of In-Context Learning(NeurIPS'25)의
head 분리 방법론을 IPI(Indirect Prompt Injection) 시나리오에 맞게 응용했습니다.

실행 방법은 [run-guide.md](run-guide.md), 실행 결과는 [../results/](../results/), 다음 할 일은
[todo.md](todo.md) 참고.

> ⚠️ 이 문서는 **코드가 하는 일**을 그대로 기술한다. 그 방법론에 어떤 교란 요인이 있는지,
> 그래서 현재 수치를 어디까지 믿을 수 있는지는 [review-2026-07-29.md](review-2026-07-29.md)를
> 반드시 함께 볼 것.

---

## 1. 전체 파이프라인

```
dataset.py  ──▶  attn_relevance.py  ──▶  head_ranking.py  ──▶  edge_ablation.py
(프롬프트 생성)   (head별 relevance 산출)  (top-K head 랭킹 +      (control head의
                                          겹침 분석)              D_inj-edge만 차단)
```

`run_pipeline.py`가 이 네 단계를 순서대로 실행합니다 (`[1/4]`~`[4/4]`).

---

## 2. Head를 찾는 방법론 — AttnLRP relevance (`attn_relevance.py`)

### 2.0 Atlas / AttnLRP / lxt — 셋의 관계

이 세 이름이 문서·코드에 섞여 나와 혼동하기 쉬운데, **경쟁 관계가 아니라 층위가 다릅니다.**

| | 무엇 | 출처 |
|---|---|---|
| **AttnLRP** | attribution **방법론**. LRP를 트랜스포머용으로 확장해, 비선형 구간(RMSNorm, softmax)에서 gradient가 왜곡되는 걸 보정한다. backward **한 번**으로 `R(x\|y)`("이 성분이 출력 로짓 y에 얼마나 기여했나")를 얻는다 | Achtibat et al., **ICML 2024** |
| **lxt** | 그 방법론의 **레퍼런스 구현 라이브러리**. `monkey_patch`가 HF 모델링 모듈의 forward를 갈아끼워, 평범한 `.backward()`가 raw gradient 대신 LRP relevance를 흘리게 만든다 | **같은 저자** (Achtibat) |
| **Atlas** | AttnLRP를 **가져다 쓰는 연구**. 새 attribution 방법을 만들지 않는다 — 논문에 *"we adopt AttnLRP [2]"*, *"AttnLRP [2], **on which our method is based**"*로 명시 | NeurIPS'25, **저자에 Achtibat 포함** |

즉 **lxt는 Atlas의 대안이 아니라, Atlas의 방법이 돌아가는 엔진**입니다.
Atlas = *무엇을 계산할지*, AttnLRP = *어떻게 relevance를 정의할지*, lxt = *그 계산을 수행하는 코드*.
세 작업이 사실상 같은 그룹에서 나와서 조합이 자연스럽습니다.

**Atlas가 AttnLRP 위에 얹은 것** — AttnLRP는 "relevance를 구하는 법"까지만 준다. Atlas는
거기에 (1) head 단위로 attribute하고, (2) key 위치별로 집계하고, (3) top-K로 고르고,
(4) knockout으로 인과성을 검증하는 절차를 추가했다. 우리가 응용한 게 이 절차다.

#### ⚠️ Atlas에는 head relevance 정의가 **두 개** 있다 — 우리가 쓰는 건 Eq. 7/8

혼동의 실제 원인이 대개 여기다.

| | 무엇에 대한 relevance | 논문에서 쓰는 곳 |
|---|---|---|
| **Eq. 4** | head의 **latent output** `z^h_i` — 차원 `d_h`와 위치 `i`를 전부 합산 → **head당 스칼라 1개** | §5, in-context vs parametric head 분리 |
| **Eq. 7/8** | **attention weight** `A^h_{i,j}` — **key 위치 `j`별** 점수 | §6, head가 *어느 토큰을 읽는지* 특화 분석 |

논문 원문:

```
ρ^h_j    = Σ_{i=1..S} R+( A^h_{i,j} | y_t )                       ... Eq. 7  (p.7)
ρ^h_task = Σ_{j∈J_task} ρ^h_j ,   ρ^h_ret = Σ_{j∈J_ret} ρ^h_j     ... Eq. 8  (p.8)
```

**우리는 Eq. 7/8을 쓴다.** "어느 head가 주입된 span을 읽는가"를 알아야 하므로 key 위치별
분해가 필수인데, Eq. 4는 head당 스칼라라 그걸 줄 수 없다. 논문의 `J_task`/`J_ret`(질문 토큰 /
검색된 답 토큰) 자리에 우리는 `data_benign`/`data_inj`를 넣었다 — 2.3절 표 참고.

> 이 선택의 부작용: **Eq. 7이 key 위치별 분해라서, attribution이 특정 위치로 쏠리는
> 버그에 직접 노출된다.** lxt가 Qwen3에 붙인 "attribution skewed toward first token" 경고가
> 우리에게 치명적인 이유가 이것이다 (2.2절). Eq. 4 방식이었다면 위치를 합쳐버리니
> 덜 민감했을 것이다.

### 2.1 어떤 head가 "기여했는가"를 재는 방법

attention 확률(얼마나 쳐다봤는지)만 보면 "쳐다보긴 했지만 실제로 안 썼다"를 구분하지
못합니다. Atlas 논문(AttnLRP, Achtibat et al. ICML'24 기반)은 여기에 **gradient**를 곱해서
"실제로 최종 출력에 기여한 정도"를 잽니다:

```
relevance = attention_weight * attention_weight.grad     # post-softmax, per-head, per-(i,j)
ρ^h_j     = Σ_i ReLU(relevance)[i, j]                     # key 위치 j에 대해 모든 query i를 합산
```

구현 순서 (`compute_head_relevance`):
1. `output_attentions=True` + `attn_implementation="eager"`로 모델을 호출해, 모든 layer의
   attention 텐서 `(batch, heads, seq_i, seq_j)`를 그대로 받음 (`sdpa`/`flash_attention_2`는
   이 텐서를 아예 만들지 않으므로 반드시 `eager`가 필요).
2. 각 attention 텐서에 `.retain_grad()`를 걸어, backward 이후에도 gradient가 남게 함.
3. 관심 있는 출력 토큰(예: 악성 토큰 "attacker")의 로짓 **하나**에 대해 `backward()`.
4. `attention * attention.grad`를 계산하고 음수는 버림(ReLU) → 이게 `(layer, head, i, j)`별
   relevance.
5. 우리가 관심 있는 key 위치 그룹(예: D_inj 토큰들)에 대해 `j` 축을 합산 → `(layer, head)`
   하나당 점수 하나.

### 2.2 AttnLRP를 얻는 방법 — `lxt` 패키지

Atlas 원 코드는 HF 모델링 코드를 통째로 포크해서 backward 규칙을 손으로 다시 짠 버전을
씁니다. 여기서는 같은 저자 그룹이 배포하는 **`lxt.efficient.monkey_patch`**를 대신 씁니다:

```python
from lxt.efficient import monkey_patch
from transformers.models.qwen2 import modeling_qwen2
monkey_patch(modeling_qwen2)   # RMSNorm/gated-MLP/attention의 backward 규칙을 교정
```

표준 `Qwen2ForCausalLM`을 그대로 쓰면서 AttnLRP-정합적인 backward를 얻습니다.

**지원 범위** (lxt 2.1 소스 `lxt/efficient/models/__init__.py`의 `DEFAULT_MAP` 직접 확인,
2026-08-21):

| lxt가 지원하는 것 | 우리 코드가 분기하는 것 |
|---|---|
| llama · qwen2 · **qwen3** · **gemma3** · bert · gpt2 · vit | **qwen2 · llama 뿐** (`attn_relevance.py:37-43`) |

`monkey_patch(module)`은 `DEFAULT_MAP`에서 patch를 자동으로 찾으므로 **lxt 쪽에 할 일은
없습니다.** 다른 모델을 쓰려면 `attn_relevance.py`에 분기를 추가하면 되고, 모델당 ~6줄입니다.

다만 lxt README의 지원 상태 표에 단서가 있습니다:

- **Qwen3 = 🧪 "Attribution skewed toward first token"** — 2.0절 마지막 경고 참고.
  우리 방법론(Eq. 7, key 위치별 분해)에 직접 타격이므로 **쓰기 전에 position별 relevance
  분포 진단이 선행돼야 합니다.**
- **Gemma 3 = ✅** (lxt의 대표 예시로 쓰일 만큼 검증됨). 단 4B 이상은 멀티모달 래퍼
  (`Gemma3ForConditionalGeneration`)라 텍스트 타워 로드 경로를 따로 잡아야 하고,
  대부분 레이어가 sliding-window local attention이라 **긴 프롬프트에서 D_inj span이
  윈도우 밖으로 나가면 그 레이어 head의 relevance가 구조적으로 0**이 됩니다.
- lxt 2.1은 `transformers==4.52.4`로 테스트됐다고 명시. 우리 핀은 4.51.3 — qwen2/llama
  경로는 실측으로 검증됐지만 qwen3/gemma3 경로는 이 조합에서 확인된 적이 없습니다.

**참고**: Atlas 원 논문이 쓴 모델은 Llama-3.1-8B-Instruct / Mistral-7B-Instruct-v0.3 /
Gemma-2-9B-it 세 개인데, 이 중 현재 lxt로 바로 되는 건 **Llama-3.1-8B 하나뿐**입니다
(Mistral·Gemma-2는 `DEFAULT_MAP`에 없음).

**⚠️ gradient checkpointing과 상극**: checkpointing을 켜면 backward 시점의 attention
텐서가 forward 때 잡아둔 것과 다른(재계산된) tensor가 되어 `.grad`가 채워지지 않습니다.
`attn_relevance.py`는 항상 checkpointing을 끈 상태로 씁니다.

### 2.3 세 종류의 relevance = 세 종류의 "무엇에 기여했는가"

`run_pipeline.py`가 각 템플릿마다 3번의 relevance 계산을 돌립니다 (타깃 토큰과 key 그룹이
다름):

| 이름 | 타깃 로짓(무엇의 확률에 backward) | key 그룹(어디를 봤는지 합산) | 의미 |
|---|---|---|---|
| `read_score` | 정상 답 토큰 (예: " 3pm") | `data_benign` (정상 내용) | 이 head가 **정상 내용을 읽는 데** 기여했는가 |
| `internal_score` | 악성 토큰 (예: "attacker"), 자유 텍스트 응답 | `data_inj` (주입된 명령) | 이 head가 **자유 텍스트 응답 오염에** 기여했는가 |
| `external_score` | 악성 토큰, tool-call JSON 인자 | `data_inj` | 이 head가 **tool 호출 오염에** 기여했는가 |

`read_score`는 주입문이 아예 없는 `read_clean` 프롬프트에서 재서, 공격의 영향을 받지 않은
"순수 read 능력" 기준선으로 삼습니다.

---

## 3. Head 랭킹 & 겹침 분석 (`head_ranking.py`)

1. 30개 템플릿 전체에서 각 relevance를 평균 → `(layer, head)`별 점수 하나씩 3세트.
2. 점수 상위 K개(기본 20개)를 뽑아 `read_heads` / `internal_heads` / `external_heads` 세
   집합을 만듦.
3. 세 집합끼리 **Jaccard 유사도**(교집합/합집합)를 계산:
   - `jaccard(read, internal)`, `jaccard(read, external)`이 낮다 → read와
     instruction-following이 서로 다른 head에서 일어난다는 신호.
   - `jaccard(internal, external)`이 높다 → 자유 텍스트 오염과 tool-call 오염이 **같은
     head 집합**을 공유한다는 신호 (control head 하나만 찾아도 두 공격 경로를 동시에
     방어할 수 있다는 근거).
4. `internal_heads ∩ external_heads` = **control head 후보 리스트** (`control_heads_both`).
   다음 단계(edge 차단)의 대상이 됨.
5. `functional_map.png`에 layer×head 좌표로 색을 찍어 시각화 (빨강 = control head 후보).

**아직 다루지 않는 것**: `internal_heads`와 `external_heads`의 **차집합**(대칭차) — "명령을
따르기로 한 뒤, 자유 텍스트로 낼지 tool-call로 낼지를 가르는 head"는 이 분석에 없음.
[todo.md](todo.md) 참고.

---

## 4. Edge를 차단하는 방법론 (`edge_ablation.py`)

### 4.1 head를 끄는 게 아니라, **edge**를 끊는다

여기서 가장 중요한 설계 선택: control head 후보가 나와도 **그 head 전체를 죽이지 않습니다.**
대신 **"그 head가 D_inj(주입된 명령 텍스트) 위치를 보는 attention 연결선(edge)만"**
차단합니다. head는 다른 위치(D_benign, 질문 등)에 대해서는 그대로 정상 작동합니다.

이렇게 하는 이유: head를 통째로 끄면 그 head가 담당하던 다른 정상 기능까지 같이 사라질 수
있습니다. edge 단위로 좁게 개입하면, "이 head가 명령이 심긴 부분을 보는 것"만 막고 나머지
기능(예: 같은 head가 D_benign을 보는 것)은 그대로 둘 수 있습니다.

### 4.2 구현: pre-softmax 마스킹

`attn_implementation="eager"`로 로드하면 HF는 attention 계산을 다음 함수로 위임합니다:

```python
attention_interface = eager_attention_forward   # modeling_qwen2 모듈의 전역 함수
```

`edge_knockout()` 컨텍스트 매니저는 이 전역 함수 자체를 바꿔치기해서, 지정한
`{layer: [head, ...]}` 목록에 대해서만 다음을 수행합니다:

1. 정상적으로 `Q·Kᵀ`를 계산해 pre-softmax attention score를 얻음.
2. 대상 (layer, head) 조합에 대해, **key 위치가 D_inj 토큰 범위인 열만** `-inf`로 채움
   (`masked_fill`).
3. 그 상태로 softmax를 취함 → `-inf`였던 위치들의 확률이 정확히 0이 됨. 즉 그 head는
   D_inj 토큰들을 attention 확률 0으로 "보이지 않게" 만들지만, 나머지 위치에 대한 attention
   분포는 정상적으로 재정규화됨.
4. 나머지 layer/head는 원래 `eager_attention_forward`와 동일하게 동작.

GQA(grouped-query attention, Qwen2.5가 사용) 대응을 위해 `repeat_kv`로 key/value를 query
head 수에 맞춰 복제한 뒤 계산합니다 (원본 HF 구현과 동일한 처리).

### 4.3 Sweep — knockout하는 head 개수를 늘려가며 효과 측정

`sweep_knockout()`은 `head_ranking.py`에서 나온 `control_heads_both`를 점수 순으로 정렬한
뒤, 상위 `k`개(기본 `0, 5, 10, 20, 40, 80`)의 D_inj-edge를 차단하면서 두 지표를 측정합니다:

- `malicious_token_prob`: 악성 토큰이 나올 확률 (external 프롬프트 기준) — **공격 성공률
  proxy**. 이게 k가 커질수록 떨어져야 방어가 되는 것.
- `read_token_prob`: 정상 답 토큰이 나올 확률 (read_injected 프롬프트 기준, 즉 주입문이
  있는 상태에서 여전히 정답을 말하는지) — **정상 기능 보존(utility) proxy**. 이게 k가
  커져도 유지되어야 "부작용 없이" 방어된 것.

이 sweep은 순수 forward 연산(gradient 불필요)이라 4bit 양자화 모델에도 그대로 적용할 수
있습니다 (`attn_relevance.py`의 backward 요구와 달리, checkpointing/양자화와 무관하게
항상 안전).

---

## 5. 데이터셋 설계 (`dataset.py`)

- 콘텐츠 도메인 6종(이메일/문서/캘린더/주문/HR/인보이스) × 공격 문구 스타일 5종(명령형/
  권위사칭/가짜승인/가짜주석/긴급성) = 템플릿 30개.
- 각 템플릿마다 4가지 mode로 프롬프트를 만듦: `read_clean`(주입 없음, read 기준선),
  `read_injected`(주입 있음, utility 측정용), `internal`(자유 텍스트 오염 측정),
  `external`(tool-call 오염 측정).
- **세그먼트별로 미리 토크나이즈하며 이어붙여 span을 구성 시점에 정확한 정수 인덱스로
  기록**합니다 (`PromptBuilder`). 문자열을 통째로 만든 뒤 나중에 substring 위치를 다시
  찾는 방식은 토큰화 경계 문제로 버그가 나기 쉬워 피했습니다.
- `assistant_prefix`(예: `"Answer:"`)와 타깃 토큰의 앞 공백 유무를 반드시 짝지어야
  합니다 — BPE에서 단어 시작 토큰은 앞 공백을 포함해서 인코딩되는 경우가 많기 때문.

---

## 6. 알려진 함정 / 구현 노트

- **`attn_implementation="eager"` 필수**: `sdpa`/`flash_attention_2`는 attention 행렬을
  아예 만들지 않아 relevance도, edge knockout도 불가능.
- **gradient checkpointing과 `retain_grad()`는 상극** (§2.2) — `attn_relevance.py`에서는
  항상 끔.
- **`READ_PREFIX` 설계가 모델 크기에 민감함**: 원래 `"The answer is"`를 썼는데, 3B 이상
  모델은 그 뒤에 곧바로 정답을 말하지 않고 `"The answer is that ~~~"`처럼 우회하는 경우가
  많아 즉시-다음-토큰 확률이 노이즈 수준으로 낮게 측정되는 문제가 있었습니다. `"Answer:"` +
  system prompt에 "부연설명 없이 값만 답하라"는 지시를 추가해 해결 (자세한 경위는
  [../results/2026-07-27_colab_phase1to3/README.md](../results/2026-07-27_colab_phase1to3/README.md) 참고).
- **`model.generate()`에는 `attention_mask`를 명시적으로 넘길 것**: 안 넘기면 HF가
  "input_ids 중 pad_token_id와 같은 값은 패딩"이라고 자동 추정하는데, pad_token_id ==
  eos_token_id인 모델에서는 프롬프트 안에 정상적으로 반복 등장하는 `<|im_end|>` 같은
  토큰이 진짜 내용인데도 마스킹되어버립니다 (`debug_read_target.py`에서 확인된 이슈,
  `run_pipeline.py`는 `generate()`를 안 써서 영향 없음).

---

## 7. 파일 구성

- `dataset.py` — synthetic IPI 템플릿/프롬프트 생성 (§5)
- `attn_relevance.py` — lxt monkey-patch + AttnLRP 기반 head relevance 산출 (§2)
- `head_ranking.py` — top-K head 랭킹 + Jaccard 겹침 분석 + functional map 시각화 (§3)
- `edge_ablation.py` — control head의 D_inj-edge 차단 + knockout sweep (§4)
- `run_pipeline.py` — 위 네 개를 엮은 진입점
- `debug_read_target.py` — read_target 토큰이 실제 모델 응답과 맞는지 직접 확인하는
  디버깅 스크립트
- `head_poc.ipynb` — Colab 실행 노트북 (VS Code Colab GPU 연결용)
- `adapters/injecagent.py` — 외부 벤치마크(InjecAgent) 어댑터

문서 전체 목록은 [프로젝트 루트 README](../README.md) 참고.
