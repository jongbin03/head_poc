# 교수님 피드백 대응 결과 (2026-07-29 피드백 → 2026-07-31 결과 정리)

> 원 피드백은 [feedback-2026-07-29.md](feedback-2026-07-29.md), 피드백 다음날 나온 자체
> 방법론 검토는 [review-2026-07-29.md](review-2026-07-29.md), 작업 진행 로그(버그/시행착오
> 포함 상세 기록)는 [todo.md](todo.md) 참고. 이 문서는 그 세 문서에 흩어진 내용을 "무엇을
> 물었고, 무엇을 했고, 무엇이 나왔고, 무엇이 문제인가" 순서로 한 번에 정리한 결과 보고서다.

## 0. 원 피드백 요약

1. AgentDojo 데이터셋으로 다시 돌려보고, 성능(utility) 측정 방법도 제대로 된 걸로 할 것
2. 키 그룹을 2개로 했는데 합성 데이터셋 모드는 왜 4개인지, `read_injected`는 왜 있는지
3. Attention head 찾는 방법을 다시 체계화할 것

아래 1~4절이 순서대로 이 세 가지에 대한 답이다.

---

## 1. "키 그룹 2개 vs 데이터셋 모드 4개" — 정의 재확인

**결론**: 헷갈렸던 이유는 "키 그룹"이라는 말이 실은 4가지 다른 것(조사 개수, 선정에 쓰는
그룹 수, span 종류, 데이터셋 모드 수)을 동시에 가리키고 있었기 때문이다.

| 구분 | 개수 | 목록 |
|---|---|---|
| head를 찾는 "조사"(relevance 계산 단위) | 3개 | read, internal, external |
| 그중 control head 선정(교집합)에 쓰이는 것 | 2개 | internal, external |
| 조사가 보는 "창문"(span 종류) | 2개 | `data_benign`(정상 내용), `data_inj`(숨은 명령) |
| `dataset.py`의 프롬프트 모드 | 4개 | `read_clean`, `read_injected`, `internal`, `external` |

- **internal/external은 "정상 vs 침입"이 아니다.** 둘 다 `data_inj`(침입 문구) 창문을
  보고, 차이는 오직 **backward 타깃**(주입된 명령이 자유 텍스트로 새는 경로 vs tool-call로
  새는 경로)뿐이다. 즉 "2 groups = 정상 vs 주입"이 아니라 "2 groups = 침입의 두 실행
  경로"다.
- **read**는 세 번째 조사로 따로 계산되지만(`data_benign` 창문, 주입 없는 `read_clean`
  프롬프트) control head 선정 교집합에는 참여하지 않는다 — read/internal/external이
  얼마나 겹치는지 보는 대조군이다.
- **`read_injected`는 head를 찾는 데 안 쓰인다.** head 탐색이 끝난 뒤, 찾은 head를
  실제로 knockout했을 때 "정상 업무는 유지되는가"(utility)를 재는 2단계 검증에서만
  쓰인다. `read_clean`은 숨은 명령 자체가 없어서 knockout할 대상(D_inj 엣지)이 없으므로
  이 검증에 못 쓰고, 숨은 명령이 있으면서 질문은 정상 질문인 `read_injected`가 필요하다.
- InjecAgent/AgentDojo(실제 벤치마크 2종)는 tool-calling 단일 포맷이라 애초에
  internal/external 구분이 구조상 없다(P2-d에서 확인) — 즉 이 2×2 구조는 우리 합성
  데이터셋에서만 성립하고, 실제 벤치마크로 갈수록(4절) "채널 축"을 유지하기 어렵다는
  것도 이번에 명확해졌다.

## 2. Head 탐색 방법론 — 무엇을, 어떻게 찾는가

### 2.1 방법론의 뿌리 (Atlas, NeurIPS'25)

Head를 찾는 핵심 도구는 [Atlas of In-Context Learning](https://arxiv.org/html/2505.15807v2)
(NeurIPS'25)의 AttnLRP 기반 relevance 방식이다:

```
relevance = attention_weight * attention_weight.grad   # post-softmax, per-head, per-(i,j)
ρ^h_j     = Σ_i ReLU(relevance)[i, j]                    # key 위치 j로 들어오는 relevance 합
```

관심 있는 출력 토큰 하나의 로짓에서 backward한 뒤, 관심 있는 key 위치 그룹(span)에 대해
합산하면 head 하나당 점수 하나가 나온다. 이걸 30개 템플릿(또는 InjecAgent/AgentDojo
케이스)에서 평균 내 top-K를 뽑는다.

Atlas 논문 자체도 "기준 하나"로 head를 찾지 않는다 — (1) 개방형/폐쇄형 두 조건의 차이로
in-context/parametric head를 가르고, (2) in-context head 안에서 다시 task-토큰/retrieval-
토큰 두 그룹에 대해 따로 relevance를 계산해 task-head/retrieval-head를 가른다. 우리는 이
"여러 기준으로 계산해서 비교"하는 틀을 그대로 가져오되, 기준의 내용만 IPI 방어라는
문제에 맞게 새로 설계했다 — "정상 읽기(read) vs 공격 실행(internal/external)", 그리고
공격 실행 안에서 "어떤 형태로 실행되는지(말 vs 버튼)"라는 축.

### 2.2 실제 탐색 절차

1. `attn_relevance.compute_head_relevance`로 read/internal/external 세 종류의 relevance를
   계산 (모델별 전체 (layer, head) 점수).
2. `head_ranking.py`가 top-K(기본 20) 랭킹 + Jaccard 겹침 계산.
3. `internal_heads ∩ external_heads` = `control_heads_both` — 두 실행 경로 모두에서
   공통으로 상위권인 head, 다음 단계(edge 차단)의 대상.
4. `edge_ablation.py`가 그 head들의 **D_inj로 가는 attention edge만** 차단(head 전체를
   끄지 않음 — 같은 head가 정상 내용을 보는 기능은 보존)한 뒤, 악성 토큰 확률(공격 성공
   proxy)과 정상 답 토큰 확률(utility proxy)을 sweep.

### 2.3 방법론 진단 (P7, 2026-07-31, `compare_head_sources.py` 이전 버전)

교수님이 지적한 "체계화"에 정면으로 답하기 위해, 지금까지의 결론이 실제로 유효한지부터
진단했다 (1.5B, 템플릿 30개 전체):

| 진단 항목 | 결과 |
|---|---|
| 랜덤 head 기준선 | 실제 선정 head(9~14개)는 k=10 안에 `malicious_token_prob` 0.91→**0.0000**(완전 억제). 동일 개수의 랜덤 head는 k=40까지 늘려도 0.91→**0.6448**까지만 하락 — head 선정이 랜덤보다 뚜렷이 유효함 |
| jaccard 우연 기준선 | jaccard(internal,external)=0.538, 우연 기대값=0.031 → **17.6배**, 명백히 유의 |
| `dual_use_candidates`(신규 노출) | `control_heads_both` 14개 중 **9개**가 read top-20과도 겹침 — "read와 control이 완전히 분리된 회로"라는 주장은 유지 불가, "internal-external끼리의 겹침이 read와의 겹침보다 크다"는 상대적 주장만 성립 |
| layer 0 지배 | 14개 중 **6개가 layer 0** — "정보 대역폭 차단" 대안 가설을 아직 배제 못 함 |
| 문서-코드 불일치 | 메인 sweep이 실제로 쓰는 top-40(union 기반) 랭킹과 `control_heads_both`(교집합) sweep 결과가 거의 동일 — 헤드라인 수치 자체는 안전함 확인 |

이 결과로 "head 탐색 자체는 유효하다"는 결론을 내리고, `docs/todo.md`의 P8(합성
데이터셋의 content-availability 교란 제거)은 **보류**하기로 했다 — synthetic은
discovery(head 찾기)에만 계속 쓰고, 발표용 헤드라인 성능 수치는 AgentDojo 네이티브
채점(3절)에 맡기기로 함.

---

## 3. AgentDojo로 재현 + 3개 소스 head 비교

### 3.1 왜 "재현" 대신 "탐색 소스"로 썼는가

P2-d(2026-07-28)에서 이미 synthetic-only / InjecAgent-only / 교집합 세 조건이 거의 동일한
성능(jaccard=0.36)을 보인 바 있다 — "synthetic으로 찾은 head를 실제 벤치마크에 전이"시키는
구조가 synthetic의 한계(노골적 문구, 단순 문서 구조)를 상속할 수 있다는 신호였다. 그래서
AgentDojo를 "단순 재현 대상"이 아니라 **head 탐색 자체의 세 번째 소스**로 다루기로
설계를 바꿨다.

### 3.2 데이터셋 3종

| 소스 | 성격 | 규모 | internal/external 구분 |
|---|---|---|---|
| synthetic (`dataset.py`) | 자체 제작, 이메일/문서 등 6도메인×5문구스타일=30템플릿 | 30 | 있음(설계상 유일하게 있음) |
| InjecAgent (`adapters/injecagent.py`) | 외부 벤치마크, tool-calling 단일 포맷 | 1,054 case | 없음 |
| AgentDojo (`adapters/agentdojo.py`, 신규) | 외부 벤치마크, `pip install agentdojo`, banking/slack/travel/workspace 4 suite | 949 (user_task×injection_task) 조합 중 필터 통과 **220개** | 없음 |

AgentDojo 어댑터(Track A, head 탐색 전용) 설계: `GroundTruthPipeline`(LLM 없이
`task.ground_truth()`의 tool 호출을 그대로 실행)으로 "완벽한 에이전트"의 메시지 시퀀스를
얻고, `[assistant(tool_call), tool(응답), assistant(다음 tool_call)]` 3-메시지 패턴(공격
문구가 첫 tool 응답에 바로 등장하는 단순 case)만 골라 InjecAgent 어댑터와 같은 단일 턴
프롬프트로 만든다. 구현 중 실제로 부딪힌 문제 3가지:

1. **YAML 개행 정규화**: `environment.yaml`이 Python `.format()` 삽입 후 `yaml.safe_load`를
   거치며 주입 문구의 연속 개행이 미세하게 바뀌어, 공격 문구 리터럴 그대로 매칭하면 전부
   실패 → `<INFORMATION>`/`</INFORMATION>` 태그를 앵커로 찾는 방식으로 수정.
2. **인자 오염형 공격**: 일부 case는 "다음 정상 행동"과 "공격자 목표 행동"이 **같은
   tool**(인자만 다름, 예: `send_money`를 정상 수취인 vs 공격자 수취인으로)이라 tool-이름
   기준 next-token proxy로 공격 성공/실패를 구분할 수 없음 — feedback.md 1-1절이 예상했던
   문제가 실제로 존재함을 확인, 이런 case는 건너뛰고 tool **이름 자체**가 달라지는
   "tool 선택형" case만 사용.
3. **토큰 충돌**: tool 이름 문자열은 달라도 토큰화 후 첫 토큰이 같은 경우(예:
   `get_balance`/`get_iban`이 공통 접두사 공유)가 있어, 문자열이 아니라 실제 토큰 id
   기준으로 다시 필터링.

### 3.3 3소스 head 비교 결과 (1.5B, `compare_head_sources.py`)

InjecAgent/AgentDojo는 채널 구분이 없으므로, internal/external 축 대신 **소스 축**을
교집합·비교의 기본 틀로 삼았다 — "어느 소스로 찾은 head가 다른 소스에도 잘 전이되는가."

| 비교 | jaccard | 우연 대비 |
|---|---|---|
| synthetic ↔ InjecAgent | 0.308 | 12.2배 |
| synthetic ↔ AgentDojo | 0.214 | 8.5배 |
| InjecAgent ↔ AgentDojo | 0.290 | 9.5배 |

- head 집합 크기: synthetic 14개, InjecAgent 20개, AgentDojo 20개.
- **3소스 교집합 = 5개, 전부 layer 0**: `(0,1) (0,3) (0,6) (0,7) (0,10)`. 2.3절에서 나온
  "layer 0 지배" 우려가 소스를 3개로 늘려도 그대로 남아있다는 뜻 — 다음 단계(4절)에서
  이 5개만 knockout했을 때 정말 효과가 있는지 반드시 확인해야 함.
- 2소스씩 교집합은 6~9개, 3소스 합집합은 36개. 세 jaccard 모두 우연 대비 8.5~12.2배로
  유의하지만, 세 소스가 완전히 같은 head를 찾은 것도 아님(교집합 5~9개 vs 각 소스
  14~20개) — "어느 정도는 공통, 어느 정도는 소스별로 다름"이 정확한 요약.

**부수 성과**: AgentDojo head 탐색이 (긴 시퀀스로 인해) 기존에 알려진
`compute_head_relevance`의 GPU 메모리 누적 버그를 InjecAgent보다 훨씬 심하게 유발함을
발견 — `todo.md`에 이미 적혀 있던 근본 해결책(배치마다 완전히 새 서브프로세스=새 CUDA
컨텍스트)을 `discover-parallel`로 실제 구현해 단일 프로세스 12%(18/150) → 배치 분리
79%(118/150)로 수율을 끌어올림. 이 해결책이 실제로 효과가 있다는 걸 이번에 처음 확인했고,
향후 다른 대량 relevance 계산(예: InjecAgent head_n 확장)에도 재사용 가능.

---

## 4. AgentDojo 네이티브 평가 (Track B) — 진짜 utility 측정

### 4.1 왜 필요한가

`review-2026-07-29.md` 2절이 지적한 문제: InjecAgent의 "utility 상승" 수치는 독립적인
증거가 아니라 공격 성공률 하락의 산술적 뒷면이다(같은 프롬프트, 같은 위치에서 정답
tool 이름과 공격자 tool 이름이 서로 경쟁하는 토큰이라서). 합성 데이터셋도 `read_injected`
축이 있어 구조적으로는 이 문제가 없지만, 애초에 "정말 명령을 따르기로 결정하는가"가
아니라 "이미 forward하기로 정해진 상태에서 대상만 받아쓰는가"를 재고 있다는 게 P8의
우려였다(보류됨). **AgentDojo의 네이티브 채점**(모델 출력이 아니라, 실제로 tool을
실행한 뒤 환경 상태 — 계좌 잔액, 전송된 메일 등 — 을 검사하는 결정론적 함수)은 이 두
문제를 한꺼번에 해결한다.

### 4.2 하네스 구성

- `adapters/agentdojo_pipeline.py`(`KnockoutLocalLLM`): AgentDojo의 `AgentPipeline`/
  `ToolsExecutionLoop`(실제 멀티턴 tool 호출 루프, agentdojo가 이미 제공)에 우리 HF 모델을
  직접 끼워 넣는 커스텀 파이프라인 요소. `model.generate()`를 `edge_knockout()` 컨텍스트
  안에서 호출해, 롤아웃 내내(여러 번의 `generate()` 호출에 걸쳐) D_inj 엣지를 끊는다.
  ⚠️ agentdojo 기본 tool-call 문법(`<function=name>{...}</function>`)은 1.5B가 안 따라서
  (마크다운 코드블록으로 응답, 파싱 실패) **Qwen2.5 네이티브 `<tool_call>{"name":...,
  "arguments":{...}}</tool_call>` 포맷**(`tokenizer.apply_chat_template(..., tools=...)`)
  으로 교체.
- `run_agentdojo_eval.py`: (user_task, injection_task) 쌍마다 k=0(방어 없음) vs
  k=지정 head(knockout)로 두 번 롤아웃해 `TaskSuite.run_task_with_pipeline`의 네이티브
  utility/security를 비교. `--attack`으로 agentdojo에 등록된 공격 기법(아래 4.4) 아무거나
  지정 가능.

### 4.3 실행 결과

**1.5B, banking 15쌍**: 공격 성공률 **0/15** (k=0에도 0). 모델이 tool을 안 부르고
되묻기만 해서, 애초에 주입 문구가 담긴 tool 응답에 도달을 못 함 — knockout 효과를 볼
공격 표본 자체가 없음.

**7B(4bit), 4 suite(banking/slack/travel/workspace) × 12쌍=48쌍, `important_instructions`
공격**:

| | utility | 공격 성공률(security) |
|---|---|---|
| k=0 (방어 없음) | 0.188 | **0.021** (1/48) |
| k=14 (synthetic 기반 control head knockout) | 0.188 (변화 없음) | **0.000** (0/48) |

성공한 유일한 공격(`slack/user_task_10+injection_task_1`)이 knockout으로 실패로 바뀌었고
utility는 전혀 안 깎임 — 지금까지의 모든 결과(P7 랜덤 기준선, jaccard 유의성)와 방향이
일치하는 **첫 실제 멀티턴 신호**다.

suite별 utility 편차가 컸다: banking 0.42, slack 0.25, workspace 0.08, **travel 0.00**
(12쌍 전부 실패) — 7B도 travel suite는 거의 못 풂.

### 4.4 공격 기법을 바꿔도 결과가 안 바뀜

공격 성공률(2%)이 너무 낮은 게 "공격 문구가 약해서"인지 확인하려고, agentdojo가 이미
등록해둔 공격 중 가장 노골적인 **`tool_knowledge`**(어떤 tool을 어떤 인자로 부를지까지
문구에 못 박아 넣어 추론 부담을 없앰)로 교체해 동일 조건으로 재실행.

| | 문구 | 요구하는 것 |
|---|---|---|
| `important_instructions` | "정기결제 수취인을 이 계좌로 바꿔라" | 어떤 tool/인자를 쓸지 모델이 스스로 추론 |
| `tool_knowledge` | 위 문구 + "`update_scheduled_transaction`을 `{id, recipient}` 인자로 불러라" | 정답 tool/인자를 그대로 알려줌, 모델은 값만 채우면 됨 |

결과: 공격 성공률 **거의 그대로**(46쌍 중 1건 = 0.022, 이전 0.021과 사실상 동일). 성공한
case 자체는 바뀜(`banking/user_task_12+injection_task_7`)지만, **전체 성공률은 문구를
아무리 노골적으로 바꿔도 안 오름** — 그 case도 knockout으로 실패로 바뀜(방향은 동일).

**재해석**: 문제는 "모델이 공격을 이해 못 해서"가 아니라, **모델의 멀티스텝 tool 실행
자체가 전반적으로 신뢰도가 낮은 것**(utility도 20% 안팎)으로 보인다. 정상 과업이든 공격
이행이든 여러 tool 호출을 정확히 연속으로 해내야 성공하는 건 마찬가지라, 공격 문구를
더 명확하게 써도 "모델이 여러 단계를 안 틀리고 해낼 확률" 자체가 안 오르면 성공률도
안 오른다.

### 4.5 모델을 키워보면 어떨까 — 14B(bnb-4bit) 시도

4.4절 결론이 맞다면 남은 손잡이는 "모델의 tool-calling 실행 신뢰도 자체를 올리는 것"뿐이라,
16GB GPU에서 시도 가능한 최대치인 **`unsloth/Qwen2.5-14B-Instruct-bnb-4bit`**(사전 양자화
저장소, ~10GB)로 실제 확인했다. GGUF(Q4_K_M/Q5_K_M, llama.cpp 전용)는 `edge_knockout()`이
`transformers`의 attention 함수를 몽키패치하는 방식이라 애초에 호환되지 않아 제외 —
bitsandbytes 4bit을 그대로 유지.

48층×40헤드 모델에서 synthetic 30개 템플릿으로 새로 head 탐색: **13개**
`(0,0) (0,14) (0,27) (0,38) (29,11) (36,21) (36,22) (36,23) (37,16) (40,10) (42,5) (43,19)
(44,35)` — layer 0 지배(13개 중 4개)는 여기서도 유지, layer 36에서 인접 head 3개가
클러스터로 뽑히는 새로운 패턴도 관찰됨.

같은 조건(4 suite×12쌍, `important_instructions`)으로 7B와 비교:

| | 7B | 14B |
|---|---|---|
| k=0 utility | 0.188 | **0.267** |
| knockout 후 utility | 0.188 | 0.222 |
| k=0 공격 성공률 | 0.021 (1/48) | 0.022 (1/45) |
| knockout 후 공격 성공률 | 0.000 | 0.000 |

**결론**: utility는 실제로 오른다(banking 0.42→0.50, travel은 처음으로 0.00→0.11, workspace
0.08→0.17) — 모델을 키우면 정상 과업 수행력은 개선된다. 하지만 **공격 성공률은 그대로다
(~2%)** — 모델을 키운다고 "공격에 걸리는 비율" 자체가 줄거나 늘지 않는다. 즉 5.2절의
"모델 크기가 유일한 손잡이"라는 진단은 **절반만 맞다** — 정상 과업 능력은 모델 크기를
따라가지만, 이 데이터셋의 공격 성공률은 그와 독립적으로 낮게 유지된다.

(다운로드가 심하게 정체되거나(`hf_transfer` 미사용 시) 평가 도중 GPU 메모리 파편화로
급격히 느려지는(2시간 30분째 진행 정체) 문제를 겪었는데, 둘 다 해결법을 찾아 문서화함 —
자세한 내용은 `todo.md` 해당 절 참고.)

### 4.6 proxy 지표 vs 네이티브 채점 — 나란히 대조

같은 14B 모델·같은 13개 head로, 이번엔 synthetic/InjecAgent **proxy 지표**(다음 토큰 확률
기반, `edge_ablation.sweep_knockout`)도 재실행해 AgentDojo 네이티브 결과와 바로 대조했다:

| | k=0 (방어 없음) | k=13 (knockout) |
|---|---|---|
| synthetic (30개) 공격 성공 확률 | 1.0000 | **0.0000** |
| synthetic utility(read) | 0.9360 | 0.9515 |
| InjecAgent (1,054개) 공격 성공 확률 | 0.3347 | **0.0463** (7.2배↓) |
| AgentDojo (45~48쌍) 공격 **성공률** | 0.021~0.022 | 0.000 |

같은 모델, 같은 head인데 **synthetic/InjecAgent proxy 지표는 강하고 깔끔한 억제 효과를
보이고, AgentDojo 네이티브 평가는 애초에 baseline 공격 성공률 자체가 2%대로 매우
낮다.** 이 표 하나가 review-2026-07-29.md/P8이 처음 제기했던 "proxy 지표가 실제 공격
성공률을 부풀렸을 수 있다"는 우려를 정량적으로 뒷받침하는 가장 직접적인 증거다 —
"synthetic·InjecAgent가 보여주는 90%대/30%대 공격 성공률"과 "AgentDojo 멀티턴 환경의
실제 2%대 공격 성공률" 사이의 간극이 이 프로젝트가 처음부터 우려했던 "proxy 지표 vs
진짜 utility 측정" 문제 그 자체다.

---

## 5. 문제점 / 한계 정리

### 5.1 통계적 유의성 부족 (가장 중요)

48~45쌍 중 성공한 공격이 각 1건뿐이라 "knockout이 억제했다"는 결과 자체는 정직하지만
표본이 너무 얇아 통계적으로 거의 의미가 없다. 지금까지 나온 모든 정황(P7 랜덤 기준선,
jaccard 유의성, Track B의 1건, 4.6절의 proxy-vs-native 대조)이 같은 방향을 가리키긴
하지만, 신뢰 구간을 논할 수 있는 수준의 표본은 아직 없다. 표본이 얇은 근본 원인 자체가
5.6절 관찰(AgentDojo baseline 공격 성공률이 원래 2%대로 낮음)과 같은 뿌리다.

### 5.2 모델 크기는 utility만 개선하고, 공격 성공률은 안 바꿈

- 1.5B: tool을 거의 안 부르고 되묻기만 함 — 공격에 노출될 기회 자체가 없음.
- 7B(4bit): tool은 부르지만 인자를 잘못 타이핑(`bill-december-22023.txt` 같은 오타)하거나
  `arguments`를 dict가 아니라 리스트로 잘못 생성하는 등 실행이 자주 깨짐(파싱 실패는
  `adapters/agentdojo_pipeline.py`에서 방어적으로 처리해 sweep이 죽지는 않지만, 그
  case의 정보는 못 씀).
- **14B(bnb-4bit)로 실제 확인(4.5절)**: utility는 0.188→0.267로 개선되지만 **공격
  성공률은 0.021→0.022로 그대로**. 즉 "모델을 키우면 다 해결된다"는 예상은 절반만
  맞다 — 정상 과업 수행력은 모델 크기를 따라가지만, 낮은 baseline 공격 성공률(5.6절
  참고)은 모델 크기와 별개의 원인(아마도 AgentDojo 환경/공격 자체의 특성)일 가능성이
  높음. 4.4절(공격 문구 교체 무효)과 종합하면, 지금 시도한 두 손잡이(공격 문구·모델
  크기)로는 표본 부족 문제를 근본적으로 못 풀고, 하드웨어가 허용하는 한 최대 모델
  크기(16GB GPU에서 14B-4bit이 실용적 한계)까지 이미 시도했다는 뜻.

### 5.3 layer 0 지배 — 미해결 대안 가설

`control_heads_both`(synthetic)의 14개 중 6개, 3소스 교집합 5개 전부가 layer 0이다.
"명령을 명령으로 인식해 실행하는 회로"라는 해석과, "그냥 층 0의 정보 대역폭을 막은
것"이라는 대안 가설을 아직 실험으로 구분하지 못했다.

### 5.4 read/control 완전 분리 아님

`dual_use_candidates` 진단 결과 `control_heads_both` 14개 중 9개가 read top-20과도 겹친다
— "read와 명령 실행이 완전히 분리된 회로"라는 강한 주장은 못 하고, "internal-external
끼리의 겹침이 read와의 겹침보다 뚜렷이 크다"는 상대적 주장만 유지 가능.

### 5.5 synthetic 데이터셋의 content-availability 교란 (P8, 보류 상태)

`exec_target`이 30개 템플릿 전부 `"attacker"` 하나로 고정돼 있고 그 문자열이 오직
`data_inj` 안에만 존재해서, D_inj 엣지를 끊으면 그 글자 자체가 사라져 확률 붕괴가 거의
자동이라는 우려(review.md 1절)가 아직 안 풀렸다. P7의 랜덤 기준선 결과(랜덤도 어느 정도는
억제 효과가 있었음, 0.91→0.64)가 이 우려를 완전히 반박하지 못한다 — 오히려 방향이
일치한다. 지금은 "discovery 전용으로만 쓰고 헤드라인 수치는 AgentDojo에 맡긴다"로
봉합했지만, synthetic 자체의 신뢰도 문제는 남아있다.

### 5.6 InjecAgent utility 지표는 못 씀

`review-2026-07-29.md` 2절: InjecAgent의 exec/read 두 타깃이 같은 프롬프트, 같은 위치에서
경쟁하는 토큰이라 "utility 상승"이 공격 성공률 하락의 산술적 뒷면일 뿐 독립 정보가 없다.
이 프로젝트에서 InjecAgent의 utility 수치는 어디에도 인용하지 않기로 했다 — 이게
AgentDojo(4절)로 가야 했던 핵심 이유 중 하나.

### 5.7 인프라 안정성

`compute_head_relevance`의 프로세스 전역 GPU 메모리 누적 버그가 여전히 근본적으로는
안 고쳐졌다(배치 분리로 우회만 함). 7B/14B 평가 중 travel suite에서 CUDA OOM 여러 건
발생(스킵 처리는 되지만 그 case의 정보는 손실). 14B 다운로드/평가 과정에서 두 가지
운영 문제를 추가로 겪음: ① 기본 다운로더가 연결 수십 개를 열어둔 채 정체되는 문제
(`hf_transfer` + 사전 양자화 저장소로 해결), ② 평가 중 CUDA 메모리 파편화가 누적되며
갈수록 느려지는 문제(매 쌍마다 `gc.collect()`/`empty_cache()` 추가 + 
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`로 완화). 둘 다 근본적으로 안정된
인프라라기보다 그때그때 우회한 것이라, 표본을 훨씬 크게 늘리면 다시 부딪힐 가능성이
있다.

---

## 6. 다음 단계

1. **표본 확대**: 지금 표본(suite당 12쌍)은 신뢰도 있는 결론을 내기엔 부족 — 특히
   travel처럼 utility가 낮고 OOM도 잦은 suite는 배분 비중을 낮추고, 신호가 나오는
   suite(banking/slack)에 더 배분.
2. ~~**모델 크기**: 더 큰 모델을 시도~~ — 14B(bnb-4bit)까지 시도 완료(4.5절). utility는
   개선되지만 공격 성공률은 그대로라, 모델 크기만으로는 표본 부족 문제가 안 풀림이
   확인됨 — 이 손잡이는 사실상 소진.
3. **최종 head 집합 결정**: 세 소스 단독(특히 3소스 교집합 5개 layer-0 head) vs 합집합
   vs 교집합을 Track B로 평가해 최종적으로 어떤 조합을 쓸지 결정 — 지금은 synthetic
   유래 head(1.5B 14개/14B 13개)만 평가했고 InjecAgent/AgentDojo 유래 head, 합집합/
   교집합은 아직 안 돌려봄.
4. Top-K sweep + random-head baseline(P7 인프라)을 세 소스 전부에 동일 적용.
5. layer 0 지배(5.3) 원인 규명 — 정보 대역폭 차단 vs 명령 인식 회로.
6. AgentDojo baseline 공격 성공률 자체가 왜 낮은지(모델 크기와 무관하다는 게 4.5절로
   확인됐으므로) 원인 규명 — 공격 시나리오 자체의 특성인지, harness의 다른 한계인지.
7. 결과를 `methodology.md`/`run-guide.md`에 "Head Selection Methodology" 섹션으로 통합.

자세한 실행 로그(버그 수정 경위, 커밋 단위 진행 상황)는 [todo.md](todo.md)의 P4/P7/P8
절 참고.
