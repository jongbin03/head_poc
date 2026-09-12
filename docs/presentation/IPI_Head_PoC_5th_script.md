# 5차 발표 스크립트 (초안) — Llama-70B 전이헤드 confound 정정 + 세대축(Qwen3-8B) 교차검증

> 4차 발표(`IPI_Head_PoC_4th_script.md`, 확정 2026-09-07) 이후 피드백(A) "다른 모델
> 교차검증 — Qwen3-8B, Llama-70B"(`feedback-2026-09-06.md` §0)에 대응한 사이클의 결과.
> 4차와 같은 톤 — 배경→결과 순, 표 중심, 슬라이드당 결론 한 줄.
>
> ⚠️ **이 문서는 초안이다.** S10-S13(2026-09-12 세션에서 시작한 실험)은 세션 진행 중
> 작성돼 **[결과 대기]** 표시가 있는 자리가 있다 — 실험이 끝나는 대로 이 세션에서 채운다.
> `build_deck_5th.py`는 아직 없음 — 수치가 전부 확정된 뒤에 4th 패턴을 본떠 작성.
>
> 수치 출처: `docs/status-2026-09-08.md`, `docs/status-2026-09-09.md`(§1~§4, 2026-09-12
> 정정 포함), `results/2026-09-08_p12_llama70b/`(RUN_NOTES.md), `docs/feedback-2026-09-06.md`
> §1, `tools/diag_qwen3_relevance.py` 실행 로그(2026-09-12), `results/2026-09-12_p12_llama70b_headn80/`.

---

## S1. 타이틀

**Read Head, Control Head 분리 PoC — 스케일축 정정(Llama-70B) + 세대축(Qwen3-8B) 교차검증**

---

## S2. 서론 — 4차 발표 요약

- 파서(`agentdojo_default`) confound 아님 확정, 기본값 채택.
- 표본 확대(n=148)로 "스케일업 반례" 재확정 — **Qwen2.5-7B/Llama-8B는 slack knockout
  전량 억제, Qwen2.5-32B만 절반 이상 persist(8~9→5, backfire 1)**.
- 양자화: `fp4` 아티팩트 확정(`nf4+double_quant`로 해소, 기본값 교체). 32B bf16(양자화
  배제)에서도 slack knockout 불완전 — "32B 스케일 효과"로 읽었음.
- 다음 피드백(A): **Qwen3-8B·Llama-70B로 모델 교차검증** — "다른 모델·다른 스케일에서도
  같은 패턴인가?"

---

## S3. 이번 사이클 배경 — Llama-70B 1차 결과가 "스케일업 반례"를 더 강하게 재현했다

`results/2026-09-08_p12_llama70b/` — Llama-3.1-70B, **8B에서 찾은 헤드를 그대로 전이**해
slack held-out 35쌍 knockout.

| 모델 | k0 성공 | knockout 효과 | backfire | net |
|---|---|---|---|---|
| Qwen2.5-7B / Llama-8B bf16 | 6 | 6 → 0 전량 억제 | 0 | 0 |
| Qwen2.5-32B bf16 (자체 헤드) | 8~9 | 절반 이상 persist | 1 | 5 |
| **Llama-3.1-70B nf4dq (8B 헤드 전이)** | **5** | 1 억제 / 4 persist | **1** | **5 (순 억제 0)** |

- 언뜻 "대형 모델일수록 knockout에 저항한다"는 결론을 강화하는 것처럼 보였다 — 70B가
  세 지점 중 가장 극단.
- **그런데 이 70B 행은 70B 자신의 헤드가 아니라 8B에서 찾은 헤드를 그대로 썼다**(70B
  자체 Track A는 이 시점까지 `--device_map auto`의 backward OOM으로 실행 불가 — 아래
  S4). → "70B가 저항하는가" vs "8B의 (70B 기준으론 틀렸을 수 있는) 헤드를 껐을 뿐인가"를
  구분할 수 없는 상태로 다음 사이클에 들어감.

---

## S4. 실험① — Llama-70B Track A(헤드 탐색)를 막던 것은 하드웨어가 아니라 배선이었다

- 기존 결론("70B AttnLRP backward는 이 하드웨어에서 불가")은 `--device_map auto` 경로
  한정이었다. auto는 (a) 타이트한 `--max_memory`면 `CPU/disk 분산` 에러, (b) 느슨하면
  레이어가 한 GPU에 몰려 backward OOM — 그 사이에 열리는 창이 없었다.
- **해결**: `--device_map_plan "0:32,1:30,2:18"`(수동 device_map) 신설. embed/norm/
  lm_head는 첫 GPU에 몰아두고(backward가 양 끝에서 시작·수렴), 레이어는 순서대로 분배.
- 3-GPU 배선(A6000 48G root / Blackwell 32G / 4090 24G)으로 **130/130 성공, 0 oom, 0
  nan** — 부수 발견: Blackwell nf4 커널이 32B greedy 출력을 안 건드리는 데 이어 70B
  backward도 정상(4bit=A6000 고정 관례가 계속 완화됨).

---

## S5. 실험① 결과 — 70B 자체 헤드는 8B와 같은 상대적 깊이(레이어 위치)에서 나온다

`results/2026-09-08_p12_llama70b/heads_agentdojo.json`

| 항목 | Llama-8B | Llama-70B |
|---|---|---|
| head 위치 | layer 11–22 / 32 | layer 26–44 / 80 |
| 상대 깊이 | ≈ 38–47% | ≈ 35–44% |
| 재현성 | — | smoke(16)∩full(130) = 17/20 |

**같은 상대 깊이 대역** — 스케일이 8배 커져도 "주입 신호를 담당하는 레이어"의 상대
위치는 유지된다는 신호.

---

## S6. 실험② — "70B 자체 헤드" vs "8B 전이 헤드" knockout 정면 대조

같은 모델(Llama-3.1-70B nf4dq)·같은 slack 105쌍(`--eval_split all`)·같은 공격
(important_instructions)에서 **knockout에 쓰는 헤드 집합만** 8B 전이 ↔ 70B 자체로 교체.

| knockout 헤드 | k0 sec | kN sec | 억제/backfire | net | kN utility | parse_ok |
|---|---|---|---|---|---|---|
| 8B 전이 (S3과 동일 헤드) | 0.276 | 0.257 | 4 / 2 | **−2 (효과 없음)** | 0.190 | 0.782 |
| **70B 자체** | 0.276 | **0.181** | **13 / 3** | **−10 (ASR 34%↓)** | 0.190 | 0.783 |
| 70B 자체 (누수 없는 heldout 15쌍) | 0.533 | **0.200** | 5 / 0 | −5 | 0.333(무손상) | 0.853 |

- parse_ok율 두 조건이 동일(0.78) → security 하락이 "tool-call을 못 뱉어서"가 아님.
- utility는 오히려 소폭 상승 → 모델 손상이 아니라 **injection-following만 선택적으로
  억제**.
- **누수 없는 heldout이 누수 있는 all105보다 더 강한 효과** → all105의 결과가 누수로
  부풀려진 게 아님.

---

## S7. 실험② 결과 — 2번째 공격 축(tool_knowledge)으로 교차 확인

같은 구성, `--attack tool_knowledge`(32B에서 baseline ASR이 ~2배였던 더 강한 공격).

| knockout 헤드 | k0 sec | kN sec | 억제/backfire | net | kN utility |
|---|---|---|---|---|---|
| 8B 전이 | 0.402 | 0.392 | 3 / 2 | −1 (효과 없음) | 0.186 |
| **70B 자체** | 0.398 | **0.223** | **18 / 0** | **−18 (ASR 44%↓)** | 0.204 |
| 70B 자체 (heldout 15쌍) | 0.733 | **0.400** | 5 / 0 | −5 | 0.333(무손상) |

**공격을 바꿔도 같은 패턴** — 70B 자체 헤드는 backfire **0건**(더 깨끗함). → 헤드가 특정
공격 문구가 아니라 **일반 injection 신호**를 담는다는 근거가 공격-독립적으로 확정.

---

## S8. 판정 — "대형 모델이 저항한다"가 아니라 "전이 헤드는 스케일이 안 된다"

- **정정**: 4차 발표 시점의 "스케일업 반례" 서술 중 **Llama 계열 부분**은 70B의 저항이
  아니라 **8B 헤드가 70B에 전이되지 않았기 때문**이었다. 70B 자신의 헤드로 끄면 knockout이
  정상 작동한다(ASR 34~44%↓, utility 손상 0, 공격 2종 모두 재현).
- **Qwen 계열은 사정이 다르다** — Qwen2.5-32B의 "8~9→5" 결과는 처음부터 **32B 자체 헤드**
  (`results/2026-08-24_s4_32b/`)를 썼다(전이 아님, `feedback-2026-08-31.md:289`). 남은
  caveat은 그 헤드가 **fp4로 탐색**돼 nf4dq eval과 양자화가 안 맞는다는 것뿐 — 별개 축
  (`docs/todo.md` §4-1, 이번 사이클에선 미실행).
- **헤드 분리 가설 자체는 두 스케일(8B/70B)에서 성립** — 바뀐 건 "대형=저항"이 아니라
  "knockout은 모델별 자체 헤드 탐색이 필요, 전이 헤드는 스케일이 안 된다"는 방법론적
  교훈.

---

## S9. 실험③ 배경 — heldout 표본이 너무 얇았다

- §S6~S7의 "누수 없는 heldout"은 **slack 15쌍뿐**이었다 — head_n=200(8B 탐색과 동일
  값)이 slack user_task 대부분을 head 선정에 써버렸기 때문.
- 15쌍은 판정을 뒤집을 만큼 얇지는 않지만(누수 있는 all105보다 오히려 강한 효과), 표본을
  늘리면 신뢰도가 올라간다. → **head_n을 낮춰 재탐색하면 heldout user_task가 늘어난다**
  (대가: 탐색 예시 감소로 헤드 노이즈 소폭 증가).

---

## S10. 실험③ 결과 — head_n 80 재탐색으로 slack heldout 15쌍 → 48쌍

`results/2026-09-12_p12_llama70b_headn80/`

| head_n | slack head 그룹 | slack heldout 쌍 |
|---|---|---|
| 200 (기존) | user_task 대부분 소진 | 15 |
| **80 (신규)** | 7/20 | **48** |

- **재현성**: head_n=200과 head_n=80 두 헤드 집합의 jaccard = **0.82**(20개 중 18개
  일치) — 탐색 예시를 54개로 줄여도(vs 137쌍 중 149) 헤드가 거의 그대로 재현됨. 탐색
  풀 축소 우려(§S9)는 기우였음.
- 탐색 자체는 0 oom / 0 nan(54/54 성공, `results/2026-09-12_p12_llama70b_headn80/`).

**[결과 대기]** — 확대된 heldout 48쌍으로 knockout 재평가(important_instructions +
tool_knowledge) 진행 중. S6~S7의 판정(억제 우세, utility 무손상)이 더 큰 표본에서도
유지되는지가 이 슬라이드의 결론.

---

## S11. 실험④ 배경 — 세대 축(Qwen3-8B), lxt의 "첫 토큰 쏠림" 경고 선(先)진단

- 피드백(A) 두 번째 축: Qwen3-8B — 아키텍처는 그대로, **세대만 바뀐** 대조군.
- lxt README가 Qwen3에서 "attribution이 첫 토큰으로 쏠린다"고 경고 — 우리 방법은
  relevance를 D_inj span에 group-sum하므로, 질량이 position 0에 흡수되면 head 점수가
  계통적으로 눌릴 위험. **배선(6줄)보다 진단이 먼저** — `tools/diag_qwen3_relevance.py`.
- 판단 기준: position 0 비중이 0이 아닌 것 자체는 문제가 아니다(causal LM의 흔한
  attention sink) — **같은 프롬프트로 qwen2 대조군과 나란히 돌려 상대적으로 얼마나 더
  쏠리는지**가 기준.

---

## S12. 실험④ 결과 — 쏠림은 실재하지만 D_inj 신호를 지우지는 않는다

`tools/diag_qwen3_relevance.py` (2026-09-12, 같은 프롬프트, target=주입 응답 tool-call 토큰)

| family | position 0 비중 | data_inj span 비중 (22 tokens) |
|---|---|---|
| qwen2 (Qwen2.5-7B-Instruct, 대조군) | 0.49% | 37.71% |
| **qwen3 (Qwen3-8B)** | **16.71%** | 32.78% |

- position 0 쏠림은 **qwen2 대비 ~34배** — 경고가 우리 세팅에서도 실재함을 확인.
- 그러나 **data_inj span 비중은 qwen2와 비슷한 수준을 유지**(32.78% vs 37.71%)하고
  여전히 단일 position 0보다 2배 이상 크다 — 우리 head 탐색은 span 단위 group-sum이라
  position 0은 애초에 그 합산에 안 들어간다.
- **진단 통과(캐비엇과 함께)** — Track A/B 진행.

**[결과 대기]** — Qwen3-8B Track A(헤드 탐색) + Track B(knockout 평가) 진행 중.

---

## S13. 결과 요약 & 다음 단계

**이번 사이클 확정된 것**

- Llama-70B Track A(수동 device_map)로 자체 헤드 탐색 가능 확인.
- "스케일업 반례"의 Llama 부분 = 전이헤드 confound였음을 2개 공격 축(important_
  instructions/tool_knowledge)에서 공격-독립적으로 확정, backfire 0(tool_knowledge).
- Qwen2.5-32B 쪽은 애초에 전이 문제가 아니었음(자체 헤드, fp4↔nf4dq 양자화 불일치만
  남음) — 4차 발표 서술의 착오를 정정(§S8, `docs/status-2026-09-09.md` 2026-09-12 정정).
- Qwen3-8B lxt "첫 토큰 쏠림" 경고 — 실재하나 D_inj 신호를 지우지 않음, 진단 통과.

**진행 중 (이 세션 안에 채울 것)**

- head_n 80 재탐색 heldout 48쌍으로 knockout 재평가(§S10).
- Qwen3-8B Track A/B(§S12).

**다음 사이클 후보**

- Qwen2.5-32B 자체 헤드 nf4dq 재탐색(`docs/todo.md` §4-1) — fp4↔nf4dq 탐색 일관성,
  Qwen 쪽 "스케일업 반례"가 여전히 유효한지.
- 70B k-sweep(topk 20→40→60) — 자체 헤드가 통함을 확인했으니 억제력 곡선.
