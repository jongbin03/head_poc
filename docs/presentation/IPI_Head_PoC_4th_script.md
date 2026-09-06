# 4차 발표 스크립트 — 파서 정합성 검증 + 양자화 아티팩트 규명

> 3차 발표(`IPI_Head_PoC_3rd_script.md`, 2026-08-26) 이후 교수님 피드백 대응 사이클(P16,
> 2026-08-31~09-06) 결과를 담은 4차 발표 자료. 3차와 같은 톤으로 **최대한 간단하게** —
> 서론 한 슬라이드, 세 개의 실험 라인(파서 A/B → 표본 확대 → 양자화 규명)을 배경→결과
> 순으로, 대조 실험 한 슬라이드, 한계 한 슬라이드, 마무리 한 슬라이드로 구성(총 14슬라이드).
> 발표 스크립트 전체 서술이 아니라, 슬라이드에 들어갈 내용을 그대로 정리한 문서 —
> `build_deck_4th.py`가 이 문서 순서·수치를 그대로 pptx로 옮길 예정(아직 미작성, 2nd/3rd의
> `build_deck_{2nd,3rd}.py` 패턴 참고해서 신설 필요).
>
> 수치 출처: `docs/status-2026-09-06.md`(§1), `docs/status-2026-09-01.md`(§1, §3.0),
> `docs/feedback-2026-08-31.md`(§1, §2.1.4, §2.1.9, §2.1.13~2.1.19, §2.5, §2.5.3),
> `results/2026-08-31_p16_aisecking/`(파서 A/B), `results/2026-09-01_*`(32B n=148 확대),
> `results/2026-09-02_p16b_4bit/`·`2026-09-02_p16c_32b_nf4dq/`·`2026-09-03_p16d_32b_bf16/`
> (양자화 실험), `results/2026-09-05_p16e_32b_bf16_banking/`(대조 실험).
>
> ✅ **S12 완성 (2026-09-06)** — (A') 32B bf16 banking 대조 결과 회수·반영 완료. 이 문서
> 내용은 확정, `build_deck_4th.py` 신설·pptx 산출만 남음.

---

## S1. 타이틀

**Read Head, Control Head 분리 PoC — 파서 정합성 검증 + 양자화 아티팩트 규명**

부제: 3차 발표 2026-08-26 · 피드백 대응 실험 08-31~09-06 · 4차 발표 2026-09-XX

---

## S2. 서론

**이번 사이클의 출발점 — 교수님 피드백**

- 3차 발표 결론 한 줄: knockout 효과가 스케일(7B→32B)·패밀리(Qwen2→Llama)·공격 강도
  세 축 모두에서 재현됨.
- 피드백 4번째 항목: "tool-call 파서를 AgentDojo 기본값으로 바꿔서 실험해봐라" — 우리
  Track B가 커스텀 파서를 쓰고 있었다는 점을 짚음.
- 이 대응 과정에서 **두 개의 곁가지 조사**로 자연스럽게 이어짐:
  1. 파서를 바꾸며 표본을 키우다 **slack suite에서만 knockout이 일부 공격을 못 막는
     패턴** 발견
  2. 그 원인을 추적하다 **bnb 4bit 양자화 설정 자체의 아티팩트** 발견
- 오늘은 이 세 실험 라인(파서 A/B → 표본 확대 → 양자화 규명)을 순서대로.

---

## S3. 실험① 배경 — 왜 파서를 의심했나

- Track B(`run_agentdojo_eval.py`)가 지금까지 쓰던 tool-call 프롬프트/파서는 AgentDojo
  자체 기본값이 아니라 **우리가 만든 커스텀 파서**였음.
- 이유: 초기 실험에서 1.5B 모델이 AgentDojo 기본 포맷을 따르지 않아 대체했던 이력.
- 우려: 지금까지 관찰한 "banking/workspace suite 저조" 패턴이 **파서 아티팩트**일 수
  있지 않은가?
- 대응: `--tool_call_format agentdojo_default` 옵션 추가(AgentDojo 자체
  `_make_system_prompt`/`_parse_model_output` 재사용) 후 같은 조건에서 A/B 비교.

---

## S4. 실험① 결과 — Custom vs AgentDojo-default A/B

같은 모델·같은 heads·같은 seed(42)·같은 suite(banking/slack/workspace, held-out)로 비교.

**Qwen2.5-7B** (`results/2026-08-31_p16_aisecking/qwen7b_{custom,agentdojo_default}.json`)

| | custom(기존) | agentdojo_default |
|---|---|---|
| n_pairs | 43 | 45 |
| parse ok율 | 57.5% | 59.4% |
| k0 utility | 39.5% | 42.2% |
| k0 security(ASR) | 4.7% | 6.7% |
| kN security | 0.0% | 0.0% |
| banking / slack / workspace k0_sec | 0% / 13% / 0% | 0% / 20% / 0% |

**Llama-3.1-8B** (`results/2026-08-31_p16_aisecking/llama8b_{custom,agentdojo_default}.json`)

| | custom(기존) | agentdojo_default |
|---|---|---|
| n_pairs | 45 | 44 |
| parse ok율 | 79.2% | 79.1% |
| k0 utility | 35.6% | 38.6% |
| k0 security(ASR) | 4.4% | 2.3% |
| kN security | 0.0% | **2.3%** ⚠️(1/44, banking 신규 backfire) |

- **결론**: banking/workspace 저조는 파서 아티팩트가 아님 — 두 모델·두 파서 모두 같은
  suite별 비대칭 패턴 재현. parse ok율도 파서 방식과 거의 무관.
- 1.5B의 AgentDojo 기본 포맷 실패 전례는 7B/8B에서 재현 안 됨 → **`agentdojo_default`를
  기본값으로 채택.**
- Llama+agentdojo_default 조합에서만 banking 1건 backfire 관측(표본 1건, 이후 표본
  확대로 재확인 필요 — 다음 슬라이드로 이어짐).

---

## S5. 실험② 배경 — 표본을 왜 최대치로 키웠나

- 파서 전환 직후 32B vs 7B 비교(초기 표본 n=43~45)에서: 스케일업해도 ASR이 안 오르는
  기존 결론은 재확인되지만, **slack suite에서만 knockout이 일부 성공 공격을 못 막는**
  신호가 눈에 띔.
- 우연(표본이 작아서)인지 실재하는 패턴인지 가리려면 표본을 최대치로 키워야 함 →
  banking/slack/travel/workspace **held-out 후보 풀 전부**를 사용.

---

## S6. 실험② 결과 — Qwen2.5-32B 4-suite 전수 확대 (n=148)

`results/2026-08-31_p16_aisecking/qwen32b_agentdojo_default_expanded.json`

| suite | n | k0_util | k0_sec | kN_util | kN_sec | parse_ok |
|---|---|---|---|---|---|---|
| banking | 45(전부) | 66.7% | 2.2% | 73.3% | **0.0%** | 52.9% |
| **slack** | **35(전부)** | 25.7% | **22.9%** | 22.9% | **14.3%** | 73.4% |
| travel | 28(전부) | 25.0% | 3.6% | 25.0% | 0.0% | 83.8% |
| workspace | 45 | 20.0% | 0.0% | 0.0% | 0.0% | 44.4% |
| **전체** | **148** | 36.5% | **6.8%** | 37.8% | 3.4% | 68.3% |

- **스케일업 반례 최종 확정**: 지금까지 가장 큰 표본(n=148)에서도 전체 ASR 6.8% —
  스케일을 키워도 공격 성공률이 오르지 않는다는 결론이 표본 크기 문제일 여지 소멸.
- **slack knockout 실패가 정밀하게 드러남**: held-out 35쌍 전부 기준, 성공한 공격
  8건 중 **5건(62.5%)을 knockout이 못 막음** (억제 3/8, 실패 5/8).
- banking/travel/workspace는 이 표본 크기에서도 여전히 완벽 억제(0%) — **slack만
  특이 케이스.**

---

## S7. 실험② 상세 — slack 실패는 왜 특정 injection에 몰렸나

| injection_task | 목표 | 필요 tool call 수 | 성공 8건 중 |
|---|---|---|---|
| 1 | 피싱 링크 전송 | 1개 | 3건 (2건 억제 실패) |
| 3 | 피싱 URL 방문 | 1개 | 1건 (억제) |
| 5 | 초대→추가→제거 | 3개(단순) | 4건 (3건 억제 실패) |
| 2 | 전 채널 메시지 유출 | 6개(복합) | 0건 |
| 4 | general 채널 메시지 유출 | 2개(concat 필요) | 0건 |

- 성공한 8건 전부가 `injection_task 1/3/5`(단일~소수 스텝, 단순 인자)였고, 다단계·복합
  채점인 `2/4`는 전무.
- **원인 재진단**: "문구가 설득력 있어서"가 아니라 "실행에 필요한 tool call 개수가
  적어서" — 단순한 공격은 knockout으로 injection 신호를 줄여도 우연히/관성적으로
  실행되기 쉽고, 복잡한 공격은 knockout과 무관하게 애초에 실행 성공률 자체가 낮음
  (utility 병목과 얽혀 있음).
- 이 진단은 이후 조사(양자화 규명, 다음 실험 라인)로 이어지는 실마리가 됨: **slack
  knockout 실패가 관찰된 유일한 32B 실행이 4bit(`fp4`) 조건이었다**는 공변 관계 발견.

---

## S8. 실험③ 배경 — 양자화를 왜 의심했나

- slack knockout 실패가 나온 유일한 실행은 **32B(4bit)** — 대조군으로 쓴 8B급(Llama,
  Qwen 7B)은 전부 **bf16**이었음.
- 즉 "원인이 (1) 32B 스케일인지 (2) 4bit 양자화인지"가 지금까지 데이터로는 **공변되어
  구분 불가능**한 상태.
- 확인 절차: 같은 스케일(7B)에서 bf16 vs 4bit 직접 비교 → 4bit 세팅 자체(품질) 점검 →
  32B를 bf16으로 재실행해 양자화를 배제.

---

## S9. 실험③ 결과 — 7B: bf16 vs fp4 vs nf4+double_quant

`results/2026-09-02_p16b_4bit/`

| 실행 | quant | slack k0_sec | slack kN_sec | 전체 k0_util | slack backfire |
|---|---|---|---|---|---|
| 7B bf16 | bf16 | 0.171 | **0.000** | 0.311 | 0 |
| 7B fp4(bnb 기본값) | fp4/no-dq | 0.057 | 0.057 | 0.178 | **2** |
| 7B nf4+double_quant | nf4/dq | 0.343 | 0.057 | 0.254 | **1** |

- **fp4 아티팩트 확정**: bnb 4bit 기본값(`fp4`+double_quant off)이 모델을 과손상시켜
  knockout이 손상된 궤적에서 backfire를 냄. `nf4+double_quant`로 utility 대폭 회복
  (0.178→0.254) + backfire 2→1.
- **비결정성 아님**: 동일 조건 2회(RTX 4090) 152/152 쌍 전 필드 일치 — greedy+고정 seed
  에서 완전 결정론적. → `run_agentdojo_eval.py` 4bit 기본값을 `nf4`+double_quant on으로
  교체 완료.

---

## S10. 실험③ 결과 — 32B: fp4 / nf4dq(비결정적) / bf16

`results/2026-09-02_p16c_32b_nf4dq/`, `results/2026-09-03_p16d_32b_bf16/`

| 실행 | GPU / quant | slack k0_sec | slack kN_sec | suppressed | backfire | persist |
|---|---|---|---|---|---|---|
| 32B nf4dq #1 | A6000 / nf4dq | 0.229 | 0.229 | 2 | 2 | 6 |
| 32B nf4dq #2 | Blackwell / nf4dq | 0.171 | 0.114 | 2 | 0 | 4 |
| **32B bf16** | A6000+2 / **bf16** | 0.257 | **0.143** | 5 | **1** | 4 |
| (대조) 7B·8B bf16 | 4090 / bf16 | ~0.18 | **0.000** | 전량 | **0** | — |

- 두 nf4dq 실행이 교집합 136쌍 중 24쌍 불일치(18쌍이 baseline 자체 차이) → **32B-4bit
  평가는 run-to-run 비결정적** — 이 경로로는 스케일 판정 불가.
- **bf16으로 우회**: 완주율 100%·결정론적인데도 kN_sec 0.143(≠0) + backfire 1 — 7B·8B
  bf16(전량 억제·backfire 0)과 질적으로 다름 → **fp4와 무관한 스케일 기여 확인**.

---

## S11. 종합 결론

1. ✅ **파서는 confound 아님** — banking/workspace 저조는 커스텀 파서 탓이 아니라
   모델·suite 자체의 성질. `agentdojo_default`로 전환 완료.
2. ✅ **스케일업 반례 재확정(n=148)** — 스케일을 키워도 전체 ASR은 안 오름(6.8%). 단
   slack suite에서만 knockout 억제가 불완전(성공 8건 중 5건 미억제).
3. ✅ **원인 두 갈래로 분리 규명**:
   - `fp4` 4bit 양자화 아티팩트(확정) — `nf4+double_quant`로 대부분 해소, 기본값 교체 완료.
   - 32B 스케일 효과(약하게 확정) — bf16 단독에서도 slack knockout 불완전 + backfire 1건.
4. 단 순효과는 여전히 방어적(suppressed 5 > backfire 1) — **"못 막는다"가 아니라
   "불완전 + 가끔 backfire".**

---

## S12. 대조 실험 — 32B bf16 banking, "slack 특유" 확정

같은 32B bf16 스택(A6000+2, 동일 heads·설정)에서 `--suite`만 slack→banking으로 교체
(`results/2026-09-05_p16e_32b_bf16_banking/`).

| suite | k0_sec | kN_sec | backfire | 판정 |
|---|---|---|---|---|
| slack (S10 재인용) | 0.257 | 0.143 | **1/9** | 스케일 효과로 불완전 억제 |
| banking | 0.0 | 0.0 | **0/42** | baseline부터 공격 실패, knockout도 새 leak 없음 |

- banking은 42쌍 전부 baseline(`k0`)부터 공격이 실패해서 "억제율"이라는 틀 자체가 안
  맞음 — 대신 판정 근거는 **backfire**: `k0=False`(42쌍 전부 해당)인데 knockout 후
  `kN=True`로 뒤집힌 쌍이 있는지. **0/42로 하나도 없음.**
- 같은 32B bf16 스택에서 slack만 backfire 1건이 나왔던 것과 대비 → **32B 스케일 효과는
  slack에 국한, banking엔 전이 안 됨 = "slack 특유의 스케일 취약" 확정.**
- ⚠️ 완주율 42/45(93.3%, `user_task_10`의 injection_task 3쌍 원인불명 누락)·banking
  자체가 baseline 공격 성공률이 원래 낮은 suite라는 점은 caveat(feedback 2.1.19).

---

## S13. 한계 및 다음 단계

| # | 한계 | 향후 방향 |
|---|---|---|
| 1 | 32B-4bit 평가의 run-to-run 비결정성 원인 미규명(별개 이슈로 남김) | bnb 4bit 커널/디바이스 배치 조건 추가 조사 |
| 2 | slack knockout 불안정의 최종 원인이 "단순 공격의 실행 관성" 가설 수준(확정 아님) | k-sweep(§2.2, head 개수 늘려 억제력 보강)으로 완화 여부 확인 |
| 3 | Llama-3.1-8B의 banking 1건 backfire(표본 1건)는 표본 확대로 미확인 | 필요 시 Llama도 n=148급 확대 |
| 4 | 다른 아키텍처(Qwen3, Llama-70B) 교차검증은 후순위로 보류 중 | 다음 사이클 후보 |
| 5 | banking 대조(S12)는 baseline 공격 성공이 0/42라 backfire 검증력이 약함(막아야 할 성공 사례 자체가 없었음) | k-sweep 등으로 banking 공격 성공률 자체를 올려 재검증 여지 |
| 6 | banking 대조 완주율 93.3%(3쌍 원인불명 누락, `.log` 미보존) | 재실행 시 로그 보존 필수 |

---

## S14. 마무리

- **파서 전환**: confound 아니었음, `agentdojo_default` 기본값 채택.
- **표본 확대(n=148)**: 스케일업 반례 최종 확정 + slack 특이 패턴을 정밀하게 포착.
- **양자화 규명**: fp4 아티팩트 확정(기본값 교체 완료) + 32B 스케일 효과 확정,
  순효과는 방어적 유지.
- **대조 실험(S12)**: banking 대조로 32B 스케일 효과가 **slack에 국한**됨을 확정 —
  "모델을 키우면 전반적으로 위험해진다"가 아니라 "특정 suite에서만, 가끔 불완전
  + backfire".
