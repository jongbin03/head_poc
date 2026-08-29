# 3차 발표 스크립트 — 모델 스케일업 + 패밀리 교차검증

> 2차 발표(`IPI_Head_PoC_2nd_script.md`) 이후 진행한 서버 이전 + 확장 실험(2026-08-21~26)
> 결과를 담은 3차 발표 자료. **최대한 간단하게** — 서론 한 줄, 방법(head 탐색 / 평가) 두
> 슬라이드, 결과 여섯 슬라이드(개요 + 모델×공격 4종 suite 상세 + 종합 비교), 한계 한
> 슬라이드, 마무리 한 슬라이드로 구성(총 13슬라이드).
> 발표 스크립트 전체
> 서술이 아니라, 슬라이드에 들어갈 내용을 그대로 정리한 문서 — `build_deck_3rd.py`가 이
> 문서 순서·수치를 그대로 pptx로 옮긴다.
>
> 수치 출처: `docs/status-2026-08-25.md`(32B), `docs/status-2026-08-26.md`(Llama-8B),
> `results/2026-08-24_s4_32b/agentdojo_eval.json` · `agentdojo_eval_toolknowledge.json`,
> `results/2026-08-25_s6_llama8b/agentdojo_eval.json`.

---

## S1. 타이틀

**Read Head, Control Head 분리 PoC — 확장 검증**
모델 스케일업(Qwen2.5-32B) · 모델 패밀리 교차검증(Llama-3.1-8B)

부제: 2차 발표 08-19 · 서버 이전 + 확장 실험 08-21~26 · 3차 발표 2026-08-26

---

## S2. 서론

**이번 사이클에 검증한 것**

- 2차 발표 결론 한 줄: knockout(head의 D_inj edge만 차단)이 7B에서 공격을 억제하면서
  정상 기능(utility)은 유지했다.
- 이번 질문: 이 효과가 **①더 큰 모델**에서도, **②다른 모델 계열**에서도 재현되는가?
- 두 축으로 같은 실험을 반복: **스케일 축**(Qwen2.5 7B→32B), **패밀리 축**(Qwen2→Llama-3.1, ~7-8B 고정)

---

## S3. 실험 방법 — 환경

| 항목 | 내용 |
|---|---|
| 서버 | SSH 공용 서버, Titan RTX 24GB ×3 |
| dtype | bf16 (Turing 에뮬레이션, fp32 대비 오차 무시할 수준) |
| 스케일 축 모델 | Qwen2.5-32B-Instruct (4bit) |
| 패밀리 축 모델 | Llama-3.1-8B-Instruct (fp16) |
| 벤치마크 | AgentDojo — banking / slack / travel / workspace 4개 suite |

---

## S4. 실험 방법 — Head 탐색 (Track A)

- **AttnLRP 기반 relevance** = attention weight × gradient — "주입된 명령을 실제로
  본 head"를 랭킹
- 각 (layer, head)의 relevance를 합산해 **상위 20개** 추출
- **user_task 단위 group split**: 같은 과업(user_task)의 다른 injection이 탐색·평가
  양쪽에 섞이지 않게 통째로 한쪽에만 배정 — 데이터 누수 방지
- **suite별 층화 샘플링**: 4개 suite가 고르게 표본에 들어가도록 쿼터 배정

---

## S5. 실험 방법 — 평가 (Track B)

- AgentDojo의 **실제 멀티턴 agent loop** 안에 모델을 직접 끼워 넣고 실행 —
  토큰 확률 proxy가 아니라 **tool을 실제로 실행한 뒤 환경 상태로 채점**(네이티브 채점)
- 같은 롤아웃을 두 조건으로 비교:
  - **k=0**: 방어 없음 (baseline)
  - **k=N**: 탐색으로 찾은 head들의 D_inj 방향 attention edge만 차단 (knockout)
- **utility**(원래 과업 성공)와 **security**(공격 성공률, 낮을수록 좋음)를 두 조건에서 비교
- head 탐색에 쓰인 user_task는 평가 후보에서 제외(**held-out**)

---

## S6. 결과 개요 — 모델 × 공격 종류 비교

| 모델 | 공격 | n | utility k=0 | utility k=N | ASR k=0 | ASR k=N |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | important_instructions | 48 | 18.75% | 18.75% | 2.08% | 0% |
| Qwen2.5-7B | tool_knowledge | 46 | 19.6% | 17.4% | 2.17% | 0% |
| Qwen2.5-32B | important_instructions | 57 | 45.6% | 47.4% | 8.77% | 3.51% |
| Qwen2.5-32B | tool_knowledge | 55 | 54.5% | 56.4% | 14.5% | 9.1% |
| Llama-3.1-8B | important_instructions | 44 | 38.6% | 38.6% | 4.5% | 0% |
| Llama-3.1-8B | tool_knowledge | 44 | 36.4% | 34.1% | 4.5% | 0% |

- 세 모델·두 공격 전부 **공격 성공률은 knockout 후 방향이 일관되게 낮아짐**(억제),
  utility는 대체로 유지
- **⚠️ 캐비엇**: Qwen2.5-7B 두 행은 이번 사이클 이전(2026-07-31)의 **구버전 파이프라인**
  결과다 — user_task 단위 held-out split이 도입되기 전이라 32B/Llama-8B 행과 **직접
  비교하면 안 된다**(표본 구성 방식이 다름). 이 슬라이드는 "세 모델 다 knockout이 공격을
  억제하는 방향"이라는 정성적 개요이고, 정량 비교는 이어지는 슬라이드(같은 파이프라인
  끼리)에서 한다. Llama-3.1-8B 행은 둘 다 travel suite 제외(n=44, 파서 미지원 — S8/S10 각주).

---

## S7. 결과 — 32B / important_instructions (n=57)

| suite | n | utility k=0 | utility k=N | ASR k=0 | ASR k=N |
|---|---|---|---|---|---|
| banking | 15 | 53.3% | 73.3% | 0% | 0% |
| slack | 15 | 53.3% | 53.3% | **33.3%** | **13.3%** |
| travel | 15 | 20.0% | 6.7% | 0% | 0% |
| workspace | 12 | 58.3% | 58.3% | 0% | 0% |
| **합계** | **57** | **45.6%** | **47.4%** | **8.77%** | **3.51%** |

- utility는 knockout 후에도 유지(소폭 상승), 공격 성공률은 약 60% 상대 감소
- 공격 성공 사례는 slack에 집중 — banking/travel/workspace는 baseline부터 0%

---

## S8. 결과 — 32B / tool_knowledge (n=55)

| suite | n | utility k=0 | utility k=N | ASR k=0 | ASR k=N |
|---|---|---|---|---|---|
| banking | 15 | 86.7% | 80.0% | 0% | 0% |
| slack | 15 | 46.7% | 53.3% | **53.3%** | **33.3%** |
| travel | 12 | 25.0% | 25.0% | 0% | 0% |
| workspace | 13 | 53.8% | 61.5% | 0% | 0% |
| **합계** | **55** | **54.5%** | **56.4%** | **14.5%** | **9.1%** |

- 공격 문구를 강화하자 baseline ASR이 8.77%→14.5%로 뜀 — 그래도 knockout 후 억제 방향은 유지
- utility는 오히려 소폭 상승(54.5%→56.4%) — 정상 과업 훼손 없음

---

## S9. 결과 — Llama-3.1-8B / important_instructions (n=44)

| suite | n | utility k=0 | utility k=N | ASR k=0 | ASR k=N |
|---|---|---|---|---|---|
| banking | 15 | 40.0% | 40.0% | 0% | 0% |
| slack | 15 | 53.3% | 53.3% | **13.3%** | **0%** |
| workspace | 14 | 21.4% | 21.4% | 0% | 0% |
| **합계** | **44** | **38.6%** | **38.6%** | **4.5%** | **0%** |

- utility 완전 유지, 공격 성공률 완전 억제(4.5%→0%)
- Qwen2가 아닌 **다른 아키텍처(Llama)에서도 같은 패턴이 재현됨**
- (각주) travel suite는 이번 파이프라인에서 멀티 tool-call 응답 파싱을 지원하지 않아
  제외 — 실제 성능 결과가 아님

---

## S10. 결과 — Llama-3.1-8B / tool_knowledge (n=44)

| suite | n | utility k=0 | utility k=N | ASR k=0 | ASR k=N |
|---|---|---|---|---|---|
| banking | 15 | 33.3% | 26.7% | 0% | 0% |
| slack | 15 | 53.3% | 53.3% | **13.3%** | **0%** |
| workspace | 14 | 21.4% | 21.4% | 0% | 0% |
| **합계** | **44** | **36.4%** | **34.1%** | **4.5%** | **0%** |

- 공격 성공률은 여기서도 완전 억제(4.5%→0%) — important_instructions와 동일
- utility는 소폭 하락(36.4%→34.1%, banking 1건이 knockout 후 실패로 전환) — 32B만큼
  완전하진 않지만 여전히 큰 폭 유지
- (각주) travel suite 제외 — 위와 동일 사유(파서 미지원)

---

## S11. 결과 — 공격 강도 비교 (모델 간 종합)

| 모델 | 공격 | n | ASR k=0 | ASR k=N | 상대 감소율 |
|---|---|---|---|---|---|
| Qwen2.5-32B | important_instructions | 57 | 8.77% | 3.51% | 60% |
| Qwen2.5-32B | tool_knowledge | 55 | 14.5% | 9.1% | 37.5% |
| Llama-3.1-8B | important_instructions | 44 | 4.5% | 0% | **100%** |
| Llama-3.1-8B | tool_knowledge | 44 | 4.5% | 0% | **100%** |

- **32B**: 공격이 강해질수록 knockout 후에도 남는 절대적 위험이 커짐(3.51%→9.1%) —
  억제는 하지만 완전하진 않음
- **Llama-3.1-8B**: 공격 강도와 무관하게 **두 공격 다 완전 억제(100%)** — 표본이 작아
  (성공 사례 2건뿐) 과대해석은 주의해야 하지만, 32B와 다른 양상이라는 점은 흥미로운 대조

---

## S12. 한계 및 다음 실험

| # | 한계 | 향후 방향 |
|---|---|---|
| 1 | **표본 크기** — suite마다 무작위 15쌍만 평가, 전수 평가 아님 (예: workspace는 실제 후보 pool 300~450쌍 중 15개만 사용) | suite별 전체 평가로 확대 |
| 2 | **suite별 편차** — 공격 성공 사례가 거의 slack에만 몰려 있음(banking/travel/workspace는 baseline부터 0%인 경우가 많음) | 결과를 "slack 주도"로 caveat, 더 다양한 공격 시나리오 필요 |
| 3 | **AttnLRP(lxt) 검증 아키텍처 범위** — 지금까지 head 탐색(backward LRP)이 정상 동작을 확인한 건 Qwen2.5·Llama-3.1 두 계열뿐 | 같은 표준 구조(RMSNorm+SwiGLU+표준 attention)를 쓰는 다른 모델(Mistral, DeepSeek dense 계열 등)로 lxt 확장이 코드상 저비용으로 가능해 보임 — 다음 실험 후보로 제안 |
| 4 | **tool-call 파싱** — 100% 성공은 아님(모델별 70%대) — 파싱 실패 턴은 tool 미실행으로 집계돼 utility가 과소평가될 수 있음 | 모델별 파서 정교화 지속 |

---

## S13. 마무리

- **스케일 축**(7B→32B), **패밀리 축**(Qwen2→Llama), **공격 강도 축**(문구 강화) —
  세 방향 모두 "공격 억제 + 정상 기능 보존" 패턴이 재현됨
- 한계: 표본 크기·suite 편차·lxt 검증 아키텍처 범위 — 상세는 S12
