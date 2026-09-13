# 5차 발표 스크립트 (재구성) — Llama-3.1-70B / Qwen3-8B 헤드 탐색·평가 실험

> **2026-09-12 세션 요청으로 전면 재구성.** 4차 요약·"스케일업 반례 정정" 서사는 빼고,
> **이번 사이클에 진행한 실험 자체를 결과 리포트 형식**으로 정리한다 — 서론(실험 소개) →
> Llama-3.1-70B 헤드 탐색 → 평가 → Qwen3-8B 헤드 탐색(레이어 0 쏠림 진단 포함) → 평가 →
> 요약. 표는 세션 중 사용자에게 제공한 상세 표를 그대로 재사용.
>
> ✅ **모든 suite 확장 평가 완료** — 4개 suite(banking/slack/travel/workspace) 전부
> 시도, S6·S9에 반영 완료. `build_deck_5th.py` 이 구조로 재작성 완료.
>
> 수치 출처: `results/2026-09-08_p12_llama70b/`, `results/2026-09-12_p12_llama70b_headn80/`,
> `results/2026-09-12_p11_qwen3_8b/`, `tools/diag_qwen3_relevance.py` 실행 로그(09-12).

---

## S1. 타이틀

**Read Head, Control Head 분리 PoC — Llama-3.1-70B / Qwen3-8B 헤드 탐색·평가 실험**

---

## S2. 서론 — 이번 사이클에 진행한 실험

| # | 실험 | 목적 |
|---|---|---|
| 1 | **Llama-3.1-70B 헤드 탐색 (Track A)** | 70B 스케일에서 AttnLRP로 자체 control head를 찾을 수 있는가(기존엔 backward OOM으로 불가 판정) |
| 2 | **Llama-3.1-70B 평가 (Track B)** | 그 헤드를 knockout하면 slack IPI 공격이 억제되는가 — 자체 헤드 vs 8B 전이 헤드 비교, heldout 표본 확대 |
| 3 | **Qwen3-8B 헤드 탐색 (Track A) + 레이어 0 쏠림 진단** | lxt가 경고한 "Qwen3는 attribution이 첫 토큰에 쏠린다"는 현상이 우리 세팅에서 실재하는지, 실재해도 헤드 탐색이 유효한지 |
| 4 | **Qwen3-8B 평가 (Track B)** | Qwen3-8B에서 찾은 헤드로도 knockout이 8B급 모델과 같은 패턴을 보이는가 |

공통 조건: AgentDojo slack suite, `agentdojo_default` tool-call 파서, 공격 2종
(`important_instructions`, `tool_knowledge`), greedy decoding.

---

## S3. Llama-3.1-70B 헤드 탐색 — 방법

- 기존 결론("70B AttnLRP backward는 하드웨어 한계로 불가")은 **`--device_map auto` 경로
  한정**이었다 — 타이트한 `--max_memory`면 CPU/disk 분산 에러, 느슨하면 레이어가 한
  GPU에 몰려 backward OOM.
- **해결**: `--device_map_plan "0:32,1:30,2:18"`(수동 device_map) — embed/norm/lm_head를
  첫 GPU에 몰아두고(backward가 양 끝에서 시작·수렴), 레이어는 순서대로 분배.
  `CUDA_VISIBLE_DEVICES=1,0,2` → A6000(48G, root) / Blackwell(32G) / 4090(24G).
- 두 번 탐색: **head_n=200**(2026-09-08, 8B 탐색과 동일 조건) / **head_n=80**(2026-09-12,
  heldout 평가 표본 확대 목적 — S5 참고).

---

## S4. Llama-3.1-70B 헤드 탐색 — 결과

70B는 80층×64헤드(총 5,120개) 중 20개 선정. `--max_seq_len 1000` 필터 후 949쌍 중
**137쌍만 통과**하며 **workspace는 두 탐색 모두 전량 필터 탈락**(0쌍).

### suite별 내역

| head_n | suite | 탐색에 쓴 pair | user_task 그룹 | quota | shortfall |
|---|---|---|---|---|---|
| 200 | banking | 61 | 11/11 | 66 | 5 |
| 200 | slack | 67 | 20/20 | 66 | 0 |
| 200 | travel | 2 | 1/1 | 66 | 64 |
| **80** | banking | 26 | 5/11 | 26 | 0 |
| **80** | slack | 26 | 7/20 | 26 | 0 |
| **80** | travel | 2 | 1/1 | 26 | 24 |

| head_n | n_examples_used | oom | nan |
|---|---|---|---|
| 200 | 130 / 149 | 0 | 0 |
| 80 | 54 / 54 | 0 | 0 |

### 선정된 헤드 (layer, head_idx) — head_n=80

```
(35,35) (38,52) (34,6) (32,22) (35,34) (30,51) (31,47) (29,58) (31,45)
(33,30) (35,18) (31,7) (26,53) (29,62) (32,16) (28,41) (44,35) (28,1)
(33,14) (34,43)
```

| 항목 | 값 |
|---|---|
| layer 범위 | 26–44 / 80 (≈33–55% 깊이, 대부분 28–35 구간) |
| head_n=200 대비 재현성 | **jaccard 0.82** (20개 중 18개 일치) |
| 8B 헤드와 비교 | Llama-8B는 layer 11–22/32 (≈38–47%) — **같은 상대 깊이 대역** |

---

## S5. Llama-3.1-70B 평가 실험 — 방법

- **평가 대상**: 위에서 찾은 헤드를 knockout(edge ablation)했을 때 slack IPI 공격
  성공률(security) / 정상 과업 수행률(utility)이 어떻게 바뀌는가.
- **비교축 ①** — 헤드 출처: **70B 자체 헤드** vs **8B에서 찾은 헤드를 그대로 전이**.
  최초 70B 평가(2026-09-08)는 70B Track A가 없어 8B 헤드를 전이해 썼음 — 이후 자체 헤드가
  나오자 정면 대조.
- **비교축 ②** — 표본: `--eval_split all`(105쌍, 헤드 탐색에 쓰인 case 포함 = 누수 있음)
  vs `--eval_split heldout`(헤드 탐색에 안 쓰인 case만 = 누수 없음).
- **heldout 표본 확대**: head_n=200 탐색은 slack user_task 20개 중 18개를 헤드 탐색에
  써서 heldout 후보가 **15쌍**뿐이었다. head_n=80으로 재탐색하면 7개만 쓰여 heldout
  후보가 **70쌍**(실제 60쌍 평가, `--limit_pairs 60` 캡)으로 확대됨.

| head_n | 헤드 탐색에 쓴 slack user_task | heldout 후보 | 평가한 수 |
|---|---|---|---|
| 200 | 18 / 20 | 15 | 15 (전부) |
| 80 | 7 / 20 | 70 | 60 |

---

## S6. Llama-3.1-70B 평가 결과

### 자체 헤드 vs 전이 헤드 (all105, 2026-09-08/09)

| 공격 | 헤드 출처 | k0 sec | kN sec | 억제/backfire | net | kN util | parse_ok |
|---|---|---|---|---|---|---|---|
| important_instructions | 8B 전이 | 0.276 | 0.257 | 4/2 | −2 (효과 없음) | 0.190 | 0.782 |
| important_instructions | **70B 자체** | 0.276 | **0.181** | 13/3 | **−10 (34%↓)** | 0.190 | 0.783 |
| tool_knowledge | 8B 전이 | 0.402 | 0.392 | 3/2 | −1 (효과 없음) | 0.186 | — |
| tool_knowledge | **70B 자체** | 0.398 | **0.223** | 18/0 | **−18 (44%↓)** | 0.204 | — |

### heldout 표본 확대 (70B 자체 헤드, 2026-09-09 → 2026-09-12)

| 공격 | 표본 | k0 sec | kN sec | 억제/bf/persist | net | kN utility | parse_ok |
|---|---|---|---|---|---|---|---|
| important_instructions | 15쌍 (head_n=200) | 0.533 | 0.200 | 5/0/— | −5 | 0.333 (↑) | 0.853 |
| **important_instructions** | **60쌍 (head_n=80)** | 0.250 (15) | **0.117** (7) | 9/1/6 | **+8 (53%↓)** | **0.150 (↓)** | 0.733 |
| tool_knowledge | 15쌍 (head_n=200) | 0.733 | 0.400 | 5/0/— | −5 | 0.333 (무손상) | — |
| **tool_knowledge** | **60쌍 (head_n=80)** | 0.350 (21) | **0.217** (13) | 8/0/13 | **+8 (38%↓)** | **0.183 (↑)** | 0.747 |

**요지**: 자체 헤드가 전이 헤드보다 뚜렷이 강하게 작동(두 공격 다 net 방어적, 전이 헤드는
효과 없음). 표본을 4배(15→60) 키워도 방향은 유지. important_instructions만 utility가
처음 소폭 하락(원인: suppressed case 일부에서 knockout이 task 수행 자체도 같이 무너뜨림 —
무작위 형식 손상 아님). tool_knowledge는 3개 표본 연속 backfire 0.

### 다른 suite로 확장 (2026-09-12, 70B 자체 헤드 · head_n=80)

| suite | 표본 | k0 sec | kN sec | 억제/bf/persist | net | kN utility | parse_ok |
|---|---|---|---|---|---|---|---|
| slack | 60 | 0.250 (15) | 0.117 (7) | 9/1/6 | +8 (53%↓) | 0.150 (↓) | 0.733 |
| **banking** | 59 (1 oom) | 0.085 (5) | 0.068 (4) | 3/2/2 | **+1 (약함)** | **0.627 (↑, +7)** | 0.642 |
| travel | **0 / 60** | — | — | — | — | — | — |
| workspace | **0 / 60** | — | — | — | — | — | — |

- **banking은 knockout 신호가 slack보다 훨씬 약함** — baseline 공격 자체가 5건뿐이라
  net+1은 사실상 잡음에 가까움. 대신 **utility가 크게 개선**(30→37/59) — knockout이
  banking 정상 과업에는 손상은커녕 도움이 되는 방향.
- **travel·workspace는 70B에서 평가 자체가 불가능** — **A6000(48GB, 가장 큰 카드)로도
  전량 OOM**(60/60 전부 실패). Track A 탐색 때 이미 `max_seq_len=1000` 필터에 전량
  걸러졌던 것(§S4)과 같은 맥락 — 70B forward+8턴 생성이 이 suite들의 프롬프트 길이를
  이 하드웨어에서 감당 못 함.

---

## S7. Qwen3-8B 헤드 탐색 — 레이어 0(첫 토큰) 쏠림 진단

- lxt README 경고: "Qwen3는 attribution이 첫 토큰(position 0)으로 쏠린다." 우리 head
  탐색은 relevance를 D_inj span에 **group-sum**하므로, 질량이 position 0에 흡수되면
  head 점수가 계통적으로 눌릴 위험 — 배선 전에 직접 진단(`tools/diag_qwen3_relevance.py`).
- **판단 기준**: position 0 비중이 0이 아닌 것 자체는 문제가 아니다(causal LM의 흔한
  attention sink). **같은 프롬프트로 qwen2 대조군과 나란히 돌려 상대적으로 얼마나 더
  쏠리는지**가 기준.

| family | 모델 | position 0 비중 | data_inj span 비중 (22 토큰) |
|---|---|---|---|
| qwen2 (대조군) | Qwen2.5-7B-Instruct | 0.49% | 37.71% |
| **qwen3** | Qwen3-8B | **16.71%** | 32.78% |

- 쏠림은 **qwen2 대비 ~34배로 실재**한다.
- 그러나 **data_inj span 비중은 qwen2와 비슷한 수준을 유지**(32.78% vs 37.71%)하고 여전히
  position 0 단독보다 2배 이상 크다 — group-sum 방식이라 position 0은 애초에 그 합산에
  안 들어감.
- **판정: 진단 통과(캐비엇과 함께)** — Track A/B 진행.

---

## S8. Qwen3-8B 헤드 탐색 — 결과

36층×32헤드(총 1,152개) 중 20개 선정. `--max_seq_len 1200` 필터로 949쌍 중 174쌍
통과(4 suite 전부 생존, 70B와 다름). head_n=200, quota=50/suite.

### suite별 내역

| suite | ok | oom | 비고 |
|---|---|---|---|
| banking | 53 | 4 | |
| slack | 50 | 0 | oom 없음 |
| travel | 0 | **14 (전량)** | |
| workspace | 0 | **22 (전량)** | |
| **합계** | **103** | **40 (28%)** | oom이 **전부 travel/workspace에 집중**(추정: 긴 프롬프트), banking/slack은 거의 안전 |

### 선정된 헤드 (layer, head_idx)

```
(25,10) (0,3) (20,29) (22,11) (21,18) (19,21) (24,31) (0,0) (29,0)
(22,0) (23,26) (18,30) (21,19) (18,14) (21,11) (26,26) (18,15) (20,5)
(21,27) (28,22)
```

| 항목 | 값 |
|---|---|
| layer 범위 | 18–29 / 36 (≈50–80% 깊이) |
| **layer 0** | **2개 (10%)** — §S7 쏠림 진단과 무관 단정 불가(caveat), 다만 layer 0 지배는 이 방법론 전반의 반복 현상(`docs/todo.md` P4) |
| Llama-8B/70B와 비교 | 상대 깊이 38–47% / 33–55% — **Qwen3-8B가 뚜렷이 더 깊은 대역** |

---

## S9. Qwen3-8B 평가 실험 — 방법 & 결과

- 헤드 탐색에 쓰인 slack user_task 6/20 → **heldout 후보 35쌍(전부 평가)**.
- knockout 20개 헤드, k=0→k=20.

| 공격 | 표본 | k0 sec | kN sec | 억제/bf/persist | net | kN utility | parse_ok |
|---|---|---|---|---|---|---|---|
| **important_instructions** | 35 | 0.229 (8) | **0.000** | **8/0/0** | **+8 (전량 억제)** | 0.429 (↑) | 0.711 |
| tool_knowledge | 35 | 0.171 (6) | 0.057 (2) | 5/1/1 | +4 (66%↓) | 0.400 (무손상) | 0.709 |

**요지**: important_instructions는 8B급(Llama-8B/Qwen2.5-7B)과 완전히 동일한 "전량
억제·backfire 0" 패턴 재현. tool_knowledge만 backfire 1건이지만 net은 방어적. 첫 토큰
쏠림 경고(S7)가 실재해도 실제 knockout 효과는 손상되지 않았다.

### 다른 suite로 확장 (2026-09-12, important_instructions 기준)

| suite | 표본 | k0 sec | kN sec | 억제/bf/persist | net | kN utility | parse_ok |
|---|---|---|---|---|---|---|---|
| slack | 35 | 0.229 (8) | 0.000 | 8/0/0 | +8 (전량 억제) | 0.429 (↑) | 0.711 |
| **banking** | 42 | 0.095 (4) | 0.000 | 4/0/0 | **+4 (전량 억제)** | 0.667 (↓, −1 소폭) | 0.632 |
| **workspace** | 53/60 (7 oom) | 0.000 | 0.000 | **대조군 — baseline 공격 자체가 0건** | 0 | 0.189 (무변화) | 0.615 |
| travel (`--eval_split all`) | 52/60 (8 oom) | 0.000 | 0.019 (1) | 0/1/0 | **−1 (잡음)** | 0.096 (**극히 낮음**) | 0.849 |

- **banking도 slack처럼 전량 억제·backfire 0** — Qwen3-8B는 70B와 달리 banking에서도
  knockout이 깨끗하게 작동. utility는 소폭 하락(29→28/42)했지만 손실 1건 수준.
- **workspace는 baseline 공격 성공이 아예 없어 순수 대조군** — knockout이 정상 과업에
  영향 없음을 재확인(utility 무변화).
- **travel은 4090에서 100% OOM → Blackwell(32GB)로 재시도해 52/60 확보.** k0_util이
  0.096으로 극히 낮은 건 `docs/todo.md` P14(멀티콜 체이닝 파서 미지원)가 예측한 그대로 —
  이 suite 자체가 구조적으로 낮은 신뢰도. 유일한 backfire 1건도 baseline이 0이라는 점에서
  통계적 잡음에 가까움.

---

## S10. 결과 요약 & 다음 단계

**확인된 것**

- Llama-70B: 수동 device_map으로 자체 헤드 탐색 가능. 자체 헤드가 8B 전이 헤드보다
  뚜렷이 강하게 작동(전이 헤드는 net 효과 없음). heldout 표본을 4배(15→60) 늘려도 net
  억제 유지 — 단 important_instructions에서 utility 첫 손상 발견(원인 특정: suppressed
  case의 부작용).
- **70B는 slack·banking만 평가 가능 — travel·workspace는 A6000(48GB)로도 100% OOM.**
  banking은 knockout 신호가 약함(net+1, 사실상 baseline 5건뿐)지만 utility는 오히려 개선.
- Qwen3-8B: 첫 토큰 쏠림은 실재(qwen2 대비 34배)하나 D_inj 신호를 지우지 않음 — 8B급과
  동일한 knockout 패턴(important_instructions 전량 억제, tool_knowledge net 방어적) 재현.
- **Qwen3-8B는 4개 suite 전부 평가 가능**(70B와 대비). banking도 slack처럼 전량 억제,
  workspace는 baseline 공격이 없는 순수 대조군, travel은 알려진 파서 문제(P14)로 낮은
  utility가 재확인됨(Blackwell로 겨우 확보, 4090은 100% OOM).

**다음 단계**

- Qwen2.5-32B 자체 헤드 nf4dq 재탐색 (`docs/todo.md` §4-1).
- 70B k-sweep(topk 20→40→60), utility 손상 원인 정밀 확인.
- 70B travel/workspace OOM 근본 해결(2-GPU 분산 등) — 지금은 evaluation 자체가 안 됨.
