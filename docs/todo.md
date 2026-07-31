# TODO

## 우선순위

| 순위 | 항목 | 비고 |
|---|---|---|
| ~~P0~~ | ~~로컬 5070Ti 환경에서 파이프라인 전체 재현~~ | **완료** (2026-07-28, `../results/2026-07-28_local_5070ti`) |
| ~~P1~~ | ~~Qwen2.5-7B(8B급)로 본 실험 확장~~ | **완료** (2026-07-28, 4bit, 같은 결과 폴더) |
| ~~P2~~ | ~~외부/추가 데이터셋으로 기존 control head의 edge knockout 효과 검증~~ | **완료** (2026-07-28, P2-a/b/c/d 전부) |
| ~~P7~~ | ~~방법론 진단 — 랜덤 head 기준선 / top-K sweep / jaccard 우연 기준선 / 문서-코드 불일치 정정~~ | **완료** (2026-07-31, `../results/2026-07-31_Qwen-Qwen2-5-1-5B-Instruct`) |
| P8 | 합성 데이터셋의 content-availability 교란 제거 | **보류 (2026-07-31 결정)**. head 탐색은 synthetic 그대로 써도 유효하다고 판단, 발표용 헤드라인은 AgentDojo 네이티브 채점(P4)에 맡기기로 함 |
| **P4** | **(교수님 피드백) Head 탐색 방법론 재설계 — synthetic/InjecAgent/AgentDojo 3소스 비교, Track A(탐색)/Track B(평가) 하이브리드** | **최우선.** 구 P4+P6 통합. 가장 비중 큰 작업 |
| P5 | (교수님 피드백) 키 그룹 2개 vs 데이터셋 모드 4개 문서 정비 | P4 결과로 서술이 또 바뀔 수 있어 그 뒤에 |
| P3 | control head 내 internal-only vs external-only 채널 분기 검증 | **보류**. P4가 channel 축을 source 축으로 대체해 자동 흡수되지 않게 됨 |

아래는 우선순위 순서대로 자세한 내용, 그 뒤에 보류 항목.
자세한 대응 계획(특히 "키 그룹" 정의 재확인)은 `feedback-2026-07-29.md`,
**현재 방법론의 교란 요인 분석은 `review-2026-07-29.md`** 참고.

> ⚠️ **2026-07-29 자체 리뷰로 우선순위가 크게 바뀌었다.** `review-2026-07-29.md`에서
> 합성 데이터의 완전 억제(`0.0000`)가 content-availability 교란으로 부풀려졌을 가능성,
> InjecAgent utility 지표가 ASR의 산술적 뒷면이라 독립 정보가 없다는 점이 확인됐다.
> 측정을 고치기 전에 AgentDojo(P4)로 넘어가면 신뢰할 수 없는 숫자만 하나 더 늘어난다.

---

## ~~P0. 로컬 5070Ti 환경에서 파이프라인 전체 재현~~ — 완료 (2026-07-28)

`transformers==4.51.3` 고정으로 lxt import 이슈 해결 후, 0.5B(smoke)/1.5B/3B 전체 30개
템플릿 재실행. Colab(T4) 결과와 수치 근접(1.5B/3B 오차 범위 내) — 환경 재현성 확인됨.
자세한 로그: `../results/2026-07-28_local_5070ti/README.md`.

## ~~P1. Qwen2.5-7B(8B급)로 본 실험 확장 (5070Ti)~~ — 완료 (2026-07-28)

P0와 같은 세션에서 `--four_bit`로 7B까지 바로 실행, knockout sweep 정상 수행
(`malicious_token_prob`이 k=10~20 안에 0으로 붕괴, `read_token_prob`은 k=20까지
오히려 소폭 상승). 결과는 P0와 같은 폴더(`../results/2026-07-28_local_5070ti/README.md`)에
통합 기록됨 — 별도 폴더 분리하지 않음.

## P2. 외부/추가 데이터셋으로 기존 control head의 edge knockout 효과 검증

**배경**: 지금까지 head를 찾은 데이터와 knockout 효과를 검증한 데이터가 완전히 동일함
(30개 템플릿 전체). "이 30개 조합에서만 통하는 head"일 위험이 있어, 우리가 찾은
control head가 **새로운/외부 데이터**에서도 여전히 먹히는지 확인이 필요.

### ~~P2-a. Held-out 분리 (같은 데이터셋 안에서)~~ — 완료 (2026-07-28)

`dataset.py`의 `sample_templates()`/`build_phase0_batch()`에 `style_indices` 필터를,
`run_pipeline.py`에 `--heldout_style_idx {0..4}` 옵션을 추가 (해당 스타일을 head 선정
`[2/4]`/`[3/4]`에서 완전히 배제하고 `[4/4]` knockout sweep을 in-distribution/held-out
양쪽으로 나눠 실행). 1.5B로 스타일 0~4 전부, 7B(4bit)로 대표 스타일(0번) 재확인 — 5개
스타일 전부에서 held-out 데이터에도 knockout 효과(`malicious_token_prob`이 k=10~20 안에
0으로 붕괴, `read_token_prob` 유지/상승)가 그대로 재현됨. "30개 템플릿에만 통하는 head"
과적합 우려 해소. 결과: `../results/2026-07-28_Qwen-Qwen2-5-1-5B-Instruct_heldout{0..4}/`,
`../results/2026-07-28_Qwen-Qwen2-5-7B-Instruct_4bit_heldout0/`.

**작업 중 발견한 버그**: `run_pipeline.py`의 `run_dir` 기본값이 `<날짜>_<모델명>`만 써서
같은 날 같은 모델로 `--heldout_style_idx`만 바꿔 여러 번 돌리면 결과가 서로 덮어써짐.
`_heldout{N}` 접미사를 추가해 수정 (1.5B 5회 실행 중 발견, 다행히 콘솔 로그에서 5개 결과
전부 복구 가능했음 — `functional_map.png`는 텍스트 로그에 없어 마지막 1개만 복구됨).

### ~~P2-c. 검증된 외부 IPI 벤치마크로 재현~~ — 완료 (2026-07-28)

`adapters/injecagent.py`를 새로 추가 — InjecAgent(github.com/uiuc-kang-lab/InjecAgent)의
`Tool Response Template`에서 `<Attacker Instruction>` placeholder 위치로 D_benign/D_inj
span을 분리하고, `exec_target`=Attacker Tool 이름 첫 토큰(공격 성공 proxy), `read_target`=
User Tool 이름 첫 토큰(주입에도 원래 요청한 tool을 올바르게 호출하는지, utility proxy)으로
매핑. `run_pipeline.py --injecagent` 옵션으로 우리 합성 데이터셋에서 찾은
`control_heads_both`를 그대로 재사용해 (head 선정에는 전혀 안 씀, 순수 평가 전용)
InjecAgent 전체 1,054개 test case(dh_base+ds_base)에 knockout sweep 실행.

| 모델 | k=0 malicious | k=40 malicious | k=0 read | k=40 read |
|---|---|---|---|---|
| 1.5B | 0.1894 | 0.0406 (~4.7배 ↓) | 0.7834 | 0.9474 |
| 7B(4bit) | 0.5237 | 0.1300 (~4.0배 ↓) | 0.4881 | 0.8849 |

**결론**: 완전히 다른 실제 벤치마크(다른 도메인 구조 + 훨씬 은근한 문구)에서도 knockout
효과가 두 스케일 모두 일관되게(4~5배) 재현됨 — utility(read_token_prob)도 knockout 후
오히려 상승. 다만 우리 합성 데이터셋만큼 완전한 억제(k=10 안에 0)는 아니고 잔여 확률이
남음. baseline 공격 확률이 스케일이 커질수록 오히려 올라간 것(0.19→0.52)으로 보아, 남은
효과 차이는 스케일 문제가 아니라 **도메인/문구 구조 차이**(은근한 tool-call 관찰 텍스트
vs 우리의 노골적인 "IGNORE ALL..." 스타일) 때문으로 추정 — 아래 P2-b 재개 검토 근거.
결과: `../results/2026-07-28_Qwen-Qwen2-5-1-5B-Instruct/summary.txt`,
`../results/2026-07-28_Qwen-Qwen2-5-7B-Instruct_4bit/summary.txt`.
InjecAgent 원본 데이터는 git에 커밋하지 않음(`external_injecagent/`, `.gitignore`에 추가) —
재현하려면 `git clone https://github.com/uiuc-kang-lab/InjecAgent.git external_injecagent`.

### ~~P2-b. 미지의 공격 문구로 확장~~ — 완료 (2026-07-28)

P2-c 결과가 부분적 전이(완전한 억제 아님)로 나와서, "문구 때문인지 도메인 구조 때문인지"
분리하기 위해 재개. `dataset.py`에 `_INJECTION_STYLES_UNSEEN`(다국어 혼용/코드블록 위장/
유니코드 난독화/짧고 우회적인 표현, 4종) + `sample_unseen_templates()`/
`build_unseen_style_batch()` 추가, `run_pipeline.py --unseen_styles` 옵션으로 기존 5종에서
찾은 `control_heads_both`를 그대로(재선정 없이) 새 스타일 4종 x 도메인 6종(24개)에 적용.

| 모델 | 기존 5종 baseline→k=10 | 미지 4종 baseline→k=10 |
|---|---|---|
| 1.5B | 0.9118 → 0.0000 | 0.9228 → 0.0000 |
| 7B(4bit) | 0.9690 → 0.0000 | 0.9998 → 0.0000 |

**결론**: 완전히 새로운 표현 방식(다국어/코드블록 위장/유니코드 난독화/극단적으로 짧은
문구)에도 두 스케일 모두 기존 5종과 **동일하게 k=10 안에 완전히 붕괴** — 문구는 knockout
효과를 전혀 약화시키지 않음. 이는 P2-c의 부분적 전이가 문구 때문이 아니라 **도메인/데이터
구조 차이**(우리 데이터셋의 단순 이메일/문서 구조 vs InjecAgent의 tool-call observation
내부에 심어진 구조) 때문이라는 가설을 뒷받침함. 결과:
`../results/2026-07-28_Qwen-Qwen2-5-1-5B-Instruct_unseen/summary.txt`,
`../results/2026-07-28_Qwen-Qwen2-5-7B-Instruct_4bit_unseen/summary.txt`.

### ~~P2-d. InjecAgent 일부로 직접 head 탐색 → 합성 head와 교집합 → 나머지로 검증~~ — 완료 (2026-07-28)

**배경**: P2-c/P2-b 결론상 "InjecAgent 잔여 효과는 문구가 아니라 도메인 구조 차이" 였는데,
그렇다면 "InjecAgent 도메인 자체에서 직접 찾은 head는 우리 합성 head와 얼마나 겹치고,
그 교집합이 오히려 더 잘 통하는가"를 봐야 원인을 한 단계 더 좁힐 수 있음 (사용자 제안).

`adapters/injecagent.py`에 `build_injecagent_clean_example`(주입 없는 read baseline)/
`build_injecagent_pairs`/`split_pairs`(고정 seed로 head 선정용/평가용 분리) 추가.
`run_pipeline.py --injecagent_headsplit --injecagent_head_n N`으로 InjecAgent 중 N개만
써서 (교집합 없이 그 채널 하나에서 바로) top-K head를 뽑고, 우리 합성 `control_heads_both`와
교집합한 뒤, 나머지 InjecAgent case에 (a) 합성 단독 (b) InjecAgent 단독 (c) 교집합 3가지를
나란히 knockout 평가. `--injecagent_headsplit_from <json>`으로 head 탐색 결과를 저장/재사용해
비싼 relevance 계산 없이 eval sweep만 새 프로세스에서 재실행 가능 (아래 버그의 우회책).

**1.5B, head_n=60(seed=42), eval n=994 결과**:

| 조건 | head 개수 | k=0 malicious | k=full malicious | k=0 read | k=full read |
|---|---|---|---|---|---|
| synthetic-only (`control_heads_both`) | 14 | 0.1907 | 0.0450 | 0.7834 | 0.9306 |
| injecagent-only (InjecAgent 60개로 직접 탐색) | 20 | 0.1907 | 0.0436 | 0.7834 | 0.9353 |
| 교집합 | 9 | 0.1907 | 0.0447 | 0.7834 | 0.9309 |

(jaccard(synthetic_heads, ia_heads) = 0.36 — 서로 상당수 겹치지만 완전히 같지는 않은 상태에서 세 조건 성능이 사실상 동일)

**결론**: 세 조건이 거의 동일한 성능(억제 후 malicious_token_prob ~0.044~0.045)을 보임.
교집합(9개, 가장 적은 개입)이 나머지 둘과 동등한 효과를 내는 건 "두 도메인에서 공통으로
뽑힌 head가 핵심이고 나머지는 군더더기"라는 신호. 동시에 InjecAgent 자체에서 독자적으로
찾은 head(우리 것과 일부만 겹침, jaccard 0.36)도 비슷한 성능이라는 건 "이 도메인엔 우리가 못 찾은
다른 유효한 control head들도 존재"한다는 뜻. 무엇보다, head 선정 방법을 바꿔도(합성 전용/
InjecAgent 전용/교집합) 전부 P2-c에서 본 것과 같은 수준(~0.04~0.05)의 잔여 확률에서
멈춘다는 점이 중요함 — **P2-c의 부분적 전이는 "head를 잘못 골라서"가 아니라 정말
도메인 구조 자체의 한계**라는 결론에 더 무게가 실림. 결과:
`../results/2026-07-28_Qwen-Qwen2-5-1-5B-Instruct_headsplit/summary.txt`.

**작업 중 발견한 심각한 버그 (미해결, 다음 섹션에 근본 해결책 기록)**:
`attn_relevance.compute_head_relevance`(backward pass 기반 relevance 계산)를 한 프로세스
안에서 여러 번 호출하면 호출 1번당 GPU 메모리가 ~0.12GB씩 계속 누적되다가 약 80쌍(=160회
호출) 근처에서 OOM. `gc.collect()`/`torch.cuda.empty_cache()` 빈도를 늘려도, **relevance
루프가 끝난 뒤 한 번 더 강하게 정리해도, 모델 인스턴스를 통째로 다시 로드해도 전혀 안
줄어듦**(전부 재현 실험으로 확인 — 정리 전/후 GPU 메모리 수치가 완전히 동일했음). 즉 모델
객체나 파이썬 참조 문제가 아니라 프로세스 전역/CUDA 드라이버 레벨(혹은 `lxt` monkey-patch
내부)의 문제로 보임. 기존 [2/4] 단계(30개 템플릿 x 3회 = 90번 호출)에서도 같은 비율로 이미
새고 있었는데, 호출 수가 적어서 지금까지 안 터졌던 것뿐임 — InjecAgent에만 있는 버그가
아니라 `compute_head_relevance`를 많이 호출하는 모든 실험에 잠재된 문제.

**추가로 발견한 부작용**: relevance 루프 직후 같은 프로세스에서 eval sweep(forward-only)을
이어 돌리면, 물리 VRAM이 이미 거의 꽉 찬 상태(약 97%)로 시작해 스와핑/스래싱이 나 (GPU-Util
100%인데도) 극도로 느려짐 — 실제로 30분 이상 안 끝났음. **대응**: 위에서 언급한
`--injecagent_headsplit_from`으로 head 탐색 결과를 json으로 저장해두고 완전히 새 프로세스로
eval sweep만 재실행 — 새 프로세스는 메모리가 깨끗해서 정상 속도로 끝남 (수 초~수십 초).
이 과정에서 `run_dir` 이름에 `--injecagent_headsplit_from`을 빠뜨려서 `_headsplit` 접미사가
안 붙는 버그도 발견/수정함 — 접미사가 없어서 이전 P2-c 결과 폴더를 한 번 덮어썼다가 git으로
복구했음 (커밋된 파일이라 다행히 복구 가능했음, 커밋 전 결과물은 이런 사고에 취약하다는 교훈).

## ~~P7. 방법론 진단 — 지금까지의 결론이 유효한지 판별~~ — 완료 (2026-07-31)

**배경**: 2026-07-29 자체 리뷰(`review-2026-07-29.md`) 결과. 합성 데이터에서 나온
완전 억제(`malicious_token_prob = 0.0000`)가 "control head를 껐기 때문"인지 "공격 대상
문자열을 못 보게 했기 때문"인지 구분할 실험이 지금 하나도 없다. 전부 forward-only라
`compute_head_relevance`의 메모리 누수와 무관하고 비용이 거의 없다.

**한 일**: `head_ranking.py`에 `random_heads()`/`expected_jaccard_by_chance()` 추가,
`edge_ablation.py` 기본 `ks`에서 k=80 제거, `run_pipeline.py`에 랜덤 head 기준선 /
`control_heads_both`(교집합) sweep / top-K(5/10/20/40) sweep을 모두 추가해 union
top-40 sweep과 나란히 출력·`summary.txt`에 기록하도록 확장. 1.5B, 템플릿 30개 전체로
실행 (`../results/2026-07-31_Qwen-Qwen2-5-1-5B-Instruct/summary.txt`).

**결과**:
1. **랜덤 head 기준선**: 실제 선정 head(교집합 9~14개)는 k=10 안에
   `malicious_token_prob` 0.9118 → **0.0000**(완전 억제). 동일 개수의 랜덤 head는
   k=40까지 늘려도 0.9118 → **0.6448**까지만 떨어짐 — head 선정이 랜덤보다 훨씬
   적은 개입으로 훨씬 강하게 억제한다는 점은 확인됐다. 다만 랜덤도 어느 정도
   떨어진다는 점은 아래 P8의 content-availability 교란 가설과 방향이 일치 — P8이
   여전히 필요하다는 근거가 됨.
2. **top-K sweep**: K=10(교집합 9개)에서 이미 완전 억제, K=5(교집합 3개)에서도 부분
   억제(0.35). K=20이 특별한 임계점은 아니고 K≥10이면 충분해 보임.
3. **jaccard 우연 기준선**: jaccard(internal,external)=0.538, 우연 기대값=0.031 →
   **17.6배**, 명백히 유의함 (`head_ranking.py`의 `jaccard_chance_at_k`로 상시 계산됨).
4. **`dual_use_candidates`**: `control_heads_both` 14개 중 **9개**가 read top-20과도
   겹침 — "read와 control이 완전히 분리된 회로"라는 주장은 유지하기 어렵고, 상대적
   주장("internal-external끼리의 겹침 0.538이 read와의 겹침 0.290/0.333보다 뚜렷이
   크다")만 성립.
5. **layer 0 지배 재확인**: `control_heads_both` 14개 중 6개가 layer 0 — review.md
   3-4의 수치와 일치, 정보 대역폭 차단 대안 가설을 아직 배제 못 함.
6. **문서-코드 불일치**: union top-40 sweep과 intersection sweep이 거의 동일하게(둘 다
   k=10~14 안에 0.0000) 억제되므로 기존 헤드라인 수치는 안전함이 확인됨 — 코드는
   그대로 두고 두 sweep을 나란히 보고하는 쪽으로 정리.

## P8. 합성 데이터셋의 content-availability 교란 제거 (보류, 2026-07-31)

**보류 결정**: P7에서 head 탐색이 랜덤 기준선 대비 뚜렷이 유효함이 확인되어("head 찾을 때는
문제 없었다"), synthetic 데이터셋은 이 교란을 고치지 않은 상태 그대로 P4의 head 탐색
소스 중 하나로만 계속 쓰기로 함. 대신 발표용 헤드라인 성능 수치는 P4의 AgentDojo 네이티브
채점(Track B)에 맡긴다. 아래는 참고용으로 남겨둔 기존 배경/설계 — 나중에 synthetic을
평가(evaluation)에도 다시 쓰게 되면 참고.

**배경**: `review-2026-07-29.md` 1절. `exec_target`이 30개 템플릿 전부 `"attacker"` 하나로
고정돼 있고(`dataset.py:294`) 이 문자열이 **오직 `data_inj` 안에만** 존재한다. 게다가
assistant prefix가 이미 tool 호출을 열어둔 상태(`dataset.py:73`)라, 측정 대상이 "명령을
따르기로 결정하는가"가 아니라 "이미 정해진 forward의 수신자를 받아쓰는가"다. D_inj 엣지를
끊으면 그 문자열이 시야에서 사라지므로 확률 붕괴가 거의 자동이다. P2-a/P2-b도 문구만
바꿨을 뿐 이 교란을 그대로 물려받는다 (unseen 4종 전부 `attacker@evil.com` 포함).

**할 일**:
1. 공격 대상 문자열을 D_inj 밖에도 배치 — system prompt의 주소록이나 tool 목록에 넣어,
   D_inj를 가려도 대상 자체는 보이게 (InjecAgent가 이미 이렇게 돼 있고
   `adapters/injecagent.py:110-118`, 정확히 거기서만 억제가 부분적이다).
2. assistant prefix를 InjecAgent처럼 중립적으로 바꿔 모델이 **결정**하게 만들기.
3. 공격 목표를 템플릿마다 다르게 (지금은 30개 전부 같은 주소·같은 행동·같은 타깃 토큰).

**예상 결과**: 여기서 `0.0000`이 안 나올 가능성이 높은데, **그 자체가 정직하고 발표 가능한
결과**다. 이 단계를 거쳐야 합성 지표가 비로소 "명령 추종"을 측정하고 InjecAgent와 공정하게
비교된다.

## P5. (교수님 피드백) 키 그룹 2개 vs 데이터셋 모드 4개 문서 정비

**배경**: 2026-07-29 피드백 — head 선정에 쓰는 "키 그룹"은 `internal`/`external` 2개뿐인데
(`head_ranking.py:69`, `control_heads_both = internal_heads ∩ external_heads`), 합성 데이터셋
모드는 `read_clean`/`read_injected`/`internal`/`external` 4개(`dataset.py:75`)라 왜 4개인지,
특히 `read_injected`는 왜 있는지 설명이 부족했음.

**중요한 확인 사항(2026-07-29 논의)**: 이 2 groups는 "정상 토큰 vs injected 토큰"이 아니라
**둘 다 `data_inj`(injected) 스팬 기준이고, 차이는 backward 타깃(주입 지시가 자유 텍스트로
새는 경로 vs tool-call로 새는 경로)**임. `read`(정상 토큰, `data_benign` 스팬)는 세 번째로
따로 계산되지만 교집합엔 안 들어감. 이 오해가 "2 vs 4" 질문의 근본 원인 중 하나로 보임.

**할 일**: `dataset.py` MODES 주석 정비(위 구분 명시), `methodology.md`/`presentation-notes.md`/
`presentation/`의 "4개 모드" 서술을 "head 탐색용 2 groups(둘 다 injected 스팬, 경로만 다름) +
utility 측정용 2 baselines(read_clean/read_injected, 정상 스팬)" 구조로 재서술,
데이터셋 커버리지(스타일x도메인) 표 추가. 자세한 내용: `feedback-2026-07-29.md` 0절, 2절.

## P4. (교수님 피드백, 구 P4+P6 통합) Head 탐색 방법론 재설계 — AgentDojo를 탐색 소스로

**배경**: 2026-07-29 피드백 — (a) AgentDojo로 재현 + utility 측정 방법 정비, (b) attention
head 탐색 방법 체계화, 두 요구를 논의 끝에 하나로 통합. 이유: P2-d에서 이미 synthetic-only/
InjecAgent-only/교집합 세 조건이 거의 동일한 성능(jaccard=0.36)을 보였는데, 이는 "synthetic
데이터로 찾은 head를 실제 벤치마크에 전이시키는" 지금 구조가 synthetic의 한계(노골적 문구,
단순 문서 구조)를 상속할 수 있다는 신호. 그렇다면 AgentDojo를 단순 재현 대상으로만 쓰기보다
**head 탐색 자체의 소스로 삼는 실험을 먼저 설계**하는 게 순서상 맞음.

**설계 확정 (2026-07-31 논의)**:

1. **synthetic 데이터셋은 폐기하지 않고 head 탐색(discovery)에는 계속 쓴다.** P7에서 랜덤
   기준선 대비 뚜렷한 차이가 확인됐으므로("head 찾을 때는 문제 없었다") discovery 자체는
   유효하다고 보고 세 번째 탐색 소스로 유지. 다만 **P8(content-availability 교란 제거)은
   진행하지 않음** — synthetic은 지금 상태 그대로 discovery 전용으로만 쓰고, 발표용
   헤드라인 성능 수치는 AgentDojo 네이티브 채점(아래 Track B) 쪽에 둔다. `methodology.md`
   등에 synthetic을 인용할 때는 "discovery 전용, P8 미적용 상태"라는 caveat을 달 것.
2. **Track A(탐색) / Track B(평가) 하이브리드로 AgentDojo를 붙인다.**
   - **Track A (head 탐색 전용)**: `adapters/injecagent.py` 패턴을 그대로 재사용 —
     AgentDojo injection task 중 공격 문구가 삽입된 tool 응답 턴 하나만 잘라 단일
     프롬프트로 만들고(`"Thought:...\nAction:"` 유도), `compute_head_relevance`로 head
     탐색. 멀티턴/환경 실행이 필요 없어 기존 인프라(`head_ranking.py`, P7에서 만든
     `random_heads`/`expected_jaccard_by_chance`/top-K sweep)를 그대로 재사용 가능.
   - **Track B (평가 전용, 신규 구현 필요)**: knockout 효과의 최종 수치는 AgentDojo의
     실제 멀티턴 agent loop + 네이티브 `utility_function`/`security_function`(모델 출력이
     아니라 tool 실행 후 환경 상태를 검사하는 결정론적 함수)으로 채점한다. `edge_knockout()`을
     롤아웃 전체에 걸친 여러 `generate()` 호출 동안 유지해야 하므로(위치 인덱스가 턴마다
     늘어나는 input_ids 기준으로 안정적이어야 함) 신규 구현이 필요. 이게 review-2026-07-29.md
     2절이 지적한 "InjecAgent utility 지표가 ASR의 산술적 뒷면이라 독립 정보 없음" 문제의
     근본 해결책이자 AgentDojo로 가는 진짜 이유.
3. **head 탐색 소스는 3개, 비교 후 합집합/교집합 결정.**
   - synthetic으로 찾은 head (`control_heads_both`, 기존 결과 재사용)
   - InjecAgent로 찾은 head (P2-d 패턴, `build_injecagent_pairs`/`split_pairs` 재사용)
   - AgentDojo(Track A)로 찾은 head (신규)
   - 세 집합의 jaccard 겹침을 비교하고, 각 집합(+ 합집합 + 교집합)의 knockout 성능을
     Track B로 평가해 최종적으로 어떤 조합을 쓸지 결정한다.
4. **internal/external 채널 축은 AgentDojo에 이식하지 않는다.** InjecAgent와 마찬가지로
   AgentDojo도 tool-calling 단일 포맷이라 "자유 텍스트로 새는 경로"가 구조적으로 없음(P2-d에서
   이미 확인된 사실과 동일). 채널(internal/external) 축 대신 **소스(synthetic/InjecAgent/
   AgentDojo) 축**을 교집합·비교의 기본 틀로 삼는다 — "어느 소스로 찾은 head가 다른 소스에도
   잘 전이되는가"가 새로운 핵심 질문.

**agentdojo 패키지 코드 조사 결과 (2026-07-31, `pip install agentdojo==0.1.35`)**: 예상보다
Track B 공수가 작다.

- `BaseUserTask.utility()`/`BaseInjectionTask.security()`는 정확히 예상대로 "모델 출력 +
  실행 전/후 환경 상태(pydantic 객체) 비교"로 판정하는 결정론적 함수 — 토큰 확률이 아님.
- `AgentPipeline`은 `config.llm`에 문자열(OpenAI 등) 대신 **커스텀 `BasePipelineElement`
  객체를 직접 꽂을 수 있게** 설계돼 있음. 기본 제공 `LocalLLM`은 OpenAI 호환 API 서버(vLLM,
  포트 8000)를 호출하는 방식이라 우리 `edge_knockout` 몽키패치와 안 맞지만(별도 프로세스라
  attention 함수를 바꿔치기할 수 없음), **vLLM 서버 없이 우리 LLM 요소로 통째로 교체**하면
  됨 — `agentdojo/agent_pipeline/llms/local_llm.py`를 템플릿으로 HTTP 대신 `model.generate()`
  직접 호출 + `edge_knockout` 적용.
- 멀티턴 루프는 `ToolsExecutionLoop`가 이미 구현("tool_call 없을 때까지 LLM↔tool 반복") —
  우리는 LLM 요소만 만들어 끼우면 나머지 턴 관리는 agentdojo가 대신 함.
- tool 실행은 `FunctionsRuntime.run_function`이 순수 파이썬으로 pydantic `TaskEnvironment`
  객체를 조작하는 방식 — 네트워크/외부 서비스 불필요, 로컬에서 완전히 재현 가능.
- 주입 방식도 우리와 개념적으로 동일: `environment.yaml`에 이름 붙은 placeholder(예:
  `{injection_incoming_transaction}`)가 있고 공격 문구가 문자열 치환됨 —
  `PromptBuilder`/`adapters/injecagent.py`의 placeholder 방식 그대로 D_inj 토큰 위치 추적 가능.
- task suite 4종 확인: banking(16)/slack(21)/travel(20)/workspace(40 user task).

즉 직접 새로 짜야 하는 건 (a) 우리 모델을 감싸는 커스텀 LLM 파이프라인 요소, (b) 롤아웃
동안 D_inj 위치를 추적하는 로직 정도 — 멀티턴 루프·tool 실행·utility/security 채점은
agentdojo가 이미 제공.

**`adapters/agentdojo.py` 작성 결과 (2026-07-31)**: `adapters/injecagent.py` 패턴대로
`build_agentdojo_example`/`build_agentdojo_clean_example`/`build_agentdojo_pairs` 구현,
`adapters/injecagent.py`의 `split_pairs()`를 그대로 import해 재사용(중복 구현 없음).
4개 suite(banking/slack/travel/workspace) 전체 949개 (user_task, injection_task) 조합 중
**220개**가 필터를 통과해 pair로 만들어짐 (banking 61 / slack 47 / travel 8 / workspace 104).

구현 중 발견/수정한 것:
- `GroundTruthPipeline`(LLM 없이 `task.ground_truth()`를 그대로 실행)으로 "완벽한 에이전트"의
  메시지 시퀀스를 얻은 뒤, `[assistant(tool_call), tool(응답), assistant(다음 tool_call)]`
  3-메시지 패턴만 다룸 — 멀티턴 prefix가 필요한 case는 건너뜀(Track A는 단일 턴만 다룬다는
  설계 그대로).
- ⚠️ **일부 case는 read_target == exec_target** — 예: banking user_task_0 + injection_task_0은
  둘 다 다음 행동이 `send_money`이고 차이가 tool 이름이 아니라 인자(수신 IBAN)뿐인 "인자
  오염형" 공격. `feedback-2026-07-29.md` 1-1절이 예상했던 문제가 실제로 존재함을 확인 —
  지금은 이런 case를 건너뛰고 tool **이름**이 달라지는 "tool 선택형" case만 씀(인자 오염형은
  향후 별도 분석 대상으로 남김).
- ⚠️ **문자열 그대로 매칭이 안 됨**: `environment.yaml`이 Python `.format()` 삽입 후
  `yaml.safe_load`를 거치면서 주입 문구의 연속 개행(`\n\n\n<INFORMATION>\n\n`)이
  `\n\n\n<INFORMATION>\n`으로 정규화됨(YAML 개행 접기 규칙) — attack_text 리터럴로 `in`
  검사하면 전부 실패. `<INFORMATION>`/`</INFORMATION>` 태그를 앵커로 찾는 방식으로 수정.
- ⚠️ **tool 이름 문자열이 달라도 첫 토큰이 같을 수 있음**(예: `get_balance`/`get_iban`이
  둘 다 공통 접두사 토큰 공유) — tool 이름 비교가 아니라 토큰화된 `exec_target`/`read_target`
  id 비교로 다시 필터링(수정 전 256쌍 중 36쌍이 이 문제였음, 수정 후 220쌍 전부 확인됨).
- `ImportantInstructionsAttack._JB_STRING`(agentdojo가 실제 벤치마크에서 쓰는 표준 jailbreak
  템플릿)을 클래스 속성만 가져와 재사용 — `BaseAttack` 인스턴스화는 `target_pipeline.name`이
  agentdojo가 아는 모델 이름이어야 해서 우리 커스텀 모델과 안 맞아 피함.

**할 일** (구체 순서):
1. ~~`pip install agentdojo` 설치 + API 코드 조사~~ — 완료 (위 참고).
2. ~~`adapters/agentdojo.py` 작성 (Track A)~~ — 완료 (위 참고).
3. synthetic / InjecAgent / AgentDojo(Track A) 세 소스 각각 head 탐색 실행, jaccard 비교표
   작성 (P2-d/P7 코드 재사용) — **다음 단계**.
4. Track B 평가 하네스 구현: 커스텀 `BasePipelineElement` LLM 요소(`edge_knockout` 적용)
   + D_inj 위치 추적 로직 + `TaskSuite.run_task_with_pipeline`으로 네이티브 utility/security
   채점 연동.
5. 세 소스 단독 vs 합집합 vs 교집합을 Track B로 평가해 최종 head 집합 결정.
6. Top-K sweep + random-head baseline(P7 인프라 재사용)을 세 소스 전부에 동일 적용.
7. 결과를 `methodology.md`/`run-guide.md`에 "Head Selection Methodology" 섹션으로 통합.

별도 세션에서 단계별 진행 권장 (1. Track A 어댑터 → 2. 소스 비교 → 3. Track B 하네스 →
4. 최종 조합 결정 → 5. K/baseline → 6. 문서화 순).

## P3. control head 내 internal-only vs external-only 채널 분기 검증 (보류)

**2026-07-31 갱신**: P4에서 AgentDojo에 internal/external 채널 축을 이식하지 않기로
결정(channel 축 → source 축 전환)했으므로, 이 항목은 더 이상 P4에 자동으로 흡수되지
않는다. internal/external 구분이 남아있는 건 synthetic 데이터셋뿐이라, 진행한다면
synthetic 한정 분석으로 남을 것 — 우선순위는 낮음(보류). 아래는 기존 배경 기록.

**배경**: 현재 `head_ranking.py`의 `summarize_overlap`은 `internal_heads ∩ external_heads`
(=`control_heads_both`, "명령을 명령으로 받아들여 실행하는" 공통 회로)만 계산한다.
"명령을 따르기로 한 뒤, 그걸 자유 텍스트로 내보낼지 tool-call JSON으로 내보낼지를
가르는 head"는 아직 식별되지 않았다 — 이건 교집합이 아니라 **차집합(대칭차)**을 봐야 나온다.

**할 일**:
1. `head_ranking.py`의 `summarize_overlap`에 다음 두 집합 계산/반환 추가
   - `external_only = external_heads - internal_heads` (tool-call 포맷팅/라우팅 전용 후보)
   - `internal_only = internal_heads - external_heads` (자유 텍스트 경로 전용 후보)
2. `run_pipeline.py`에 위 두 집합 출력 추가 (`[3/4]` 로그)
3. `edge_ablation.py`로 `external_only` head들만 knockout해서:
   - tool-call 쪽 `malicious_token_prob`(external)은 떨어지는데
   - internal 쪽 자유 텍스트 오염(`internal` mode의 exec_target 확률)은 안 떨어지는지 검증
   - (반대 방향도 `internal_only`로 대칭 검증)
4. 결과가 "채널 전용 head가 실제로 존재"로 나오면, idea1의 knockout 방어 범위를
   프롬프트 포맷별로 따로 설계해야 한다는 뜻이므로 methodology.md/run-guide.md에 반영

**참고**: 2026-07-27 대화에서 나온 논의. `../results/2026-07-27_colab_phase1to3/README.md`의
`control_heads_both` 결과가 이 실험의 출발점(내부/외부 공통 control head 후보)이 됨.

---

## 보류 (우선순위 낮음)

### P2-d 서브프로세스 배치로 head_n 확장 — compute_head_relevance 메모리 누수 근본 해결

**배경**: P2-d 작업 중 `attn_relevance.compute_head_relevance`를 한 프로세스에서 반복 호출하면
(backward pass 기반이라 gradient 계산용 텐서가 관여) 호출당 ~0.12GB씩 GPU 메모리가
누적되다 약 160회 호출 근처에서 OOM 나는 버그를 발견함. `gc.collect()`/`empty_cache()`는
물론 **모델을 통째로 재로드해도, relevance 루프가 끝난 뒤 한 번 더 세게 정리해도 전혀
리셋 안 됨**(둘 다 재현 실험으로 확인 — 정리 전/후 수치가 완전히 동일) — 모델 인스턴스
문제가 아니라 프로세스 전역(CUDA 드라이버 또는 `lxt` monkey-patch 내부 전역 상태)의
문제로 보임.

**현재 임시 대응(이미 구현됨)**: `run_pipeline.py --injecagent_headsplit_from <p2d_heads.json>`
— head 탐색 결과(JSON)를 저장해두고, eval sweep만 완전히 새 프로세스에서 재실행해 메모리
압박(물리 VRAM 거의 꽉 찬 상태에서 스와핑/스래싱)을 피함. 다만 이건 "eval 단계"의 느려짐만
우회할 뿐, **head 탐색 자체(head_n)를 60 이상으로 못 올리는 근본 제약은 그대로 남아있음.**

**진짜 할 일**: head 탐색 자체를 여러 개의 **완전히 새로운 서브프로세스**로 나눠서 돌린다
(각 배치가 새 CUDA 컨텍스트를 갖게 되므로, 프로세스 전역 누수든 뭐든 배치 경계에서 강제로
리셋됨). 예: 50개씩 나눠서 `subprocess.run(...)`으로 각 배치를 짧은 헬퍼 스크립트로 실행,
배치별 partial score(`{"data_benign": Tensor[num_layers,num_heads], ...}`)를 `torch.save`로
임시 파일에 저장, 메인 프로세스가 다 모아서 `aggregate_scores`로 합산. 이렇게 하면 P2-d의
`--injecagent_head_n`을 60 같은 안전 범위 대신 원래 의도한 200(혹은 그 이상)까지 올릴 수
있음. **이 버그는 P2-d에만 국한된 게 아니라 `compute_head_relevance`를 대량으로 호출하는
모든 미래 실험(예: dataset_limit을 크게 늘리는 경우)에 잠재된 문제이므로, 장기적으로는
`attn_relevance.py` 자체에 배치 실행 유틸리티를 추가하는 것도 고려할 만함.**

### 부작용(collateral damage) 범위 측정 — 이 실험 밖의 일반 능력까지 깎이는지

**배경**: 지금 "정상 기능 보존"의 유일한 증거는 우리 데이터셋 안의 `read_token_prob` 하나뿐.
control head의 D_inj-edge를 끊는 게 이 시나리오 밖의 일반적인 언어 능력(요약, 산술, 일반
QA 등)까지 깎아먹는지는 전혀 안 봤음.

**할 일**: 공격과 무관한 일반 벤치마크(예: MMLU 일부 subset, 또는 간단한 자체 QA 세트)를
같은 모델에 knockout 적용 전/후로 돌려서 정확도 차이를 측정. 차이가 거의 없어야
"이 head들이 IPI-특이적이고, 일반 능력과는 분리되어 있다"는 주장이 완성됨.

### Llama 계열로 교차 검증 (코드는 이미 지원, 실행만 안 해봄)

**배경**: `attn_relevance.py`/`edge_ablation.py` 둘 다 `--family llama` 옵션이 있지만
지금까지 전부 Qwen2.5로만 돌렸음. Qwen 전용 결과인지 아키텍처 일반적인 결과인지 구분 안 됨.

**할 일**: `Llama-3.2-1B-Instruct`(빠른 확인용) → `Llama-3.1-8B-Instruct`(선택) 순으로
동일 파이프라인 재실행, Qwen2.5 결과와 패턴(jaccard 구조, knockout 효과) 비교.

### Path patching으로 인과관계 정밀 검증 (원래 설계에 있던 Phase 2, 아직 미착수)

**배경**: 지금 edge knockout은 "이 head들을 다 같이 끄면 효과가 있다"는 집합 수준의
증거임. 어떤 head가 정말 인과적으로 핵심인지(다른 head로 대체 불가능한지) 개별적으로는
아직 안 봄.

**할 일**: `control_heads_both`에서 나온 head들을 하나씩만 개별 knockout해서 개별 기여도를
측정하거나, activation patching(clean run의 activation을 corrupted run에 이식)으로 인과
방향을 더 정밀하게 검증. run-guide.md/methodology.md의 원래 계획(Phase 2)에 있던 것.

### 실전 배포 형태로 전환

**배경**: 지금 `edge_knockout()`은 평가 스크립트 안에서만 쓰는 context manager. 실제
추론 서버에 상시 적용 가능한 형태(예: 모델 로드 시 항상 적용되는 forward hook, 또는
vLLM/TGI 같은 서빙 프레임워크에 끼워 넣는 방법)로 바꾸려면 별도 설계가 필요.
이건 방어 효과 검증이 어느 정도 끝난 뒤(위 항목들) 고려할 일.
