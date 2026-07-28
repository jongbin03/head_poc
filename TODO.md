# TODO

## 우선순위

| 순위 | 항목 | 비고 |
|---|---|---|
| ~~P0~~ | ~~로컬 5070Ti 환경에서 파이프라인 전체 재현~~ | **완료** (2026-07-28, `results/2026-07-28_local_5070ti`) |
| ~~P1~~ | ~~Qwen2.5-7B(8B급)로 본 실험 확장~~ | **완료** (2026-07-28, 4bit, 같은 결과 폴더) |
| ~~P2~~ | ~~외부/추가 데이터셋으로 기존 control head의 edge knockout 효과 검증~~ | **완료** (2026-07-28, P2-a/b/c/d 전부) |
| P3 | control head 내 internal-only vs external-only 채널 분기 검증 | 기존 채널 분기 실험 |

아래는 우선순위 순서대로 자세한 내용, 그 뒤에 보류 항목.

---

## ~~P0. 로컬 5070Ti 환경에서 파이프라인 전체 재현~~ — 완료 (2026-07-28)

`transformers==4.51.3` 고정으로 lxt import 이슈 해결 후, 0.5B(smoke)/1.5B/3B 전체 30개
템플릿 재실행. Colab(T4) 결과와 수치 근접(1.5B/3B 오차 범위 내) — 환경 재현성 확인됨.
자세한 로그: `results/2026-07-28_local_5070ti/README.md`.

## ~~P1. Qwen2.5-7B(8B급)로 본 실험 확장 (5070Ti)~~ — 완료 (2026-07-28)

P0와 같은 세션에서 `--four_bit`로 7B까지 바로 실행, knockout sweep 정상 수행
(`malicious_token_prob`이 k=10~20 안에 0으로 붕괴, `read_token_prob`은 k=20까지
오히려 소폭 상승). 결과는 P0와 같은 폴더(`results/2026-07-28_local_5070ti/README.md`)에
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
과적합 우려 해소. 결과: `results/2026-07-28_Qwen-Qwen2-5-1-5B-Instruct_heldout{0..4}/`,
`results/2026-07-28_Qwen-Qwen2-5-7B-Instruct_4bit_heldout0/`.

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
결과: `results/2026-07-28_Qwen-Qwen2-5-1-5B-Instruct/summary.txt`,
`results/2026-07-28_Qwen-Qwen2-5-7B-Instruct_4bit/summary.txt`.
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
`results/2026-07-28_Qwen-Qwen2-5-1-5B-Instruct_unseen/summary.txt`,
`results/2026-07-28_Qwen-Qwen2-5-7B-Instruct_4bit_unseen/summary.txt`.

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

(jaccard(synthetic_heads, ia_heads) = 0.081 — 서로 거의 안 겹침에도 세 조건 성능이 사실상 동일)

**결론**: 세 조건이 거의 동일한 성능(억제 후 malicious_token_prob ~0.044~0.045)을 보임.
교집합(9개, 가장 적은 개입)이 나머지 둘과 동등한 효과를 내는 건 "두 도메인에서 공통으로
뽑힌 head가 핵심이고 나머지는 군더더기"라는 신호. 동시에 InjecAgent 자체에서 독자적으로
찾은 head(우리 것과 거의 안 겹침)도 비슷한 성능이라는 건 "이 도메인엔 우리가 못 찾은
다른 유효한 control head들도 존재"한다는 뜻. 무엇보다, head 선정 방법을 바꿔도(합성 전용/
InjecAgent 전용/교집합) 전부 P2-c에서 본 것과 같은 수준(~0.04~0.05)의 잔여 확률에서
멈춘다는 점이 중요함 — **P2-c의 부분적 전이는 "head를 잘못 골라서"가 아니라 정말
도메인 구조 자체의 한계**라는 결론에 더 무게가 실림. 결과:
`results/2026-07-28_Qwen-Qwen2-5-1-5B-Instruct_headsplit/summary.txt`.

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

## P3. control head 내 internal-only vs external-only 채널 분기 검증

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
   프롬프트 포맷별로 따로 설계해야 한다는 뜻이므로 README.md/RUN.md에 반영

**참고**: 2026-07-27 대화에서 나온 논의. `results/2026-07-27_colab_phase1to3/README.md`의
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
방향을 더 정밀하게 검증. RUN.md/README.md의 원래 계획(Phase 2)에 있던 것.

### 실전 배포 형태로 전환

**배경**: 지금 `edge_knockout()`은 평가 스크립트 안에서만 쓰는 context manager. 실제
추론 서버에 상시 적용 가능한 형태(예: 모델 로드 시 항상 적용되는 forward hook, 또는
vLLM/TGI 같은 서빙 프레임워크에 끼워 넣는 방법)로 바꾸려면 별도 설계가 필요.
이건 방어 효과 검증이 어느 정도 끝난 뒤(위 항목들) 고려할 일.
