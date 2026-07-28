# TODO

## 우선순위

| 순위 | 항목 | 비고 |
|---|---|---|
| ~~P0~~ | ~~로컬 5070Ti 환경에서 파이프라인 전체 재현~~ | **완료** (2026-07-28, `results/2026-07-28_local_5070ti`) |
| ~~P1~~ | ~~Qwen2.5-7B(8B급)로 본 실험 확장~~ | **완료** (2026-07-28, 4bit, 같은 결과 폴더) |
| ~~P2~~ | ~~외부/추가 데이터셋으로 기존 control head의 edge knockout 효과 검증~~ | **완료** (2026-07-28, P2-a/b/c 전부) |
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
