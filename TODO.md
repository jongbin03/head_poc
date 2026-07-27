# TODO

## 우선순위

| 순위 | 항목 | 비고 |
|---|---|---|
| P0 | 로컬 5070Ti 환경에서 파이프라인 전체 재현 | 아래 P1(7B)의 전제 조건 |
| P1 | Qwen2.5-7B(8B급)로 본 실험 확장 | 5070Ti에서 진행 |
| P2 | 외부/추가 데이터셋으로 기존 control head의 edge knockout 효과 검증 | held-out + 미지 문구 + 외부 벤치마크 |
| P3 | control head 내 internal-only vs external-only 채널 분기 검증 | 기존 채널 분기 실험 |

아래는 우선순위 순서대로 자세한 내용, 그 뒤에 보류 항목.

---

## P0. 로컬 5070Ti 환경에서 파이프라인 전체 재현

**배경**: 지금까지 결과는 전부 Colab 무료 T4에서 나왔음. 7B 이상으로 올리려면 Colab 무료
세션(세션 끊김, 디스크 휘발성, VRAM/시간 제약)보다 로컬 5070Ti(16GB, 상시 가동)가 유리함.
`RUN.md`에 로컬 실행 가이드는 이미 있지만 실제로 이 환경에서 실행 검증된 적은 없음
(README.md 문서에도 명시된 사실).

**할 일**:
1. `RUN.md` 0~2단계대로 pyenv 3.11.9 + cu128 PyTorch 환경 구성, `torch.cuda.is_available()`
   / `get_device_name(0)` 확인
2. Phase 0 스모크 테스트(0.5B, `--dataset_limit 2`)부터 돌려서 로컬 환경 자체가 Colab과
   동일하게 동작하는지 확인 (lxt/transformers 버전 조합도 로컬에서 다시 검증 — Colab에서
   쓴 `transformers==4.51.3` 고정이 로컬 cu128 PyTorch와도 호환되는지 체크)
3. 1.5B/3B 30개 템플릿 전체를 로컬에서 재실행해 Colab 결과(`results/2026-07-27_colab_phase1to3`)와
   수치가 재현되는지 대조 — 환경 차이로 인한 편차가 있는지 확인

## P1. Qwen2.5-7B(8B급)로 본 실험 확장 (5070Ti)

**배경**: RUN.md Phase 4 계획에 있던 단계. 1.5B/3B에서 확인한 패턴(read/control 분리,
internal∩external 공유, edge knockout으로 공격 억제+read 보존)이 7B로 스케일을 올려도
재현되는지가 다음 확인 대상. 5070Ti 16GB는 Colab 무료 T4보다 크고 안정적이라 이 스케일
검증에 적합.

**할 일**:
1. `--four_bit` 없이 bf16으로 먼저 시도 (7B bf16 weight만 ~14GB — backward 포함하면 VRAM
   빠듯할 수 있으니 `nvidia-smi -l 1`로 실측하며 진행, OOM 시 2번으로)
2. OOM이면 `--four_bit` 4bit 양자화 + checkpointing 끔 조합으로 head-level relevance
   추출 시도 (README.md §6 "알려진 함정" 참고 — checkpointing 켜면 embedding-level
   relevance만 가능해지므로 head relevance가 필요하면 반드시 checkpointing 끄고 VRAM 확인)
3. 최소한 `edge_ablation.py`의 knockout sweep(forward-only, 4bit/checkpointing 무관하게
   항상 안전)만이라도 반드시 수행 — 1.5B/3B에서 찾은 `control_heads_both`가 7B에서도
   knockout으로 먹히는지 검증 (head relevance를 7B에서 새로 뽑는 것보다 우선순위 높음)
4. 결과를 `results/YYYY-MM-DD_5070ti_7b/README.md`로 기록 (Colab 결과와 비교 표 포함)

## P2. 외부/추가 데이터셋으로 기존 control head의 edge knockout 효과 검증

**배경**: 지금까지 head를 찾은 데이터와 knockout 효과를 검증한 데이터가 완전히 동일함
(30개 템플릿 전체). "이 30개 조합에서만 통하는 head"일 위험이 있어, 우리가 찾은
control head가 **새로운/외부 데이터**에서도 여전히 먹히는지 확인이 필요.

### P2-a. Held-out 분리 (같은 데이터셋 안에서)
1. `dataset.sample_templates()`의 스타일 5종 중 4종(24개)으로 control head를 찾고,
   나머지 1종(6개)은 head 선정에서 완전히 배제했다가 knockout 검증에만 사용
2. `run_pipeline.py`에 head 선정용 템플릿과 평가용 템플릿을 분리하는 옵션 추가
3. held-out에서도 같은 패턴(공격 억제 + read 보존)이 재현되면 과적합 우려 해소

### P2-b. 미지의 공격 문구로 확장
1. `_INJECTION_STYLES`에 없는 새로운 스타일(다국어 혼용, 코드블록 위장, 유니코드
   난독화, 더 짧고 우회적인 표현 등) 추가 작성
2. 기존 5종으로 찾은 `control_heads_both`를 그대로 써서 새 스타일에 edge knockout 적용
3. 여전히 `malicious_token_prob`이 죽는지 확인 — 안 죽으면 "특정 문구 패턴에 과적합된
   head"라는 뜻이므로 방어 범위에 대한 중요한 negative result

### P2-c. 검증된 외부 IPI 벤치마크로 재현
**후보**: **InjecAgent**(tool-calling 에이전트 대상, 우리 `external` mode와 구조 가장
유사 — 우선 추천) / **BIPIA**(RAG/요약 시나리오, `internal` mode와 유사) / **AgentDojo**
(멀티스텝 에이전트, 스케일 검증용)

1. InjecAgent부터 시작 — 그 데이터셋 프롬프트에서 D_benign/D_inj span을 추출하는
   어댑터를 `dataset.py` 옆에 추가
2. 우리가 찾은 control head가 그 벤치마크의 공격 성공률(ASR)을 실제로 낮추는지 측정

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
