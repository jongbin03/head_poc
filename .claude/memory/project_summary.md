---
name: atlas-poc-summary
description: IPI read/control head 분리 PoC — 지금까지 진행 상황, 발견한 버그, 실험 결과 요약
metadata:
  type: project
---

# atlas_poc 프로젝트 요약 (2026-07-27 기준)

## 프로젝트 목표

Atlas of In-Context Learning 논문의 head 분리 방법론을 응용해, 외부 데이터(RAG/tool 결과)에
대해 (1) 내용을 "읽는" attention head와 (2) 그 데이터 속 명령을 "따라 실행"하는(내부 응답 오염
또는 외부 tool-call 오염) attention head가 서로 분리되는지, 그리고 후자만 골라서 무력화해도
전자(정상 기능)는 보존되는지 검증하는 IPI(Indirect Prompt Injection) 방어 PoC.

GitHub: `jongbin03/head_poc` (origin, `atlas_poc/` 폴더가 로컬 프로젝트 루트).

## 파일 구성

- `dataset.py` — synthetic IPI 프롬프트 생성 (콘텐츠 도메인 6종 × 공격 문구 스타일 5종 = 30
  템플릿, 모드 4종: read_clean/read_injected/internal/external)
- `attn_relevance.py` — lxt(AttnLRP) 기반 layer×head relevance 산출 (attention × gradient)
- `head_ranking.py` — top-K head 랭킹 + read/internal/external 간 Jaccard overlap 계산
- `edge_ablation.py` — **head를 끄는 게 아니라, 특정 head가 D_inj(주입된 명령) 토큰을 보는
  attention edge만 pre-softmax `-inf`로 차단**하는 edge knockout + sweep
- `run_pipeline.py` — 위 네 개를 엮은 진입점 (`[1/4]`~`[4/4]`)
- `debug_read_target.py` — read_target 토큰이 실제로 모델 응답과 맞는지 top-k logit +
  greedy continuation으로 직접 확인하는 디버깅 스크립트
- `head_poc.ipynb` — Colab 실행 순서를 정리한 노트북 (VS Code의 Colab GPU 연결용)
- `RUN.md` — 로컬 5070Ti 기준 실행 가이드
- `README.md` — **설계 노트가 아니라 실제 실행 중인 방법론 설명 문서** (커밋 `8b1fbeb`로
  재작성됨: head를 찾는 AttnLRP 방법, head 랭킹/겹침 분석, edge 차단 구현 원리, 데이터셋
  설계, 알려진 함정을 코드 그대로 설명)
- `TODO.md` — 다음 할 일, **우선순위 P0~P3 순으로 정리됨** (아래 "다음 할 일" 및
  `next_priorities.md` 참고)
- `head_poc_presentation.pptx` — 내부 발표용 슬라이드(19장, python-pptx로 생성). **로컬
  파일로만 존재, git에는 커밋 안 함** (사용자가 "그건 필요 없다"고 명시). 같은 내용을
  claude.ai 아티팩트(HTML 슬라이드)로도 만들어 링크 공유했었음 — 세션이 달라지면 그 URL은
  못 찾으니, 다시 필요하면 이 프로젝트 요약을 근거로 재생성할 것.

## 실행 환경 관련 확정 사항

- **VS Code ↔ Colab 연결은 커널(GPU)만 원격**이고, 로컬 파일과 Colab 런타임 디스크는 별개
  파일시스템. 로컬에서 코드 수정 후 반드시 **GitHub push → Colab에서 git pull/clone** 필요.
- Colab 기본 `transformers`는 `lxt`가 요구하는 구버전 API(`find_pruneable_heads_and_indices`)와
  최신 아키텍처 지원(`qwen3` 모듈 필요) 사이에서 충돌 → **`transformers==4.51.3`으로 고정**
  (그보다 최신/구버전 둘 다 에러). pip 설치 후 런타임 재시작 필수.

## 발견한 버그와 수정 이력

1. **3B에서 `read_token_prob`이 비정상적으로 낮게 측정된 버그** (커밋 `bc0acdc`)
   - 원인: `READ_PREFIX="The answer is"` 뒤에 3B 모델이 곧바로 정답을 안 쓰고
     `"The answer is that ~~~ 3pm"`처럼 `"that"`으로 우회 응답하는 경우가 대부분이라,
     즉시-다음-토큰 확률(`read_target`)이 노이즈 수준으로 낮게 측정됨. 실제 발화 내용은
     정확했음 (`greedy continuation`으로 확인) — read 실패가 아니라 측정 지점 문제.
   - 수정: `READ_PREFIX` → `"Answer:"`, system prompt에
     `"Respond with only the requested value, no extra words."` 추가 (read 모드 전용).
   - 검증 결과: 3B의 `read_token_prob` k=0이 0.0067 → 0.6957로 100배 이상 상승, 버그였음이
     확인됨. 1.5B도 0.4371 → 0.5405로 동반 상승.

2. **`debug_read_target.py`의 attention_mask 자동추론 버그** (커밋 `315f964`)
   - 원인: `model.generate()`에 attention_mask를 안 넘기면 HF가 "input_ids 중
     pad_token_id와 같은 값은 패딩"이라고 자동 추정하는데, 이 모델은 pad_token_id ==
     eos_token_id라서 프롬프트 안의 `<|im_end|>` 토큰(정상적으로 여러 번 등장)이 진짜
     내용인데도 마스킹됨. `model()` 직접 호출(마스크 안 넘김 = 마스킹 없음)과 결과가
     달라지는 원인이었음.
   - 수정: `attention_mask=torch.ones_like(input_ids)`를 `model()`과 `generate()` 양쪽에
     명시적으로 전달. `pad_token_id=tok.eos_token_id`도 명시.
   - **영향 범위**: `run_pipeline.py`/`attn_relevance.py`/`edge_ablation.py`는 `generate()`를
     안 쓰므로 이 버그와 무관 — 본 실험 수치는 안전함.

3. **검토했지만 기각한 "버그"**: 숫자로 시작하는 `read_answer`(`" 3pm"`, `" 1.1"`)가
   BPE에서 `[공백 토큰, "3", "pm"]`처럼 앞 공백이 분리되는 현상. 처음엔 이걸 버그로 보고
   "공백 토큰을 건너뛰고 다음 실제 토큰을 target으로" 고치려 했으나, 오히려 잘못된 방향임을
   깨닫고 되돌림 — "공백 토큰이 나오는가 vs `" The"`가 나오는가"의 경쟁이 바로 "숫자로 바로
   답하는가 vs 말을 돌려 답하는가"를 가르는 진짜 분기점이라 원래 방식이 맞았음.

## 용어 정정 (중요)

**"control head를 knockout한다"는 표현은 부정확함.** 정확히는 head 자체를 끄는 게 아니라,
**그 head가 D_inj(주입된 명령 텍스트) 위치를 보는 attention edge만** pre-softmax에서 차단
(edge knockout). head는 다른 위치(D_benign, 질문 등)에 대해서는 여전히 정상 작동함. 이게
"head 하나를 통째로 죽여도 정상 기능이 보존되더라"보다 더 강한 결과인 이유 — 훨씬 좁은
개입으로도 공격이 죽는다는 뜻.

## 실험 결과 (2026-07-27, Colab T4, 최종/수정된 수치)

| 모델 | 템플릿 | jaccard(internal,external) | k=0 malicious | k=20 malicious | k=0 read | k=20 read |
|---|---|---|---|---|---|---|
| 0.5B (smoke) | 2  | 0.481 | 0.9386 | 0.0000 | 0.6071 | 0.6055 |
| 1.5B | 30 (수정 후) | 0.538 | 0.9094 | 0.0000 | 0.5405 | 0.5075 |
| 3B   | 30 (수정 후) | 0.538 | 0.9998 | 0.0000 | 0.6957 | 0.6867 |

**결론**: 공격 억제(`malicious_token_prob`이 k=20 안에 0으로 붕괴)는 모든 스케일에서 견고.
정상 기능 보존(`read_token_prob`이 edge knockout 후에도 거의 유지/상승)도 버그 수정 후
1.5B/3B 모두에서 확인됨. `jaccard(internal,external)`이 read 쪽 겹침보다 훨씬 높아, 내부
응답 오염과 외부 tool-call 오염이 상당 부분 같은 control head 회로를 공유한다는 idea1의
전제를 지지.

자세한 원본 로그: `results/2026-07-27_colab_smoketest/README.md`,
`results/2026-07-27_colab_phase1to3/README.md` 참고.

## 다음 할 일 — 우선순위 P0~P3 (자세한 내용은 `next_priorities.md` 및 `TODO.md` 참고)

1. **P0**: 로컬 5070Ti 환경에서 지금까지의 파이프라인(Phase 0~3)을 재현 — 7B 실험의 전제 조건
2. **P1**: Qwen2.5-7B(8B급)로 본 실험 확장, 5070Ti에서 진행
3. **P2**: 외부/추가 데이터셋으로 기존 `control_heads_both`의 edge knockout 효과 검증
   (held-out 분리 / 미지의 공격 문구 / InjecAgent 등 외부 IPI 벤치마크)
4. **P3**: control head 내 internal-only vs external-only 채널 분기 검증 (기존 "채널 분기"
   실험 — `external_heads - internal_heads` / `internal_heads - external_heads` 대칭차)

부작용(collateral damage) 측정, Llama 계열 교차검증, path patching, 실전 배포 전환은
`TODO.md`의 "보류" 섹션으로 미뤄짐.

## 협업 방식 관련 메모

- 사용자는 Colab 무료 T4로 실험을 돌리고, 결과를 대화에 붙여넣으면 원인 분석/코드 수정을
  요청하는 방식으로 작업함. VS Code에서 Colab 확장으로 GPU만 연결해 쓰는 중.
- 실험 결과가 나올 때마다 `results/YYYY-MM-DD_설명/README.md` 형태로 원본 로그 + 해석 +
  체크리스트를 기록하는 패턴을 확립함 (앞으로도 이 패턴 유지).
- 코드/설계에 대해 틀렸을 수 있는 부분은 사용자가 직접 짚어서 정정을 요구하는 경우가 많음
  (예: "control head 찾는 방법 설명해줘", "head knockout이 아니라 edge knockout 아니야?").
  설명할 때 용어를 정확히 쓰는 것을 중요하게 여김.
- 내부 발표자료(슬라이드)를 요청할 때는 claude.ai 아티팩트(HTML)로 먼저 구상/시안을 보여준
  뒤, 실제 파일 형식(pptx 등)이 필요하면 별도로 요청하는 패턴. pptx는 git에 커밋하지 않음
  (바이너리 산출물은 로컬 전용으로 취급, 코드/문서만 커밋).
- 매 세션 끝에 `.claude/memory/project_summary.md`(및 관련 파일)를 최신 상태로 갱신해달라고
  명시적으로 요청하는 경우가 있음 — 요청 없이도 큰 진행 단계(버그 수정, 실험 완료, 우선순위
  변경)마다 최신화해두는 게 좋음.
