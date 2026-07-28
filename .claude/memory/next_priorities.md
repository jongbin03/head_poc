---
name: atlas-poc-next-priorities
description: atlas_poc 다음 실험 우선순위(P2~P3, P0/P1 완료) — TODO.md 요약, 다음 세션 시작 시 바로 참고
metadata:
  type: project
---

# 다음 할 일 (우선순위 순, 2026-07-28 갱신)

전체 상세 내용은 프로젝트 루트 `TODO.md` 참고. 이 파일은 다음 세션에서 "지금 뭐부터 하면
되지?"를 바로 답하기 위한 요약.

## ~~P0/P1 — 로컬 5070Ti 재현 + 7B 확장~~ — 완료 (2026-07-28)

로컬 5070Ti에서 `transformers==4.51.3`으로 lxt 호환성 이슈 해결 후 0.5B/1.5B/3B/7B(4bit)
전부 실행 완료. Colab 결과와 수치 근접(재현성 확인), 7B(4bit)도 같은 패턴(공격 억제 +
read 보존) 재현됨. 결과: `results/2026-07-28_local_5070ti/README.md`,
요약: [[atlas-poc-summary]]. **다음 우선순위는 P2부터.**

## P2 — 외부/추가 데이터셋으로 기존 control head의 edge knockout 효과 검증

지금까지 head를 찾은 데이터와 knockout 효과를 검증한 데이터가 완전히 동일(30개 템플릿
전체) — "이 30개에서만 통하는 head"일 위험. 세 갈래로 검증:

- ~~**P2-a Held-out**~~ — **완료 (2026-07-28)**. 스타일 5종 중 4종으로 head를 찾고
  나머지 1종은 평가에만 사용 — 1.5B로 5종 전부, 7B(4bit)로 대표 1종 재확인, 전부 재현됨.
  자세한 내용: [[atlas-poc-summary]].
- ~~**P2-c 외부 IPI 벤치마크**~~ — **완료 (2026-07-28)**. InjecAgent 1,054개 실제 test
  case에 우리 데이터셋으로 찾은 control_heads_both를 그대로(재선정 없이) 적용 —
  knockout 효과가 두 스케일(1.5B/7B)에서 일관되게 재현(malicious 4~5배 감소, read 상승)
  되지만 완전한 억제는 아님. 자세한 수치: [[atlas-poc-summary]].
- **P2-b 미지의 공격 문구 — 재개 검토 (다음 작업)**: 원래 "P2-c 성공하면 불필요"로
  보류했으나 P2-c가 부분적 전이로 나와서, 잔여 공격 확률이 문구 때문인지 도메인 구조
  때문인지 분리하는 통제 실험으로 다시 필요해짐. TODO.md P2-b 섹션 참고.

## P3 — control head 내 internal-only vs external-only 채널 분기 검증 (P2-b 다음)

`internal_heads ∩ external_heads`(공통 control head)만 봤지, "명령을 따르되 자유 텍스트로
낼지 tool-call로 낼지를 가르는 head"(교집합이 아니라 **대칭차**)는 아직 안 봄.

- `head_ranking.py`에 `external_only`/`internal_only` 집합 계산 추가
- `edge_ablation.py`로 각각 knockout해서 채널별 분리 효과 검증

## 보류 (지금 우선순위 아님)

부작용(collateral damage: 일반 QA/MMLU 능력 훼손 여부) 측정, Llama 계열 교차검증(코드는
`--family llama` 지원, 실행만 안 함), path patching(개별 head 인과관계 정밀 검증), 실전
배포 형태 전환(`edge_knockout()`을 상시 forward hook/서빙 프레임워크로) — 자세한 배경은
`TODO.md` 하단 "보류" 섹션 참고.

## 관련 메모

[[atlas-poc-summary]] — 프로젝트 전체 배경, 발견한 버그, 실험 결과, 협업 방식 메모.
[[atlas-poc-feedback-language]] — 이 프로젝트에서는 항상 한국어로 응답.
