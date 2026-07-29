---
name: atlas-poc-next-priorities
description: atlas_poc 다음 실험 우선순위(P3, P0~P2(a/b/c/d) 완료) — TODO.md 요약, 다음 세션 시작 시 바로 참고
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

## ~~P2 — 외부/추가 데이터셋으로 기존 control head의 edge knockout 효과 검증~~ — 완료 (2026-07-28, P2-a/b/c/d 전부)

지금까지 head를 찾은 데이터와 knockout 효과를 검증한 데이터가 완전히 동일(30개 템플릿
전체)했던 "이 30개에서만 통하는 head"일 위험을 검증:

- ~~**P2-a Held-out**~~ — 스타일 5종 중 4종으로 head를 찾고 나머지 1종은 평가에만 사용 —
  1.5B로 5종 전부, 7B(4bit)로 대표 1종 재확인, 전부 재현됨.
- ~~**P2-c 외부 IPI 벤치마크**~~ — InjecAgent 1,054개 실제 test case에 우리 데이터셋으로
  찾은 control_heads_both를 그대로(재선정 없이) 적용 — knockout 효과가 두 스케일
  (1.5B/7B)에서 일관되게 재현(malicious 4~5배 감소, read 상승)되지만 완전한 억제는 아님.
- ~~**P2-b 미지의 공격 문구**~~ — P2-c가 부분적 전이로 나와서 "문구 때문인지 도메인 구조
  때문인지" 분리하려고 재개. 도메인은 그대로 두고 완전히 새로운 문구 4종(다국어 혼용/
  코드블록 위장/유니코드 난독화/짧고 우회적인 표현)으로 knockout 적용 — 두 스케일 모두
  기존 5종과 **동일하게 k=10 안에 완전히 붕괴**. 즉 P2-c의 잔여 효과는 문구가 아니라
  **도메인/데이터 구조 차이** 때문이라는 결론.
- ~~**P2-d InjecAgent 자체 head 탐색**~~ — 사용자 제안: InjecAgent 60개로 직접 control head를
  찾아 우리 합성 `control_heads_both`와 교집합한 뒤 나머지 994개로 (a) 합성 단독(14개)
  (b) InjecAgent 단독(20개) (c) 교집합(9개) 3가지를 비교. jaccard(두 head 집합)=0.36으로
  일부만 겹치는데도 **셋 다 거의 동일한 성능**(억제 후 malicious ~0.044~0.045) — 교집합이
  가장 적은 개입(9개)으로 동등한 효과를 내는 것과, head 선정 방법을 뭘 바꿔도 P2-c와 같은
  수준에서 멈추는 것 둘 다 "P2-c 잔여 효과는 head를 잘못 골라서가 아니라 도메인 구조 자체의
  한계"라는 결론을 강화함. **작업 중 `compute_head_relevance`의 심각한 GPU 메모리 누수
  버그를 발견함**(모델 재로드로도 전혀 안 없어짐, 프로세스 전역 문제로 추정, 미해결) —
  자세한 내용과 우회책/근본 해결책은 [[atlas-poc-summary]] 버그 4번 및 TODO.md "보류" 섹션
  참고. 결과: `results/2026-07-28_Qwen-Qwen2-5-1-5B-Instruct_headsplit/summary.txt`.

자세한 수치는 [[atlas-poc-summary]] 참고.

## P3 — control head 내 internal-only vs external-only 채널 분기 검증 (다음 작업)

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
