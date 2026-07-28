---
name: atlas-poc-code-organization
description: 실험 코드/결과물을 phase별로 어떻게 나누는지에 대한 결정 (디렉토리 복제 X, git 태그 + results/ 폴더)
metadata:
  type: project
---

# 코드/결과 조직 방식 (2026-07-28 결정)

## 결정

**코드는 복제하지 않고 루트에 단일 코드베이스로 유지한다.** phase(P1/P2/P3...) 구분은
코드 디렉토리가 아니라 다음 두 가지로 한다:

1. **결과물**: `run_pipeline.py` 실행마다 `results/<날짜>_<모델명>[_4bit]/`가 자동 생성되어
   `functional_map.png` + `summary.txt`가 그 안에 저장됨 (덮어쓰기 방지, `--out_dir`로
   수동 지정도 가능). 커밋 `<이 결정을 적용한 커밋>`에서 `run_pipeline.py`에 구현.
2. **코드 히스토리**: phase가 완료될 때마다 `git tag`로 표시 (예: `p1-local-5070ti-repro`).
   나중에 "그 시점 코드가 정확히 뭐였는지" 필요하면 `git checkout <tag>`로 복원.
   각 `results/.../README.md`에도 실행 시점 커밋 해시를 기록하는 관례를 유지 (이미
   `bc0acdc`, `315f964` 등으로 해오던 패턴을 필수 항목으로 정착).

## 왜 디렉토리 복제를 안 하는지

`dataset.py`/`head_ranking.py` 같은 핵심 모듈이 phase마다 **그 자체로 계속 진화**한다
(예: P2-a는 `dataset.py`에 held-out split 옵션 추가, P3는 `head_ranking.py`에
`external_only`/`internal_only` 계산 추가). phase 디렉토리마다 코드를 복제해두면:
- 다음 phase에서 공유 모듈을 고칠 때 어느 사본을 고쳐야 하는지 헷갈림
- 버그 수정을 여러 사본에 반복 적용해야 해서 금방 어긋남(drift)

git 히스토리가 원래 이 역할(특정 시점의 코드 스냅샷)을 하도록 만들어졌으므로, 그걸 쓰는 게
더 안전함.

## phase 전용 신규 파일은 새 파일로 분리

P2/P3에서 추가하는 코드 중 재사용 없이 그 phase 전용인 것(예: P2-c의 InjecAgent 어댑터)은
`adapters/injecagent.py`처럼 새 파일로 분리 — 자연스럽게 "그 phase에서 추가된 것"으로
구분됨. 공유 모듈 수정과는 별개로 다룸.

## 관련 메모

[[atlas-poc-summary]] — 프로젝트 전체 배경.
[[atlas-poc-next-priorities]] — P2/P3 작업 내용 (이 파일들을 실제로 만들면서 위 원칙 적용).
