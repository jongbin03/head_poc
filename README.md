# IPI Read/Control Head 분리 PoC

**질문**: 외부 데이터(툴 결과/RAG retrieval)를 다룰 때, 모델 안에서 (1) 그 데이터의
**내용을 읽는 데** 관여하는 attention head와 (2) 그 데이터 속 **명령을 실행에 옮기는 데**
관여하는 attention head가 서로 분리되는가? 분리된다면, (2)만 골라 무력화해도 (1)은
보존되는가?

Atlas of In-Context Learning(NeurIPS'25)의 head 분리 방법론(AttnLRP 기반 relevance)을
IPI(Indirect Prompt Injection) 방어 시나리오에 응용한 PoC.

```
dataset.py  ──▶  attn_relevance.py  ──▶  head_ranking.py  ──▶  edge_ablation.py
(프롬프트 생성)   (head별 relevance 산출)  (top-K 랭킹 + 겹침)   (control head의
                                                              D_inj-edge만 차단)
```

`run_pipeline.py`가 위 네 단계를 순서대로 실행한다 (`[1/4]`~`[4/4]`).

## 문서

| 문서 | 내용 |
|---|---|
| [docs/methodology.md](docs/methodology.md) | 실제로 구현되어 돌아가는 방법론 상세 |
| [docs/run-guide.md](docs/run-guide.md) | 실행 가이드 (로컬 5070Ti / Colab) |
| [docs/review-2026-07-29.md](docs/review-2026-07-29.md) | **방법론 자체 검토** — 현재 결과의 교란 요인과 권장 작업 순서 |
| [docs/feedback-2026-07-29.md](docs/feedback-2026-07-29.md) | 교수님 피드백 대응 계획 |
| [docs/feedback-response-2026-07-31.md](docs/feedback-response-2026-07-31.md) | **교수님 피드백 대응 결과 정리** — 데이터셋/head 탐색/AgentDojo 평가 결과와 한계 |
| [docs/todo.md](docs/todo.md) | 작업 이력(P0~P2 완료) + 다음 할 일(P3~P6) |
| [docs/presentation-notes.md](docs/presentation-notes.md) | 발표 준비용 정리 (개요→방법론→결과) |
| [docs/presentation/](docs/presentation/) | 발표 슬라이드 및 스크립트 |
| [results/](results/) | 날짜별 실행 결과 원본 로그 |

> ⚠️ `docs/presentation-notes.md`와 `results/`의 일부 결론은
> [docs/review-2026-07-29.md](docs/review-2026-07-29.md)에서 지적한 교란 요인
> (합성 데이터의 content-availability, InjecAgent utility 지표의 비독립성) 검토 전에
> 작성된 것이다. 인용 전에 리뷰 문서를 먼저 확인할 것.

## 코드 구성

- `dataset.py` — synthetic IPI 템플릿/프롬프트 생성
- `attn_relevance.py` — lxt monkey-patch + AttnLRP 기반 head relevance 산출
- `head_ranking.py` — top-K head 랭킹 + Jaccard 겹침 분석 + functional map 시각화
- `edge_ablation.py` — control head의 D_inj-edge 차단 + knockout sweep
- `run_pipeline.py` — 위 네 개를 엮은 진입점
- `adapters/injecagent.py` — 외부 벤치마크(InjecAgent) 어댑터
- `debug_read_target.py` — read_target 토큰이 실제 모델 응답과 맞는지 확인하는 디버깅 스크립트
- `head_poc.ipynb` — Colab 실행 노트북

## 빠른 실행

```bash
python run_pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct --family qwen2 --topk 20
```

자세한 옵션과 환경 세팅은 [docs/run-guide.md](docs/run-guide.md) 참고.
