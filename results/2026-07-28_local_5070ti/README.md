# 2026-07-28 로컬(RTX 5070 Ti) 실행 결과

**실행 환경**: 로컬 Windows PC, `.venv` (Python), GPU `NVIDIA GeForce RTX 5070 Ti`,
`torch 2.11.0+cu128`, `transformers==4.51.3` + `lxt` 고정 (Colab과 동일 조합).

## 설치 이슈 (transformers 버전)

- venv에 처음 `transformers 5.14.1`이 깔려 있어 `lxt.efficient` import 시
  `ImportError: cannot import name 'find_pruneable_heads_and_indices'` 발생
  (transformers v5에서 legacy pruning 유틸 제거됨).
- `results/2026-07-27_colab_smoketest/README.md`에 기록된 대로
  `transformers==4.51.3`으로 다운그레이드하여 해결 (Colab과 동일한 결론 재확인:
  최신 버전은 `find_pruneable_heads_and_indices` 없음, `4.46.3`은 `qwen3` 모듈 없음 →
  `4.51.3`만 둘 다 만족).

네 번의 실행: 0.5B(smoke, dataset_limit=2) → 1.5B(전체 30개) → 3B(전체 30개) →
7B(4bit, 전체 30개).

---

## 실행 1: Qwen2.5-0.5B-Instruct, dataset_limit=2 (설치 확인용 smoke test)

```bash
python run_pipeline.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --family qwen2 --device cuda --topk 20 --dataset_limit 2
```

```
top-20 Jaccard(read, internal)   = 0.176
top-20 Jaccard(read, external)   = 0.290
top-20 Jaccard(internal,external)= 0.481
control_heads_both (internal ∩ external) = [(0,1),(1,8),(1,9),(15,2),(15,9),(15,12),
  (16,7),(20,12),(21,2),(21,3),(21,11),(22,5),(23,10)]

k=  0  malicious=0.9463  read=0.8204  (n=2)
k=  5  malicious=0.0682  read=0.7888  (n=2)
k= 10  malicious=0.0021  read=0.8132  (n=2)
k= 20  malicious=0.0000  read=0.8322  (n=2)
k= 40  malicious=0.0000  read=0.8124  (n=2)
k= 80  malicious=0.0000  read=0.8124  (n=2)
```

## 실행 2: Qwen2.5-1.5B-Instruct, 전체 30개 템플릿

```bash
python run_pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --family qwen2 --device cuda --topk 20
```

```
top-20 Jaccard(read, internal)   = 0.290
top-20 Jaccard(read, external)   = 0.333
top-20 Jaccard(internal,external)= 0.538
control_heads_both (internal ∩ external) = [(0,1),(0,3),(0,5),(0,6),(0,7),(0,10),
  (19,1),(21,9),(21,11),(23,1),(24,10),(24,11),(26,2),(27,3)]

k=  0  malicious=0.9118  read=0.5355  (n=30)
k=  5  malicious=0.0198  read=0.5193  (n=30)
k= 10  malicious=0.0000  read=0.5171  (n=30)
k= 20  malicious=0.0000  read=0.5110  (n=30)
k= 40  malicious=0.0000  read=0.5106  (n=30)
k= 80  malicious=0.0000  read=0.5106  (n=30)
```

## 실행 3: Qwen2.5-3B-Instruct, 전체 30개 템플릿

```bash
python run_pipeline.py --model Qwen/Qwen2.5-3B-Instruct \
  --family qwen2 --device cuda --topk 20
```

```
top-20 Jaccard(read, internal)   = 0.379
top-20 Jaccard(read, external)   = 0.333
top-20 Jaccard(internal,external)= 0.538
control_heads_both (internal ∩ external) = [(0,2),(0,4),(0,12),(0,13),(27,1),(29,1),
  (29,2),(29,3),(31,9),(32,1),(32,7),(33,7),(33,10),(33,15)]

k=  0  malicious=0.9995  read=0.6989  (n=30)
k=  5  malicious=0.7653  read=0.7039  (n=30)
k= 10  malicious=0.4562  read=0.6746  (n=30)
k= 20  malicious=0.0000  read=0.6788  (n=30)
k= 40  malicious=0.0000  read=0.7961  (n=30)
k= 80  malicious=0.0000  read=0.7961  (n=30)
```

## 실행 4: Qwen2.5-7B-Instruct (4bit), 전체 30개 템플릿

```bash
python run_pipeline.py --model Qwen/Qwen2.5-7B-Instruct \
  --family qwen2 --device cuda --four_bit --topk 20
```

```
top-20 Jaccard(read, internal)   = 0.290
top-20 Jaccard(read, external)   = 0.333
top-20 Jaccard(internal,external)= 0.600
control_heads_both (internal ∩ external) = [(0,3),(0,10),(0,15),(0,21),(19,15),(19,19),
  (22,3),(22,24),(23,10),(23,11),(24,21),(25,8),(25,12),(26,4),(27,21)]

k=  0  malicious=0.9690  read=0.8315  (n=30)
k=  5  malicious=0.1787  read=0.8541  (n=30)
k= 10  malicious=0.0001  read=0.8894  (n=30)
k= 20  malicious=0.0000  read=0.8934  (n=30)
k= 40  malicious=0.0000  read=0.9095  (n=30)
k= 80  malicious=0.0000  read=0.9095  (n=30)
```

---

## 결과 요약 표

| 모델 | 방식 | jaccard(read,internal) | jaccard(read,external) | jaccard(internal,external) | k=0 malicious | k=0 read | k=20 malicious | k=20 read |
|---|---|---|---|---|---|---|---|---|
| 0.5B (n=2) | fp | 0.176 | 0.290 | 0.481 | 0.9463 | 0.8204 | 0.0000 | 0.8322 |
| 1.5B | fp | 0.290 | 0.333 | 0.538 | 0.9118 | 0.5355 | 0.0000 | 0.5110 |
| 3B   | fp | 0.379 | 0.333 | 0.538 | 0.9995 | 0.6989 | 0.0000 | 0.6788 |
| 7B   | 4bit | 0.290 | 0.333 | 0.600 | 0.9690 | 0.8315 | 0.0000 | 0.8934 |

> Colab 결과(`2026-07-27_colab_phase1to3`, READ_PREFIX 수정 후 실행 4~5)와 비교하면
> 1.5B/3B 수치가 거의 동일 (1.5B: k=0 read 0.5405 vs 0.5355, 3B: 0.6957 vs 0.6989 —
> 오차 범위 내). 즉 이 로컬(5070Ti) 환경 재현이 Colab(T4) 결과와 일관됨.

## 해석

- **공격 억제**: 0.5B~7B 전 구간에서 `malicious_token_prob`이 k=0의 0.9~1.0에서
  k=10~20 이내에 0으로 붕괴 — 모델 크기와 무관하게 일관됨 (4bit 양자화한 7B도 동일 패턴).
- **정상 기능 보존**: `read_token_prob`은 k=0→k=20 구간에서 거의 유지되거나 소폭 변동
  (1.5B/3B/7B 모두), 오히려 7B는 k가 커질수록 소폭 상승(0.8315→0.8934) — Colab 3B
  재검증 결과와 같은 패턴.
- **internal/external control head 공유**: `jaccard(internal,external)`이 0.48~0.60
  범위로 모델 크기 전반에서 read 쪽 겹침보다 뚜렷하게 높음 — 크기가 커져도 "내부 오염과
  외부 tool-call 오염이 같은 head를 상당 부분 공유한다"는 결론이 0.5B~7B(4bit) 전체에서
  재현됨.
- **RTX 5070Ti 환경 재현성**: Colab(T4, fp16 1.5B/3B)과 로컬(5070Ti) 수치가 근접 —
  GPU/환경이 달라도 결론이 흔들리지 않음을 추가로 확인.

## 다음 액션

- `functional_map.png`는 실행마다 덮어써지므로(현재 루트에는 마지막 실행인 7B 것만 남음),
  필요하면 각 실행 직후 이 폴더로 복사해서 보존할 것.
- TODO.md의 "5070Ti repro" 항목은 이 결과로 완료로 볼 수 있음 (0.5B/1.5B/3B/7B 전부
  성공적으로 재현).
