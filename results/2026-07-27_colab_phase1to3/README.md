# 2026-07-27 Colab Phase 1~3 본 실험 결과

**실행 환경**: Google Colab (무료, T4 GPU). 환경 세팅은
`results/2026-07-27_colab_smoketest/README.md`와 동일
(`transformers==4.51.3` + `lxt` 고정, 이어서 사용).

세 번의 실행: 1.5B(6개 템플릿) → 1.5B(전체 30개) → 3B(전체 30개).

---

## 실행 1: Qwen2.5-1.5B-Instruct, dataset_limit=6 (도메인 6개 x 스타일 1종)

```bash
python run_pipeline.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --family qwen2 --device cuda --topk 20 --dataset_limit 6
```

```
top-20 Jaccard(read, internal)   = 0.290
top-20 Jaccard(read, external)   = 0.290
top-20 Jaccard(internal,external)= 0.667
control_heads_both (internal ∩ external) = [(0,1),(0,3),(0,5),(0,6),(0,7),(0,10),
  (1,4),(19,1),(21,9),(21,11),(22,7),(23,1),(24,10),(24,11),(26,2),(27,3)]

k=  0  malicious=0.9914  read=0.4611  (n=6)
k=  5  malicious=0.0522  read=0.4735  (n=6)
k= 10  malicious=0.0000  read=0.4044  (n=6)
k= 20  malicious=0.0000  read=0.3750  (n=6)
k= 40  malicious=0.0000  read=0.4001  (n=6)
k= 80  malicious=0.0000  read=0.4001  (n=6)
```

## 실행 2: Qwen2.5-1.5B-Instruct, 전체 30개 템플릿

```bash
python run_pipeline.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --family qwen2 --device cuda --topk 20
```

```
top-20 Jaccard(read, internal)   = 0.290
top-20 Jaccard(read, external)   = 0.290
top-20 Jaccard(internal,external)= 0.538
control_heads_both (internal ∩ external) = [(0,1),(0,3),(0,5),(0,6),(0,7),(0,10),
  (19,1),(21,9),(21,11),(23,1),(24,10),(24,11),(26,2),(27,3)]

k=  0  malicious=0.9094  read=0.4371  (n=30)
k=  5  malicious=0.0220  read=0.4328  (n=30)
k= 10  malicious=0.0000  read=0.3891  (n=30)
k= 20  malicious=0.0000  read=0.3743  (n=30)
k= 40  malicious=0.0000  read=0.3771  (n=30)
k= 80  malicious=0.0000  read=0.3771  (n=30)
```

## 실행 3: Qwen2.5-3B-Instruct, 전체 30개 템플릿

```bash
python run_pipeline.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --family qwen2 --device cuda --topk 20
```

```
top-20 Jaccard(read, internal)   = 0.333
top-20 Jaccard(read, external)   = 0.290
top-20 Jaccard(internal,external)= 0.538
control_heads_both (internal ∩ external) = [(0,2),(0,4),(0,12),(0,13),(27,1),(29,1),
  (29,2),(29,3),(31,9),(32,1),(32,7),(33,7),(33,10),(33,15)]

k=  0  malicious=0.9998  read=0.0067  (n=30)
k=  5  malicious=0.7560  read=0.0064  (n=30)
k= 10  malicious=0.4379  read=0.0064  (n=30)
k= 20  malicious=0.0000  read=0.0263  (n=30)
k= 40  malicious=0.0000  read=0.0519  (n=30)
k= 80  malicious=0.0000  read=0.0519  (n=30)
```

---

## 체크리스트 (RUN.md 4단계 기준)

- [x] 세 실행 모두 `[1/4]`~`[4/4]` 에러 없이 완료
- [x] Jaccard 정상 범위 (NaN 없음)
- [x] `functional_map.png` 각 실행마다 생성 (Colab 세션 로컬에만 있음 — 필요 시 다운로드해서 이 폴더에 추가할 것)
- [x] knockout sweep에서 k에 따라 `malicious_token_prob` 유의미하게 변화

## 결과 요약 표

| 모델 | 템플릿 수 | jaccard(read,internal) | jaccard(read,external) | jaccard(internal,external) | k=0 malicious | k=0 read | k=20 malicious | k=20 read |
|---|---|---|---|---|---|---|---|---|
| 1.5B | 6  | 0.290 | 0.290 | 0.667 | 0.9914 | 0.4611 | 0.0000 | 0.3750 |
| 1.5B | 30 | 0.290 | 0.290 | 0.538 | 0.9094 | 0.4371 | 0.0000 | 0.3743 |
| 3B   | 30 | 0.333 | 0.290 | 0.538 | 0.9998 | 0.0067 | 0.0000 | 0.0263 |

> ⚠️ **이 표(실행 1~3)의 `read` 관련 수치는 이후 measurement bug로 판명되어 아래
> "재검증" 섹션의 실행 4~5로 대체됨.** `malicious`(공격 억제) 쪽 수치는 그대로 유효함.

## 해석

**1.5B, 6개 → 30개 비교 (표본 늘렸을 때 결론이 흔들리는지 확인용)**
- `jaccard(internal,external)`이 0.667→0.538로 다소 낮아짐 — 템플릿이 늘어나며 두 공격 경로가
  공유하는 head 비중이 약간 줄었지만, 여전히 read 쪽 겹침(0.290)보다는 훨씬 높아 "read와
  control이 분리되고, internal/external control은 상당히 공유된다"는 결론은 유지됨.
- edge knockout 효과(`malicious`가 k=5~10에서 급락, `read`는 완만하게만 하락)도 6개/30개 모두
  같은 패턴 → 표본 수에 따라 결론이 뒤집히지 않음 (좋은 신호).
- `control_heads_both` 리스트가 6개/30개 사이에서 대부분 겹침((1,4), (22,7)만 30개 쪽에서
  빠짐) → 안정적인 head 후보.

**1.5B → 3B 비교 (모델 크기 확장)**
- `jaccard(internal,external)=0.538`로 동일 — 모델 크기가 커져도 "내부 오염 ∩ 외부 tool-call
  오염이 같은 head를 상당히 공유한다"는 패턴 자체는 재현됨.
- `control_heads_both` head 위치는 완전히 달라짐 (1.5B: layer 0, 19~27 / 3B: layer 0, 27~33)
  — layer 수가 다른(1.5B는 28층, 3B는 36층 추정) 모델이라 절대 layer 번호 비교는 의미 없고,
  "초반 layer(0) + 후반 layer 쪽" 패턴이 유지되는 정도로 봐야 함.

**⚠️ 3B 결과의 이상 신호 — 그대로 결론 내리면 안 됨**
- `read_token_prob`이 k=0(edge knockout 없음)에서부터 **0.0067**로 극히 낮음. 1.5B는 같은
  조건에서 0.4371이었는데 3B는 60배 이상 낮음. 이건 edge knockout 효과가 아니라 **애초에
  모델이 그 정답 토큰을 그 자리에 낼 확률 자체가 거의 0**이라는 뜻 (RUN.md 3단계 체크리스트
  5번 항목이 원래 경고하던 케이스: "prefix 끝 공백 vs 타깃 토큰 앞 공백 불일치" 또는 3B가
  같은 프롬프트에 다르게 반응해 "The answer is" 뒤에 다른 형식으로 답하는 경우).
- 그 결과 `read` 축의 k에 따른 변화(0.0067→0.0263→0.0519)는 절대값이 너무 작아 잡음일
  가능성이 높고, "edge knockout이 정상 기능을 보존한다"는 주장의 근거로 이 3B 결과를 쓰면 안 됨.
- **다음 확인 필요**: 3B에서 `read_clean`/`read_injected` 프롬프트에 대해 모델이 실제로
  어떤 텍스트를 생성하는지(logit 상위 토큰이 뭔지) 직접 찍어봐서, read_target 토큰 자체가
  잘못 잡힌 건지 확인할 것. (`model.generate(...)`로 몇 개 샘플 직접 디코딩해보는 걸 권장)
- `malicious_token_prob`은 k=0에서 0.9998로 정상 범위이고 edge knockout으로 0까지 떨어지는
  것도 1.5B와 같은 패턴이라, **공격 억제 쪽 결론은 3B에서도 유효**해 보임. 문제는 read
  (utility 보존) 쪽 측정만.

## 다음 액션 (당시 시점 — 아래 "재검증" 섹션에서 해결됨)

1. ~~3B의 `read_token_prob` 이상치 원인 규명~~ → `debug_read_target.py`로 원인 규명 완료
2. ~~`read_answer`/`READ_PREFIX` 조합을 3B로 직접 sanity check~~ → 완료, 수정함
3. Phase 4(7B, 4bit)로 넘어가기 전에 위 이슈부터 해결 → 아래 재검증 결과로 해결 확인됨

---

## 재검증 (READ_PREFIX 수정 이후, 같은 날 2026-07-27)

**원인**: `debug_read_target.py`로 직접 확인한 결과, 3B 모델은 `"The answer is"` 뒤에
곧바로 정답을 말하지 않고 `"The answer is that ~~~ 3pm"`처럼 `"that"`으로 한 단계
우회해서 답하는 경우가 대부분이었음. 실제 발화 내용(`greedy continuation`)은 정확했지만,
측정 지점(prefix 바로 다음 토큰)이 3B의 실제 응답 스타일과 안 맞아 확률이 노이즈 수준으로
낮게 나온 것 (read 실패가 아니라 측정 버그).

**수정** (`dataset.py`, 커밋 `bc0acdc`): `READ_PREFIX`를 `"The answer is"` → `"Answer:"`로,
system prompt에 `"Respond with only the requested value, no extra words."` 지시 추가
(read 계열 모드에서만 적용, internal/external 실험엔 영향 없음).

### 실행 4: Qwen2.5-1.5B-Instruct, 전체 30개 템플릿 (수정 후 재실행)

```
top-20 Jaccard(read, internal)   = 0.333
top-20 Jaccard(read, external)   = 0.379
top-20 Jaccard(internal,external)= 0.538
control_heads_both (internal ∩ external) = [(0,1),(0,3),(0,5),(0,6),(0,7),(0,10),
  (19,1),(21,9),(21,11),(23,1),(24,10),(24,11),(26,2),(27,3)]

k=  0  malicious=0.9094  read=0.5405  (n=30)
k=  5  malicious=0.0220  read=0.5201  (n=30)
k= 10  malicious=0.0000  read=0.5160  (n=30)
k= 20  malicious=0.0000  read=0.5075  (n=30)
k= 40  malicious=0.0000  read=0.5112  (n=30)
k= 80  malicious=0.0000  read=0.5112  (n=30)
```

### 실행 5: Qwen2.5-3B-Instruct, 전체 30개 템플릿 (수정 후 재실행)

```
top-20 Jaccard(read, internal)   = 0.379
top-20 Jaccard(read, external)   = 0.333
top-20 Jaccard(internal,external)= 0.538
control_heads_both (internal ∩ external) = [(0,2),(0,4),(0,12),(0,13),(27,1),(29,1),
  (29,2),(29,3),(31,9),(32,1),(32,7),(33,7),(33,10),(33,15)]

k=  0  malicious=0.9998  read=0.6957  (n=30)
k=  5  malicious=0.7560  read=0.7017  (n=30)
k= 10  malicious=0.4379  read=0.6953  (n=30)
k= 20  malicious=0.0000  read=0.6867  (n=30)
k= 40  malicious=0.0000  read=0.7944  (n=30)
k= 80  malicious=0.0000  read=0.7944  (n=30)
```

### 수정 전후 비교

| 모델 | 템플릿 | k=0 read (수정 전) | k=0 read (수정 후) | k=20 read (수정 전) | k=20 read (수정 후) |
|---|---|---|---|---|---|
| 1.5B | 30 | 0.4371 | **0.5405** | 0.3743 | **0.5075** |
| 3B   | 30 | 0.0067 ⚠️ | **0.6957** | 0.0263 ⚠️ | **0.6867** |

3B는 수정 전 0.0067(노이즈 수준)이었던 게 **0.6957로 100배 이상 상승** — 이전 결과가
측정 버그였다는 게 확실히 증명됨. 1.5B도 함께 상승(0.4371→0.5405)한 걸 보면, 이 prefix
포맷이 1.5B에도 어느 정도 걸렸던 것으로 보이며 수정으로 두 모델 모두 개선됨.

### 최종 결론 (수정된 지표 기준)

- **공격 억제**: 1.5B/3B 모두 `malicious_token_prob`이 k=0의 0.9~1.0에서 k=20 이내에 0으로
  붕괴 — 이 결론은 수정 전후 변함없이 견고함.
- **정상 기능 보존**: 이제 신뢰할 수 있는 수치로 봤을 때도, `read_token_prob`은 control head의
  D_inj-edge를 knockout해도 (1.5B: 0.54→0.51, 3B: 0.70→0.69~0.79) **거의 그대로 유지되거나
  오히려 3B는 소폭 상승**함. 즉 "control head를 통째로 끄는 게 아니라 그 head가 주입된 명령
  (D_inj)을 보는 attention edge만 끊어도 공격은 막히고 정상 read 기능은 거의 안 다친다"는
  핵심 주장이 이제 신뢰할 수 있는 수치로도 확인됨.
- 3B의 `jaccard(internal,external)=0.538`은 1.5B와 동일 — 모델 크기가 커져도 내부/외부
  control head 공유 패턴이 재현되는 것도 그대로 유지.

**남은 이슈**: `debug_read_target.py`의 `attention_mask` 자동추론 버그(커밋 `315f964`)는
`model.generate()`를 쓰는 디버깅 스크립트에만 해당되고, `run_pipeline.py`/`attn_relevance.py`/
`edge_ablation.py`는 `generate()`를 쓰지 않으므로 이 위 표의 본 실험 수치엔 영향 없음.
