# 2026-07-27 Colab 스모크 테스트 결과

**실행 환경**: Google Colab (무료, T4 GPU), 로컬 5070 Ti 아님.

## 실행 명령

```bash
python run_pipeline.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --family qwen2 \
  --device cuda \
  --topk 20 \
  --dataset_limit 2
```

## 설치 환경 메모 (Colab 기본 환경과 lxt 호환성 이슈)

Colab 기본 설치된 `transformers`는 lxt(`lxt.efficient`)가 요구하는 구버전 API
(`find_pruneable_heads_and_indices` 등)와 최신 아키텍처 지원(`qwen3` 모듈) 사이에서
버전 충돌이 있었음:

- 최신 transformers → `find_pruneable_heads_and_indices` 없음 (제거됨)
- `transformers==4.46.3` → `qwen3` 모듈 없음 (lxt가 무조건 import)

**해결**: `transformers==4.51.3`으로 고정 후 정상 동작 확인.

```bash
pip install -q "transformers==4.51.3" lxt accelerate bitsandbytes matplotlib
```

(`gradio`가 요구하는 huggingface-hub 버전과 충돌 경고가 뜨지만, 본 파이프라인은
gradio를 쓰지 않으므로 무시 가능.)

## 출력 로그 (원본)

```
[1/4] loading Qwen/Qwen2.5-0.5B-Instruct (family=qwen2, four_bit=False) ...
Sliding Window Attention is enabled but not implemented for `eager`; unexpected results may be encountered.
[2/4] building synthetic IPI dataset ...
[3/4] ranking heads & computing overlap ...
  top-20 Jaccard(read, internal)   = 0.143
  top-20 Jaccard(read, external)   = 0.250
  top-20 Jaccard(internal,external)= 0.481
  control_heads_both (internal ∩ external)  = [(0, 1), (1, 8), (1, 9), (15, 2), (15, 9), (15, 12), (16, 7), (20, 12), (21, 2), (21, 3), (21, 11), (22, 5), (23, 10)]
  functional map saved to functional_map.png
[4/4] edge-knockout sweep on control head candidates ...
  k=  0  malicious_token_prob=0.9386  read_token_prob=0.6071  (n=2)
  k=  5  malicious_token_prob=0.0628  read_token_prob=0.6027  (n=2)
  k= 10  malicious_token_prob=0.0021  read_token_prob=0.5757  (n=2)
  k= 20  malicious_token_prob=0.0000  read_token_prob=0.6055  (n=2)
  k= 40  malicious_token_prob=0.0000  read_token_prob=0.6572  (n=2)
  k= 80  malicious_token_prob=0.0000  read_token_prob=0.6572  (n=2)
```

## 체크리스트 결과 (RUN.md 3단계 기준)

- [x] `[1/4]`~`[4/4]` 네 단계 에러 없이 순서대로 출력
- [x] Jaccard 세 줄이 0~1 사이 정상값 (NaN 아님)
- [x] `functional_map.png` 생성됨
- [x] `k=0`→`k=80` 구간에서 `malicious_token_prob` 유의미하게 변화 (knockout이 실제로 적용됨)
- [x] `k=0`에서 `malicious_token_prob`, `read_token_prob` 둘 다 0 아님

→ Phase 0 스모크 테스트 통과.

## 해석 (n=2, 통계적 의미 없는 스모크 테스트 수준)

- `malicious_token_prob`: 0.9386 → 0.0000 (control head 5~10개 knockout만으로 급격히 붕괴)
- `read_token_prob`: 0.6071 → 0.6572 (거의 유지, 오히려 소폭 상승) — 정상 기능 보존 신호
- `jaccard(internal, external)=0.481` > `jaccard(read, internal)=0.143`, `jaccard(read, external)=0.250`
  → 내부 응답 오염과 외부 tool-call 오염이 상당 부분 같은 control head 회로를 공유한다는
  idea1의 전제와 방향이 일치.

**주의**: `--dataset_limit 2`(템플릿 2개)짜리 스모크 테스트이므로 코드가 정상 동작함을
확인한 것이지 실험 결론으로 쓸 수 없음. 다음 단계는 RUN.md 4단계
(`--dataset_limit 6` 이상, `Qwen2.5-1.5B-Instruct` 이상, 전체 30개 템플릿)로 진행.
