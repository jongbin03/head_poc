# Llama-3.1-70B 실험 준비 노트 (2026-09-08)

P12 (docs/todo.md) / feedback-2026-09-06.md §1. **스케일 축**: Llama-3.1-8B → Llama-3.1-70B
(같은 3.1 버전 안에서 스케일만). Qwen2.5 7B→32B에 대응.

## 검증 질문

32B(Qwen)에서 나온 반례 — "knockout이 slack의 달성 가능 공격을 절반 이상 못 막는다"
(feedback §1, 8~9→5) — 가 Llama family 70B에서도 재현되는가?
8B는 slack 포함 전 suite에서 knockout 전량 억제(6→0)였음.

## 사전 확인 (2026-09-08 완료)

- ✅ GPU 3장 전부 유휴 (PRO4500 32G / A6000 48G / 4090 24G)
- ✅ `meta-llama/Llama-3.1-70B-Instruct` HF 접근 OK (`jongbeen212` 계정)
  - ⚠️ `Llama-3.3-70B-Instruct`는 403 (gated 승인 안 됨). 3.1이 스케일 축엔 오히려 정답
    (8B도 3.1이라 버전 고정됨).
- ✅ P13(llama tool-call 파서 0% 버그) **이미 서버 재검증됨** — 2026-08-31 Llama-3.1-8B
  A/B에서 parse ok 79%(custom)·79%(agentdojo_default). feedback 2.5.3. status-09-07/P12의
  "서버 재검증 전" 서술은 낡음.
- ✅ 디스크 1.5T 여유 (fp16 70B ~140GB 받아도 됨)

## 방침

**Track B(knockout 평가) 먼저.** forward-only라 4bit·OOM 걱정 적음. 8B에서 찾은 헤드
(`results/2026-08-25_s6_llama8b/heads_agentdojo.json`, 20개)를 70B에 전이 평가.
Track A(70B 자체 헤드 탐색)는 `attn_relevance.py` nf4dq 배선(feedback-09-06 §3) 후 별도.

**양자화**: `--four_bit` = nf4 + double_quant (run_agentdojo_eval.py 기본값, P16 확정).
A6000에서 nf4dq는 bf16 baseline과 일치(feedback 2.1.20) — bf16 70B는 이 하드웨어에
물리적으로 안 들어감(~140GB > 104GB). Blackwell(PRO4500) nf4 커널은 32B 손상 이력이
있으므로 **A6000 고정**.

**GPU**: 70B nf4 가중치 ~40GB → A6000 48GB 단독 시도. 멀티턴 롤아웃 KV 캐시로 OOM나면
`--device_map auto --max_memory`로 A6000+4090 분산 (순차라 느림).

## held-out 표본 주의

8B 헤드의 split_info는 head 탐색에 user_task를 거의 다 씀:
slack eval_pairs=24, banking=7, **travel=0, workspace=0**.
→ `--eval_split heldout`이면 slack만 표본이 됨(≈5~24쌍). travel/workspace는 held-out 0.
`--limit_pairs`를 크게(50) 줘서 남은 held-out 전수 사용. 필요시 `--eval_split all`도
나란히 돌려 누수 영향 비교(feedback §1 방식).

## 실행 (run.sh 참고)

1. **다운로드** (백그라운드, ~수 시간): `hf download meta-llama/Llama-3.1-70B-Instruct`
2. **스모크**: slack 3쌍, A6000 단독, 로드·OOM·완주 확인
3. **본 실행 slack**: `--eval_split heldout --limit_pairs 50`, tmux
4. OOM이면: `--device_map auto --max_memory 0:44GiB 1:22GiB` (CUDA_VISIBLE_DEVICES=1,2)
5. 결과 나오면 8B(6→0) / Qwen 32B(8~9→5)와 slack k0_sec→kN_sec, backfire, utility 비교

## 결과 파일

- `eval_slack_heldout.json` — 본 실행
- `eval_slack_all.json` — 누수 비교용 (선택)
- `smoke_slack.json` — 스모크
- `console_*.log` — tee 로그
