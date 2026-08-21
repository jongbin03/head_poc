"""
tools/diag_agentdojo_pairs.py

AgentDojo의 (user_task × injection_task) 조합 949개 중 **왜 220개만 pair가 되는지**,
그리고 **왜 travel만 8개인지**를 사유별로 센다. GPU 불필요 (토크나이저만 씀).

배경 (실측 2026-08-21):
  suite별 전조합 대비 실제 생성:
    banking   144 -> 61 (42%)
    slack     105 -> 47 (45%)
    workspace 560 -> 104 (19%)
    travel    140 ->  8 (5.7%)   ← 압도적으로 적다

  8/19 발표에서 "travel이 OOM으로 빠졌다"고 본 것은 **원인 오진**일 가능성이 크다.
  길이 필터(`--max_seq_len 2000`)는 220 -> 207로 6%만 자르고, OOM은 그 뒤 단계다.
  즉 travel은 relevance 계산에 들어가기도 전에 이미 8개로 줄어 있었다.

`build_agentdojo_example`이 None을 반환하는 사유:
  no_inj_ground_truth      injection_task.ground_truth(env)가 비어 있음 (무효 조합)
  no_injected_tool_turn    주입이 포함된 tool 턴이 없거나, 그 뒤에 이어질 행동이 없음
  same_tool_name           정상 다음 tool == 공격자 tool (tool-이름 proxy로 구분 불가)
  first_token_collision    tool 이름은 다른데 **첫 토큰이 같음**
                           (예: get_balance / get_iban). next-token proxy의 구조적 한계.
  clean_turn_missing       clean 롤아웃이 더 짧아 같은 turn_idx가 없음
  clean_no_next_action     clean 쪽에서 다음 행동을 못 찾음

**2026-08-21 어댑터 일반화 이후 기대값** (`_find_injected_tool_turn`이 첫 턴이 아니라
"주입이 포함된 첫 턴"을 찾도록 바뀜):

    suite      before -> after
    banking     61   ->  64
    slack       47   ->  74
    travel       8   ->  59
    workspace  104   -> 110
    합계       220   -> 307

이 표와 실제 출력이 어긋나면 일반화 구현에 문제가 있는 것이다. `injection_text_not_found`는
이제 논리적으로 0이어야 한다 (앵커로 턴을 찾은 뒤 같은 앵커로 다시 자르므로).

사용:
    python tools/diag_agentdojo_pairs.py
    python tools/diag_agentdojo_pairs.py --model Qwen/Qwen2.5-7B-Instruct --suites travel
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REASONS = [
    "ok",
    "no_injected_tool_turn",   # 주입 턴이 없거나 그 뒤에 이어질 행동이 없음
    "same_tool_name",
    "first_token_collision",
    "injection_text_not_found",
    "no_inj_ground_truth",
    "clean_turn_missing",      # clean 롤아웃이 더 짧아 같은 턴이 없음
    "clean_no_next_action",
    "clean_build_failed",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="토크나이저만 쓴다 (가중치 다운로드 없음).")
    p.add_argument("--suites", nargs="*", default=None)
    p.add_argument("--benchmark_version", default="v1.2.2")
    p.add_argument("--show_tools", action="store_true",
                   help="suite별 tool 이름과 첫 토큰을 찍어 충돌을 눈으로 확인한다.")
    p.add_argument("--report_lengths", action="store_true",
                   help="생성된 프롬프트의 토큰 길이 분포를 turn_idx(0 / >=1)별로 낸다. "
                        "일반화로 앞선 턴이 붙으면서 길어졌는지 확인용 — max_seq_len 재설정 근거.")
    p.add_argument("--history_max_chars", type=int, default=None,
                   help="앞선 턴 tool 응답을 이 길이로 자른다 (기본 None=자르지 않음).")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from adapters.agentdojo import build_agentdojo_pairs, DEFAULT_SUITE_NAMES

    tok = AutoTokenizer.from_pretrained(args.model)
    suites = args.suites if args.suites else DEFAULT_SUITE_NAMES

    stats: dict = {}
    pairs = build_agentdojo_pairs(
        tok, device="cpu", suite_names=suites,
        benchmark_version=args.benchmark_version, skip_stats=stats,
        history_max_chars=args.history_max_chars,
    )

    width = max(len(r) for r in REASONS) + 2
    header = f"{'suite':<11}" + "".join(f"{r:>{width}}" for r in REASONS)
    print("\n" + header)
    print("-" * len(header))
    totals: Counter = Counter()
    for s in suites:
        row = stats.get(s, {})
        attempted = row.get("attempted", 0)
        line = f"{s:<11}"
        for r in REASONS:
            v = row.get(r, 0)
            totals[r] += v
            line += f"{v:>{width}}"
        ok = row.get("ok", 0)
        rate = (ok / attempted * 100) if attempted else 0.0
        print(line + f"   (attempted={attempted}, 생성률 {rate:.1f}%)")
    print("-" * len(header))
    print(f"{'합계':<11}" + "".join(f"{totals[r]:>{width}}" for r in REASONS))
    print(f"\n최종 pair 수: {len(pairs)}")

    # turn_idx 분포 — 일반화로 새로 살아난 게 몇 개인지
    turn_dist = Counter(p["injected"].meta.get("turn_idx", 0) for p in pairs)
    by_suite_turn: dict = {}
    for p_ in pairs:
        m = p_["injected"].meta
        by_suite_turn.setdefault(m["suite"], Counter())[m.get("turn_idx", 0)] += 1
    print(f"\nturn_idx 분포 (0=기존 경로, >=1=일반화로 회복): {dict(sorted(turn_dist.items()))}")
    for s in suites:
        c = by_suite_turn.get(s, Counter())
        t0 = c.get(0, 0)
        print(f"  {s:<11} turn0={t0:4d}  turn>=1={sum(c.values()) - t0:4d}  합계={sum(c.values()):4d}")

    if args.report_lengths:
        import statistics as st

        groups: dict = {"turn0": [], "turn>=1": []}
        per_suite: dict = {}
        for p_ in pairs:
            m = p_["injected"].meta
            n = int(p_["injected"].input_ids.shape[-1])
            key = "turn0" if m.get("turn_idx", 0) == 0 else "turn>=1"
            groups[key].append(n)
            per_suite.setdefault(m["suite"], {}).setdefault(key, []).append(n)

        def _fmt(v):
            if not v:
                return "  (없음)"
            v = sorted(v)
            p95 = v[min(len(v) - 1, int(len(v) * 0.95))]
            over = {t: sum(1 for x in v if x > t) for t in (2000, 4000, 8000)}
            return (f"n={len(v):4d}  중앙={st.median(v):6.0f}  p95={p95:6.0f}  최대={v[-1]:6.0f}  "
                    f">2000:{over[2000]:3d} >4000:{over[4000]:3d} >8000:{over[8000]:3d}")

        print("\n=== 프롬프트 토큰 길이 (injected) ===")
        for k, v in groups.items():
            print(f"  전체 {k:<8} {_fmt(v)}")
        for s in suites:
            print(f"  [{s}]")
            for k in ("turn0", "turn>=1"):
                print(f"    {k:<8} {_fmt(per_suite.get(s, {}).get(k, []))}")
        print("\n  → `>2000` 개수가 크면 --max_seq_len을 그대로 두었을 때 일반화로 살린 것을"
              "\n    다시 필터로 버리게 된다. --history_max_chars로 앞선 턴을 줄이거나"
              "\n    절단(truncation)을 앞당길 것 (docs/plan-2026-08-26.md 4.5절).")

    if args.show_tools:
        from agentdojo.task_suite.load_suites import get_suite
        print("\n=== suite별 tool 이름의 첫 토큰 (충돌 확인용) ===")
        for s in suites:
            suite = get_suite(args.benchmark_version, s)
            names = sorted(t.name for t in suite.tools)
            first = {}
            for n in names:
                fid = tok(" " + n, add_special_tokens=False)["input_ids"][0]
                first.setdefault(fid, []).append(n)
            n_collide = sum(len(v) for v in first.values() if len(v) > 1)
            print(f"\n[{s}] tools={len(names)}  "
                  f"첫 토큰이 겹치는 tool={n_collide}  고유 첫 토큰={len(first)}")
            for fid, group in sorted(first.items(), key=lambda kv: -len(kv[1])):
                if len(group) > 1:
                    print(f"   {tok.decode([fid])!r:>12} ({len(group)}개): {', '.join(group)}")


if __name__ == "__main__":
    main()
