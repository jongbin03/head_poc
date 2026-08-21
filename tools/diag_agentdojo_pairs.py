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

`build_agentdojo_example`이 None을 반환하는 사유는 5가지다:
  no_inj_ground_truth     injection_task.ground_truth(env)가 비어 있음
  no_first_tool_turn      [assistant tool_call, tool, assistant tool_call] 패턴 불성립
                          (멀티턴 prefix가 필요하거나 다음 행동이 없는 case)
  same_tool_name          정상 다음 tool == 공격자 tool (tool-이름 proxy로 구분 불가)
  injection_text_not_found  공격 문구가 첫 tool 응답에 그대로 안 나타남
  first_token_collision   tool 이름은 다른데 **첫 토큰이 같음**
                          (예: get_balance / get_iban). next-token 확률 proxy의 구조적 한계.

마지막 두 개(same_tool_name, first_token_collision)가 travel에서 크다면, travel 불균형은
메모리 문제가 아니라 **타깃 정의(=tool 이름 첫 토큰)의 한계**다. 그 경우 VRAM을 늘리거나
max_seq_len을 올려도 절대 해결되지 않으며, 타깃 토큰 정의를 바꿔야 한다.

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
    "no_first_tool_turn",
    "same_tool_name",
    "first_token_collision",
    "injection_text_not_found",
    "no_inj_ground_truth",
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
    args = p.parse_args()

    from transformers import AutoTokenizer
    from adapters.agentdojo import build_agentdojo_pairs, DEFAULT_SUITE_NAMES

    tok = AutoTokenizer.from_pretrained(args.model)
    suites = args.suites if args.suites else DEFAULT_SUITE_NAMES

    stats: dict = {}
    pairs = build_agentdojo_pairs(
        tok, device="cpu", suite_names=suites,
        benchmark_version=args.benchmark_version, skip_stats=stats,
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
