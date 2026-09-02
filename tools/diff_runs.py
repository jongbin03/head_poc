"""
tools/diff_runs.py

run_agentdojo_eval.py 결과 JSON 두 개(이상)를 **쌍(row) 단위로** 비교한다.

주 용도 — docs/feedback-2026-08-31.md 2.1.13 "4bit knockout 불안정" 추정 검증:
  (1) 완전 동일 조건 2회 실행을 넣어 "같은 쌍이 실행 간에 뒤집히는" 비결정성의
      크기를 잰다. greedy decoding인데도 kN_security 등이 흔들리면 그 개수가
      knockout 결함이 아니라 4bit 수치 노이즈의 규모다 (todo.md 709~735행).
  (2) fp4 vs nf4, double_quant on/off 처럼 한 축만 바꾼 두 실행을 넣어
      불안정성(backfire/persist)이 그 설정에 붙어 있는지 본다.

비교 대상 4필드: k0_utility, k0_security, kN_utility, kN_security
  - k0_* : knockout 없음(baseline). 동일 조건 2회면 이론상 완전히 같아야 한다.
  - kN_* : knockout 적용. 여기가 흔들리는 게 2.1.13의 관심사.

사용:
    python tools/diff_runs.py A.json B.json
    python tools/diff_runs.py fp4.json nf4.json --label fp4 nf4
    python tools/diff_runs.py run1.json run2.json run3.json     # 3회 이상도 가능

출력:
    - 실행별 메타(commit / quantization 블록 / n_pairs)
    - 실행별 overall·suite별 k0_sec / kN_sec / k0_util / kN_util 나란히
    - 완주 쌍 교집합 크기 + 한쪽에만 있는 쌍(OOM/ERROR 스킵) 목록
    - **비결정성**: 교집합 쌍 중 ≥1 필드가 실행 간 불일치인 쌍 수 (필드별 분해 + 개별 목록)
    - backfire(k0_sec=F & kN_sec=T) / suppressed(k0_sec=T & kN_sec=F) 실행별 집계
"""
import argparse
import json
import os
import sys
from collections import defaultdict

try:  # Windows 콘솔(cp949)에서도 em-dash 등이 깨지지 않게. Linux는 이미 utf-8.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FIELDS = ["k0_utility", "k0_security", "kN_utility", "kN_security"]
SUITES = ["banking", "slack", "travel", "workspace"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def key(row):
    return (row["suite"], row["user_task"], row["injection_task"])


def rows_by_key(summary):
    return {key(r): r for r in summary.get("rows", [])}


def fmt_meta(label, path, summary):
    env = summary.get("env", {}) or {}
    commit = (env.get("git", {}) or {}).get("commit", "?")[:10]
    q = summary.get("quantization") or env.get("quantization")
    argv_four_bit = "--four_bit" in (env.get("argv") or [])
    if q:
        qs = (f"4bit[{q.get('bnb_4bit_quant_type')}, "
              f"double_quant={q.get('bnb_4bit_use_double_quant')}, "
              f"compute={q.get('bnb_4bit_compute_dtype')}]")
    elif summary.get("four_bit") or argv_four_bit:
        qs = "4bit[설정 미기록 — 구버전 결과, fp4+double_quant off로 추정]"
    else:
        qs = env.get("dtype", "bf16?")
    return (f"  [{label}] {os.path.basename(path)}\n"
            f"        commit={commit}  {qs}  n_pairs={summary.get('n_pairs')}  "
            f"attack={summary.get('attack')}  heads_json={os.path.basename(summary.get('heads_json',''))}")


def rate(rows, field):
    return (sum(1 for r in rows if r[field]) / len(rows)) if rows else 0.0


def print_rate_table(labels, summaries):
    print("\n=== 실행별 rate (완주 쌍 전체 기준, 각 실행의 자기 표본) ===")
    hdr = f"{'suite':<10} " + " ".join(f"{lab:>22}" for lab in labels)
    print(hdr)
    print(f"{'':<10} " + " ".join(f"{'k0sec / kNsec / k0u / kNu':>22}" for _ in labels))
    for suite in SUITES + ["<ALL>"]:
        cells = []
        for s in summaries:
            rs = [r for r in s.get("rows", []) if suite == "<ALL>" or r["suite"] == suite]
            if not rs:
                cells.append(f"{'-':>22}")
                continue
            cells.append(f"{rate(rs,'k0_security'):.2f}/{rate(rs,'kN_security'):.2f}/"
                         f"{rate(rs,'k0_utility'):.2f}/{rate(rs,'kN_utility'):.2f}".rjust(22))
        n = ""
        print(f"{suite:<10} " + " ".join(cells) + n)


def census(labels, maps):
    print("\n=== backfire / suppressed / persist 집계 (각 실행 자기 표본, slack 중심) ===")
    for suite in SUITES:
        line = [f"  {suite:<10}"]
        any_row = False
        for lab, m in zip(labels, maps):
            rs = [r for k, r in m.items() if k[0] == suite]
            if not rs:
                line.append(f"{lab}: -")
                continue
            any_row = True
            backfire = sum(1 for r in rs if (not r["k0_security"]) and r["kN_security"])
            suppressed = sum(1 for r in rs if r["k0_security"] and not r["kN_security"])
            persist = sum(1 for r in rs if r["k0_security"] and r["kN_security"])
            line.append(f"{lab}: bf={backfire} sup={suppressed} per={persist}")
        if any_row:
            print("   ".join(line))


def nondeterminism(labels, maps):
    common = set(maps[0])
    for m in maps[1:]:
        common &= set(m)
    all_keys = set()
    for m in maps:
        all_keys |= set(m)
    only = {lab: sorted(set(m) - common) for lab, m in zip(labels, maps)}

    print("\n=== 완주 쌍 교집합 ===")
    print(f"  전체 쌍(합집합): {len(all_keys)}   모든 실행에 있는 쌍(교집합): {len(common)}")
    for lab, ks in only.items():
        if ks:
            print(f"  [{lab}]에만 있는 쌍 {len(ks)}개 (다른 실행에서 OOM/ERROR 스킵):")
            for k in ks:
                print(f"       {k[0]}/{k[1]}+{k[2]}")

    print("\n=== 비결정성 — 교집합 쌍 중 실행 간 값이 다른 것 ===")
    field_diff = defaultdict(list)
    any_diff_keys = set()
    for k in sorted(common):
        for field in FIELDS:
            vals = [maps[i][k][field] for i in range(len(maps))]
            if len(set(vals)) > 1:
                field_diff[field].append((k, vals))
                any_diff_keys.add(k)
    print(f"  ≥1 필드가 불일치인 쌍: {len(any_diff_keys)} / {len(common)}")
    for field in FIELDS:
        diffs = field_diff[field]
        tag = "  <-- knockout 적용 필드" if field.startswith("kN") else ""
        print(f"    {field:<12}: {len(diffs)} 쌍 불일치{tag}")
        for k, vals in diffs:
            vs = "  ".join(f"{lab}={v}" for lab, v in zip(labels, vals))
            print(f"        {k[0]}/{k[1]}+{k[2]}   {vs}")
    if not any_diff_keys:
        print("  (교집합 쌍 전부 모든 필드에서 동일 — 이 조건에서는 결정론적)")

    # 동일 조건 2회 실행 판정 힌트
    k0_diffs = len(field_diff["k0_utility"]) + len(field_diff["k0_security"])
    if len(maps) == 2 and k0_diffs:
        print(f"\n  ⚠️ k0(baseline, knockout 없음) 필드도 {k0_diffs}건 흔들림 — "
              f"두 실행이 완전 동일 조건이라면 이건 순수 4bit 커널 비결정성의 하한이다.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="run_agentdojo_eval.py 결과 JSON 2개 이상")
    ap.add_argument("--label", nargs="+", default=None, help="각 결과의 짧은 이름 (기본: A B C ...)")
    args = ap.parse_args()

    if len(args.results) < 2:
        ap.error("결과 JSON을 최소 2개 넣어야 비교가 된다")
    labels = args.label or [chr(ord("A") + i) for i in range(len(args.results))]
    if len(labels) != len(args.results):
        ap.error(f"--label 개수({len(labels)})가 결과 개수({len(args.results)})와 다르다")

    summaries = [load(p) for p in args.results]
    maps = [rows_by_key(s) for s in summaries]

    print("=" * 78)
    print("diff_runs — 결과 JSON 쌍 단위 비교")
    print("=" * 78)
    for lab, path, s in zip(labels, args.results, summaries):
        print(fmt_meta(lab, path, s))

    print_rate_table(labels, summaries)
    census(labels, maps)
    nondeterminism(labels, maps)


if __name__ == "__main__":
    main()
