"""
tools/diag_asb_pairs.py

`adapters/asb.py`가 실제로 유효한 쌍을 만드는지, span/target이 말이 되는지 확인하는
스모크 테스트. GPU 불필요 (토크나이저만 씀). `tools/diag_agentdojo_pairs.py`와 같은 역할.

사용:
    python tools/diag_asb_pairs.py
    python tools/diag_asb_pairs.py --model meta-llama/Llama-3.1-8B-Instruct --limit 20 --show
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="토크나이저만 쓴다 (가중치 다운로드 없음).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--attack_type", default="context_ignoring",
                    choices=["naive", "fake_completion", "escape_characters",
                             "context_ignoring", "combined_attack"])
    p.add_argument("--aggressive", default=None, choices=["true", "false"],
                    help="미지정이면 both(기본, all_attack_tools.jsonl 전체 사용).")
    p.add_argument("--show", action="store_true",
                    help="쌍 몇 개를 디코딩해서 눈으로 확인한다.")
    args = p.parse_args()

    aggressive = None
    if args.aggressive is not None:
        aggressive = args.aggressive == "true"

    from transformers import AutoTokenizer
    from adapters.asb import build_asb_pairs, load_agent_tasks, load_normal_tools, load_attack_tools

    tok = AutoTokenizer.from_pretrained(args.model)

    agent_tasks = load_agent_tasks()
    normal_tools = load_normal_tools()
    attack_tools = load_attack_tools(aggressive=aggressive)
    print(f"agents: {len(agent_tasks)}, "
          f"normal_tools/agent: {[len(v) for v in normal_tools.values()][:3]}..., "
          f"attack_tools/agent: {[len(v) for v in attack_tools.values()][:3]}...")

    pairs = build_asb_pairs(
        tok, device="cpu", attack_type=args.attack_type,
        aggressive=aggressive, limit=args.limit,
    )
    print(f"pairs built: {len(pairs)}")

    if not pairs:
        return

    # 첫 토큰 충돌/데이터 sanity를 agent별로 집계
    from collections import Counter
    by_agent = Counter(p["injected"].meta["agent"] for p in pairs)
    print("pairs by agent:", dict(by_agent))

    if args.show:
        for pair in pairs[: min(3, len(pairs))]:
            inj = pair["injected"]
            clean = pair["clean"]
            print("\n" + "=" * 80)
            print("meta:", inj.meta)
            print("read_target token:", repr(tok.decode([inj.read_target])))
            print("exec_target token:", repr(tok.decode([inj.exec_target])))
            print("--- injected prompt tail (last 400 chars) ---")
            print(tok.decode(inj.input_ids[0].tolist())[-400:])
            print("--- clean prompt tail (last 400 chars) ---")
            print(tok.decode(clean.input_ids[0].tolist())[-400:])
            print("data_inj span len:", len(inj.spans.get("data_inj", [])))
            print("data_benign span len (injected):", len(inj.spans.get("data_benign", [])))
            print("data_benign span len (clean):", len(clean.spans.get("data_benign", [])))


if __name__ == "__main__":
    main()
