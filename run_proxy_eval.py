"""
run_proxy_eval.py

`compare_head_sources.py`(Track A)로 찾은 head 집합을 synthetic/InjecAgent의 **proxy 지표**
(edge_ablation.sweep_knockout — 다음 토큰 확률 기반, forward-only)로 knockout 평가한다.
`run_agentdojo_eval.py`(AgentDojo 네이티브 멀티턴 채점)와 같은 모델·같은 head 집합으로
나란히 비교하기 위한 목적 — "proxy 지표 vs 네이티브 채점"에서 결론이 얼마나 같은지/다른지
직접 볼 수 있다.

`run_pipeline.py`와 다른 점: 이 스크립트는 head를 새로 찾지 않고(그래서 `attn_relevance`의
lxt backward가 필요 없음) `--heads_json`으로 이미 찾아둔 head 집합을 그대로 받아 평가만
한다 — synthetic/InjecAgent/AgentDojo 세 소스에서 찾은 head를 서로 바꿔가며 평가할 때도
그대로 재사용 가능.

사용 예:
    python run_proxy_eval.py --model unsloth/Qwen2.5-14B-Instruct-bnb-4bit --family qwen2 \\
        --heads_json results/2026-07-31_source_compare/heads_synthetic_14b.json \\
        --out_json results/.../proxy_eval_synthetic_14b.json
"""
import argparse
import json
from typing import Dict, List, Tuple

from attn_relevance import load_model_for_relevance
from dataset import build_phase0_batch
from edge_ablation import sweep_knockout


def _load_heads(heads_json: str) -> List[Tuple[int, int]]:
    with open(heads_json, encoding="utf-8") as f:
        data = json.load(f)
    return [tuple(h) for h in data["heads"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--family", default="qwen2", choices=["qwen2", "llama"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--four_bit", action="store_true")
    parser.add_argument("--heads_json", required=True, help="compare_head_sources.py discover(-parallel) 결과 JSON")
    parser.add_argument("--injecagent_repo_dir", default="external_injecagent")
    parser.add_argument("--injecagent_limit", type=int, default=None)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    heads = _load_heads(args.heads_json)
    print(f"[run_proxy_eval] {len(heads)} heads loaded from {args.heads_json}: {heads}")
    ks = sorted({0, len(heads)})

    print(f"[run_proxy_eval] loading {args.model} (family={args.family}, four_bit={args.four_bit}) ...")
    model, tok = load_model_for_relevance(
        model_path=args.model, four_bit=args.four_bit, device=args.device, model_family=args.family
    )
    if args.family == "qwen2":
        from transformers.models.qwen2 import modeling_qwen2 as modeling_mod
    else:
        from transformers.models.llama import modeling_llama as modeling_mod

    results = {"model": args.model, "heads_json": args.heads_json, "n_heads": len(heads), "heads": heads}

    print("[run_proxy_eval] synthetic 데이터셋 (30개 템플릿) sweep ...")
    pairs = build_phase0_batch(tok, device=args.device, limit=None)
    exec_examples = [ex["external"] for ex in pairs]
    read_examples = [ex["read_injected"] for ex in pairs]
    synthetic_sweep = sweep_knockout(model, modeling_mod, exec_examples, read_examples, heads, ks=ks)
    for row in synthetic_sweep:
        print(
            f"  [synthetic] k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
            f"  read_token_prob={row['read_token_prob']:.4f}  (n={row['n_examples']})"
        )
    results["synthetic"] = synthetic_sweep

    print(f"[run_proxy_eval] InjecAgent sweep (repo_dir={args.injecagent_repo_dir}) ...")
    from adapters.injecagent import build_injecagent_batch

    ia_examples = build_injecagent_batch(
        tok, device=args.device, repo_dir=args.injecagent_repo_dir, limit=args.injecagent_limit
    )
    print(f"  {len(ia_examples)} InjecAgent examples loaded.")
    injecagent_sweep = sweep_knockout(model, modeling_mod, ia_examples, ia_examples, heads, ks=ks)
    for row in injecagent_sweep:
        print(
            f"  [InjecAgent] k={row['k']:>3}  malicious_token_prob={row['malicious_token_prob']:.4f}"
            f"  read_token_prob={row['read_token_prob']:.4f}  (n={row['n_examples']})"
        )
    results["injecagent"] = injecagent_sweep

    out_json = args.out_json or "proxy_eval_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nsummary saved to {out_json}")


if __name__ == "__main__":
    main()
