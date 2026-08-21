"""
debug_read_target.py

3B(혹은 임의 모델)에서 read_token_prob이 비정상적으로 낮게 나올 때, 원인이
"read_target 토큰 자체가 잘못 잡힌 것"인지 확인하기 위한 디버깅 스크립트.

하는 일 (dataset.py/attn_relevance.py의 lxt monkey-patch 없이, 순수 HF만 사용):
  1. 각 템플릿의 read_clean / read_injected 프롬프트에 대해
     - assistant_prefix 다음에 모델이 실제로 이어 쓰는 상위 후보 토큰 top-N과 확률을 출력
     - 우리가 기대하는 read_target 토큰이 그 안에 있는지, 몇 위인지, 확률이 얼마인지 출력
  2. model.generate로 몇 토큰 더 생성해서 실제 텍스트를 눈으로 확인

사용:
    python debug_read_target.py --model Qwen/Qwen2.5-3B-Instruct --family qwen2 --device cuda
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime_env import add_runtime_args, describe, resolve_dtype

from dataset import build_phase0_batch


def show_topk(logits: torch.Tensor, tok, target_id: int, topn: int = 10):
    probs = torch.softmax(logits.float(), dim=-1)
    top_probs, top_ids = torch.topk(probs, topn)
    lines = []
    for rank, (p, tid) in enumerate(zip(top_probs.tolist(), top_ids.tolist()), start=1):
        piece = tok.decode([tid])
        marker = "  <-- read_target" if tid == target_id else ""
        lines.append(f"    #{rank:>2} id={tid:<7} p={p:.4f}  {piece!r}{marker}")

    target_rank = (top_ids == target_id).nonzero()
    if target_rank.numel() > 0:
        rank = target_rank.item() + 1
        target_p = probs[target_id].item()
    else:
        # top-N 밖에 있으면 전체에서의 순위를 따로 계산
        all_sorted = torch.argsort(probs, descending=True)
        rank = (all_sorted == target_id).nonzero().item() + 1
        target_p = probs[target_id].item()

    target_piece = tok.decode([target_id])
    print(f"    [target] id={target_id} piece={target_piece!r} prob={target_p:.6f} rank={rank}")
    for line in lines:
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--family", default="qwen2", choices=["qwen2", "llama"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset_limit", type=int, default=3)
    parser.add_argument("--gen_tokens", type=int, default=12)
    add_runtime_args(parser)
    args = parser.parse_args()

    torch_dtype, dtype_name = resolve_dtype(args.dtype, args.device)
    print(describe(dtype_name))
    print(f"loading {args.model} (plain HF, no lxt patch) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map=args.device,
        attn_implementation="eager",
    )
    model.eval()

    pairs = build_phase0_batch(tok, device=args.device, limit=args.dataset_limit)

    for i, ex in enumerate(pairs):
        for mode in ("read_clean", "read_injected"):
            sample = ex[mode]
            print(f"\n=== template #{i} mode={mode} ===")
            prompt_text = tok.decode(sample.input_ids[0])
            print(f"  prompt tail: ...{prompt_text[-120:]!r}")

            # attention_mask를 명시적으로 전부 1(패딩 없음)로 넘긴다. 안 넘기면
            # HF가 "input_ids 중 pad_token_id와 같은 값은 패딩"이라고 자동 추정하는데,
            # 이 모델은 pad_token_id == eos_token_id라서 프롬프트 안에 우연히 eos와
            # 같은 토큰 id가 있으면 그 부분이 실제 내용인데도 마스킹돼 attention이
            # 깨진다 (model()과 generate()가 서로 다른 로짓/토큰을 내는 원인이었음).
            attention_mask = torch.ones_like(sample.input_ids)

            with torch.no_grad():
                out = model(
                    input_ids=sample.input_ids, attention_mask=attention_mask, use_cache=False,
                )
                next_logits = out.logits[0, -1]

            show_topk(next_logits, tok, sample.read_target, topn=10)

            with torch.no_grad():
                gen = model.generate(
                    input_ids=sample.input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.gen_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            gen_text = tok.decode(gen[0, sample.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"  greedy continuation: {gen_text!r}")


if __name__ == "__main__":
    main()
