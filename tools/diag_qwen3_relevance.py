"""
tools/diag_qwen3_relevance.py

plan-2026-08-26.md 3.1.1: lxt README가 Qwen3에서 "attribution이 첫 토큰으로 쏠린다"고
경고한다. 배선(attn_relevance.py에 qwen3 분기 추가, 2026-08-31)은 됐지만, 이 경고가
우리 세팅에서 실제로 문제가 되는지는 직접 진단해야 한다 — 원문 그대로 "쓰려면 배선보다
진단이 먼저다". 지금까지 이 경고는 lxt 자체 README를 인용만 했을 뿐 우리 프로젝트에서
직접 실측한 적이 없다.

`attn_relevance.compute_head_relevance()`와 같은 backward-hook 메커니즘을 쓰되,
key_spans로 group-sum하지 않고 **key position별 relevance를 그대로 남겨서** position 0이
실제로 관련도 질량을 지배하는지 확인한다.

**판단 기준**: position 0 비중이 절대적으로 0이 아니라고 문제인 게 아니다(causal LM은
흔히 어느 정도의 "attention sink"가 position 0에 있다 — 이건 일반적인 현상). **같은
프롬프트/구조로 다른 family(qwen2)와 나란히 돌려서, Qwen3가 그 baseline 대비 유의미하게
더 쏠리는지**가 진짜 판단 기준이다. qwen2 쪽 비중이 이미 크다면 이건 Qwen3만의 문제가
아니라 우리 방법론 일반의 특성일 수 있다.

사용 (서버에서, lxt/transformers 설치된 환경):
    python tools/diag_qwen3_relevance.py --model Qwen/Qwen3-8B --family qwen3
    python tools/diag_qwen3_relevance.py --model Qwen/Qwen2.5-7B-Instruct --family qwen2  # 대조군
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_position_relevance(model, input_ids, target_token_id):
    """compute_head_relevance()(attn_relevance.py)와 같은 backward-hook 메커니즘이지만,
    key_spans로 group-sum하지 않고 (num_layers, seq_len) 형태로 **position별** relevance를
    남긴다 (head 축은 여기서 바로 합산 — head별 분포는 이 진단의 관심사가 아님).
    구현 세부(forward 훅 위치, out.attentions 대신 모듈 원본 텐서를 쓰는 이유 등)는
    attn_relevance.compute_head_relevance()의 주석 참고 — 동일 로직."""
    import torch

    embed = model.get_input_embeddings()
    with torch.no_grad():
        base_embeds = embed(input_ids)
    inputs_embeds = base_embeds.clone().detach().requires_grad_(True)

    seq_len = input_ids.shape[1]
    layers = model.model.layers
    num_layers = len(layers)
    pos_scores = torch.zeros(num_layers, seq_len)
    hooked = [False] * num_layers
    fwd_handles, bwd_handles = [], []

    def _make_fwd_hook(l):
        def _fwd(module, args, output):
            a = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if a is None or not a.requires_grad:
                return

            def _bwd(grad):
                rel = (a[0].detach().float() * grad[0].float()).clamp(min=0)  # [heads, q, k]
                pos_scores[l] = rel.sum(dim=(0, 1)).cpu()  # heads+query 합산 -> [k]
                hooked[l] = True
                return None

            bwd_handles.append(a.register_hook(_bwd))
        return _fwd

    for l, layer in enumerate(layers):
        fwd_handles.append(layer.self_attn.register_forward_hook(_make_fwd_hook(l)))

    try:
        out = model(inputs_embeds=inputs_embeds, output_attentions=True, use_cache=False)
        target_logit = out.logits[0, -1, target_token_id]
        model.zero_grad(set_to_none=True)
        if inputs_embeds.grad is not None:
            inputs_embeds.grad = None
        target_logit.backward()
    finally:
        for h in fwd_handles:
            h.remove()
        for h in bwd_handles:
            h.remove()

    n_hooked = sum(hooked)
    if n_hooked != num_layers:
        missing = [i for i, v in enumerate(hooked) if not v]
        raise RuntimeError(
            f"relevance가 {n_hooked}/{num_layers} 레이어에서만 계산됨 (누락: {missing[:10]})"
        )

    return pos_scores  # [num_layers, seq_len]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--family", required=True, choices=["qwen2", "llama", "qwen3"])
    p.add_argument("--four_bit", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import torch
    from attn_relevance import load_model_for_relevance
    from dataset import build_phase0_batch

    print(f"[diag] loading {args.model} (family={args.family}) ...")
    model, tok, dtype_name = load_model_for_relevance(
        args.model, four_bit=args.four_bit, device=args.device, model_family=args.family,
    )

    batch = build_phase0_batch(tok, device=args.device, limit=1)
    ex = batch[0]["external"]  # data_inj span + exec_target(tool-call 토큰) 있는 버전

    pos_scores = compute_position_relevance(model, ex.input_ids, ex.exec_target)
    per_position = pos_scores.sum(dim=0)  # 레이어 합산 -> [seq_len]
    total = per_position.sum().item()

    data_inj = set(ex.spans.get("data_inj", []))
    pos0 = per_position[0].item()
    inj_mass = sum(per_position[i].item() for i in data_inj)

    print(f"\nmodel={args.model}  family={args.family}  seq_len={pos_scores.shape[1]}  "
          f"total_relevance={total:.4f}")
    print(f"position 0 비중: {pos0 / total * 100:.2f}%  ({pos0:.4f}/{total:.4f})")
    print(f"data_inj span 비중: {inj_mass / total * 100:.2f}%  "
          f"({inj_mass:.4f}/{total:.4f}, {len(data_inj)}개 토큰)")

    topk = torch.topk(per_position, min(10, per_position.numel()))
    print("\n상위 10개 position (index: 비중%):")
    for val, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        tag = " <- data_inj" if idx in data_inj else (" <- position 0" if idx == 0 else "")
        print(f"  {idx:5d}: {val / total * 100:6.2f}%{tag}")


if __name__ == "__main__":
    main()
