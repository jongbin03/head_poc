"""
tools/diag_dtype.py

fp16에서 relevance가 NaN이 되는 원인을 특정하고, loss scaling으로 살릴 수 있는지 판정한다.

배경 (실측 2026-08-21, 서버 Titan RTX / Qwen2.5-0.5B / fp16):
  템플릿 3개 중 1개의 `external` 모드에서만 relevance가 NaN(140/336)이 됐다.
  `run_pipeline.py`에 가드가 없어 aggregate_scores 평균이 통째로 NaN이 되고,
  topk_heads가 점수 순서가 아닌 **인덱스 순서**((0,0),(0,1),...)를 반환해
  jaccard(*,external)이 0.000으로 찍혔다 — 에러도 경고도 없이.

가려야 할 것:
  (a) **overflow**  — a.grad가 fp16 최대(65504)를 넘어 inf가 되고, a*inf에서 0*inf=NaN.
                      → 해법은 loss를 **줄여서** backward (down-scaling).
  (b) **underflow** — a.grad가 fp16 최소 정규값(~6e-5) 밑으로 떨어져 0으로 뭉갬.
                      → 해법은 loss를 **키워서** backward (통상적 loss scaling).
  두 경우의 처방이 정반대라 반드시 먼저 구분해야 한다.

relevance는 gradient에 대해 1차 동차이고(ReLU·합산 모두 양의 스칼라배와 교환됨),
backward는 seed gradient에 선형이다. 따라서 타깃 로짓에 상수 c를 곱해 backward한 뒤
fp32로 올려서 c로 나누면 **랭킹도 절대값도 보존**된다 — c는 순전히 표현범위를 옮기는 용도다.

사용:
    python tools/diag_dtype.py                      # 기본: 0.5B, 템플릿 3개, fp16/fp32 대조
    python tools/diag_dtype.py --model ... --limit 5 --scales 1 0.01 100
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attn_relevance import load_model_for_relevance  # noqa: E402
from dataset import build_phase0_batch  # noqa: E402


@torch.enable_grad()
def relevance_with_scale(model, input_ids, target_token_id, positions, scale: float):
    """compute_head_relevance를 그대로 재현하되, backward 전에 타깃 로짓에 scale을 곱하고
    fp32로 올린 뒤 다시 나눈다. 중간 단계(attention grad)의 inf/nan도 같이 집계한다."""
    embed = model.get_input_embeddings()
    with torch.no_grad():
        base = embed(input_ids)
    inputs_embeds = base.clone().detach().requires_grad_(True)

    out = model(inputs_embeds=inputs_embeds, output_attentions=True, use_cache=False)
    attn_maps = out.attentions
    for a in attn_maps:
        if a.requires_grad:
            a.retain_grad()

    logit = out.logits[0, -1, target_token_id]
    model.zero_grad(set_to_none=True)
    (logit * scale).backward()

    stats = {
        "logit": float(logit.detach().float()),
        "grad_inf_layers": [], "grad_nan_layers": [],
        "grad_absmax": 0.0, "grad_absmin_nonzero": float("inf"),
        "grad_all_zero_layers": [],
    }
    num_layers = len(attn_maps)
    num_heads = attn_maps[0].shape[1]
    rel_sum = torch.zeros(num_layers, num_heads)

    for l, a in enumerate(attn_maps):
        if a.grad is None:
            continue
        g = a.grad[0]
        if torch.isinf(g).any():
            stats["grad_inf_layers"].append(l)
        if torch.isnan(g).any():
            stats["grad_nan_layers"].append(l)
        gf = g.float().abs()
        finite = gf[torch.isfinite(gf)]
        if finite.numel():
            stats["grad_absmax"] = max(stats["grad_absmax"], float(finite.max()))
            nz = finite[finite > 0]
            if nz.numel():
                stats["grad_absmin_nonzero"] = min(stats["grad_absmin_nonzero"], float(nz.min()))
            else:
                stats["grad_all_zero_layers"].append(l)
        # fp32로 올린 뒤 scale을 되돌린다 → scale과 무관하게 같은 값이 나와야 정상
        rel = (a[0].float() * a.grad[0].float()).clamp(min=0) / scale
        rel_sum[l] = rel[:, :, positions].sum(dim=(1, 2)).cpu()

    return rel_sum, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--family", default="qwen2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=3)
    # fp32를 먼저 돌려야 scale=1 결과가 rel_err의 기준으로 잡힌다 (순서 중요)
    p.add_argument("--dtypes", nargs="+", default=["fp32", "fp16"])
    p.add_argument("--scales", nargs="+", type=float, default=[1.0, 0.01, 100.0],
                   help="타깃 로짓에 곱할 상수. <1이면 overflow 회피, >1이면 underflow 회피.")
    p.add_argument("--mode", default="external", choices=["read", "internal", "external"])
    args = p.parse_args()

    ref = {}
    for dt in args.dtypes:
        # 프로세스 하나에서 모델을 두 번 로드하면 lxt가 "already patched" 경고를 낸다.
        # 패치는 멱등이라 결과에는 영향이 없지만, 엄밀히 보려면 dtype마다 따로 실행할 것.
        model, tok, name = load_model_for_relevance(
            args.model, device=args.device, model_family=args.family, dtype=dt)
        pairs = build_phase0_batch(tok, device=args.device, limit=args.limit)

        for scale in args.scales:
            print(f"\n===== dtype={name}  scale={scale:g} =====")
            for i, ex in enumerate(pairs):
                key = "read_clean" if args.mode == "read" else args.mode
                span = "data_benign" if args.mode == "read" else "data_inj"
                e = ex[key]
                target = e.read_target if args.mode == "read" else e.exec_target
                rel, st = relevance_with_scale(
                    model, e.input_ids, target, e.spans[span], scale)

                n_nan = int(torch.isnan(rel).sum())
                amin = st["grad_absmin_nonzero"]
                amin_s = "(all zero)" if amin == float("inf") else f"{amin:.3e}"
                mark = "   <<<< NaN" if n_nan else ""
                print(
                    f"  tmpl{i}: logit={st['logit']:+.3f}  rel_nan={n_nan:3d}  "
                    f"grad|max|={st['grad_absmax']:.3e}  grad|min>0|={amin_s}{mark}"
                )
                if st["grad_inf_layers"]:
                    print(f"          ⚠ grad inf at layers {st['grad_inf_layers']}  → OVERFLOW")
                if st["grad_nan_layers"]:
                    print(f"          ⚠ grad nan at layers {st['grad_nan_layers']}")
                if st["grad_all_zero_layers"]:
                    print(f"          ⚠ grad all-zero at layers {st['grad_all_zero_layers']}  → UNDERFLOW")

                # fp32/scale=1 결과를 기준으로 상대오차 — scaling이 값을 보존하는지 확인
                if name == "fp32" and scale == 1.0:
                    ref[i] = rel.clone()
                elif i in ref and torch.isfinite(rel).all():
                    base = ref[i]
                    den = base.abs().max().clamp(min=1e-12)
                    err = float((rel - base).abs().max() / den)
                    print(f"          rel_err vs fp32/scale=1: {err:.2e}")
        del model
        torch.cuda.empty_cache()

    print(
        "\n판정 가이드:\n"
        "  - `grad inf` 가 뜨고 scale<1에서 NaN이 사라지면 → OVERFLOW. down-scaling으로 fp16 사용 가능.\n"
        "  - `grad all-zero` 가 뜨고 scale>1에서 개선되면 → UNDERFLOW. up-scaling으로 사용 가능.\n"
        "  - 어떤 scale에서도 NaN이면 → fp16 포기. fp32 또는 4bit+fp32 compute로 갈 것.\n"
        "  - rel_err가 1e-2 이하이면 scaling이 값을 보존한 것 (랭킹은 그보다 더 안정적)."
    )


if __name__ == "__main__":
    main()
