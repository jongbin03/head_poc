"""
attn_relevance.py

Atlas (NeurIPS'25) 의 AttnLRP 기반 head relevance 산출 방식을 lxt(pip install lxt)로
재구현한다. Atlas 원 코드(preprocessing/filter_heads_composition.py)의 핵심은:

    rel = attn_weights * attn_weights.grad     # post-softmax, per-head, per-(i,j)
    ρ^h_j = Σ_i ReLU(rel)[i, j]                 # Eq. 7
    score(J) = Σ_{j in J} ρ^h_j                 # Eq. 8, J = 관심있는 key-token 그룹

를 그대로 따르되, `self_attn.softmax`라는 커스텀 서브모듈을 만들 필요 없이
`output_attentions=True` (+ `attn_implementation="eager"`) 로 얻은 attention tensor에
바로 retain_grad()를 걸어서 구현한다.

⚠️ gradient checkpointing을 켜면 (`model.gradient_checkpointing_enable()`) 이 함수의
attention tensor는 backward 시점에 재계산된 별개의 tensor가 되어 `.grad`가 채워지지
않는다. head-level relevance가 필요한 이 모듈에서는 checkpointing을 반드시 꺼둘 것.
(embedding-level relevance만 필요하면 checkpointing을 켜도 무방 — README 참고)
"""
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer


def load_model_for_relevance(
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    four_bit: bool = False,
    device: str = "cuda",
    model_family: str = "qwen2",
    dtype: str = "auto",
):
    """
    model_family: "qwen2" | "llama"  (lxt가 공식 지원하는 아키텍처만)
    dtype: "auto" | "bf16" | "fp16" | "fp32" — runtime_env.resolve_dtype 참고.
           Turing(Titan RTX 등)에는 bf16이 없어 auto가 fp16으로 내려간다.
           ⚠️ fp16은 loss scaling이 없어 아래 backward에서 relevance가 NaN이 될 수 있다.
           호출자는 `runtime_env.has_nonfinite`로 검사하고 OOM 스킵과 구분해 카운트할 것.
    checkpointing은 여기서 켜지 않는다 — head-level relevance와 상극이기 때문.

    반환: (model, tokenizer, dtype_name) — dtype_name은 "auto"가 실제로 무엇으로
    해결됐는지이며, 호출자가 env.json에 기록해야 한다.
    """
    from lxt.efficient import monkey_patch

    from runtime_env import resolve_dtype

    if model_family == "qwen2":
        from transformers.models.qwen2 import modeling_qwen2 as modeling_mod
        model_cls = modeling_mod.Qwen2ForCausalLM
    elif model_family == "llama":
        from transformers.models.llama import modeling_llama as modeling_mod
        model_cls = modeling_mod.LlamaForCausalLM
    else:
        raise ValueError(f"unsupported model_family={model_family}")

    monkey_patch(modeling_mod, verbose=False)

    torch_dtype, dtype_name = resolve_dtype(dtype, device)

    kwargs = dict(torch_dtype=torch_dtype, device_map=device, attn_implementation="eager")
    if four_bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype
        )

    model = model_cls.from_pretrained(model_path, **kwargs)
    model.eval()  # gradient checkpointing 쓰지 않으므로 train() 불필요
    for p in model.parameters():
        p.requires_grad_(False)

    tok = AutoTokenizer.from_pretrained(model_path)
    return model, tok, dtype_name


def _clamp_pos(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(min=0)


@torch.enable_grad()
def compute_head_relevance(
    model,
    input_ids: torch.Tensor,
    target_token_id: int,
    key_spans: Dict[str, List[int]],
) -> Dict[str, torch.Tensor]:
    """
    한 샘플에 대해 (layer, head)별 relevance를 그룹별로 계산.

    key_spans: {"data_benign": [...], "data_inj": [...], ...} 처럼
               관심있는 key(token) position 목록. dataset.py의 IPIExample.spans를
               그대로 넣으면 된다 (system/question/assistant_prefix 등도 같이 들어
               있어도 무방 — 여기서 쓰는 것만 골라서 aggregate 함).

    반환: {"data_benign": Tensor[num_layers, num_heads], "data_inj": Tensor[...], ...}
    """
    embed = model.get_input_embeddings()
    with torch.no_grad():
        base_embeds = embed(input_ids)
    inputs_embeds = base_embeds.clone().detach().requires_grad_(True)

    out = model(inputs_embeds=inputs_embeds, output_attentions=True, use_cache=False)
    attn_maps = out.attentions  # tuple(num_layers) of [batch, heads, seq_i, seq_j]

    for a in attn_maps:
        if a.requires_grad:
            a.retain_grad()

    target_logit = out.logits[0, -1, target_token_id]

    model.zero_grad(set_to_none=True)
    if inputs_embeds.grad is not None:
        inputs_embeds.grad = None
    target_logit.backward()

    num_layers = len(attn_maps)
    num_heads = attn_maps[0].shape[1]
    group_scores = {g: torch.zeros(num_layers, num_heads) for g in key_spans}

    for l, a in enumerate(attn_maps):
        if a.grad is None:
            # checkpointing이 켜져 있거나 그래프가 끊긴 경우 — 여기 걸리면 설정을 점검할 것
            continue
        rel = _clamp_pos(a[0].float() * a.grad[0].float())  # [heads, seq_i, seq_j]
        for g, positions in key_spans.items():
            if len(positions) == 0:
                continue
            group_scores[g][l] = rel[:, :, positions].sum(dim=(1, 2)).cpu()

    return group_scores


def relevance_for_example(model, example, groups: Optional[List[str]] = None):
    """
    dataset.IPIExample 하나에 대해, read/internal/external 타깃 중 example.meta에
    맞는 것 하나에 대한 relevance를 계산하는 편의 함수.
    (read용과 exec용은 target_token_id가 다르므로 호출자가 두 번 불러야 함 —
    run_pipeline.py 참고)
    """
    groups = groups or list(example.spans.keys())
    key_spans = {g: example.spans.get(g, []) for g in groups}
    return key_spans
