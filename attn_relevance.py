"""
attn_relevance.py

Atlas (NeurIPS'25) 의 AttnLRP 기반 head relevance 산출 방식을 lxt(pip install lxt)로
재구현한다. Atlas 원 코드(preprocessing/filter_heads_composition.py)의 핵심은:

    rel = attn_weights * attn_weights.grad     # post-softmax, per-head, per-(i,j)
    ρ^h_j = Σ_i ReLU(rel)[i, j]                 # Eq. 7
    score(J) = Σ_{j in J} ρ^h_j                 # Eq. 8, J = 관심있는 key-token 그룹

를 그대로 따르되, `self_attn.softmax`라는 커스텀 서브모듈을 만들 필요 없이
`output_attentions=True` (+ `attn_implementation="eager"`) 로 얻은 attention tensor에
backward 훅을 걸어서 구현한다. (이전 판은 `retain_grad()`로 모든 레이어의 gradient를
backward가 끝날 때까지 통째로 들고 있었다 — 훅으로 바꿔 gradient가 계산되는 즉시
축약·전송하고 버림으로써 peak 메모리를 절반으로 줄인다. 2026-08-21 status 문서 참고.)

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
    device_map: Optional[str] = None,
):
    """
    model_family: "qwen2" | "llama"  (lxt가 공식 지원하는 아키텍처만)
    dtype: "auto" | "bf16" | "fp16" | "fp32" — runtime_env.resolve_dtype 참고.
           실측으로 CUDA에서는 항상 bf16이 선택된다(fp16은 NaN으로 배제, 1.3절).
    device: **입력 텐서를 올릴 device.** device_map="auto"로 모델을 쪼개도 입력은
           첫 device(보통 cuda:0)에 있어야 하므로, 모델 배치와 분리해서 받는다.
    device_map: 모델 가중치 배치. None이면 `device`를 그대로 쓴다(기존 동작).
           **"auto"를 주면 여러 GPU에 레이어를 분산**한다 — 32B처럼 단일 카드에
           안 들어가는 모델용. 이때 `device`는 "cuda:0"으로 두는 게 안전하다.
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

    resolved_device_map = device_map or device
    # dtype 판정에는 **device_map**을 넘긴다. "auto"를 device처럼 취급해 fp32로 떨어지면
    # 분산 로드의 목적(메모리 절감) 자체가 무너지기 때문 (runtime_env.resolve_dtype 주석 참고).
    torch_dtype, dtype_name = resolve_dtype(dtype, resolved_device_map)

    kwargs = dict(torch_dtype=torch_dtype, device_map=resolved_device_map,
                  attn_implementation="eager")
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

    num_layers = len(attn_maps)
    num_heads = attn_maps[0].shape[1]
    group_scores = {g: torch.zeros(num_layers, num_heads) for g in key_spans}
    hooked = [False] * num_layers

    def _make_hook(l, a):
        def _hook(grad):
            # a는 어차피 autograd가 그래프 안에 들고 있으므로 detach가 메모리를 더 쓰지 않음.
            # rel을 [heads] 그룹 스칼라로 즉시 축약해 CPU로 보내고, grad 자체는 리턴 없이 버림
            # (return None -> 훅이 끝나면 이 grad 텐서는 즉시 해제됨).
            rel = (a[0].detach().float() * grad[0].float()).clamp(min=0)  # [heads, seq_i, seq_j]
            for g, positions in key_spans.items():
                if len(positions) == 0:
                    continue
                group_scores[g][l] = rel[:, :, positions].sum(dim=(1, 2)).cpu()
            hooked[l] = True
            return None
        return _hook

    handles = []
    for l, a in enumerate(attn_maps):
        if a.requires_grad:
            handles.append(a.register_hook(_make_hook(l, a)))

    target_logit = out.logits[0, -1, target_token_id]

    model.zero_grad(set_to_none=True)
    if inputs_embeds.grad is not None:
        inputs_embeds.grad = None
    target_logit.backward()

    for h in handles:
        h.remove()

    # hooked[l] == False인 레이어는 checkpointing이 켜져 있거나 그래프가 끊긴 경우 —
    # 여기 걸리면 설정을 점검할 것 (group_scores는 0으로 남아 조용히 통과한다).

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
