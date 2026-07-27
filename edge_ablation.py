"""
edge_ablation.py

순수 forward-pass 개입 (gradient 불필요, 기존 계획의 "Phase 1: Edge Blocking").
head_ranking.py에서 나온 top control head 후보들에 대해, D_inj 쪽 attention edge만
pre-softmax에서 -inf 처리(=edge knockout, Cutting-off-the-Head §4.2 방식)한 뒤:

  1. 악성 토큰(y_exec) 확률이 떨어지는지 (ASR proxy 감소)
  2. 정상 read 토큰(y_read) 확률은 유지되는지 (utility 보존)

를 측정한다. lxt monkey-patch나 backward가 전혀 필요 없으므로 4bit 양자화 모델에도
그대로 쓸 수 있다 (attn_relevance.py와 달리 이 파일은 4bit에서도 안전).
"""
from contextlib import contextmanager
from typing import Dict, List, Optional
import torch


def _build_head_key_mask(num_heads: int, seq_len: int, heads: List[int], key_positions: List[int]) -> torch.Tensor:
    mask = torch.zeros(num_heads, seq_len, dtype=torch.bool)
    if heads and key_positions:
        # 주의: mask[heads][:, key_positions] = True 는 fancy indexing이 만든 '복사본'에
        # 쓰는 것이라 원본 mask가 조용히 안 바뀌는 전형적인 PyTorch/numpy 함정이다.
        # 반드시 단일 __setitem__ 호출(advanced indexing)로 broadcasting해서 써야 한다.
        head_idx = torch.tensor(heads, dtype=torch.long).unsqueeze(1)     # [H, 1]
        key_idx = torch.tensor(key_positions, dtype=torch.long).unsqueeze(0)  # [1, K]
        mask[head_idx, key_idx] = True
    return mask


def _make_knockout_forward(modeling_mod, knockout_map: Dict[int, List[int]], key_positions: List[int]):
    """
    knockout_map: {layer_idx: [head_idx, ...]} — 이 (layer, head)들에 대해서만
    key_positions 열을 -inf 처리한다. 나머지 layer/head는 정상 eager attention과
    동일하게 동작 (수식은 HF의 표준 eager_attention_forward를 그대로 재현).
    """
    # GQA 대응: num_key_value_heads < num_attention_heads인 모델(Qwen2.5 등)은
    # key/value를 query의 head 수에 맞춰 repeat해야 matmul이 성립한다 (원본 HF
    # eager_attention_forward와 동일한 처리). modeling_mod에서 직접 가져온다 —
    # qwen2/llama 각 모듈이 자기 repeat_kv를 갖고 있으므로 호출자가 넘긴 모듈과
    # 반드시 짝을 맞춰야 한다.
    repeat_kv = modeling_mod.repeat_kv

    def knockout_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        layer_idx = getattr(module, "layer_idx", None)

        n_rep = getattr(module, "num_key_value_groups", 1)
        if n_rep != 1:
            key = repeat_kv(key, n_rep)
            value = repeat_kv(value, n_rep)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn_weights = attn_weights + causal_mask

        heads_to_knock = knockout_map.get(layer_idx, []) if layer_idx is not None else []
        if heads_to_knock and key_positions:
            num_heads = attn_weights.shape[1]
            seq_len = attn_weights.shape[-1]
            head_key_mask = _build_head_key_mask(num_heads, seq_len, heads_to_knock, key_positions)
            head_key_mask = head_key_mask.to(attn_weights.device)
            attn_weights = attn_weights.masked_fill(head_key_mask[None, :, None, :], float("-inf"))

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return knockout_forward


@contextmanager
def edge_knockout(modeling_mod, knockout_map: Dict[int, List[int]], key_positions: List[int]):
    """
    modeling_mod: 예) transformers.models.qwen2.modeling_qwen2
    사용 예:
        with edge_knockout(modeling_qwen2, {12: [3, 7]}, example.spans["data_inj"]):
            out = model(input_ids=example.input_ids)

    ⚠️ attn_implementation="eager"로 모델을 로드한 경우, HF의 attention 선택 코드는
        attention_interface: Callable = eager_attention_forward   # 모듈 전역 함수 직접 참조
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    처럼 되어 있어서, eager일 때는 ALL_ATTENTION_FUNCTIONS 딕셔너리를 아예 조회하지 않고
    모듈 전역 이름 `eager_attention_forward`를 직접 참조한다. 따라서 knockout이 실제로
    적용되려면 이 모듈 전역 함수 자체를 바꿔치기해야 한다 (lxt의 monkey_patch가 하는 방식과
    동일). ALL_ATTENTION_FUNCTIONS도 혹시 sdpa/flash 등 다른 구현을 쓸 경우를 대비해 같이
    패치해두지만, eager 경로에서 실제로 knockout을 작동시키는 건 아래 eager_attention_forward
    속성 패치다.
    """
    new_forward = _make_knockout_forward(modeling_mod, knockout_map, key_positions)

    original_attr_map = {}
    for name in ("eager_attention_forward",):
        if hasattr(modeling_mod, name):
            original_attr_map[name] = getattr(modeling_mod, name)
            setattr(modeling_mod, name, new_forward)

    original_registry = dict(modeling_mod.ALL_ATTENTION_FUNCTIONS)
    modeling_mod.ALL_ATTENTION_FUNCTIONS = {k: new_forward for k in original_registry}

    try:
        yield
    finally:
        modeling_mod.ALL_ATTENTION_FUNCTIONS = original_registry
        for name, fn in original_attr_map.items():
            setattr(modeling_mod, name, fn)


@torch.no_grad()
def next_token_prob(model, input_ids: torch.Tensor, target_token_id: int) -> float:
    out = model(input_ids=input_ids, use_cache=False)
    probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
    return probs[target_token_id].item()


@torch.no_grad()
def sweep_knockout(
    model,
    modeling_mod,
    example,  # dataset.IPIExample (external/internal 버전)
    read_example,  # 같은 데이터의 read 버전 (tool_calling=False, question이 benign 질문)
    ranked_control_heads: List[tuple],  # head_ranking.topk_heads 결과, (layer, head) 내림차순
    ks: Optional[List[int]] = None,
) -> List[dict]:
    """
    Atlas Fig.2 스타일 sweep: ranked_control_heads의 상위 k개를 순서대로 늘려가며
    knockout했을 때 (악성 토큰 확률, 정상 read 토큰 확률)이 어떻게 변하는지 기록.
    """
    ks = ks if ks is not None else [0, 5, 10, 20, 40, 80]
    key_positions = example.spans.get("data_inj", [])

    results = []
    for k in ks:
        knockout_map: Dict[int, List[int]] = {}
        for layer, head in ranked_control_heads[:k]:
            knockout_map.setdefault(layer, []).append(head)

        with edge_knockout(modeling_mod, knockout_map, key_positions):
            asr_prob = next_token_prob(model, example.input_ids, example.exec_target)
        with edge_knockout(modeling_mod, knockout_map, key_positions):
            read_prob = next_token_prob(model, read_example.input_ids, read_example.read_target)

        results.append({"k": k, "malicious_token_prob": asr_prob, "read_token_prob": read_prob})

    return results
