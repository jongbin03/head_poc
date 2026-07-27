"""
head_ranking.py

attn_relevance.py 에서 나온 샘플별 relevance를 데이터셋 전체에 대해 평균 내고,
Atlas Eq.(9) 스타일로 top-K head를 뽑은 뒤, read / internal / external 세 집합
사이의 overlap(Jaccard)을 계산한다. 교수님 질문("read와 instruction-following이
구분되는가?")에 대한 1차 정량적 답이 여기서 나온다.
"""
from typing import Dict, List, Tuple
import torch


def aggregate_scores(list_of_group_scores: List[Dict[str, torch.Tensor]], group: str) -> torch.Tensor:
    """여러 샘플의 group_scores 리스트에서 특정 그룹(예: 'data_inj')만 평균."""
    stacked = torch.stack([gs[group] for gs in list_of_group_scores if group in gs])
    return stacked.mean(dim=0)  # [num_layers, num_heads]


def topk_heads(score: torch.Tensor, k: int) -> List[Tuple[int, int]]:
    """score: [num_layers, num_heads] -> top-k (layer, head) 인덱스 리스트 (내림차순)."""
    flat = score.flatten()
    k = min(k, flat.numel())
    top_vals, top_idx = torch.topk(flat, k)
    num_heads = score.shape[1]
    return [(int(idx // num_heads), int(idx % num_heads)) for idx in top_idx.tolist()]


def jaccard(set_a: List[Tuple[int, int]], set_b: List[Tuple[int, int]]) -> float:
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def summarize_overlap(
    read_score: torch.Tensor,
    internal_score: torch.Tensor,
    external_score: torch.Tensor,
    k: int = 20,
) -> dict:
    read_heads = topk_heads(read_score, k)
    internal_heads = topk_heads(internal_score, k)
    external_heads = topk_heads(external_score, k)

    return {
        "top_k": k,
        "read_heads": read_heads,
        "internal_heads": internal_heads,
        "external_heads": external_heads,
        "jaccard_read_internal": jaccard(read_heads, internal_heads),
        "jaccard_read_external": jaccard(read_heads, external_heads),
        "jaccard_internal_external": jaccard(internal_heads, external_heads),
        # idea1의 "control head"는 internal/external 양쪽에서 다 상위권인 head들
        "control_heads_both": sorted(set(internal_heads) & set(external_heads)),
        # read/control 어느 쪽에도 강하게 안 걸치는지 확인용 (dual-use head 후보)
        "dual_use_candidates": sorted(
            (set(read_heads) & set(internal_heads)) | (set(read_heads) & set(external_heads))
        ),
    }


def plot_functional_map(
    read_score: torch.Tensor,
    internal_score: torch.Tensor,
    external_score: torch.Tensor,
    save_path: str = "functional_map.png",
    top_k: int = 40,
):
    """Atlas Fig.1 / Fig.3 스타일의 layer x head scatter. read=회색, internal=파랑,
    external=주황, 둘 다 top-k인 head(=control head 후보)는 빨강으로 강조."""
    import matplotlib.pyplot as plt

    num_layers, num_heads = read_score.shape
    read_top = set(topk_heads(read_score, top_k))
    internal_top = set(topk_heads(internal_score, top_k))
    external_top = set(topk_heads(external_score, top_k))
    control_both = internal_top & external_top

    fig, ax = plt.subplots(figsize=(7, 8))
    for l in range(num_layers):
        for h in range(num_heads):
            pt = (l, h)
            if pt in control_both:
                color, size, z = "red", 40, 3
            elif pt in internal_top:
                color, size, z = "tab:blue", 25, 2
            elif pt in external_top:
                color, size, z = "tab:orange", 25, 2
            elif pt in read_top:
                color, size, z = "tab:green", 15, 1
            else:
                color, size, z = "lightgray", 4, 0
            ax.scatter(h, l, c=color, s=size, zorder=z)

    ax.set_xlabel("Head ID")
    ax.set_ylabel("Layer ID")
    ax.set_title("Read vs Control head map (red = internal ∩ external top-K)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
