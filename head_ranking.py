"""
head_ranking.py

attn_relevance.py 에서 나온 샘플별 relevance를 데이터셋 전체에 대해 평균 내고,
Atlas Eq.(9) 스타일로 top-K head를 뽑은 뒤, read / internal / external 세 집합
사이의 overlap(Jaccard)을 계산한다. 교수님 질문("read와 instruction-following이
구분되는가?")에 대한 1차 정량적 답이 여기서 나온다.
"""
import random
from typing import Dict, List, Optional, Tuple
import torch


def aggregate_scores(list_of_group_scores: List[Dict[str, torch.Tensor]], group: str) -> torch.Tensor:
    """여러 샘플의 group_scores 리스트에서 특정 그룹(예: 'data_inj')만 평균."""
    stacked = torch.stack([gs[group] for gs in list_of_group_scores if group in gs])
    return stacked.mean(dim=0)  # [num_layers, num_heads]


def normalize_score(score: torch.Tensor) -> torch.Tensor:
    """
    relevance를 전체 합=1로 정규화한다.

    왜 필요한가: internal/external은 프롬프트 길이와 타깃 토큰이 달라 relevance의
    절대 스케일이 다르다. 정규화 없이 `internal_score + external_score`로 합치면
    스케일이 큰 쪽이 control head 랭킹을 독점해서, "양쪽 모두에서 상위인 head"를
    찾겠다는 원래 의도가 깨진다.
    """
    total = score.sum()
    if total <= 0:
        return score
    return score / total


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


def expected_jaccard_by_chance(k: int, num_layers: int, num_heads: int) -> float:
    """
    두 독립적인(무작위) top-k 부분집합이 우연히 겹칠 것으로 기대되는 Jaccard (근사치).

    N=num_layers*num_heads개 슬롯에서 무작위로 뽑은 두 k-부분집합의 기대 교집합
    크기는 k^2/N (비복원추출 하이퍼기하분포 기댓값). Jaccard는 비율의 기댓값이라
    E[X]/E[Y]로 정확히 안 나오지만(E[X/Y] != E[X]/E[Y]), E[교집합]과
    E[합집합]=2k-E[교집합]을 그대로 대입한 근사값을 쓴다 — review-2026-07-29.md
    3-2절에서 이미 이 방식으로 "관측 jaccard가 우연 대비 몇 배인가"를 계산했으므로
    기존 수치와 일관되게 맞춘다.
    """
    n_slots = num_layers * num_heads
    if n_slots <= 0 or k <= 0:
        return 0.0
    expected_overlap = (k * k) / n_slots
    expected_union = 2 * k - expected_overlap
    if expected_union <= 0:
        return 0.0
    return expected_overlap / expected_union


def random_heads(
    num_layers: int, num_heads: int, n: int, seed: Optional[int] = None
) -> List[Tuple[int, int]]:
    """전체 (layer, head) 슬롯 중 무작위로 n개를 비복원 추출 (랜덤 head 기준선용).
    topk_heads()와 같은 (layer, head) 튜플 리스트 형태라 sweep_knockout에 그대로 넘길 수 있다."""
    rng = random.Random(seed)
    all_slots = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    n = min(n, len(all_slots))
    return rng.sample(all_slots, n)


def summarize_overlap(
    read_score: torch.Tensor,
    internal_score: torch.Tensor,
    external_score: torch.Tensor,
    k: int = 20,
) -> dict:
    read_heads = topk_heads(read_score, k)
    internal_heads = topk_heads(internal_score, k)
    external_heads = topk_heads(external_score, k)
    num_layers, num_heads = read_score.shape

    return {
        "top_k": k,
        "read_heads": read_heads,
        "internal_heads": internal_heads,
        "external_heads": external_heads,
        "jaccard_read_internal": jaccard(read_heads, internal_heads),
        "jaccard_read_external": jaccard(read_heads, external_heads),
        "jaccard_internal_external": jaccard(internal_heads, external_heads),
        # review-2026-07-29.md 3-2: 우연으로 기대되는 jaccard 대비 관측값이 몇 배인지
        # 보려면 이 값과 위 세 jaccard_* 값을 나누면 된다.
        "jaccard_chance_at_k": expected_jaccard_by_chance(k, num_layers, num_heads),
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
