from __future__ import annotations

from typing import List, Tuple

from mcp_router.config import RouterConfig


def adaptive_top_k(
    scores: List[float],
    cfg: RouterConfig,
) -> int:
    """
    Choose how many tools to return from a sorted-descending score list.
    Large gap between rank-1 and rank-2 => return fewer tools.
    """
    if not scores:
        return cfg.adaptive_min_tools
    n = len(scores)
    if n == 1:
        return 1
    top = scores[0]
    second = scores[1]
    gap = top - second
    if gap >= cfg.adaptive_gap_confident:
        return cfg.adaptive_min_tools
    # More ambiguity => return more candidates (capped)
    if gap >= cfg.adaptive_gap_confident * 0.5:
        return min(3, n, cfg.adaptive_max_tools)
    return min(cfg.adaptive_max_tools, n)


def take_k_pairs(pairs: List[Tuple[int, float]], k: int) -> List[Tuple[int, float]]:
    return pairs[:k] if k > 0 else []
