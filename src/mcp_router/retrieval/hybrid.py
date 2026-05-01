from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from mcp_router.config import RouterConfig
from mcp_router.registry.schema import RegisteredServer
from mcp_router.retrieval.splade_lite import SpladeLiteRetriever
from mcp_router.retrieval.tokenization import tokenize


def build_tool_documents(servers: Sequence[RegisteredServer]) -> List[str]:
    """One BM25 document per tool: server context + tool name + description."""
    docs: List[str] = []
    for server in servers:
        header = f"{server.name} {server.summary} {server.description}"
        for tool in server.tools:
            docs.append(f"{header} {tool.name} {tool.description}")
    return docs


class HybridRetriever:
    def __init__(self, servers: Sequence[RegisteredServer], cfg: RouterConfig):
        self.servers = list(servers)
        self.cfg = cfg
        self._docs = build_tool_documents(self.servers)
        tokenized = [tokenize(d) for d in self._docs]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._splade = SpladeLiteRetriever(self._docs) if cfg.splade_enabled else None

    def bm25_rank(self, query: str, top_n: int) -> List[Tuple[int, float]]:
        if not self._bm25 or not query.strip():
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        out: List[Tuple[int, float]] = []
        for i in order[:top_n]:
            idx = int(i)
            out.append((idx, float(scores[idx])))
        return out

    def splade_rank(self, query: str, top_n: int) -> List[Tuple[int, float]]:
        if self._splade is None:
            return []
        return self._splade.rank(query, top_n)

    @staticmethod
    def rrf_fusion(
        ranked_lists: List[List[int]],
        k: int = 60,
        max_rank: int = 500,
        weights: List[float] | None = None,
    ) -> Dict[int, float]:
        """Reciprocal Rank Fusion over multiple ordered lists of tool row ids."""
        scores: Dict[int, float] = {}
        if weights is None:
            weights = [1.0] * len(ranked_lists)
        if len(weights) != len(ranked_lists):
            raise ValueError("weights length must match ranked_lists length")
        for list_idx, lst in enumerate(ranked_lists):
            w = float(weights[list_idx])
            for rank, tool_row in enumerate(lst, start=1):
                if rank > max_rank:
                    break
                scores[tool_row] = scores.get(tool_row, 0.0) + w * (1.0 / (k + rank))
        return scores
