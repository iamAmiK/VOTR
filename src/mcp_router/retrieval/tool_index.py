from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from mcp_router.registry.schema import RegisteredServer, RegisteredTool
from mcp_router.retrieval.overlap import build_overlap_groups


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


class ToolIndex:
    """
    Hierarchical tool index (MCP-Zero-aligned server stage):
    - Server routing: max(cos(q, desc_emb), cos(q, summary_emb)) per server.
    - Tool scoring: normalized tool embeddings + hierarchical combination score.

    Uses NumPy (not FAISS): server count is modest; tool scoring runs on the
    candidate subset after server filtering.
    """

    def __init__(
        self,
        servers: List[RegisteredServer],
        server_description_embeddings: np.ndarray,
        server_summary_embeddings: np.ndarray,
        tool_embeddings: np.ndarray,
        tool_server_indices: np.ndarray,
        tool_local_indices: np.ndarray,
    ):
        self.servers = servers
        self.dim = int(server_description_embeddings.shape[1])
        self._server_desc = _l2_normalize(server_description_embeddings.astype(np.float32))
        self._server_sum = _l2_normalize(server_summary_embeddings.astype(np.float32))
        self._tool_mat = _l2_normalize(tool_embeddings.astype(np.float32))
        self._tool_server_indices = tool_server_indices.astype(np.int64)
        self._tool_local_indices = tool_local_indices.astype(np.int64)
        self._capability_signatures, self._overlap_groups = build_overlap_groups(self.servers)

    @classmethod
    def load(cls, index_dir: Path) -> "ToolIndex":
        index_dir = Path(index_dir)
        meta_path = index_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing {meta_path}; run scripts/build_index.py first.")
        with open(meta_path, encoding="utf-8") as f:
            raw = json.load(f)
        servers = [RegisteredServer.model_validate(s) for s in raw["servers"]]
        s_desc = np.load(index_dir / "server_description_embeddings.npy")
        s_sum = np.load(index_dir / "server_summary_embeddings.npy")
        tool_emb = np.load(index_dir / "tool_embeddings.npy")
        tool_srv = np.load(index_dir / "tool_server_indices.npy")
        tool_loc = np.load(index_dir / "tool_local_indices.npy")
        return cls(servers, s_desc, s_sum, tool_emb, tool_srv, tool_loc)

    @staticmethod
    def _combine_server_scores(
        d1: np.ndarray,
        d2: np.ndarray,
        max_weight: float,
        mean_weight: float,
    ) -> np.ndarray:
        mx = np.maximum(d1, d2)
        mn = (d1 + d2) / 2.0
        total = max_weight + mean_weight
        if total <= 0:
            return mx
        return ((max_weight * mx) + (mean_weight * mn)) / total

    def search_servers(
        self,
        query_embedding: List[float],
        k: int,
        max_weight: float = 1.0,
        mean_weight: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([query_embedding], dtype=np.float32)
        q = _l2_normalize(q)
        d1 = (self._server_desc @ q.T).ravel()
        d2 = (self._server_sum @ q.T).ravel()
        scores = self._combine_server_scores(d1, d2, max_weight, mean_weight)
        k = min(k, scores.shape[0])
        if k <= 0:
            return np.array([]), np.array([])
        part = np.argpartition(-scores, kth=k - 1)[:k]
        part_sorted = part[np.argsort(-scores[part])]
        return scores[part_sorted], part_sorted.astype(np.int64)

    def tools_for_servers(self, server_indices: np.ndarray) -> np.ndarray:
        if server_indices.size == 0:
            return np.array([], dtype=np.int64)
        mask = np.isin(self._tool_server_indices, server_indices)
        return np.nonzero(mask)[0].astype(np.int64)

    def server_scores_for_indices(
        self,
        query_embedding: List[float],
        server_indices: np.ndarray,
        max_weight: float = 1.0,
        mean_weight: float = 0.0,
    ) -> np.ndarray:
        q = np.array([query_embedding], dtype=np.float32)
        q = _l2_normalize(q)
        d1 = (self._server_desc @ q.T).ravel()
        d2 = (self._server_sum @ q.T).ravel()
        full = self._combine_server_scores(d1, d2, max_weight, mean_weight)
        if server_indices.size == 0:
            return np.array([])
        return full[server_indices]

    def score_tools_hierarchical(
        self,
        tool_rows: np.ndarray,
        tool_query_embedding: List[float],
        server_intent_embedding: List[float],
        server_score_max_weight: float = 1.0,
        server_score_mean_weight: float = 0.0,
    ) -> np.ndarray:
        if tool_rows.size == 0:
            return np.array([])
        q_t = np.array([tool_query_embedding], dtype=np.float32)
        q_t = _l2_normalize(q_t)
        t = self._tool_mat[tool_rows]
        tool_sims = (t @ q_t.T).ravel()
        srv_idx = self._tool_server_indices[tool_rows]
        q_s = np.array([server_intent_embedding], dtype=np.float32)
        q_s = _l2_normalize(q_s)
        s1 = (self._server_desc[srv_idx] @ q_s.T).ravel()
        s2 = (self._server_sum[srv_idx] @ q_s.T).ravel()
        server_sims = self._combine_server_scores(
            s1,
            s2,
            server_score_max_weight,
            server_score_mean_weight,
        )
        return (server_sims * tool_sims) * np.maximum(server_sims, tool_sims)

    def get_tool_record(self, tool_row: int) -> Tuple[RegisteredServer, RegisteredTool]:
        si = int(self._tool_server_indices[tool_row])
        li = int(self._tool_local_indices[tool_row])
        server = self.servers[si]
        return server, server.tools[li]

    def all_tool_rows(self) -> np.ndarray:
        return np.arange(self._tool_server_indices.shape[0], dtype=np.int64)

    def capability_signature(self, tool_row: int) -> str:
        if tool_row < 0 or tool_row >= len(self._capability_signatures):
            return ""
        return self._capability_signatures[tool_row]

    def overlap_rows(self, tool_row: int) -> list[int]:
        sig = self.capability_signature(tool_row)
        if not sig:
            return []
        return list(self._overlap_groups.get(sig, []))
