from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

import numpy as np

from mcp_router.config import RouterConfig
from mcp_router.registry.schema import RegisteredServer
from mcp_router.retrieval.embedder import OpenAIEmbedder
from mcp_router.retrieval.tool_index import ToolIndex


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (mat / norms).astype(np.float32)


class IndexRegistry:
    """Loads on-disk index; supports hot registration by append + atomic swap."""

    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.index_dir = Path(cfg.index_dir)

    def load_index(self) -> ToolIndex:
        return ToolIndex.load(self.index_dir)

    def persist_index(
        self,
        servers: List[RegisteredServer],
        server_desc: np.ndarray,
        server_sum: np.ndarray,
        tool_emb: np.ndarray,
        tool_srv: np.ndarray,
        tool_loc: np.ndarray,
    ) -> None:
        out = Path(self.cfg.index_dir)
        tmp = Path(str(out) + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)

        meta_servers = [s.model_dump() for s in servers]
        with open(tmp / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"version": 1, "servers": meta_servers}, f, ensure_ascii=False)

        np.save(tmp / "server_description_embeddings.npy", server_desc.astype(np.float32))
        np.save(tmp / "server_summary_embeddings.npy", server_sum.astype(np.float32))
        np.save(tmp / "tool_embeddings.npy", tool_emb.astype(np.float32))
        np.save(tmp / "tool_server_indices.npy", tool_srv.astype(np.int64))
        np.save(tmp / "tool_local_indices.npy", tool_loc.astype(np.int64))

        bak = out.with_name(out.name + ".bak")
        if out.exists():
            if bak.exists():
                shutil.rmtree(bak)
            shutil.move(str(out), str(bak))
        shutil.move(str(tmp), str(out))

    def register_server(self, server: RegisteredServer, embedder: OpenAIEmbedder) -> ToolIndex:
        """Append one server; keeps existing vectors, only embeds the new server and its tools."""
        current = self.load_index()
        names = {s.name for s in current.servers}
        if server.name in names:
            raise ValueError(f"Server already registered: {server.name}")

        servers = list(current.servers) + [server]
        ddir = self.index_dir
        s_desc = np.load(ddir / "server_description_embeddings.npy")
        s_sum = np.load(ddir / "server_summary_embeddings.npy")
        t_emb = np.load(ddir / "tool_embeddings.npy")
        t_srv = np.load(ddir / "tool_server_indices.npy")
        t_loc = np.load(ddir / "tool_local_indices.npy")

        new_si = len(current.servers)
        nd = np.array(
            [embedder.embed(server.description or server.name)],
            dtype=np.float32,
        )
        ns = np.array(
            [embedder.embed(server.summary or server.description or server.name)],
            dtype=np.float32,
        )
        nd = _l2_normalize_rows(nd)
        ns = _l2_normalize_rows(ns)
        s_desc = np.vstack([s_desc.astype(np.float32), nd])
        s_sum = np.vstack([s_sum.astype(np.float32), ns])

        if server.tools:
            t_texts = [t.description or t.name for t in server.tools]
            new_te = np.array(embedder.embed_batch(t_texts), dtype=np.float32)
            new_te = _l2_normalize_rows(new_te)
            new_srv = np.full(new_te.shape[0], new_si, dtype=np.int64)
            new_loc = np.arange(new_te.shape[0], dtype=np.int64)
            t_emb = np.vstack([t_emb.astype(np.float32), new_te])
            t_srv = np.concatenate([t_srv, new_srv])
            t_loc = np.concatenate([t_loc, new_loc])

        self.persist_index(servers, s_desc, s_sum, t_emb, t_srv, t_loc)
        return self.load_index()
