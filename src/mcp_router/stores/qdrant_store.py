from __future__ import annotations

"""
Qdrant adapter (optional).

Install: pip install mcp-router[qdrant]

Use when you need multi-replica serving, filtering, or incremental upserts at scale.
The default on-disk numpy index is enough for ~few thousand tools (MCP-Zero scale).
"""

from typing import Any, Protocol, Sequence


class VectorStore(Protocol):
    def upsert_server_tools(self, payload: Any) -> None: ...

    def search(self, vector: Sequence[float], limit: int) -> list[Any]: ...


class QdrantStore:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "QdrantStore is a stub. Install qdrant-client and implement collection "
            "schema + upsert/search for your deployment."
        )
