#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_router.config import load_config  # noqa: E402
from mcp_router.retrieval.embedder import OpenAIEmbedder  # noqa: E402
from mcp_router.retrieval.hybrid import tokenize  # noqa: E402


DEFAULT_PREPARED = ROOT / "evaluation" / "results" / "livemcpbench" / "paper_aligned_exact_step_subset.json"
DEFAULT_CATALOG = ROOT / "data" / "catalog_subsets" / "livemcpbench.embedding.json"
DEFAULT_OUT = ROOT / "evaluation" / "results" / "livemcpbench" / "paper_faithful_tool_to_agent_eval.json"


@dataclass(frozen=True)
class CatalogEntity:
    entity_id: str
    entity_type: str
    owner_server: str | None
    name: str
    text: str
    embedding: np.ndarray


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


def recall_at(predicted: list[str], gold: str, k: int) -> float:
    return 1.0 if gold in predicted[:k] else 0.0


def average_precision_at(predicted: list[str], gold: str, k: int) -> float:
    try:
        rank = predicted[:k].index(gold) + 1
    except ValueError:
        return 0.0
    return 1.0 / rank


def ndcg_at(predicted: list[str], gold: str, k: int) -> float:
    try:
        rank = predicted[:k].index(gold) + 1
    except ValueError:
        return 0.0
    return 1.0 / math.log2(rank + 1.0)


def rrf_fusion(
    ranked_lists: list[list[int]],
    *,
    rrf_k: int,
    weights: list[float],
    max_rank: int,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    if len(ranked_lists) != len(weights):
        raise ValueError("ranked_lists and weights must have the same length")
    for list_idx, ranked in enumerate(ranked_lists):
        weight = float(weights[list_idx])
        for rank, row_idx in enumerate(ranked, start=1):
            if rank > max_rank:
                break
            scores[row_idx] = scores.get(row_idx, 0.0) + weight * (1.0 / (rrf_k + rank))
    return scores


def build_entities(catalog_path: Path) -> list[CatalogEntity]:
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    entities: list[CatalogEntity] = []
    for server in raw:
        server_name = server["name"]
        display_name = server.get("display_name", server_name)
        description = server.get("description", "")
        summary = server.get("summary", "") or description
        server_text = f"{server_name} {display_name} {summary} {description}".strip()
        entities.append(
            CatalogEntity(
                entity_id=f"agent::{server_name}",
                entity_type="agent",
                owner_server=server_name,
                name=server_name,
                text=server_text,
                embedding=np.array(server.get("summary_embedding") or server["description_embedding"], dtype=np.float32),
            )
        )
        for tool in server.get("tools", []):
            tool_name = tool["name"]
            tool_description = tool.get("description", "")
            parameter_text = " ".join(f"{k} {v}" for k, v in (tool.get("parameter") or {}).items())
            tool_text = (
                f"{server_name} {display_name} {tool_name} {tool_description} {parameter_text}"
            ).strip()
            entities.append(
                CatalogEntity(
                    entity_id=f"tool::{server_name}::{tool_name}",
                    entity_type="tool",
                    owner_server=server_name,
                    name=tool_name,
                    text=tool_text,
                    embedding=np.array(tool["description_embedding"], dtype=np.float32),
                )
            )
    return entities


class UnifiedToolAgentRetriever:
    def __init__(
        self,
        entities: list[CatalogEntity],
        *,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        rrf_k: int = 60,
    ):
        self.entities = entities
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self._entity_matrix = _normalize_rows(
            np.vstack([entity.embedding.astype(np.float32) for entity in entities])
        )
        docs = [entity.text for entity in entities]
        tokenized = [tokenize(doc) for doc in docs]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def rank_entities(self, query: str, query_embedding: list[float], top_n: int) -> list[tuple[int, float]]:
        ranked_lists: list[list[int]] = []
        weights: list[float] = []

        if self.dense_weight > 0:
            q = _normalize_rows(np.array([query_embedding], dtype=np.float32))
            dense_scores = (self._entity_matrix @ q.T).ravel()
            dense_order = np.argsort(-dense_scores)[:top_n]
            ranked_lists.append([int(i) for i in dense_order.tolist()])
            weights.append(self.dense_weight)

        if self.bm25_weight > 0 and self._bm25 is not None and query.strip():
            bm25_scores = self._bm25.get_scores(tokenize(query))
            bm25_order = np.argsort(-bm25_scores)[:top_n]
            ranked_lists.append([int(i) for i in bm25_order.tolist()])
            weights.append(self.bm25_weight)

        if not ranked_lists:
            return []

        fused = rrf_fusion(ranked_lists, rrf_k=self.rrf_k, weights=weights, max_rank=top_n)
        ordered = sorted(fused.items(), key=lambda item: -item[1])[:top_n]
        return [(idx, score) for idx, score in ordered]

    def collapse_to_unique_agents(
        self,
        ranked_entities: list[tuple[int, float]],
        *,
        k_agents: int,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        agents: list[str] = []
        trace: list[dict[str, Any]] = []
        seen: set[str] = set()

        for row_idx, score in ranked_entities:
            entity = self.entities[row_idx]
            if entity.entity_type == "agent":
                owner = entity.name
            elif entity.entity_type == "tool" and entity.owner_server:
                owner = entity.owner_server
            else:
                continue

            trace.append(
                {
                    "entity_id": entity.entity_id,
                    "entity_type": entity.entity_type,
                    "entity_name": entity.name,
                    "owner_server": owner,
                    "score": score,
                }
            )
            if owner not in seen:
                seen.add(owner)
                agents.append(owner)
                if len(agents) >= k_agents:
                    break
        return agents, trace


def build_retriever(catalog_path: Path, dense_weight: float, bm25_weight: float, rrf_k: int) -> UnifiedToolAgentRetriever:
    entities = build_entities(catalog_path)
    return UnifiedToolAgentRetriever(
        entities,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
        rrf_k=rrf_k,
    )


def evaluate(
    prepared_path: Path,
    catalog_path: Path,
    *,
    top_n: int,
    top_k: int,
    dense_weight: float,
    bm25_weight: float,
    rrf_k: int,
) -> dict[str, Any]:
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    cfg = load_config()
    embedder = OpenAIEmbedder(cfg)
    retriever = build_retriever(catalog_path, dense_weight=dense_weight, bm25_weight=bm25_weight, rrf_k=rrf_k)

    step_records: list[dict[str, Any]] = []
    for task in prepared:
        for step in task.get("steps", []):
            step_records.append(step)

    query_embeddings = embedder.embed_batch([step["step_query"] for step in step_records]) if step_records else []

    rows: list[dict[str, Any]] = []
    for step, query_embedding in zip(step_records, query_embeddings):
        query = step["step_query"]
        ranked_entities = retriever.rank_entities(query, query_embedding, top_n=top_n)
        predicted_servers, trace = retriever.collapse_to_unique_agents(ranked_entities, k_agents=top_k)

        rows.append(
            {
                "task_id": step["task_id"],
                "question": step["question"],
                "step_index": step["step_index"],
                "step_query": query,
                "gold_tool_name": step["gold_tool_name"],
                "gold_server_name": step["gold_server_name"],
                "predicted_servers_topk": predicted_servers,
                "ranked_entities_trace": trace,
                "recall@1": recall_at(predicted_servers, step["gold_server_name"], 1),
                "recall@3": recall_at(predicted_servers, step["gold_server_name"], 3),
                "recall@5": recall_at(predicted_servers, step["gold_server_name"], 5),
                "map@1": average_precision_at(predicted_servers, step["gold_server_name"], 1),
                "map@3": average_precision_at(predicted_servers, step["gold_server_name"], 3),
                "map@5": average_precision_at(predicted_servers, step["gold_server_name"], 5),
                "ndcg@1": ndcg_at(predicted_servers, step["gold_server_name"], 1),
                "ndcg@3": ndcg_at(predicted_servers, step["gold_server_name"], 3),
                "ndcg@5": ndcg_at(predicted_servers, step["gold_server_name"], 5),
            }
        )

    n = len(rows)
    return {
        "metadata": {
            "protocol": "paper-faithful-tool-to-agent-variant",
            "prepared_subset_path": str(prepared_path),
            "catalog_path": str(catalog_path),
            "num_step_queries": n,
            "top_n_entities": top_n,
            "top_k_agents": top_k,
            "dense_weight": dense_weight,
            "bm25_weight": bm25_weight,
            "rrf_k": rrf_k,
            "note": (
                "This implements the paper-style unified tool+agent retrieval and "
                "Algorithm 1 collapse over the locally available exact-step subset."
            ),
        },
        "summary": {
            "num_step_queries": n,
            "recall": {
                "@1": sum(r["recall@1"] for r in rows) / n if n else 0.0,
                "@3": sum(r["recall@3"] for r in rows) / n if n else 0.0,
                "@5": sum(r["recall@5"] for r in rows) / n if n else 0.0,
            },
            "mAP": {
                "@1": sum(r["map@1"] for r in rows) / n if n else 0.0,
                "@3": sum(r["map@3"] for r in rows) / n if n else 0.0,
                "@5": sum(r["map@5"] for r in rows) / n if n else 0.0,
            },
            "nDCG": {
                "@1": sum(r["ndcg@1"] for r in rows) / n if n else 0.0,
                "@3": sum(r["ndcg@3"] for r in rows) / n if n else 0.0,
                "@5": sum(r["ndcg@5"] for r in rows) / n if n else 0.0,
            },
        },
        "results": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the paper-faithful Tool-to-Agent retrieval variant on LiveMCPBench.")
    ap.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--dense-weight", type=float, default=1.0)
    ap.add_argument("--bm25-weight", type=float, default=1.0)
    ap.add_argument("--rrf-k", type=int, default=60)
    args = ap.parse_args()

    result = evaluate(
        args.prepared,
        args.catalog,
        top_n=args.top_n,
        top_k=args.top_k,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        rrf_k=args.rrf_k,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved eval results to: {args.out}")


if __name__ == "__main__":
    main()
