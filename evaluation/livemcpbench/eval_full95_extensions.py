#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
THIS_DIR = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from mcp_router.config import load_config  # noqa: E402
from mcp_router.retrieval.embedder import OpenAIEmbedder  # noqa: E402

from eval_tool_to_agent_paper_variant import build_retriever  # noqa: E402


DEFAULT_PREPARED = ROOT / "evaluation" / "results" / "livemcpbench" / "paper_aligned_prepared.json"
DEFAULT_CATALOG = ROOT / "data" / "catalog_subsets" / "livemcpbench.embedding.json"
DEFAULT_TASK_OUT = ROOT / "evaluation" / "results" / "livemcpbench" / "full95_task_level_server_eval.json"
DEFAULT_RECON_OUT = ROOT / "evaluation" / "results" / "livemcpbench" / "full95_reconstructed_stepwise_eval.json"
MAX_EMBED_QUERY_CHARS = 6000


def precision_at(predicted: list[str], gold: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for item in predicted[:k] if item in gold)
    return hits / k


def recall_at(predicted: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = sum(1 for item in predicted[:k] if item in gold)
    return hits / len(gold)


def average_precision_at(predicted: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    ap = 0.0
    hits = 0
    for rank, item in enumerate(predicted[:k], start=1):
        if item in gold:
            hits += 1
            ap += hits / rank
    return ap / len(gold)


def ndcg_at(predicted: list[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    for rank, item in enumerate(predicted[:k], start=1):
        if item in gold:
            dcg += 1.0 / math.log2(rank + 1.0)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "num_queries": n,
        "precision": {
            "@1": sum(r["precision@1"] for r in rows) / n if n else 0.0,
            "@3": sum(r["precision@3"] for r in rows) / n if n else 0.0,
            "@5": sum(r["precision@5"] for r in rows) / n if n else 0.0,
        },
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
    }


def truncate_query(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= MAX_EMBED_QUERY_CHARS:
        return text
    return text[:MAX_EMBED_QUERY_CHARS]


def resolved_servers_for_tool(tool_info: dict[str, Any]) -> list[str]:
    candidates = tool_info.get("candidate_servers") or []
    return list(candidates) if len(candidates) == 1 else []


def unresolved_tool_reason(tool_info: dict[str, Any]) -> str | None:
    candidates = tool_info.get("candidate_servers") or []
    if len(candidates) == 0:
        return "unknown_tool_name"
    if len(candidates) > 1:
        return "ambiguous_tool_name"
    return None


def task_level_gold(task: dict[str, Any]) -> tuple[set[str], dict[str, Any]]:
    gold: set[str] = set()
    unresolved = []
    for tool in task.get("annotated_tools", []):
        resolved = resolved_servers_for_tool(tool)
        if resolved:
            gold.update(resolved)
        else:
            unresolved.append(
                {
                    "tool_name": tool.get("tool_name", ""),
                    "candidate_servers": tool.get("candidate_servers", []),
                    "reason": unresolved_tool_reason(tool),
                }
            )
    meta = {
        "num_annotated_tools": len(task.get("annotated_tools", [])),
        "num_resolved_tools": len(task.get("annotated_tools", [])) - len(unresolved),
        "num_unresolved_tools": len(unresolved),
        "unresolved_tools": unresolved,
    }
    return gold, meta


def proportional_slice(n_items: int, n_buckets: int, bucket_idx: int) -> tuple[int, int]:
    start = math.floor(bucket_idx * n_items / n_buckets)
    end = math.floor((bucket_idx + 1) * n_items / n_buckets)
    if end <= start:
        end = min(start + 1, n_items)
    return start, min(end, n_items)


def reconstruct_steps(task: dict[str, Any]) -> list[dict[str, Any]]:
    steps = task.get("steps", [])
    tools = task.get("annotated_tools", [])
    if not steps:
        return []

    resolved_positions = [
        (idx, resolved_servers_for_tool(tool))
        for idx, tool in enumerate(tools)
        if resolved_servers_for_tool(tool)
    ]

    rows = []
    for step_idx, step_query in enumerate(steps):
        start, end = proportional_slice(len(tools), len(steps), step_idx) if tools else (0, 0)
        assigned_tools = tools[start:end]

        gold_servers = {
            server
            for tool in assigned_tools
            for server in resolved_servers_for_tool(tool)
        }
        source = "exact" if len(steps) == len(tools) and end - start == 1 and len(gold_servers) == 1 else "reconstructed_segment"

        fallback_used = False
        if not gold_servers and resolved_positions:
            center = (start + end - 1) / 2.0 if tools else 0.0
            nearest_idx, nearest_servers = min(
                resolved_positions,
                key=lambda pair: abs(pair[0] - center),
            )
            gold_servers = set(nearest_servers)
            fallback_used = True
            source = "reconstructed_nearest_resolved_tool"
        elif not gold_servers:
            source = "reconstructed_unresolved_empty_gold"

        rows.append(
            {
                "task_id": task["task_id"],
                "question": task["question"],
                "step_index": step_idx + 1,
                "step_query": step_query,
                "gold_server_names": sorted(gold_servers),
                "reconstruction_label": source,
                "fallback_used": fallback_used,
                "assigned_tool_span": [start, end],
                "assigned_tool_names": [tool.get("tool_name", "") for tool in assigned_tools],
                "assigned_resolved_servers": sorted(
                    {
                        server
                        for tool in assigned_tools
                        for server in resolved_servers_for_tool(tool)
                    }
                ),
                "assigned_unresolved_tools": [
                    {
                        "tool_name": tool.get("tool_name", ""),
                        "candidate_servers": tool.get("candidate_servers", []),
                        "reason": unresolved_tool_reason(tool),
                    }
                    for tool in assigned_tools
                    if unresolved_tool_reason(tool) is not None
                ],
            }
        )
    return rows


def score_query(predicted_servers: list[str], gold_servers: set[str]) -> dict[str, float]:
    return {
        "precision@1": precision_at(predicted_servers, gold_servers, 1),
        "precision@3": precision_at(predicted_servers, gold_servers, 3),
        "precision@5": precision_at(predicted_servers, gold_servers, 5),
        "recall@1": recall_at(predicted_servers, gold_servers, 1),
        "recall@3": recall_at(predicted_servers, gold_servers, 3),
        "recall@5": recall_at(predicted_servers, gold_servers, 5),
        "map@1": average_precision_at(predicted_servers, gold_servers, 1),
        "map@3": average_precision_at(predicted_servers, gold_servers, 3),
        "map@5": average_precision_at(predicted_servers, gold_servers, 5),
        "ndcg@1": ndcg_at(predicted_servers, gold_servers, 1),
        "ndcg@3": ndcg_at(predicted_servers, gold_servers, 3),
        "ndcg@5": ndcg_at(predicted_servers, gold_servers, 5),
    }


def run_retrieval(
    queries: list[dict[str, Any]],
    *,
    query_key: str,
    top_n: int,
    top_k: int,
    catalog_path: Path,
    dense_weight: float,
    bm25_weight: float,
    rrf_k: int,
) -> list[dict[str, Any]]:
    cfg = load_config()
    embedder = OpenAIEmbedder(cfg)
    retriever = build_retriever(catalog_path, dense_weight=dense_weight, bm25_weight=bm25_weight, rrf_k=rrf_k)

    query_texts = [truncate_query(row[query_key]) for row in queries]
    query_embeddings = embedder.embed_batch(query_texts) if query_texts else []

    results = []
    for row, emb, embed_query in zip(queries, query_embeddings, query_texts):
        ranked_entities = retriever.rank_entities(embed_query, emb, top_n=top_n)
        predicted_servers, trace = retriever.collapse_to_unique_agents(ranked_entities, k_agents=top_k)
        gold_servers = set(row["gold_server_names"])
        scored = score_query(predicted_servers, gold_servers)
        results.append(
            {
                **row,
                "embedded_query": embed_query,
                "query_was_truncated": embed_query != row[query_key],
                "predicted_servers_topk": predicted_servers,
                "ranked_entities_trace": trace,
                **scored,
            }
        )
    return results


def evaluate_task_level(
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
    tasks = prepared["tasks"]
    queries = []
    fully_resolved = 0
    for task in tasks:
        gold_servers, meta = task_level_gold(task)
        if meta["num_unresolved_tools"] == 0:
            fully_resolved += 1
        queries.append(
            {
                "task_id": task["task_id"],
                "question": task["question"],
                "category": task.get("category", ""),
                "gold_server_names": sorted(gold_servers),
                "gold_resolution": meta,
            }
        )

    results = run_retrieval(
        queries,
        query_key="question",
        top_n=top_n,
        top_k=top_k,
        catalog_path=catalog_path,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
        rrf_k=rrf_k,
    )
    summary = summarize_rows(results)
    summary["num_tasks_with_any_gold_servers"] = sum(1 for row in results if row["gold_server_names"])
    summary["num_fully_resolved_tasks"] = fully_resolved
    return {
        "metadata": {
            "protocol": "full95-task-level-server-retrieval",
            "label_quality": "derived-from-task-tool-annotations",
            "note": (
                "Gold task server sets are derived from annotated tools. Ambiguous or unknown "
                "tool names are tracked and omitted from the resolved gold server set."
            ),
            "prepared_path": str(prepared_path),
            "catalog_path": str(catalog_path),
            "top_n_entities": top_n,
            "top_k_agents": top_k,
            "dense_weight": dense_weight,
            "bm25_weight": bm25_weight,
            "rrf_k": rrf_k,
        },
        "summary": summary,
        "results": results,
    }


def evaluate_reconstructed_stepwise(
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
    step_queries = []
    source_counts: dict[str, int] = {}
    for task in prepared["tasks"]:
        for step in reconstruct_steps(task):
            source = step["reconstruction_label"]
            source_counts[source] = source_counts.get(source, 0) + 1
            step_queries.append(step)

    results = run_retrieval(
        step_queries,
        query_key="step_query",
        top_n=top_n,
        top_k=top_k,
        catalog_path=catalog_path,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
        rrf_k=rrf_k,
    )
    summary = summarize_rows(results)
    summary["reconstruction_label_counts"] = source_counts
    summary["num_steps_with_any_gold_servers"] = sum(1 for row in results if row["gold_server_names"])
    return {
        "metadata": {
            "protocol": "full95-reconstructed-stepwise-server-retrieval",
            "label_quality": "reconstructed",
            "note": (
                "Step-level gold servers are heuristically reconstructed from ordered steps and "
                "ordered annotated tools. This is not original benchmark gold."
            ),
            "prepared_path": str(prepared_path),
            "catalog_path": str(catalog_path),
            "top_n_entities": top_n,
            "top_k_agents": top_k,
            "dense_weight": dense_weight,
            "bm25_weight": bm25_weight,
            "rrf_k": rrf_k,
        },
        "summary": summary,
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full-95 LiveMCPBench extension evaluations.")
    ap.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--task-out", type=Path, default=DEFAULT_TASK_OUT)
    ap.add_argument("--reconstructed-out", type=Path, default=DEFAULT_RECON_OUT)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--dense-weight", type=float, default=1.0)
    ap.add_argument("--bm25-weight", type=float, default=1.0)
    ap.add_argument("--rrf-k", type=int, default=60)
    args = ap.parse_args()

    task_result = evaluate_task_level(
        args.prepared,
        args.catalog,
        top_n=args.top_n,
        top_k=args.top_k,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        rrf_k=args.rrf_k,
    )
    recon_result = evaluate_reconstructed_stepwise(
        args.prepared,
        args.catalog,
        top_n=args.top_n,
        top_k=args.top_k,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        rrf_k=args.rrf_k,
    )

    args.task_out.parent.mkdir(parents=True, exist_ok=True)
    args.reconstructed_out.parent.mkdir(parents=True, exist_ok=True)
    args.task_out.write_text(json.dumps(task_result, indent=2), encoding="utf-8")
    args.reconstructed_out.write_text(json.dumps(recon_result, indent=2), encoding="utf-8")

    print(json.dumps({"task_level_summary": task_result["summary"], "reconstructed_stepwise_summary": recon_result["summary"]}, indent=2))
    print(f"Saved task-level results to: {args.task_out}")
    print(f"Saved reconstructed stepwise results to: {args.reconstructed_out}")


if __name__ == "__main__":
    main()
