#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_router.config import load_config  # noqa: E402
from mcp_router.retrieval.embedder import OpenAIEmbedder  # noqa: E402
from mcp_router.retrieval.engine import RouterEngine  # noqa: E402
from mcp_router.retrieval.tool_index import ToolIndex  # noqa: E402
from mcp_router.session.memory import SessionMemory  # noqa: E402


DEFAULT_PAYLOADS = (
    ROOT / "evaluation" / "results" / "livemcpbench" / "router_policy_reconstructed_stepwise.json"
)
DEFAULT_INDEX = ROOT / "data" / "index_livemcpbench"
DEFAULT_OUT = (
    ROOT / "evaluation" / "results" / "livemcpbench" / "router_policy_payload_eval.json"
)


def precision_at(predicted: list[str], gold: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for item in predicted[:k] if item in gold) / k


def recall_at(predicted: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return sum(1 for item in predicted[:k] if item in gold) / len(gold)


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
    import math

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


def build_router(index_dir: Path) -> RouterEngine:
    cfg = load_config()
    cfg.index_dir = index_dir
    index = ToolIndex.load(cfg.index_dir)
    embedder = OpenAIEmbedder(cfg)
    sessions = SessionMemory(cfg.session_ttl_seconds)
    return RouterEngine(cfg, index, embedder, sessions)


def dedupe_servers(route_tools: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for tool in route_tools:
        if tool.server_name in seen:
            continue
        seen.add(tool.server_name)
        out.append(tool.server_name)
    return out


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


def evaluate(payloads_path: Path, index_dir: Path) -> dict[str, Any]:
    payload_file = json.loads(payloads_path.read_text(encoding="utf-8"))
    rows = payload_file["rows"]
    router = build_router(index_dir)

    results = []
    label_counts: dict[str, int] = {}
    for idx, row in enumerate(rows):
        label = row.get("reconstruction_label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1

        payload = dict(row["payload"])
        payload["session_id"] = f"{payload['session_id']}-{idx}"
        out = router.route(**payload)
        predicted_servers = dedupe_servers(out.tools)
        gold_servers = set(row.get("gold_server_names", []))
        metrics = score_query(predicted_servers, gold_servers)
        results.append(
            {
                "task_id": row["task_id"],
                "step_index": row["step_index"],
                "question": row["question"],
                "step_query": row["step_query"],
                "gold_server_names": row["gold_server_names"],
                "reconstruction_label": row["reconstruction_label"],
                "assigned_tool_names": row.get("assigned_tool_names", []),
                "payload": payload,
                "predicted_servers_topk": predicted_servers,
                "predicted_tools_topk": [
                    {
                        "tool_key": tool.tool_key,
                        "server_name": tool.server_name,
                        "tool_name": tool.tool_name,
                        "score": tool.score,
                    }
                    for tool in out.tools
                ],
                "adaptive_k": out.adaptive_k,
                "recommended_handoff_k": out.recommended_handoff_k,
                "confidence": out.confidence,
                **metrics,
            }
        )

    summary = summarize_rows(results)
    summary["reconstruction_label_counts"] = label_counts
    summary["num_steps_with_any_gold_servers"] = sum(1 for row in results if row["gold_server_names"])
    return {
        "metadata": {
            "protocol": "router-policy-payload-eval",
            "payloads_path": str(payloads_path),
            "index_dir": str(index_dir),
            "note": "Runs the real router on Policy.md-style reconstructed payloads.",
        },
        "summary": summary,
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run router on policy-style payload JSON.")
    ap.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS)
    ap.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    result = evaluate(args.payloads, args.index_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved router policy eval to: {args.out}")


if __name__ == "__main__":
    main()
