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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_router.config import load_config  # noqa: E402
from mcp_router.retrieval.embedder import OpenAIEmbedder  # noqa: E402
from mcp_router.retrieval.engine import RouterEngine  # noqa: E402
from mcp_router.retrieval.tool_index import ToolIndex  # noqa: E402
from mcp_router.session.memory import SessionMemory  # noqa: E402


DEFAULT_PREPARED = ROOT / "evaluation" / "results" / "livemcpbench" / "paper_aligned_exact_step_subset.json"
DEFAULT_OUT = ROOT / "evaluation" / "results" / "livemcpbench" / "paper_aligned_exact_eval.json"


def dedupe_servers(tools: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        server_name = tool.get("server_name")
        if not server_name or server_name in seen:
            continue
        seen.add(server_name)
        out.append(server_name)
    return out


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


def build_engine(index_dir: Path | None = None):
    cfg = load_config()
    if index_dir is not None:
        cfg.index_dir = index_dir
    cfg.adaptive_min_tools = max(cfg.adaptive_min_tools, 30)
    cfg.adaptive_max_tools = max(cfg.adaptive_max_tools, 30)
    index = ToolIndex.load(cfg.index_dir)
    embedder = OpenAIEmbedder(cfg)
    sessions = SessionMemory(cfg.session_ttl_seconds)
    return RouterEngine(cfg, index, embedder, sessions)


def evaluate(prepared_path: Path, index_dir: Path | None = None) -> dict[str, Any]:
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    engine = build_engine(index_dir=index_dir)
    rows: list[dict[str, Any]] = []

    for task in prepared:
        for step in task.get("steps", []):
            query = step["step_query"]
            out = engine.route(
                server_intent=query,
                tool_intent=query,
                session_id=None,
                record_session=False,
            )
            predicted_servers = dedupe_servers([tool.model_dump() for tool in out.tools])
            gold_server = step["gold_server_name"]
            rows.append(
                {
                    "task_id": step["task_id"],
                    "question": step["question"],
                    "step_index": step["step_index"],
                    "step_query": query,
                    "gold_tool_name": step["gold_tool_name"],
                    "gold_server_name": gold_server,
                    "predicted_servers_top10": predicted_servers[:10],
                    "predicted_tools_top10": [
                        {
                            "server_name": tool.server_name,
                            "tool_name": tool.tool_name,
                            "tool_key": tool.tool_key,
                            "score": tool.score,
                        }
                        for tool in out.tools[:10]
                    ],
                    "recall@1": recall_at(predicted_servers, gold_server, 1),
                    "recall@3": recall_at(predicted_servers, gold_server, 3),
                    "recall@5": recall_at(predicted_servers, gold_server, 5),
                    "map@1": average_precision_at(predicted_servers, gold_server, 1),
                    "map@3": average_precision_at(predicted_servers, gold_server, 3),
                    "map@5": average_precision_at(predicted_servers, gold_server, 5),
                    "ndcg@1": ndcg_at(predicted_servers, gold_server, 1),
                    "ndcg@3": ndcg_at(predicted_servers, gold_server, 3),
                    "ndcg@5": ndcg_at(predicted_servers, gold_server, 5),
                }
            )

    n = len(rows)
    summary = {
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
    }
    return {"summary": summary, "results": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate server retrieval on the exact-step LiveMCPBench subset.")
    ap.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--index-dir", type=Path, default=None)
    args = ap.parse_args()

    result = evaluate(args.prepared, index_dir=args.index_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved eval results to: {args.out}")


if __name__ == "__main__":
    main()
