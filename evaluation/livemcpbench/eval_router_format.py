#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from mcp_router.retrieval.engine import RouterEngine  # noqa: E402
from mcp_router.retrieval.tool_index import ToolIndex  # noqa: E402
from mcp_router.session.memory import SessionMemory  # noqa: E402

from eval_full95_extensions import (  # noqa: E402
    reconstruct_steps,
    score_query,
    summarize_rows,
    task_level_gold,
)


DEFAULT_PREPARED = ROOT / "evaluation" / "results" / "livemcpbench" / "paper_aligned_prepared.json"
DEFAULT_EXACT = ROOT / "evaluation" / "results" / "livemcpbench" / "paper_aligned_exact_step_subset.json"
DEFAULT_INDEX = ROOT / "data" / "index_livemcpbench"
DEFAULT_OUT = ROOT / "evaluation" / "results" / "livemcpbench" / "router_format_eval.json"
MAX_INTENT_CHARS = 6000


def truncate_text(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= MAX_INTENT_CHARS:
        return text
    return text[:MAX_INTENT_CHARS]


def dedupe_servers_from_route_tools(tools: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        server_name = tool.server_name
        if server_name in seen:
            continue
        seen.add(server_name)
        out.append(server_name)
    return out


def build_router(index_dir: Path) -> RouterEngine:
    cfg = load_config()
    cfg.index_dir = index_dir
    index = ToolIndex.load(cfg.index_dir)
    embedder = OpenAIEmbedder(cfg)
    sessions = SessionMemory(cfg.session_ttl_seconds)
    return RouterEngine(cfg, index, embedder, sessions)


def run_queries(router: RouterEngine, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for idx, query in enumerate(queries):
        server_intent = truncate_text(query["server_intent"])
        tool_intent = truncate_text(query["tool_intent"])
        out = router.route(
            server_intent=server_intent,
            tool_intent=tool_intent,
            session_id=f"livemcpbench-router-format-{idx}",
            record_session=False,
        )
        predicted_servers = dedupe_servers_from_route_tools(out.tools)
        gold_servers = set(query["gold_server_names"])
        metrics = score_query(predicted_servers, gold_servers)
        rows.append(
            {
                **query,
                "server_intent": server_intent,
                "tool_intent": tool_intent,
                "query_was_truncated": (
                    server_intent != query["server_intent"] or tool_intent != query["tool_intent"]
                ),
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
    return rows


def evaluate_exact_subset(router: RouterEngine, exact_path: Path) -> dict[str, Any]:
    exact = json.loads(exact_path.read_text(encoding="utf-8"))
    queries = []
    for task in exact:
        for step in task.get("steps", []):
            queries.append(
                {
                    "task_id": step["task_id"],
                    "question": step["question"],
                    "step_index": step["step_index"],
                    "step_query": step["step_query"],
                    "gold_server_names": [step["gold_server_name"]],
                    "server_intent": step["question"],
                    "tool_intent": step["step_query"],
                    "label_quality": "exact",
                }
            )
    rows = run_queries(router, queries)
    return {
        "metadata": {
            "protocol": "router-format-exact-step-subset",
            "note": "Uses the real router with server_intent=task question and tool_intent=step query.",
        },
        "summary": summarize_rows(rows),
        "results": rows,
    }


def evaluate_task_level(router: RouterEngine, prepared_path: Path) -> dict[str, Any]:
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    queries = []
    fully_resolved = 0
    for task in prepared["tasks"]:
        gold_servers, meta = task_level_gold(task)
        if meta["num_unresolved_tools"] == 0:
            fully_resolved += 1
        queries.append(
            {
                "task_id": task["task_id"],
                "question": task["question"],
                "gold_server_names": sorted(gold_servers),
                "server_intent": task["question"],
                "tool_intent": task["question"],
                "label_quality": "derived-task-level",
                "gold_resolution": meta,
            }
        )
    rows = run_queries(router, queries)
    summary = summarize_rows(rows)
    summary["num_tasks_with_any_gold_servers"] = sum(1 for row in rows if row["gold_server_names"])
    summary["num_fully_resolved_tasks"] = fully_resolved
    return {
        "metadata": {
            "protocol": "router-format-full95-task-level",
            "note": "Uses the real router with server_intent=question and tool_intent=question.",
        },
        "summary": summary,
        "results": rows,
    }


def evaluate_reconstructed_stepwise(router: RouterEngine, prepared_path: Path) -> dict[str, Any]:
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    queries = []
    label_counts: dict[str, int] = {}
    for task in prepared["tasks"]:
        for step in reconstruct_steps(task):
            label_counts[step["reconstruction_label"]] = label_counts.get(step["reconstruction_label"], 0) + 1
            queries.append(
                {
                    **step,
                    "server_intent": task["question"],
                    "tool_intent": step["step_query"],
                    "label_quality": step["reconstruction_label"],
                }
            )
    rows = run_queries(router, queries)
    summary = summarize_rows(rows)
    summary["reconstruction_label_counts"] = label_counts
    summary["num_steps_with_any_gold_servers"] = sum(1 for row in rows if row["gold_server_names"])
    return {
        "metadata": {
            "protocol": "router-format-full95-reconstructed-stepwise",
            "note": (
                "Uses the real router with server_intent=task question and tool_intent=step query. "
                "Gold labels are reconstructed for non-exact steps."
            ),
        },
        "summary": summary,
        "results": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run router-format LiveMCPBench evaluations.")
    ap.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    ap.add_argument("--exact", type=Path, default=DEFAULT_EXACT)
    ap.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    router = build_router(args.index_dir)
    exact_result = evaluate_exact_subset(router, args.exact)
    task_result = evaluate_task_level(router, args.prepared)
    recon_result = evaluate_reconstructed_stepwise(router, args.prepared)

    payload = {
        "metadata": {
            "index_dir": str(args.index_dir),
            "intent_mapping": {
                "exact_step_subset": {
                    "server_intent": "task question",
                    "tool_intent": "step query",
                },
                "full95_task_level": {
                    "server_intent": "task question",
                    "tool_intent": "task question",
                },
                "full95_reconstructed_stepwise": {
                    "server_intent": "task question",
                    "tool_intent": "step query",
                },
            },
            "note": "Runs the actual deployed router path over LiveMCPBench-derived intents.",
        },
        "exact_step_subset": exact_result,
        "full95_task_level": task_result,
        "full95_reconstructed_stepwise": recon_result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "exact_step_subset": exact_result["summary"],
                "full95_task_level": task_result["summary"],
                "full95_reconstructed_stepwise": recon_result["summary"],
            },
            indent=2,
        )
    )
    print(f"Saved router-format eval to: {args.out}")


if __name__ == "__main__":
    main()
