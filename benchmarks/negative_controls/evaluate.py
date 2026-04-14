#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mcp_router.config import load_config
from mcp_router.registry.manager import IndexRegistry
from mcp_router.retrieval.embedder import OpenAIEmbedder
from mcp_router.retrieval.engine import RouterEngine
from mcp_router.session.memory import SessionMemory

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def make_engine(index_dir: Path) -> RouterEngine:
    cfg = load_config()
    cfg.index_dir = index_dir.resolve()
    registry = IndexRegistry(cfg)
    index = registry.load_index()
    embedder = OpenAIEmbedder(cfg)
    sessions = SessionMemory(cfg.session_ttl_seconds)
    return RouterEngine(cfg, index, embedder, sessions)


def confidence_at_most(actual: str, expected_max: str) -> bool:
    return CONFIDENCE_ORDER.get(actual, 99) <= CONFIDENCE_ORDER.get(expected_max, -1)


def evaluate_case(engine: RouterEngine, case: dict[str, Any]) -> dict[str, Any]:
    out = engine.route(
        server_intent=case["server_intent"],
        tool_intent=case["tool_intent"],
        session_id=f"negative-{case['id']}",
        record_session=False,
    )
    predicted = [tool.tool_key for tool in out.tools]
    predicted_servers = [tool.server_name for tool in out.tools]
    checks: dict[str, bool] = {}

    max_conf = case.get("max_confidence")
    if max_conf is not None:
        checks["max_confidence"] = confidence_at_most(out.confidence, str(max_conf))

    min_k = case.get("min_recommended_handoff_k")
    if min_k is not None:
        checks["min_recommended_handoff_k"] = int(out.recommended_handoff_k) >= int(min_k)

    max_k = case.get("max_recommended_handoff_k")
    if max_k is not None:
        checks["max_recommended_handoff_k"] = int(out.recommended_handoff_k) <= int(max_k)

    require_null_route = case.get("require_null_route")
    if require_null_route is not None:
        checks["require_null_route"] = bool(out.null_route) is bool(require_null_route)

    for idx, tool_key in enumerate(case.get("forbidden_tool_keys", [])):
        checks[f"forbidden_tool_keys[{idx}]"] = tool_key not in predicted[:5]

    for idx, server_name in enumerate(case.get("forbidden_server_names", [])):
        checks[f"forbidden_server_names[{idx}]"] = server_name not in predicted_servers[:5]

    if case.get("require_empty"):
        checks["require_empty"] = len(predicted) == 0

    passed = all(checks.values()) if checks else True
    return {
        "id": case["id"],
        "server_intent": case["server_intent"],
        "tool_intent": case["tool_intent"],
        "predicted_top5": predicted[:5],
        "predicted_servers_top5": predicted_servers[:5],
        "confidence": out.confidence,
        "recommended_handoff_k": int(out.recommended_handoff_k),
        "null_route": bool(out.null_route),
        "score_gap": float(out.score_gap),
        "checks": checks,
        "passed": passed,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    failure_counts: Counter[str] = Counter()
    by_confidence: dict[str, int] = defaultdict(int)
    handoff_values: list[int] = []

    for row in rows:
        by_confidence[row["confidence"]] += 1
        handoff_values.append(row["recommended_handoff_k"])
        for name, ok in row["checks"].items():
            if not ok:
                failure_counts[name] += 1

    return {
        "num_cases": total,
        "pass_rate": (sum(int(r["passed"]) for r in rows) / total) if total else 0.0,
        "confidence_distribution": dict(sorted(by_confidence.items())),
        "avg_recommended_handoff_k": statistics.mean(handoff_values) if handoff_values else 0.0,
        "failed_assertions": dict(failure_counts),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate negative-control routing cases.")
    ap.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "benchmarks" / "negative_controls" / "priority_cases.json",
    )
    ap.add_argument("--index-dir", type=Path, default=ROOT / "data" / "index")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    raw = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = raw.get("cases", raw)
    if not isinstance(cases, list) or not cases:
        raise SystemExit("Cases file must contain a non-empty list under `cases` or top-level.")

    engine = make_engine(args.index_dir)
    rows = [evaluate_case(engine, case) for case in cases]
    report = {"summary": summarize(rows), "rows": rows}

    print(json.dumps(report["summary"], indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report to: {args.out}")


if __name__ == "__main__":
    main()
