#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from mcp_router.config import load_config
from mcp_router.registry.manager import IndexRegistry
from mcp_router.retrieval.embedder import OpenAIEmbedder
from mcp_router.retrieval.engine import RouterEngine
from mcp_router.session.memory import SessionMemory


def hit_at(predicted: list[str], expected: str, k: int) -> bool:
    return expected in predicted[:k]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    per_conf_counts: dict[str, int] = defaultdict(int)
    per_conf_top1: dict[str, int] = defaultdict(int)
    per_conf_handoff: dict[str, int] = defaultdict(int)

    for row in rows:
        conf = row["confidence"]
        per_conf_counts[conf] += 1
        per_conf_top1[conf] += int(row["hit_at_1"])
        per_conf_handoff[conf] += int(row["hit_at_recommended_k"])

    confidence_buckets = {}
    for conf in sorted(per_conf_counts.keys()):
        c = per_conf_counts[conf]
        confidence_buckets[conf] = {
            "count": c,
            "top1_accuracy": (per_conf_top1[conf] / c) if c else 0.0,
            "handoff_accuracy_at_recommended_k": (per_conf_handoff[conf] / c) if c else 0.0,
        }

    return {
        "num_items": n,
        "top1_accuracy": (sum(int(r["hit_at_1"]) for r in rows) / n) if n else 0.0,
        "top3_accuracy": (sum(int(r["hit_at_3"]) for r in rows) / n) if n else 0.0,
        "top5_accuracy": (sum(int(r["hit_at_5"]) for r in rows) / n) if n else 0.0,
        "handoff_accuracy_at_recommended_k": (
            sum(int(r["hit_at_recommended_k"]) for r in rows) / n
        )
        if n
        else 0.0,
        "avg_recommended_handoff_k": (
            statistics.mean(r["recommended_handoff_k"] for r in rows) if n else 0.0
        ),
        "confidence_buckets": confidence_buckets,
    }


def make_engine(index_dir: Path, field_aware_enabled: bool) -> RouterEngine:
    cfg = load_config()
    cfg.index_dir = index_dir.resolve()
    cfg.field_aware_enabled = field_aware_enabled
    registry = IndexRegistry(cfg)
    index = registry.load_index()
    embedder = OpenAIEmbedder(cfg)
    sessions = SessionMemory(cfg.session_ttl_seconds)
    return RouterEngine(cfg, index, embedder, sessions)


def evaluate_step(
    engine: RouterEngine,
    item: dict[str, Any],
    session_id: str,
    keep_memory: bool,
) -> dict[str, Any]:
    out = engine.route(
        server_intent=item["server_intent"],
        tool_intent=item["tool_intent"],
        session_id=session_id,
        record_session=keep_memory,
    )
    predicted = [t.tool_key for t in out.tools]
    expected = item["expected_tool_key"]
    recommended_k = max(1, min(int(out.recommended_handoff_k), len(predicted))) if predicted else 1
    return {
        "server_intent": item["server_intent"],
        "tool_intent": item["tool_intent"],
        "expected_tool_key": expected,
        "predicted_topk": predicted[:5],
        "hit_at_1": hit_at(predicted, expected, 1),
        "hit_at_3": hit_at(predicted, expected, 3),
        "hit_at_5": hit_at(predicted, expected, 5),
        "confidence": out.confidence,
        "recommended_handoff_k": recommended_k,
        "hit_at_recommended_k": hit_at(predicted, expected, recommended_k),
        "score_gap": out.score_gap,
        "top1_score": out.top1_score,
        "top2_score": out.top2_score,
    }


def evaluate_cases(
    engine: RouterEngine,
    cases: list[dict[str, Any]],
    keep_memory: bool,
) -> dict[str, Any]:
    rows = []
    for case in cases:
        session_id = case.get("session_id") or f"fc-{case['id']}"
        rows.append(evaluate_step(engine, case, session_id, keep_memory))
    return {"summary": summarize_rows(rows), "results": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablate baseline vs field-aware retrieval.")
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--index-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--keep-memory", action="store_true")
    args = ap.parse_args()

    raw = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = raw.get("cases", raw)
    if not isinstance(cases, list) or not cases:
        raise SystemExit("Cases file must contain a non-empty list under `cases` or top-level.")
    if cases and ("hops" in cases[0] or "subtasks" in cases[0]):
        raise SystemExit("This ablation runner currently supports single-tool benchmark suites only.")

    baseline_engine = make_engine(args.index_dir, field_aware_enabled=False)
    field_engine = make_engine(args.index_dir, field_aware_enabled=True)

    report = {
        "suite": raw.get("suite", ""),
        "index_dir": str(args.index_dir.resolve()),
        "baseline": evaluate_cases(baseline_engine, cases, args.keep_memory),
        "field_aware": evaluate_cases(field_engine, cases, args.keep_memory),
    }

    delta = {
        "top1_accuracy": report["field_aware"]["summary"]["top1_accuracy"]
        - report["baseline"]["summary"]["top1_accuracy"],
        "top3_accuracy": report["field_aware"]["summary"]["top3_accuracy"]
        - report["baseline"]["summary"]["top3_accuracy"],
        "top5_accuracy": report["field_aware"]["summary"]["top5_accuracy"]
        - report["baseline"]["summary"]["top5_accuracy"],
        "handoff_accuracy_at_recommended_k": report["field_aware"]["summary"]["handoff_accuracy_at_recommended_k"]
        - report["baseline"]["summary"]["handoff_accuracy_at_recommended_k"],
    }
    report["delta"] = delta

    print(json.dumps({"baseline": report["baseline"]["summary"], "field_aware": report["field_aware"]["summary"], "delta": delta}, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved ablation report to: {args.out}")


if __name__ == "__main__":
    main()
