#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mcp_router.config import load_config
from mcp_router.registry.manager import IndexRegistry
from mcp_router.retrieval.embedder import OpenAIEmbedder
from mcp_router.retrieval.engine import RouterEngine
from mcp_router.session.memory import SessionMemory


def make_engine(index_dir: Path) -> RouterEngine:
    cfg = load_config()
    cfg.index_dir = index_dir.resolve()
    registry = IndexRegistry(cfg)
    index = registry.load_index()
    embedder = OpenAIEmbedder(cfg)
    sessions = SessionMemory(cfg.session_ttl_seconds)
    return RouterEngine(cfg, index, embedder, sessions)


def split_name(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def iter_steps(suite_name: str, path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = raw.get("cases", raw)
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["id"])
        if "hops" in case:
            for hop_index, hop in enumerate(case["hops"]):
                rows.append(
                    {
                        "suite": suite_name,
                        "case_id": case_id,
                        "hop_index": hop_index,
                        "server_intent": hop["server_intent"],
                        "tool_intent": hop["tool_intent"],
                        "expected_tool_key": hop["expected_tool_key"],
                    }
                )
        elif "subtasks" in case:
            for hop_index, hop in enumerate(case["subtasks"]):
                rows.append(
                    {
                        "suite": suite_name,
                        "case_id": case_id,
                        "hop_index": hop_index,
                        "server_intent": hop["server_intent"],
                        "tool_intent": hop["tool_intent"],
                        "expected_tool_key": hop["expected_tool_key"],
                    }
                )
        else:
            rows.append(
                {
                    "suite": suite_name,
                    "case_id": case_id,
                    "hop_index": 0,
                    "server_intent": case["server_intent"],
                    "tool_intent": case["tool_intent"],
                    "expected_tool_key": case["expected_tool_key"],
                }
            )
    return rows


def evaluate_observation(engine: RouterEngine, step: dict[str, Any]) -> dict[str, Any]:
    out = engine.route(
        server_intent=step["server_intent"],
        tool_intent=step["tool_intent"],
        session_id=f"threshold-{step['suite']}-{step['case_id']}",
        record_session=False,
    )
    predicted = [tool.tool_key for tool in out.tools]
    return {
        **step,
        "group_id": f"{step['suite']}::{step['case_id']}",
        "predicted_top5": predicted[:5],
        "top1_ok": step["expected_tool_key"] in predicted[:1],
        "top3_ok": step["expected_tool_key"] in predicted[:3],
        "top5_ok": step["expected_tool_key"] in predicted[:5],
        "score_gap": float(out.score_gap),
    }


def choose_bucket(gap: float, high: float, medium: float) -> str:
    if gap >= high:
        return "high"
    if gap >= medium:
        return "medium"
    return "low"


def evaluate_thresholds(
    rows: list[dict[str, Any]],
    high: float,
    medium: float,
    k_high: int,
    k_medium: int,
    k_low: int,
) -> dict[str, Any]:
    counts = {"high": 0, "medium": 0, "low": 0}
    top1_hits = {"high": 0, "medium": 0, "low": 0}
    handoff_hits = 0
    total_k = 0
    by_suite: dict[str, dict[str, float]] = {}

    suite_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        suite_rows[row["suite"]].append(row)
        bucket = choose_bucket(row["score_gap"], high, medium)
        counts[bucket] += 1
        top1_hits[bucket] += int(row["top1_ok"])
        if bucket == "high":
            k = k_high
            ok = row["top1_ok"]
        elif bucket == "medium":
            k = k_medium
            ok = row["top3_ok"] if k <= 3 else row["top5_ok"]
        else:
            k = k_low
            ok = row["top3_ok"] if k <= 3 else row["top5_ok"]
        handoff_hits += int(ok)
        total_k += k

    total = len(rows)
    for suite, srows in suite_rows.items():
        hits = 0
        total_suite_k = 0
        for row in srows:
            bucket = choose_bucket(row["score_gap"], high, medium)
            if bucket == "high":
                k = k_high
                ok = row["top1_ok"]
            elif bucket == "medium":
                k = k_medium
                ok = row["top3_ok"] if k <= 3 else row["top5_ok"]
            else:
                k = k_low
                ok = row["top3_ok"] if k <= 3 else row["top5_ok"]
            hits += int(ok)
            total_suite_k += k
        by_suite[suite] = {
            "num_steps": len(srows),
            "handoff_accuracy": (hits / len(srows)) if srows else 0.0,
            "avg_recommended_k": (total_suite_k / len(srows)) if srows else 0.0,
        }

    return {
        "num_steps": total,
        "handoff_accuracy": (handoff_hits / total) if total else 0.0,
        "avg_recommended_k": (total_k / total) if total else 0.0,
        "bucket_counts": counts,
        "top1_precision_by_bucket": {
            bucket: (top1_hits[bucket] / counts[bucket]) if counts[bucket] else 0.0
            for bucket in counts
        },
        "by_suite": dict(sorted(by_suite.items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Threshold sensitivity benchmark with deterministic dev/test split.")
    ap.add_argument("--index-dir", type=Path, default=ROOT / "data" / "index")
    ap.add_argument(
        "--suite",
        action="append",
        default=[],
        help="Path to a suite JSON. Can be passed multiple times.",
    )
    ap.add_argument("--dev-mod", type=int, default=5)
    ap.add_argument("--dev-remainder", type=int, default=0)
    ap.add_argument("--k-high", type=int, default=1)
    ap.add_argument("--k-medium", type=int, default=3)
    ap.add_argument("--k-low", type=int, default=5)
    ap.add_argument("--target-high-precision", type=float, default=0.95)
    ap.add_argument("--target-medium-precision", type=float, default=0.85)
    ap.add_argument("--min-high-count", type=int, default=2)
    ap.add_argument("--min-medium-count", type=int, default=2)
    ap.add_argument("--max-threshold-candidates", type=int, default=40)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    suite_paths = [Path(p) for p in args.suite] if args.suite else [
        ROOT / "benchmarks" / "functional_correctness" / "ambiguity_collision.priority.json",
        ROOT / "benchmarks" / "functional_correctness" / "robustness_safety.priority.json",
        ROOT / "benchmarks" / "functional_correctness" / "single_tool.medium_250.clean.json",
        ROOT / "benchmarks" / "functional_correctness" / "multi_tool.large.single_turn.json",
        ROOT / "benchmarks" / "functional_correctness" / "multi_hop.large.cross_app.json",
    ]

    engine = make_engine(args.index_dir)
    steps: list[dict[str, Any]] = []
    for path in suite_paths:
        steps.extend(iter_steps(path.stem, path))
    observations = [evaluate_observation(engine, step) for step in steps]

    dev_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for row in observations:
        bucket = split_name(row["group_id"]) % args.dev_mod
        if bucket == args.dev_remainder:
            dev_rows.append(row)
        else:
            test_rows.append(row)

    cfg = load_config()
    current_eval = evaluate_thresholds(
        test_rows,
        high=cfg.handoff_gap_high,
        medium=cfg.handoff_gap_medium,
        k_high=args.k_high,
        k_medium=args.k_medium,
        k_low=args.k_low,
    )

    gap_values = sorted({row["score_gap"] for row in dev_rows})
    if len(gap_values) > args.max_threshold_candidates:
        step = max(1, len(gap_values) // args.max_threshold_candidates)
        gap_values = gap_values[::step]
    gap_values = sorted(set(gap_values + [cfg.handoff_gap_medium, cfg.handoff_gap_high]))

    candidates: list[dict[str, Any]] = []
    for medium in gap_values:
        for high in gap_values:
            if high <= medium:
                continue
            eval_dev = evaluate_thresholds(
                dev_rows,
                high=high,
                medium=medium,
                k_high=args.k_high,
                k_medium=args.k_medium,
                k_low=args.k_low,
            )
            constraints_ok = (
                eval_dev["bucket_counts"]["high"] >= args.min_high_count
                and eval_dev["bucket_counts"]["medium"] >= args.min_medium_count
                and eval_dev["top1_precision_by_bucket"]["high"] >= args.target_high_precision
                and eval_dev["top1_precision_by_bucket"]["medium"] >= args.target_medium_precision
            )
            candidates.append(
                {
                    "high": high,
                    "medium": medium,
                    "constraints_ok": constraints_ok,
                    "dev": eval_dev,
                }
            )

    constrained = [c for c in candidates if c["constraints_ok"]]
    pool = constrained or candidates
    best = sorted(
        pool,
        key=lambda c: (
            -c["dev"]["handoff_accuracy"],
            c["dev"]["avg_recommended_k"],
            -c["high"],
            -c["medium"],
        ),
    )[0]

    recommended_test = evaluate_thresholds(
        test_rows,
        high=best["high"],
        medium=best["medium"],
        k_high=args.k_high,
        k_medium=args.k_medium,
        k_low=args.k_low,
    )

    report = {
        "suite_paths": [str(path) for path in suite_paths],
        "num_steps_total": len(observations),
        "num_steps_dev": len(dev_rows),
        "num_steps_test": len(test_rows),
        "current_config": {
            "handoff_gap_high": cfg.handoff_gap_high,
            "handoff_gap_medium": cfg.handoff_gap_medium,
            "test_metrics": current_eval,
        },
        "recommended_from_dev": {
            "handoff_gap_high": best["high"],
            "handoff_gap_medium": best["medium"],
            "constraints_ok": best["constraints_ok"],
            "dev_metrics": best["dev"],
            "test_metrics": recommended_test,
        },
    }

    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report to: {args.out}")


if __name__ == "__main__":
    main()
