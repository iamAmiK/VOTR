#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "benchmarks" / "results" / "confidence"
sys.path.insert(0, str(ROOT / "src"))

from mcp_router.config import load_config  # noqa: E402
from mcp_router.registry.manager import IndexRegistry  # noqa: E402
from mcp_router.retrieval.embedder import OpenAIEmbedder  # noqa: E402
from mcp_router.retrieval.engine import RouterEngine  # noqa: E402
from mcp_router.session.memory import SessionMemory  # noqa: E402


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    scale: str
    cases_path: Path
    index_dir: Path


def build_suite_specs() -> list[SuiteSpec]:
    fc = ROOT / "benchmarks" / "functional_correctness"
    data = ROOT / "data"
    return [
        SuiteSpec("single_tool.small_bloomberg", "small", fc / "single_tool.bloomberg.clean.json", data / "index_small_bloomberg"),
        SuiteSpec("single_tool.small_github", "small", fc / "single_tool.github.clean.json", data / "index_small_github"),
        SuiteSpec("single_tool.small_telegram", "small", fc / "single_tool.telegram.clean.json", data / "index_small_telegram"),
        SuiteSpec("multi_hop.small_100", "small", fc / "multi_hop.small_100.cross_app.json", data / "index_small_3"),
        SuiteSpec("multi_tool.small_100", "small", fc / "multi_tool.small_100.single_turn.json", data / "index_small_3"),
        SuiteSpec("single_tool.medium_250", "medium", fc / "single_tool.medium_250.clean.json", data / "index_medium_50"),
        SuiteSpec("multi_hop.medium_250", "medium", fc / "multi_hop.medium_250.cross_app.json", data / "index_medium_50"),
        SuiteSpec("multi_tool.medium_250", "medium", fc / "multi_tool.medium_250.single_turn.json", data / "index_medium_50"),
        SuiteSpec("single_tool.large", "large", fc / "single_tool.clean.json", data / "index"),
        SuiteSpec("multi_hop.large", "large", fc / "multi_hop.large.cross_app.json", data / "index"),
        SuiteSpec("multi_tool.large", "large", fc / "multi_tool.large.single_turn.json", data / "index"),
    ]


def make_engine(index_dir: Path) -> RouterEngine:
    cfg = load_config()
    cfg.index_dir = index_dir.resolve()
    registry = IndexRegistry(cfg)
    index = registry.load_index()
    embedder = OpenAIEmbedder(cfg)
    sessions = SessionMemory(cfg.session_ttl_seconds)
    return RouterEngine(cfg, index, embedder, sessions)


def hit_at(predicted: list[str], expected: str, k: int) -> bool:
    return expected in predicted[:k]


def evaluate_step(engine: RouterEngine, item: dict[str, Any], session_id: str, keep_memory: bool) -> dict[str, Any]:
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
        "confidence": out.confidence,
        "recommended_handoff_k": recommended_k,
        "hit_at_1": hit_at(predicted, expected, 1),
        "hit_at_3": hit_at(predicted, expected, 3),
        "hit_at_5": hit_at(predicted, expected, 5),
        "hit_at_recommended_k": hit_at(predicted, expected, recommended_k),
        "score_gap": float(out.score_gap),
    }


def iter_suite_steps(raw_cases: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for case in raw_cases:
        session_id = case.get("session_id") or f"fc-{case['id']}"
        if "subtasks" in case:
            for subtask in case["subtasks"]:
                out.append((session_id, subtask))
        elif "hops" in case:
            for hop in case["hops"]:
                out.append((session_id, hop))
        else:
            out.append((session_id, case))
    return out


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    per_conf_counts: dict[str, int] = defaultdict(int)
    per_conf_top1: dict[str, int] = defaultdict(int)
    per_conf_handoff: dict[str, int] = defaultdict(int)
    per_conf_avg_k: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        conf = row["confidence"]
        per_conf_counts[conf] += 1
        per_conf_top1[conf] += int(row["hit_at_1"])
        per_conf_handoff[conf] += int(row["hit_at_recommended_k"])
        per_conf_avg_k[conf].append(int(row["recommended_handoff_k"]))
    confidence_buckets = {}
    for conf in sorted(per_conf_counts):
        c = per_conf_counts[conf]
        confidence_buckets[conf] = {
            "count": c,
            "top1_accuracy": (per_conf_top1[conf] / c) if c else 0.0,
            "handoff_accuracy_at_recommended_k": (per_conf_handoff[conf] / c) if c else 0.0,
            "avg_recommended_handoff_k": (
                statistics.mean(per_conf_avg_k[conf]) if per_conf_avg_k[conf] else 0.0
            ),
        }
    return {
        "num_items": n,
        "top1_accuracy": (sum(int(r["hit_at_1"]) for r in rows) / n) if n else 0.0,
        "handoff_accuracy_at_recommended_k": (sum(int(r["hit_at_recommended_k"]) for r in rows) / n) if n else 0.0,
        "avg_recommended_handoff_k": (
            statistics.mean(r["recommended_handoff_k"] for r in rows) if n else 0.0
        ),
        "confidence_buckets": confidence_buckets,
    }


def write_summary_markdown(rows: list[dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# Confidence Calibration Across Scale",
        "",
        "| Scope | Top-1 | Handoff@k | Avg k | High Top-1 | Medium Top-1 | Low Top-1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        buckets = row["confidence_buckets"]
        lines.append(
            f"| {row['scope']} | {row['top1_accuracy']:.3f} | {row['handoff_accuracy_at_recommended_k']:.3f} | "
            f"{row['avg_recommended_handoff_k']:.3f} | "
            f"{buckets.get('high', {}).get('top1_accuracy', 0.0):.3f} | "
            f"{buckets.get('medium', {}).get('top1_accuracy', 0.0):.3f} | "
            f"{buckets.get('low', {}).get('top1_accuracy', 0.0):.3f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run confidence calibration benchmarks across scales.")
    ap.add_argument("--suite", action="append", default=[], help="Optional suite names to run.")
    args = ap.parse_args()

    suites = build_suite_specs()
    if args.suite:
        wanted = set(args.suite)
        suites = [suite for suite in suites if suite.name in wanted]
    if not suites:
        raise SystemExit("No suites selected.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_suite: dict[str, Any] = {}
    scale_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summary_rows: list[dict[str, Any]] = []

    for suite in suites:
        raw = json.loads(suite.cases_path.read_text(encoding="utf-8"))
        cases = raw.get("cases", raw)
        engine = make_engine(suite.index_dir)
        rows = []
        for session_id, item in iter_suite_steps(cases):
            rows.append(evaluate_step(engine, item, session_id, keep_memory=True))
        summary = summarize_rows(rows)
        per_suite[suite.name] = {
            "scale": suite.scale,
            "summary": summary,
            "rows": rows,
        }
        suite_out = RESULTS_DIR / f"{suite.name}.json"
        suite_out.write_text(json.dumps(per_suite[suite.name], indent=2), encoding="utf-8")
        scale_rows[suite.scale].extend(rows)
        summary_rows.append({"scope": suite.name, **summary})

    for scale, rows in scale_rows.items():
        summary_rows.append({"scope": f"scale.{scale}", **summarize_rows(rows)})

    report = {
        "per_suite": per_suite,
        "summary_rows": summary_rows,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary_markdown(summary_rows, RESULTS_DIR / "summary.md")
    print(json.dumps({"summary_rows": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
