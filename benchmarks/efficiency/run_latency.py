#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "benchmarks" / "results" / "efficiency"
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


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * p))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def estimated_tokens_from_chars(chars: int) -> float:
    return chars / 4.0


def make_engine(index_dir: Path) -> RouterEngine:
    cfg = load_config()
    cfg.index_dir = index_dir.resolve()
    registry = IndexRegistry(cfg)
    index = registry.load_index()
    embedder = OpenAIEmbedder(cfg)
    sessions = SessionMemory(cfg.session_ttl_seconds)
    return RouterEngine(cfg, index, embedder, sessions)


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
    latencies = [row["latency_ms"] for row in rows]
    compressed_chars = [row["compressed_chars"] for row in rows]
    compressed_tokens = [row["estimated_compressed_tokens"] for row in rows]
    return {
        "num_items": len(rows),
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "mean": (statistics.mean(latencies) if latencies else 0.0),
        },
        "returned_tools_mean": (statistics.mean(row["returned_tools"] for row in rows) if rows else 0.0),
        "avg_recommended_handoff_k": (
            statistics.mean(row["recommended_handoff_k"] for row in rows) if rows else 0.0
        ),
        "compressed_candidate_chars": {
            "mean": (statistics.mean(compressed_chars) if compressed_chars else 0.0),
            "p95": percentile(compressed_chars, 0.95),
        },
        "estimated_compressed_tokens": {
            "mean": (statistics.mean(compressed_tokens) if compressed_tokens else 0.0),
            "p95": percentile(compressed_tokens, 0.95),
        },
        "embedding_api_calls_per_route": 2.0 if rows else 0.0,
    }


def write_summary_markdown(rows: list[dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# Efficiency Benchmarks",
        "",
        "| Scope | Latency p50 ms | Latency p95 ms | Mean Returned Tools | Avg k | Mean Cand Tokens | p95 Cand Tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scope']} | {row['latency_ms']['p50']:.2f} | {row['latency_ms']['p95']:.2f} | "
            f"{row['returned_tools_mean']:.2f} | {row['avg_recommended_handoff_k']:.2f} | "
            f"{row['estimated_compressed_tokens']['mean']:.2f} | {row['estimated_compressed_tokens']['p95']:.2f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run latency / efficiency benchmarks across scales.")
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
            start = time.perf_counter()
            out = engine.route(
                server_intent=item["server_intent"],
                tool_intent=item["tool_intent"],
                session_id=session_id,
                record_session=True,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            compressed_chars = sum(len(t.compressed) for t in out.tools)
            rows.append(
                {
                    "latency_ms": latency_ms,
                    "returned_tools": len(out.tools),
                    "recommended_handoff_k": int(out.recommended_handoff_k),
                    "compressed_chars": compressed_chars,
                    "estimated_compressed_tokens": estimated_tokens_from_chars(compressed_chars),
                    "confidence": out.confidence,
                }
            )

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

    report = {"per_suite": per_suite, "summary_rows": summary_rows}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary_markdown(summary_rows, RESULTS_DIR / "summary.md")
    print(json.dumps({"summary_rows": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
