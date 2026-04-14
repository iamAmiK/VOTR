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
RESULTS_DIR = ROOT / "benchmarks" / "results" / "overlap_aware"
sys.path.insert(0, str(ROOT / "src"))

from mcp_router.config import load_config
from mcp_router.registry.manager import IndexRegistry
from mcp_router.retrieval.embedder import OpenAIEmbedder
from mcp_router.retrieval.engine import RouterEngine
from mcp_router.session.memory import SessionMemory


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    cases_path: Path
    index_dir: Path


def hit_at(predicted: list[str], expected: str, k: int) -> bool:
    return expected in predicted[:k]


def load_equivalence_lookup(path: Path | None) -> dict[str, set[str]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    groups = raw.get("equivalent_tool_groups", raw)
    if not isinstance(groups, list):
        raise SystemExit("Equivalence map must be a list or contain `equivalent_tool_groups`.")
    lookup: dict[str, set[str]] = {}
    for group in groups:
        if not isinstance(group, list) or len(group) < 2:
            continue
        members = {str(item) for item in group}
        for member in members:
            lookup[member] = set(members)
    return lookup


def equivalent_keys(expected: str, equivalence_lookup: dict[str, set[str]]) -> set[str]:
    return set(equivalence_lookup.get(expected, {expected}))


def equivalent_hit_at(
    predicted: list[str],
    expected: str,
    k: int,
    equivalence_lookup: dict[str, set[str]],
) -> bool:
    valid = equivalent_keys(expected, equivalence_lookup)
    return any(tool_key in valid for tool_key in predicted[:k])


def make_engine(index_dir: Path, overlap_aware_enabled: bool) -> RouterEngine:
    cfg = load_config()
    cfg.index_dir = index_dir.resolve()
    cfg.overlap_aware_enabled = overlap_aware_enabled
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
    equivalence_lookup: dict[str, set[str]],
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
    row = {
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
        "overlap_ambiguous": bool(out.overlap_ambiguous),
        "overlap_tool_keys": list(out.overlap_tool_keys),
        "overlap_servers": list(out.overlap_servers),
    }
    if equivalence_lookup:
        row["equivalent_tool_keys"] = sorted(equivalent_keys(expected, equivalence_lookup))
        row["equivalence_hit_at_1"] = equivalent_hit_at(predicted, expected, 1, equivalence_lookup)
        row["equivalence_hit_at_3"] = equivalent_hit_at(predicted, expected, 3, equivalence_lookup)
        row["equivalence_hit_at_5"] = equivalent_hit_at(predicted, expected, 5, equivalence_lookup)
        row["equivalence_hit_at_recommended_k"] = equivalent_hit_at(
            predicted,
            expected,
            recommended_k,
            equivalence_lookup,
        )
        equivalents = equivalent_keys(expected, equivalence_lookup)
        row["expected_or_equivalent_in_overlap_set"] = any(
            tool_key in equivalents for tool_key in row["overlap_tool_keys"]
        )
    else:
        row["expected_or_equivalent_in_overlap_set"] = expected in row["overlap_tool_keys"]
    return row


def summarize_metric_family(
    rows: list[dict[str, Any]],
    hit1_key: str,
    hit3_key: str,
    hit5_key: str,
    handoff_key: str,
) -> dict[str, Any]:
    n = len(rows)
    per_conf_counts: dict[str, int] = defaultdict(int)
    per_conf_top1: dict[str, int] = defaultdict(int)
    per_conf_handoff: dict[str, int] = defaultdict(int)

    for row in rows:
        conf = row["confidence"]
        per_conf_counts[conf] += 1
        per_conf_top1[conf] += int(row[hit1_key])
        per_conf_handoff[conf] += int(row[handoff_key])

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
        "top1_accuracy": (sum(int(r[hit1_key]) for r in rows) / n) if n else 0.0,
        "top3_accuracy": (sum(int(r[hit3_key]) for r in rows) / n) if n else 0.0,
        "top5_accuracy": (sum(int(r[hit5_key]) for r in rows) / n) if n else 0.0,
        "handoff_accuracy_at_recommended_k": (
            sum(int(r[handoff_key]) for r in rows) / n if n else 0.0
        ),
        "avg_recommended_handoff_k": (
            statistics.mean(r["recommended_handoff_k"] for r in rows) if n else 0.0
        ),
        "confidence_buckets": confidence_buckets,
    }


def summarize_overlap(rows: list[dict[str, Any]], equivalence_lookup: dict[str, set[str]]) -> dict[str, Any]:
    ambiguous = [row for row in rows if row["overlap_ambiguous"]]
    total = len(rows)
    count = len(ambiguous)
    summary = {
        "ambiguous_count": count,
        "ambiguous_rate": (count / total) if total else 0.0,
        "avg_overlap_set_size": (
            statistics.mean(len(row["overlap_tool_keys"]) for row in ambiguous) if ambiguous else 0.0
        ),
        "expected_or_equivalent_in_overlap_set_rate": (
            sum(int(row["expected_or_equivalent_in_overlap_set"]) for row in ambiguous) / count if count else 0.0
        ),
        "exact_top1_accuracy_on_ambiguous": (
            sum(int(row["hit_at_1"]) for row in ambiguous) / count if count else 0.0
        ),
        "handoff_accuracy_on_ambiguous": (
            sum(int(row["hit_at_recommended_k"]) for row in ambiguous) / count if count else 0.0
        ),
    }
    if equivalence_lookup:
        summary["equivalence_top1_accuracy_on_ambiguous"] = (
            sum(int(row["equivalence_hit_at_1"]) for row in ambiguous) / count if count else 0.0
        )
        summary["equivalence_handoff_accuracy_on_ambiguous"] = (
            sum(int(row["equivalence_hit_at_recommended_k"]) for row in ambiguous) / count if count else 0.0
        )
    return summary


def summarize_rows(rows: list[dict[str, Any]], equivalence_lookup: dict[str, set[str]]) -> dict[str, Any]:
    summary = summarize_metric_family(
        rows,
        hit1_key="hit_at_1",
        hit3_key="hit_at_3",
        hit5_key="hit_at_5",
        handoff_key="hit_at_recommended_k",
    )
    if equivalence_lookup:
        summary["equivalence_aware"] = summarize_metric_family(
            rows,
            hit1_key="equivalence_hit_at_1",
            hit3_key="equivalence_hit_at_3",
            hit5_key="equivalence_hit_at_5",
            handoff_key="equivalence_hit_at_recommended_k",
        )
    summary["overlap_aware"] = summarize_overlap(rows, equivalence_lookup)
    return summary


def evaluate_single_tool(
    engine: RouterEngine,
    cases: list[dict[str, Any]],
    keep_memory: bool,
    equivalence_lookup: dict[str, set[str]],
) -> dict[str, Any]:
    rows = []
    for case in cases:
        session_id = case.get("session_id") or f"fc-{case['id']}"
        rows.append(evaluate_step(engine, case, session_id, keep_memory, equivalence_lookup))
    return {"summary": summarize_rows(rows, equivalence_lookup), "results": rows}


def evaluate_multi_hop(
    engine: RouterEngine,
    cases: list[dict[str, Any]],
    keep_memory: bool,
    equivalence_lookup: dict[str, set[str]],
) -> dict[str, Any]:
    case_results = []
    all_rows = []
    for case in cases:
        session_id = case.get("session_id") or f"fc-{case['id']}"
        rows = [evaluate_step(engine, hop, session_id, keep_memory, equivalence_lookup) for hop in case["hops"]]
        all_rows.extend(rows)
        case_results.append(
            {
                "case_id": case["id"],
                "description": case.get("description", ""),
                "chain_success_at_1": all(r["hit_at_1"] for r in rows),
                "chain_success_at_3": all(r["hit_at_3"] for r in rows),
                "chain_success_at_5": all(r["hit_at_5"] for r in rows),
                "ambiguous_hops": sum(int(r["overlap_ambiguous"]) for r in rows),
                "hops": rows,
            }
        )

    summary = summarize_rows(all_rows, equivalence_lookup)
    summary["num_cases"] = len(case_results)
    summary["chain_success_rate_at_1"] = (
        sum(int(c["chain_success_at_1"]) for c in case_results) / len(case_results) if case_results else 0.0
    )
    summary["chain_success_rate_at_3"] = (
        sum(int(c["chain_success_at_3"]) for c in case_results) / len(case_results) if case_results else 0.0
    )
    summary["chain_success_rate_at_5"] = (
        sum(int(c["chain_success_at_5"]) for c in case_results) / len(case_results) if case_results else 0.0
    )
    return {"summary": summary, "results": case_results}


def evaluate_multi_tool(
    engine: RouterEngine,
    cases: list[dict[str, Any]],
    keep_memory: bool,
    equivalence_lookup: dict[str, set[str]],
) -> dict[str, Any]:
    case_results = []
    all_rows = []
    for case in cases:
        session_id = case.get("session_id") or f"fc-{case['id']}"
        rows = [
            evaluate_step(engine, subtask, session_id, keep_memory, equivalence_lookup)
            for subtask in case["subtasks"]
        ]
        all_rows.extend(rows)
        case_results.append(
            {
                "case_id": case["id"],
                "user_intent": case.get("user_intent", ""),
                "all_targets_at_1": all(r["hit_at_1"] for r in rows),
                "all_targets_at_3": all(r["hit_at_3"] for r in rows),
                "all_targets_at_5": all(r["hit_at_5"] for r in rows),
                "ambiguous_subtasks": sum(int(r["overlap_ambiguous"]) for r in rows),
                "subtasks": rows,
            }
        )

    summary = summarize_rows(all_rows, equivalence_lookup)
    summary["num_cases"] = len(case_results)
    summary["all_targets_rate_at_1"] = (
        sum(int(c["all_targets_at_1"]) for c in case_results) / len(case_results) if case_results else 0.0
    )
    summary["all_targets_rate_at_3"] = (
        sum(int(c["all_targets_at_3"]) for c in case_results) / len(case_results) if case_results else 0.0
    )
    summary["all_targets_rate_at_5"] = (
        sum(int(c["all_targets_at_5"]) for c in case_results) / len(case_results) if case_results else 0.0
    )
    return {"summary": summary, "results": case_results}


def evaluate_suite(
    engine: RouterEngine,
    suite_path: Path,
    keep_memory: bool,
    equivalence_lookup: dict[str, set[str]],
) -> dict[str, Any]:
    raw = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = raw.get("cases", raw)
    if cases and "subtasks" in cases[0]:
        report = evaluate_multi_tool(engine, cases, keep_memory, equivalence_lookup)
    elif cases and "hops" in cases[0]:
        report = evaluate_multi_hop(engine, cases, keep_memory, equivalence_lookup)
    else:
        report = evaluate_single_tool(engine, cases, keep_memory, equivalence_lookup)
    report["suite"] = raw.get("suite", "")
    return report


def build_suite_specs() -> list[SuiteSpec]:
    fc = ROOT / "benchmarks" / "functional_correctness"
    data = ROOT / "data"
    return [
        SuiteSpec("single_tool.small_bloomberg", fc / "single_tool.bloomberg.clean.json", data / "index_small_bloomberg"),
        SuiteSpec("single_tool.small_github", fc / "single_tool.github.clean.json", data / "index_small_github"),
        SuiteSpec("single_tool.small_telegram", fc / "single_tool.telegram.clean.json", data / "index_small_telegram"),
        SuiteSpec("single_tool.medium_250", fc / "single_tool.medium_250.clean.json", data / "index_medium_50"),
        SuiteSpec("single_tool.large", fc / "single_tool.clean.json", data / "index"),
        SuiteSpec("multi_hop.small_100", fc / "multi_hop.small_100.cross_app.json", data / "index_small_3"),
        SuiteSpec("multi_hop.medium_250", fc / "multi_hop.medium_250.cross_app.json", data / "index_medium_50"),
        SuiteSpec("multi_hop.large", fc / "multi_hop.large.cross_app.json", data / "index"),
        SuiteSpec("multi_tool.small_100", fc / "multi_tool.small_100.single_turn.json", data / "index_small_3"),
        SuiteSpec("multi_tool.medium_250", fc / "multi_tool.medium_250.single_turn.json", data / "index_medium_50"),
        SuiteSpec("multi_tool.large", fc / "multi_tool.large.single_turn.json", data / "index"),
    ]


def write_summary_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Overlap-Aware Benchmark",
        "",
        "| Suite | Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Ambiguous Rate | Exact@1 On Ambiguous | Eq@1 On Ambiguous |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['suite']} | {row['profile']} | {row['top1_accuracy']:.3f} | {row['top3_accuracy']:.3f} | {row['top5_accuracy']:.3f} | {row['handoff_accuracy_at_recommended_k']:.3f} | {row['ambiguous_rate']:.3f} | {row['exact_top1_on_ambiguous']:.3f} | {row['equivalence_top1_on_ambiguous']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run overlap-aware routing benchmark over functional-correctness suites.")
    ap.add_argument("--suite", action="append", default=[], help="Suite name to run; can be repeated.")
    ap.add_argument(
        "--equivalence-map",
        type=Path,
        default=ROOT / "benchmarks" / "functional_correctness" / "equivalence_map.json",
    )
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suites = build_suite_specs()
    if args.suite:
        wanted = set(args.suite)
        suites = [suite for suite in suites if suite.name in wanted]
    if not suites:
        raise SystemExit("No suites selected.")

    equivalence_lookup = load_equivalence_lookup(args.equivalence_map)
    summary_rows: list[dict[str, Any]] = []
    full_report: dict[str, Any] = {"suites": {}}

    for suite in suites:
        full_report["suites"][suite.name] = {}
        for profile_name, overlap_enabled in (("baseline", False), ("overlap_aware", True)):
            engine = make_engine(suite.index_dir, overlap_aware_enabled=overlap_enabled)
            report = evaluate_suite(engine, suite.cases_path, True, equivalence_lookup)
            out_path = RESULTS_DIR / f"{suite.name}.{profile_name}.json"
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            overlap_summary = report["summary"]["overlap_aware"]
            equivalence_summary = report["summary"].get("equivalence_aware", {})
            row = {
                "suite": suite.name,
                "profile": profile_name,
                "top1_accuracy": float(report["summary"]["top1_accuracy"]),
                "top3_accuracy": float(report["summary"]["top3_accuracy"]),
                "top5_accuracy": float(report["summary"]["top5_accuracy"]),
                "handoff_accuracy_at_recommended_k": float(report["summary"]["handoff_accuracy_at_recommended_k"]),
                "ambiguous_rate": float(overlap_summary["ambiguous_rate"]),
                "exact_top1_on_ambiguous": float(overlap_summary["exact_top1_accuracy_on_ambiguous"]),
                "equivalence_top1_on_ambiguous": float(
                    overlap_summary.get("equivalence_top1_accuracy_on_ambiguous", overlap_summary["exact_top1_accuracy_on_ambiguous"])
                ),
                "report_path": str(out_path.relative_to(ROOT)),
                "equivalence_top1_accuracy": float(equivalence_summary.get("top1_accuracy", report["summary"]["top1_accuracy"])),
            }
            summary_rows.append(row)
            full_report["suites"][suite.name][profile_name] = {
                "summary": report["summary"],
                "report_path": row["report_path"],
            }
            print(json.dumps(row))

    summary_json_path = RESULTS_DIR / "summary.json"
    summary_md_path = RESULTS_DIR / "summary.md"
    summary_json_path.write_text(
        json.dumps({"rows": summary_rows, "report": full_report}, indent=2),
        encoding="utf-8",
    )
    write_summary_markdown(summary_rows, summary_md_path)
    print(f"Saved summary JSON to: {summary_json_path}")
    print(f"Saved summary Markdown to: {summary_md_path}")


if __name__ == "__main__":
    main()
