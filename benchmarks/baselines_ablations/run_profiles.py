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
RESULTS_DIR = ROOT / "benchmarks" / "results" / "baselines_ablations"
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


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    overrides: dict[str, Any]
    keep_memory: bool = True


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
            sum(int(r["hit_at_recommended_k"]) for r in rows) / n if n else 0.0
        ),
        "avg_recommended_handoff_k": (
            statistics.mean(r["recommended_handoff_k"] for r in rows) if n else 0.0
        ),
        "confidence_buckets": confidence_buckets,
    }


def make_engine(index_dir: Path, overrides: dict[str, Any]) -> RouterEngine:
    cfg = load_config()
    cfg.index_dir = index_dir.resolve()
    for key, value in overrides.items():
        setattr(cfg, key, value)
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


def evaluate_single_tool(engine: RouterEngine, cases: list[dict[str, Any]], keep_memory: bool) -> dict[str, Any]:
    rows = []
    for case in cases:
        session_id = case.get("session_id") or f"fc-{case['id']}"
        rows.append(evaluate_step(engine, case, session_id, keep_memory))
    return {"summary": summarize_rows(rows), "results": rows}


def evaluate_multi_hop(engine: RouterEngine, cases: list[dict[str, Any]], keep_memory: bool) -> dict[str, Any]:
    case_results = []
    all_rows = []
    for case in cases:
        session_id = case.get("session_id") or f"fc-{case['id']}"
        rows = [evaluate_step(engine, hop, session_id, keep_memory) for hop in case["hops"]]
        all_rows.extend(rows)
        case_results.append(
            {
                "case_id": case["id"],
                "description": case.get("description", ""),
                "chain_success_at_1": all(r["hit_at_1"] for r in rows),
                "chain_success_at_3": all(r["hit_at_3"] for r in rows),
                "chain_success_at_5": all(r["hit_at_5"] for r in rows),
                "hops": rows,
            }
        )

    summary = summarize_rows(all_rows)
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


def evaluate_multi_tool(engine: RouterEngine, cases: list[dict[str, Any]], keep_memory: bool) -> dict[str, Any]:
    case_results = []
    all_rows = []
    for case in cases:
        session_id = case.get("session_id") or f"fc-{case['id']}"
        rows = [evaluate_step(engine, subtask, session_id, keep_memory) for subtask in case["subtasks"]]
        all_rows.extend(rows)
        case_results.append(
            {
                "case_id": case["id"],
                "user_intent": case.get("user_intent", ""),
                "all_targets_at_1": all(r["hit_at_1"] for r in rows),
                "all_targets_at_3": all(r["hit_at_3"] for r in rows),
                "all_targets_at_5": all(r["hit_at_5"] for r in rows),
                "subtasks": rows,
            }
        )

    summary = summarize_rows(all_rows)
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


def evaluate_suite(engine: RouterEngine, suite_path: Path, keep_memory: bool) -> dict[str, Any]:
    raw = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = raw.get("cases", raw)
    if cases and "subtasks" in cases[0]:
        report = evaluate_multi_tool(engine, cases, keep_memory)
    elif cases and "hops" in cases[0]:
        report = evaluate_multi_hop(engine, cases, keep_memory)
    else:
        report = evaluate_single_tool(engine, cases, keep_memory)
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


def build_profile_specs() -> list[ProfileSpec]:
    return [
        ProfileSpec(
            "dense_only",
            {
                "dense_retrieval_enabled": True,
                "bm25_retrieval_enabled": False,
                "splade_enabled": False,
            },
        ),
        ProfileSpec(
            "bm25_only",
            {
                "dense_retrieval_enabled": False,
                "bm25_retrieval_enabled": True,
                "splade_enabled": False,
            },
        ),
        ProfileSpec(
            "dense_bm25",
            {
                "dense_retrieval_enabled": True,
                "bm25_retrieval_enabled": True,
                "splade_enabled": False,
            },
        ),
        ProfileSpec(
            "full_stack",
            {
                "dense_retrieval_enabled": True,
                "bm25_retrieval_enabled": True,
                "splade_enabled": True,
            },
        ),
        ProfileSpec(
            "no_handoff_policy",
            {
                "dense_retrieval_enabled": True,
                "bm25_retrieval_enabled": True,
                "splade_enabled": True,
                "handoff_enabled": False,
            },
        ),
        ProfileSpec(
            "no_session_memory",
            {
                "dense_retrieval_enabled": True,
                "bm25_retrieval_enabled": True,
                "splade_enabled": True,
            },
            keep_memory=False,
        ),
    ]


def write_summary_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    header = [
        "# Baselines And Ablations",
        "",
        "| Suite | Profile | Top-1 | Top-3 | Top-5 | Handoff@k | Extra |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    body = []
    for row in rows:
        extra = row.get("case_metric_label", "")
        extra_value = row.get("case_metric_value")
        extra_text = f"{extra}={extra_value:.3f}" if extra and isinstance(extra_value, float) else ""
        body.append(
            f"| {row['suite']} | {row['profile']} | {row['top1_accuracy']:.3f} | {row['top3_accuracy']:.3f} | {row['top5_accuracy']:.3f} | {row['handoff_accuracy_at_recommended_k']:.3f} | {extra_text} |"
        )
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baseline/ablation profiles over functional-correctness suites.")
    ap.add_argument("--suite", action="append", default=[], help="Suite name to run; can be repeated.")
    ap.add_argument("--profile", action="append", default=[], help="Profile name to run; can be repeated.")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suites = build_suite_specs()
    profiles = build_profile_specs()
    if args.suite:
        wanted = set(args.suite)
        suites = [suite for suite in suites if suite.name in wanted]
    if args.profile:
        wanted = set(args.profile)
        profiles = [profile for profile in profiles if profile.name in wanted]
    if not suites:
        raise SystemExit("No suites selected.")
    if not profiles:
        raise SystemExit("No profiles selected.")

    summary_rows: list[dict[str, Any]] = []
    full_report: dict[str, Any] = {"suites": {}}

    for suite in suites:
        full_report["suites"][suite.name] = {}
        for profile in profiles:
            engine = make_engine(suite.index_dir, profile.overrides)
            report = evaluate_suite(engine, suite.cases_path, profile.keep_memory)
            out_path = RESULTS_DIR / f"{suite.name}.{profile.name}.json"
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            summary = dict(report["summary"])
            case_metric_label = ""
            case_metric_value: float | None = None
            if "chain_success_rate_at_1" in summary:
                case_metric_label = "chain@1"
                case_metric_value = float(summary["chain_success_rate_at_1"])
            elif "all_targets_rate_at_1" in summary:
                case_metric_label = "all_targets@1"
                case_metric_value = float(summary["all_targets_rate_at_1"])

            row = {
                "suite": suite.name,
                "profile": profile.name,
                "top1_accuracy": float(summary["top1_accuracy"]),
                "top3_accuracy": float(summary["top3_accuracy"]),
                "top5_accuracy": float(summary["top5_accuracy"]),
                "handoff_accuracy_at_recommended_k": float(summary["handoff_accuracy_at_recommended_k"]),
                "case_metric_label": case_metric_label,
                "case_metric_value": case_metric_value,
            }
            summary_rows.append(row)
            full_report["suites"][suite.name][profile.name] = {
                "summary": summary,
                "report_path": str(out_path.relative_to(ROOT)),
            }
            print(
                json.dumps(
                    {
                        "suite": suite.name,
                        "profile": profile.name,
                        "top1_accuracy": row["top1_accuracy"],
                        "top3_accuracy": row["top3_accuracy"],
                        "top5_accuracy": row["top5_accuracy"],
                    }
                )
            )

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
