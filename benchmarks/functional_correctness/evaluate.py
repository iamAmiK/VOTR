#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


def call_route(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/route",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


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


def evaluate_step(
    base_url: str,
    item: dict[str, Any],
    session_id: str,
    keep_memory: bool,
    equivalence_lookup: dict[str, set[str]],
) -> dict[str, Any]:
    payload = {
        "server_intent": item["server_intent"],
        "tool_intent": item["tool_intent"],
        "session_id": session_id,
        "record_session": keep_memory,
    }
    out = call_route(base_url, payload)
    predicted = [t["tool_key"] for t in out.get("tools", [])]
    expected = item["expected_tool_key"]
    recommended_k = int(out.get("recommended_handoff_k", 1))
    recommended_k = max(1, min(recommended_k, len(predicted))) if predicted else 1
    row = {
        "server_intent": item["server_intent"],
        "tool_intent": item["tool_intent"],
        "expected_tool_key": expected,
        "predicted_topk": predicted[:5],
        "hit_at_1": hit_at(predicted, expected, 1),
        "hit_at_3": hit_at(predicted, expected, 3),
        "hit_at_5": hit_at(predicted, expected, 5),
        "confidence": out.get("confidence", "unknown"),
        "recommended_handoff_k": recommended_k,
        "hit_at_recommended_k": hit_at(predicted, expected, recommended_k),
        "score_gap": out.get("score_gap", 0.0),
        "top1_score": out.get("top1_score", 0.0),
        "top2_score": out.get("top2_score", 0.0),
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
        "handoff_accuracy_at_recommended_k": (sum(int(r[handoff_key]) for r in rows) / n) if n else 0.0,
        "avg_recommended_handoff_k": (
            statistics.mean(r["recommended_handoff_k"] for r in rows) if n else 0.0
        ),
        "confidence_buckets": confidence_buckets,
    }


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
    return summary


def evaluate_single_tool(
    base_url: str,
    cases: list[dict[str, Any]],
    keep_memory: bool,
    equivalence_lookup: dict[str, set[str]],
) -> dict[str, Any]:
    rows = []
    for case in cases:
        session_id = case.get("session_id") or f"fc-{case['id']}"
        rows.append(evaluate_step(base_url, case, session_id, keep_memory, equivalence_lookup))
    return {"summary": summarize_rows(rows, equivalence_lookup), "results": rows}


def evaluate_multi_hop(
    base_url: str,
    cases: list[dict[str, Any]],
    keep_memory: bool,
    equivalence_lookup: dict[str, set[str]],
) -> dict[str, Any]:
    case_results = []
    all_rows = []
    for case in cases:
        session_id = case.get("session_id") or f"fc-{case['id']}"
        rows = [
            evaluate_step(base_url, hop, session_id, keep_memory, equivalence_lookup)
            for hop in case["hops"]
        ]
        all_rows.extend(rows)
        case_result = {
            "case_id": case["id"],
            "description": case.get("description", ""),
            "chain_success_at_1": all(r["hit_at_1"] for r in rows),
            "chain_success_at_3": all(r["hit_at_3"] for r in rows),
            "chain_success_at_5": all(r["hit_at_5"] for r in rows),
            "hops": rows,
        }
        if equivalence_lookup:
            case_result["equivalence_chain_success_at_1"] = all(
                r["equivalence_hit_at_1"] for r in rows
            )
            case_result["equivalence_chain_success_at_3"] = all(
                r["equivalence_hit_at_3"] for r in rows
            )
            case_result["equivalence_chain_success_at_5"] = all(
                r["equivalence_hit_at_5"] for r in rows
            )
        case_results.append(case_result)

    summary = summarize_rows(all_rows, equivalence_lookup)
    summary["num_cases"] = len(case_results)
    summary["chain_success_rate_at_1"] = (
        sum(int(c["chain_success_at_1"]) for c in case_results) / len(case_results)
        if case_results
        else 0.0
    )
    summary["chain_success_rate_at_3"] = (
        sum(int(c["chain_success_at_3"]) for c in case_results) / len(case_results)
        if case_results
        else 0.0
    )
    summary["chain_success_rate_at_5"] = (
        sum(int(c["chain_success_at_5"]) for c in case_results) / len(case_results)
        if case_results
        else 0.0
    )
    if equivalence_lookup:
        summary["equivalence_aware"]["num_cases"] = len(case_results)
        summary["equivalence_aware"]["chain_success_rate_at_1"] = (
            sum(int(c["equivalence_chain_success_at_1"]) for c in case_results) / len(case_results)
            if case_results
            else 0.0
        )
        summary["equivalence_aware"]["chain_success_rate_at_3"] = (
            sum(int(c["equivalence_chain_success_at_3"]) for c in case_results) / len(case_results)
            if case_results
            else 0.0
        )
        summary["equivalence_aware"]["chain_success_rate_at_5"] = (
            sum(int(c["equivalence_chain_success_at_5"]) for c in case_results) / len(case_results)
            if case_results
            else 0.0
        )
    return {"summary": summary, "results": case_results}


def evaluate_multi_tool(
    base_url: str,
    cases: list[dict[str, Any]],
    keep_memory: bool,
    equivalence_lookup: dict[str, set[str]],
) -> dict[str, Any]:
    case_results = []
    all_rows = []
    for case in cases:
        session_id = case.get("session_id") or f"fc-{case['id']}"
        rows = [
            evaluate_step(base_url, subtask, session_id, keep_memory, equivalence_lookup)
            for subtask in case["subtasks"]
        ]
        all_rows.extend(rows)
        case_result = {
            "case_id": case["id"],
            "user_intent": case.get("user_intent", ""),
            "all_targets_at_1": all(r["hit_at_1"] for r in rows),
            "all_targets_at_3": all(r["hit_at_3"] for r in rows),
            "all_targets_at_5": all(r["hit_at_5"] for r in rows),
            "subtasks": rows,
        }
        if equivalence_lookup:
            case_result["equivalence_all_targets_at_1"] = all(
                r["equivalence_hit_at_1"] for r in rows
            )
            case_result["equivalence_all_targets_at_3"] = all(
                r["equivalence_hit_at_3"] for r in rows
            )
            case_result["equivalence_all_targets_at_5"] = all(
                r["equivalence_hit_at_5"] for r in rows
            )
        case_results.append(case_result)

    summary = summarize_rows(all_rows, equivalence_lookup)
    summary["num_cases"] = len(case_results)
    summary["all_targets_rate_at_1"] = (
        sum(int(c["all_targets_at_1"]) for c in case_results) / len(case_results)
        if case_results
        else 0.0
    )
    summary["all_targets_rate_at_3"] = (
        sum(int(c["all_targets_at_3"]) for c in case_results) / len(case_results)
        if case_results
        else 0.0
    )
    summary["all_targets_rate_at_5"] = (
        sum(int(c["all_targets_at_5"]) for c in case_results) / len(case_results)
        if case_results
        else 0.0
    )
    if equivalence_lookup:
        summary["equivalence_aware"]["num_cases"] = len(case_results)
        summary["equivalence_aware"]["all_targets_rate_at_1"] = (
            sum(int(c["equivalence_all_targets_at_1"]) for c in case_results) / len(case_results)
            if case_results
            else 0.0
        )
        summary["equivalence_aware"]["all_targets_rate_at_3"] = (
            sum(int(c["equivalence_all_targets_at_3"]) for c in case_results) / len(case_results)
            if case_results
            else 0.0
        )
        summary["equivalence_aware"]["all_targets_rate_at_5"] = (
            sum(int(c["equivalence_all_targets_at_5"]) for c in case_results) / len(case_results)
            if case_results
            else 0.0
        )
    return {"summary": summary, "results": case_results}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate functional correctness benchmark suites.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8765")
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--equivalence-map", type=Path, default=None)
    ap.add_argument("--keep-memory", action="store_true")
    args = ap.parse_args()

    raw = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = raw.get("cases", raw)
    if not isinstance(cases, list) or not cases:
        raise SystemExit("Cases file must contain a non-empty list under `cases` or top-level.")

    suite = raw.get("suite", "")
    equivalence_lookup = load_equivalence_lookup(args.equivalence_map)
    if cases and "subtasks" in cases[0]:
        report = evaluate_multi_tool(args.base_url, cases, args.keep_memory, equivalence_lookup)
    elif cases and "hops" in cases[0]:
        report = evaluate_multi_hop(args.base_url, cases, args.keep_memory, equivalence_lookup)
    else:
        report = evaluate_single_tool(args.base_url, cases, args.keep_memory, equivalence_lookup)

    report["suite"] = suite
    if args.equivalence_map:
        report["equivalence_map"] = str(args.equivalence_map)
    print(json.dumps(report["summary"], indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved benchmark report to: {args.out}")


if __name__ == "__main__":
    main()
