#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class HopResult:
    expected_labels: list[str]
    topk_predicted: list[str]
    hit_at_1: bool
    hit_at_k: bool


def call_route(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/route",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def evaluate_case(base_url: str, case: dict[str, Any], k: int, keep_memory: bool) -> dict[str, Any]:
    session_id = case.get("session_id") or f"eval-{case['id']}"
    hops: list[dict[str, Any]] = case["hops"]
    hop_results: list[HopResult] = []

    for hop in hops:
        payload = {
            "server_intent": hop["server_intent"],
            "tool_intent": hop["tool_intent"],
            "session_id": session_id,
            "record_session": keep_memory,
        }
        out = call_route(base_url, payload)
        predicted = [t["tool_key"] for t in out.get("tools", [])]
        expected_keys = hop.get("expected_tool_keys")
        if expected_keys is None:
            one = hop.get("expected_tool_key")
            expected_keys = [one] if one else []
        expected_name = hop.get("expected_tool_name")
        if expected_name:
            expected_keys.append(f"*::{expected_name}")
        expected_keys = [e for e in expected_keys if isinstance(e, str) and e]
        if not expected_keys:
            raise ValueError(
                "Each hop must include expected_tool_key, expected_tool_keys, "
                "or expected_tool_name."
            )

        def _matches(pred: str, labels: list[str]) -> bool:
            for label in labels:
                if label.startswith("*::"):
                    # Name-only matching fallback for cases where server prefix is unknown
                    if pred.endswith(label[1:]):
                        return True
                elif pred == label:
                    return True
            return False

        hit1 = len(predicted) > 0 and _matches(predicted[0], expected_keys)
        hitk = any(_matches(p, expected_keys) for p in predicted[:k])
        hop_results.append(
            HopResult(
                expected_labels=expected_keys,
                topk_predicted=predicted[:k],
                hit_at_1=hit1,
                hit_at_k=hitk,
            )
        )

    acc1 = sum(1 for r in hop_results if r.hit_at_1) / len(hop_results)
    acck = sum(1 for r in hop_results if r.hit_at_k) / len(hop_results)
    chain_success = all(r.hit_at_k for r in hop_results)

    return {
        "case_id": case["id"],
        "description": case.get("description", ""),
        "hops": [
            {
                "expected_labels": r.expected_labels,
                "predicted_topk": r.topk_predicted,
                "hit_at_1": r.hit_at_1,
                "hit_at_k": r.hit_at_k,
            }
            for r in hop_results
        ],
        "hop_acc_at_1": acc1,
        "hop_acc_at_k": acck,
        "chain_success_at_k": chain_success,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate multi-hop routing accuracy.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8765")
    ap.add_argument("--cases", type=Path, required=True, help="JSON file with test cases.")
    ap.add_argument("--k", type=int, default=5, help="Top-k for hit@k / chain success.")
    ap.add_argument("--keep-memory", action="store_true", help="Enable session memory across hops.")
    ap.add_argument("--out", type=Path, default=None, help="Optional output report JSON.")
    args = ap.parse_args()

    raw = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = raw.get("cases", raw)
    if not isinstance(cases, list) or not cases:
        raise SystemExit("Cases file must contain a non-empty list under `cases` or top-level.")

    results = [evaluate_case(args.base_url, c, args.k, args.keep_memory) for c in cases]

    mean_acc1 = statistics.mean(r["hop_acc_at_1"] for r in results)
    mean_acck = statistics.mean(r["hop_acc_at_k"] for r in results)
    chain_rate = sum(1 for r in results if r["chain_success_at_k"]) / len(results)

    summary = {
        "num_cases": len(results),
        "k": args.k,
        "keep_memory": args.keep_memory,
        "mean_hop_acc_at_1": mean_acc1,
        "mean_hop_acc_at_k": mean_acck,
        "chain_success_rate_at_k": chain_rate,
    }

    report = {"summary": summary, "results": results}
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved full report to: {args.out}")


if __name__ == "__main__":
    main()
