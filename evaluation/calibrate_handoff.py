#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import urllib.request
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


def matches(pred: str, labels: list[str]) -> bool:
    for label in labels:
        if label.startswith("*::"):
            if pred.endswith(label[1:]):
                return True
        elif pred == label:
            return True
    return False


def expected_labels_for_hop(hop: dict[str, Any]) -> list[str]:
    keys = hop.get("expected_tool_keys")
    if keys is None:
        one = hop.get("expected_tool_key")
        keys = [one] if one else []
    expected_name = hop.get("expected_tool_name")
    if expected_name:
        keys.append(f"*::{expected_name}")
    return [e for e in keys if isinstance(e, str) and e]


def bucket_from_gap(gap: float, high: float, medium: float) -> str:
    if gap >= high:
        return "high"
    if gap >= medium:
        return "medium"
    return "low"


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate handoff confidence thresholds from labeled cases.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8765")
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--keep-memory", action="store_true")
    ap.add_argument("--k-high", type=int, default=1)
    ap.add_argument("--k-medium", type=int, default=3)
    ap.add_argument("--k-low", type=int, default=5)
    ap.add_argument("--target-high-precision", type=float, default=0.95)
    ap.add_argument("--target-medium-precision", type=float, default=0.85)
    ap.add_argument("--min-high-count", type=int, default=2)
    ap.add_argument("--min-medium-count", type=int, default=2)
    ap.add_argument("--max-threshold-candidates", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    raw = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = raw.get("cases", raw)
    if not isinstance(cases, list) or not cases:
        raise SystemExit("Cases file must contain a non-empty list under `cases` or top-level.")

    # Gather observations once from live router
    obs: list[dict[str, Any]] = []
    for case in cases:
        session_id = case.get("session_id") or f"eval-{case['id']}"
        for i, hop in enumerate(case["hops"]):
            labels = expected_labels_for_hop(hop)
            if not labels:
                raise ValueError("Each hop must include expected labels.")
            payload = {
                "server_intent": hop["server_intent"],
                "tool_intent": hop["tool_intent"],
                "session_id": session_id,
                "record_session": args.keep_memory,
            }
            out = call_route(args.base_url, payload)
            predicted = [t["tool_key"] for t in out.get("tools", [])]
            gap = float(out.get("score_gap", 0.0))
            top1_ok = len(predicted) > 0 and matches(predicted[0], labels)
            top3_ok = any(matches(p, labels) for p in predicted[:3])
            top5_ok = any(matches(p, labels) for p in predicted[:5])
            obs.append(
                {
                    "case_id": case["id"],
                    "hop_index": i,
                    "gap": gap,
                    "top1_ok": top1_ok,
                    "top3_ok": top3_ok,
                    "top5_ok": top5_ok,
                }
            )

    if not obs:
        raise SystemExit("No hops found.")

    gaps = sorted({o["gap"] for o in obs})
    if len(gaps) > args.max_threshold_candidates:
        # Downsample candidate thresholds across range
        step = max(1, len(gaps) // args.max_threshold_candidates)
        gaps = gaps[::step]

    def evaluate(high: float, medium: float) -> dict[str, Any]:
        counts = {"high": 0, "medium": 0, "low": 0}
        top1_hits = {"high": 0, "medium": 0, "low": 0}
        handoff_hits = 0
        total_k = 0
        for o in obs:
            b = bucket_from_gap(o["gap"], high, medium)
            counts[b] += 1
            top1_hits[b] += int(o["top1_ok"])
            if b == "high":
                k = args.k_high
                ok = o["top1_ok"] if k == 1 else (o["top3_ok"] if k <= 3 else o["top5_ok"])
            elif b == "medium":
                k = args.k_medium
                ok = o["top1_ok"] if k == 1 else (o["top3_ok"] if k <= 3 else o["top5_ok"])
            else:
                k = args.k_low
                ok = o["top1_ok"] if k == 1 else (o["top3_ok"] if k <= 3 else o["top5_ok"])
            total_k += k
            handoff_hits += int(ok)

        n = len(obs)
        high_prec = (top1_hits["high"] / counts["high"]) if counts["high"] else 0.0
        med_prec = (top1_hits["medium"] / counts["medium"]) if counts["medium"] else 0.0
        low_prec = (top1_hits["low"] / counts["low"]) if counts["low"] else 0.0
        handoff_acc = handoff_hits / n
        avg_k = total_k / n

        constraints_ok = (
            counts["high"] >= args.min_high_count
            and counts["medium"] >= args.min_medium_count
            and high_prec >= args.target_high_precision
            and med_prec >= args.target_medium_precision
        )

        return {
            "high": high,
            "medium": medium,
            "counts": counts,
            "top1_precision": {"high": high_prec, "medium": med_prec, "low": low_prec},
            "handoff_accuracy": handoff_acc,
            "avg_recommended_k": avg_k,
            "constraints_ok": constraints_ok,
        }

    all_results = []
    for medium, high in itertools.product(gaps, gaps):
        if high <= medium:
            continue
        all_results.append(evaluate(high=high, medium=medium))

    if not all_results:
        raise SystemExit("No threshold pairs evaluated.")

    constrained = [r for r in all_results if r["constraints_ok"]]
    if constrained:
        # Prefer best handoff accuracy, then smallest average k
        best = sorted(
            constrained,
            key=lambda r: (-r["handoff_accuracy"], r["avg_recommended_k"], -r["high"], -r["medium"]),
        )[0]
        selection_mode = "constraint_satisfying_best"
    else:
        # Fallback: maximize handoff, then high precision, then lower avg k
        best = sorted(
            all_results,
            key=lambda r: (
                -r["handoff_accuracy"],
                -r["top1_precision"]["high"],
                -r["top1_precision"]["medium"],
                r["avg_recommended_k"],
            ),
        )[0]
        selection_mode = "fallback_best_tradeoff"

    summary = {
        "num_cases": len(cases),
        "num_hops": len(obs),
        "selection_mode": selection_mode,
        "recommended_thresholds": {
            "handoff_gap_high": best["high"],
            "handoff_gap_medium": best["medium"],
        },
        "policy_k": {
            "handoff_k_high": args.k_high,
            "handoff_k_medium": args.k_medium,
            "handoff_k_low": args.k_low,
        },
        "metrics_at_recommended_thresholds": {
            "handoff_accuracy": best["handoff_accuracy"],
            "avg_recommended_k": best["avg_recommended_k"],
            "bucket_counts": best["counts"],
            "top1_precision_by_bucket": best["top1_precision"],
            "constraints_ok": best["constraints_ok"],
        },
        "config_snippet": {
            "handoff_gap_high": best["high"],
            "handoff_gap_medium": best["medium"],
            "handoff_k_high": args.k_high,
            "handoff_k_medium": args.k_medium,
            "handoff_k_low": args.k_low,
        },
    }

    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {"summary": summary, "all_results": all_results}
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved calibration report to: {args.out}")


if __name__ == "__main__":
    main()
