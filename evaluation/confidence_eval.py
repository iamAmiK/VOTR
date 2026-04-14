#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate confidence calibration for multi-hop routing.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8765")
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--keep-memory", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    raw = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = raw.get("cases", raw)
    if not isinstance(cases, list) or not cases:
        raise SystemExit("Cases file must contain a non-empty list under `cases` or top-level.")

    per_conf_counts: dict[str, int] = defaultdict(int)
    per_conf_top1_hits: dict[str, int] = defaultdict(int)
    per_conf_handoff_hits: dict[str, int] = defaultdict(int)
    total_hops = 0
    top1_hits = 0
    handoff_hits = 0
    rows: list[dict[str, Any]] = []

    for case in cases:
        session_id = case.get("session_id") or f"eval-{case['id']}"
        for i, hop in enumerate(case["hops"]):
            payload = {
                "server_intent": hop["server_intent"],
                "tool_intent": hop["tool_intent"],
                "session_id": session_id,
                "record_session": args.keep_memory,
            }
            out = call_route(args.base_url, payload)
            predicted = [t["tool_key"] for t in out.get("tools", [])]
            labels = expected_labels_for_hop(hop)
            if not labels:
                raise ValueError("Each hop must have expected labels.")

            conf = out.get("confidence", "unknown")
            rk = int(out.get("recommended_handoff_k", 1))
            rk = max(1, min(rk, len(predicted))) if predicted else 1

            hit1 = len(predicted) > 0 and matches(predicted[0], labels)
            hit_rk = any(matches(p, labels) for p in predicted[:rk])

            total_hops += 1
            top1_hits += int(hit1)
            handoff_hits += int(hit_rk)
            per_conf_counts[conf] += 1
            per_conf_top1_hits[conf] += int(hit1)
            per_conf_handoff_hits[conf] += int(hit_rk)

            rows.append(
                {
                    "case_id": case["id"],
                    "hop_index": i,
                    "confidence": conf,
                    "recommended_handoff_k": rk,
                    "expected_labels": labels,
                    "predicted_top1": predicted[0] if predicted else None,
                    "hit_at_1": hit1,
                    "hit_at_recommended_k": hit_rk,
                    "score_gap": out.get("score_gap", 0.0),
                    "top1_score": out.get("top1_score", 0.0),
                    "top2_score": out.get("top2_score", 0.0),
                }
            )

    confidence_buckets = {}
    for conf in sorted(per_conf_counts.keys()):
        c = per_conf_counts[conf]
        confidence_buckets[conf] = {
            "count": c,
            "top1_accuracy": (per_conf_top1_hits[conf] / c) if c else 0.0,
            "handoff_accuracy_at_recommended_k": (per_conf_handoff_hits[conf] / c) if c else 0.0,
        }

    summary = {
        "num_cases": len(cases),
        "num_hops": total_hops,
        "overall_top1_accuracy": (top1_hits / total_hops) if total_hops else 0.0,
        "overall_handoff_accuracy_at_recommended_k": (handoff_hits / total_hops) if total_hops else 0.0,
        "confidence_buckets": confidence_buckets,
    }
    report = {"summary": summary, "rows": rows}

    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved confidence report to: {args.out}")


if __name__ == "__main__":
    main()
