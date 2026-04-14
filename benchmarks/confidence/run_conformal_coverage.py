#!/usr/bin/env python3
"""
Conformal Coverage Validation
==============================
Loads a pre-computed calibration JSON produced by scripts/calibrate_conformal.py
and generates a rigorous coverage report for the paper.

For the held-out validation split it reports:

  Per-bucket coverage:
    • k=1 bucket  → Recall@1 (should be ≥ coverage target)
    • k=3 bucket  → Recall@3 (should be ≥ coverage target)
    • k=5 bucket  → Recall@5 (should be ≥ coverage target)
    • null bucket → OOD precision (fraction that were truly out-of-catalog)

  Policy comparison on the same validation set:
    conformal policy  vs  gap-threshold policy  vs  fixed-k=1/3/5

  Bootstrap 95% confidence intervals on each recall estimate.

Outputs:
  benchmarks/results/confidence/conformal_coverage.json
  benchmarks/results/confidence/conformal_coverage.md  (paper-ready table)

Usage:
    python benchmarks/confidence/run_conformal_coverage.py
    python benchmarks/confidence/run_conformal_coverage.py \\
        --cal benchmarks/results/conformal_calibration.json \\
        --coverage 0.95
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "benchmarks" / "results" / "confidence"
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    stat_fn,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boots.append(stat_fn(sample))
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


def recall_at_k(records: list[dict[str, Any]], k: int) -> float:
    if not records:
        return 0.0
    return sum(
        1 for r in records
        if r.get("correct_rank") is not None and r["correct_rank"] <= k
    ) / len(records)


# ---------------------------------------------------------------------------
# Policy simulation (operates on pre-scored validation records)
# ---------------------------------------------------------------------------

def apply_conformal_policy(
    records: list[dict[str, Any]],
    t_k1: float,
    t_k3: float,
    t_k5: float,
    t_null: float,
) -> list[dict[str, Any]]:
    """Assign handoff_k to each record according to conformal thresholds."""
    out = []
    for r in records:
        nc = float(r["nonconformity_score"])
        if nc > t_null:
            k_assigned = 0
        elif nc <= t_k1:
            k_assigned = 1
        elif nc <= t_k3:
            k_assigned = 3
        else:
            k_assigned = 5
        cr = r.get("correct_rank")
        out.append({
            **r,
            "k_assigned": k_assigned,
            "hit": (cr is not None and cr <= k_assigned) if k_assigned > 0 else False,
        })
    return out


def apply_gap_policy(
    records: list[dict[str, Any]],
    gap_high: float,
    gap_medium: float,
    k_high: int,
    k_medium: int,
    k_low: int,
    null_route_support_threshold: float = -1.0,
) -> list[dict[str, Any]]:
    """Legacy gap-threshold policy simulation on validation records."""
    out = []
    for r in records:
        gap = float(r["score_gap"])
        if gap >= gap_high:
            k_assigned = k_high
        elif gap >= gap_medium:
            k_assigned = k_medium
        else:
            k_assigned = k_low
        cr = r.get("correct_rank")
        out.append({
            **r,
            "k_assigned": k_assigned,
            "hit": (cr is not None and cr <= k_assigned),
        })
    return out


def apply_fixed_k(records: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    out = []
    for r in records:
        cr = r.get("correct_rank")
        out.append({
            **r,
            "k_assigned": k,
            "hit": cr is not None and cr <= k,
        })
    return out


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def policy_summary(assigned: list[dict[str, Any]], label: str) -> dict[str, Any]:
    n = len(assigned)
    if n == 0:
        return {"policy": label, "n": 0}

    hits = [int(r["hit"]) for r in assigned]
    ks = [r["k_assigned"] for r in assigned]
    accuracy = sum(hits) / n
    avg_k = sum(ks) / n
    null_rate = sum(1 for r in assigned if r["k_assigned"] == 0) / n

    lo, hi = bootstrap_ci(hits, lambda s: sum(s) / len(s) if s else 0.0)
    return {
        "policy": label,
        "n": n,
        "accuracy": round(accuracy, 4),
        "accuracy_ci_95": (round(lo, 4), round(hi, 4)),
        "avg_injected_k": round(avg_k, 3),
        "null_route_rate": round(null_rate, 4),
    }


def bucket_report(
    val_records: list[dict[str, Any]],
    t_k1: float, t_k3: float, t_k5: float, t_null: float,
) -> dict[str, Any]:
    """Per-bucket empirical recall + bootstrap CI on the validation split."""
    buckets: dict[str, list[dict]] = {"k1": [], "k3": [], "k5": [], "null": []}
    for r in val_records:
        nc = float(r["nonconformity_score"])
        if nc > t_null:
            buckets["null"].append(r)
        elif nc <= t_k1:
            buckets["k1"].append(r)
        elif nc <= t_k3:
            buckets["k3"].append(r)
        else:
            buckets["k5"].append(r)

    def _recall(recs: list[dict], k: int) -> float:
        if not recs:
            return 0.0
        return sum(1 for r in recs if r.get("correct_rank") is not None and r["correct_rank"] <= k) / len(recs)

    result = {}
    for bucket, k_target in [("k1", 1), ("k3", 3), ("k5", 5)]:
        recs = buckets[bucket]
        rec = _recall(recs, k_target)
        indicator = [int(r.get("correct_rank") is not None and r["correct_rank"] <= k_target) for r in recs]
        lo, hi = bootstrap_ci(indicator, lambda s: sum(s) / len(s) if s else 0.0)
        result[bucket] = {
            "count": len(recs),
            f"recall@{k_target}": round(rec, 4),
            "ci_95": (round(lo, 4), round(hi, 4)),
        }

    null_recs = buckets["null"]
    ood_precision = sum(1 for r in null_recs if r.get("correct_rank") is None) / len(null_recs) if null_recs else 0.0
    result["null"] = {
        "count": len(null_recs),
        "ood_precision": round(ood_precision, 4),
    }
    return result


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def write_markdown(
    coverage_target: float,
    bucket_rep: dict[str, Any],
    policy_rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# Conformal Coverage Validation",
        "",
        f"**Coverage target:** {coverage_target:.0%}",
        "",
        "## Per-Bucket Empirical Coverage (validation split)",
        "",
        "Each bucket uses the conformal threshold derived from the calibration split.",
        "CI = 95% bootstrap confidence interval.",
        "",
        "| Bucket | k injected | Count | Recall@k | 95% CI |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for bucket, k in [("k1", 1), ("k3", 3), ("k5", 5)]:
        b = bucket_rep.get(bucket, {})
        recall = b.get(f"recall@{k}", 0.0)
        ci = b.get("ci_95", (0.0, 0.0))
        lines.append(f"| {bucket} | {k} | {b.get('count',0)} | {recall:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    nb = bucket_rep.get("null", {})
    lines.append(f"| null (abstain) | 0 | {nb.get('count',0)} | OOD prec: {nb.get('ood_precision',0.0):.3f} | — |")

    lines += [
        "",
        "## Policy Comparison (same validation split)",
        "",
        "Accuracy = Recall@injected_k; lower avg_k = fewer tokens injected.",
        "",
        "| Policy | Accuracy | 95% CI | Avg k | Null-route % |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for r in policy_rows:
        ci = r.get("accuracy_ci_95", (0.0, 0.0))
        lines.append(
            f"| {r['policy']} "
            f"| {r['accuracy']:.3f} "
            f"| [{ci[0]:.3f}, {ci[1]:.3f}] "
            f"| {r['avg_injected_k']:.2f} "
            f"| {r['null_route_rate']*100:.1f}% |"
        )

    lines += [
        "",
        "## Calibrated Thresholds",
        "",
        f"| nc_threshold_k1 | {thresholds['nc_threshold_k1']} |",
        "| --- | --- |",
        f"| nc_threshold_k3 | {thresholds['nc_threshold_k3']} |",
        f"| nc_threshold_k5 | {thresholds['nc_threshold_k5']} |",
        f"| nc_threshold_null | {thresholds['nc_threshold_null']} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Conformal coverage validation: per-bucket recall + policy comparison"
    )
    ap.add_argument(
        "--cal",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "conformal_calibration.json",
        help="Path to conformal_calibration.json produced by scripts/calibrate_conformal.py",
    )
    ap.add_argument("--coverage", type=float, default=0.95, help="Coverage target (informational only)")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "conformal_coverage.json")
    # Gap-threshold policy reference values (taken from config defaults)
    ap.add_argument("--gap-high",   type=float, default=0.0010426450507904708)
    ap.add_argument("--gap-medium", type=float, default=0.000836960501159334)
    ap.add_argument("--k-high",     type=int,   default=1)
    ap.add_argument("--k-medium",   type=int,   default=3)
    ap.add_argument("--k-low",      type=int,   default=5)
    args = ap.parse_args()

    if not args.cal.is_file():
        raise SystemExit(
            f"Calibration file not found: {args.cal}\n"
            "Run scripts/calibrate_conformal.py first."
        )

    print(f"Loading calibration data from: {args.cal}")
    data = json.loads(args.cal.read_text(encoding="utf-8"))
    thresholds: dict[str, Any] = data["thresholds"]
    all_records: list[dict[str, Any]] = data["all_records"]

    # Reconstruct validation split using same 80/20 deterministic split
    all_records.sort(key=lambda r: r.get("id", ""))
    split = int(len(all_records) * 0.8)
    val_records = all_records[split:]
    print(f"Validation split: {len(val_records)} records (total: {len(all_records)})")

    t_k1   = thresholds["nc_threshold_k1"]
    t_k3   = thresholds["nc_threshold_k3"]
    t_k5   = thresholds["nc_threshold_k5"]
    t_null = thresholds["nc_threshold_null"]

    # --- Per-bucket coverage ---
    print("\nComputing per-bucket empirical coverage ...")
    bucket_rep = bucket_report(val_records, t_k1, t_k3, t_k5, t_null)
    for bucket, k in [("k1", 1), ("k3", 3), ("k5", 5)]:
        b = bucket_rep[bucket]
        rec = b.get(f"recall@{k}", 0.0)
        ci  = b.get("ci_95", (0.0, 0.0))
        target_met = "OK" if rec >= args.coverage else "LOW"
        print(f"  {bucket}: n={b['count']:4d}  Recall@{k}={rec:.3f}  CI=[{ci[0]:.3f},{ci[1]:.3f}]  target>={args.coverage:.2f} {target_met}")
    nb = bucket_rep["null"]
    print(f"  null: n={nb['count']:4d}  OOD-precision={nb['ood_precision']:.3f}")

    # --- Policy comparison ---
    print("\nSimulating policies on validation split ...")
    policies = []
    for k in [1, 3, 5]:
        assigned = apply_fixed_k(val_records, k)
        policies.append(policy_summary(assigned, f"fixed_k={k}"))
    assigned_gap = apply_gap_policy(
        val_records,
        args.gap_high, args.gap_medium,
        args.k_high, args.k_medium, args.k_low,
    )
    policies.append(policy_summary(assigned_gap, "gap_adaptive"))

    assigned_conf = apply_conformal_policy(val_records, t_k1, t_k3, t_k5, t_null)
    policies.append(policy_summary(assigned_conf, "conformal_adaptive"))

    print("\nPolicy comparison:")
    print(f"  {'Policy':<22} {'Accuracy':>10} {'Avg k':>8} {'Null%':>8}")
    print("  " + "-" * 52)
    for p in policies:
        ci = p.get("accuracy_ci_95", (0.0, 0.0))
        print(
            f"  {p['policy']:<22} "
            f"{p['accuracy']:>8.3f}  "
            f"[{ci[0]:.3f},{ci[1]:.3f}] "
            f"{p['avg_injected_k']:>6.2f} "
            f"{p['null_route_rate']*100:>6.1f}%"
        )

    # --- Write outputs ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "coverage_target": args.coverage,
        "thresholds": thresholds,
        "val_n": len(val_records),
        "bucket_report": bucket_rep,
        "policy_comparison": policies,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = args.out.with_suffix(".md")
    write_markdown(args.coverage, bucket_rep, policies, thresholds, md_path)
    print(f"\nSaved JSON -> {args.out}")
    print(f"Saved table -> {md_path}")


if __name__ == "__main__":
    main()
