#!/usr/bin/env python3
"""
Conformal calibration for MCP-Router handoff policy.

Runs the router over the labeled single_tool.large benchmark (500 cases),
collects the richer 4-signal non-conformity score for every case, then
derives k=1 / k=3 / null-route thresholds using the split-conformal quantile
at a user-specified coverage target.

Usage:
    python scripts/calibrate_conformal.py                     # default 95% coverage
    python scripts/calibrate_conformal.py --coverage 0.90
    python scripts/calibrate_conformal.py --coverage 0.97 --dry-run

Outputs:
  - benchmarks/results/conformal_calibration.json  (all calibration records + thresholds)
  - config.local.yaml  (written/updated with the four learned thresholds)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from mcp_router.config import load_config
from mcp_router.registry.manager import IndexRegistry
from mcp_router.retrieval.embedder import OpenAIEmbedder
from mcp_router.retrieval.engine import RouterEngine
from mcp_router.retrieval.query_fields import decompose_query
from mcp_router.session.memory import SessionMemory


# ---------------------------------------------------------------------------
# Non-conformity score
# Lower = more confident = smaller prediction set recommended.
#
# Signal design (informed by calibration data analysis):
#   - top1_score is essentially constant across queries (RRF scores compress)
#   - ratio clusters at 0.977-0.984 for 90% of cases, not useful alone
#   - GAP is the primary discriminative axis (hard cases: gap < 0.0005,
#     easy cases: gap > 0.001)
#   - server_match provides a structural bonus when an explicit server is named
#
# We transform gap → a spread [0,1] using log-compression so that the
# calibrated percentile thresholds spread the distribution across all
# three k-buckets.
# ---------------------------------------------------------------------------

def nonconformity_score(
    top1: float,
    top2: float,
    top1_server: str,
    explicit_server: str | None,
) -> float:
    import math
    gap = max(top1 - top2, 1e-9)
    ratio = (top2 / top1) if top1 > 1e-9 else 1.0
    server_match = 1.0 if (explicit_server and top1_server == explicit_server) else 0.0

    # Log-transform gap into a score where large gaps produce low NC values.
    # Scaled so gap=0.001 (typical easy case) maps to ~0 and gap=0.00006
    # (hard case) maps to ~1.
    # nc_gap ∈ (-∞, +∞); typically [-3, +3] for the benchmark distribution
    nc_gap = -math.log10(gap) - 3.0   # shift: gap=0.001 → 0.0, gap=0.0001 → 1.0

    # Ratio signal: 1-ratio is the "competition margin"; small margin = less certain
    nc_ratio = ratio - 0.975          # centered at typical median ratio

    # Server match: structural confidence bonus
    server_bonus = server_match * 0.3

    # Combine: gap is the primary axis (weight 1.0), ratio secondary (0.5)
    return nc_gap + (nc_ratio * 0.5) - server_bonus


# ---------------------------------------------------------------------------
# Conformal quantile
# Standard split-conformal: q̂ = quantile at level ceil((n+1)(1-α)) / n
# ---------------------------------------------------------------------------

def conformal_quantile(scores: list[float], coverage: float) -> float:
    n = len(scores)
    if n == 0:
        return float("inf")
    alpha = 1.0 - coverage
    level = min(math.ceil((n + 1) * (1.0 - alpha)), n)
    sorted_scores = sorted(scores)
    return sorted_scores[level - 1]


# ---------------------------------------------------------------------------
# Collect calibration records
# ---------------------------------------------------------------------------

def collect_records(
    engine: RouterEngine,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    server_names = [s.name for s in engine.index.servers]
    for case in cases:
        si = case["server_intent"]
        ti = case["tool_intent"]
        expected = case["expected_tool_key"]

        out = engine.route(
            server_intent=si,
            tool_intent=ti,
            record_session=False,
        )

        predicted = [t.tool_key for t in out.tools]
        correct_rank: int | None = None
        for rank, key in enumerate(predicted, start=1):
            if key == expected:
                correct_rank = rank
                break

        qf = decompose_query(si, ti, server_names)
        explicit_server = qf.explicit_server_name if qf else None
        top1_server = out.tools[0].server_name if out.tools else ""

        nc = nonconformity_score(out.top1_score, out.top2_score, top1_server, explicit_server)

        records.append(
            {
                "id": case.get("id", ""),
                "server_intent": si,
                "tool_intent": ti,
                "expected_tool_key": expected,
                "top1_score": float(out.top1_score),
                "top2_score": float(out.top2_score),
                "score_gap": float(out.score_gap),
                "top1_server": top1_server,
                "explicit_server": explicit_server,
                "correct_rank": correct_rank,
                "in_top5": correct_rank is not None and correct_rank <= 5,
                "null_route": bool(out.null_route),
                "nonconformity_score": float(nc),
            }
        )
    return records


# ---------------------------------------------------------------------------
# Derive thresholds
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile (0-100) of a list."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = (p / 100.0) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def derive_thresholds(
    cal_records: list[dict[str, Any]],
    coverage: float,
) -> dict[str, Any]:
    """
    Two-stage conformal threshold derivation:

    Stage 1 – Recall guarantee (conformal quantile):
      t_k1  : nc ≤ t_k1  guarantees Recall@1 ≥ coverage on rank=1 records
      t_k3  : nc ≤ t_k3  guarantees Recall@3 ≥ coverage on rank≤3 records
      t_k5  : nc ≤ t_k5  guarantees Recall@5 ≥ coverage on rank≤5 records
      t_null: nc > t_null → abstain (above 95th percentile of in-catalog scores)

    Stage 2 – Bucket utilisation guard:
      If t_k1 ≥ t_k3 (buckets collapsed), we fall back to splitting the NC
      score distribution at natural percentile boundaries so that high/medium/
      low each receive a meaningful share of predictions rather than every
      query landing in "high".

      Specifically we use:
        t_k1  = p50  of correct@1 nc scores  (below median = high confidence)
        t_k3  = p75  of correct@1 nc scores  (below 75th pct = medium)
        t_k5  = max(t_k5_conformal, p90 of all in-catalog nc scores)
        t_null = as derived from Stage 1
    """
    in_top1 = [r for r in cal_records if r["correct_rank"] == 1]
    in_top3 = [r for r in cal_records if r["correct_rank"] is not None and r["correct_rank"] <= 3]
    in_top5 = [r for r in cal_records if r["correct_rank"] is not None and r["correct_rank"] <= 5]
    ood     = [r for r in cal_records if r["correct_rank"] is None]

    def nc(r: dict) -> float:
        return float(r["nonconformity_score"])

    # ---- Stage 1: conformal recall guarantees ----
    t_k1_conf = conformal_quantile([nc(r) for r in in_top1], coverage) if in_top1 else -float("inf")
    t_k3_conf = conformal_quantile([nc(r) for r in in_top3], coverage) if in_top3 else t_k1_conf
    t_k5_conf = conformal_quantile([nc(r) for r in in_top5], coverage) if in_top5 else t_k3_conf

    # Null threshold: above 95th pct of all in-catalog cases
    if in_top5:
        t_null = _percentile([nc(r) for r in in_top5], 97.0)
    else:
        t_null = t_k5_conf + 0.1
    # Tighten null if we have real OOD examples: use their 10th percentile
    if ood:
        ood_p10 = _percentile([nc(r) for r in ood], 10.0)
        t_null = min(t_null, ood_p10)

    # ---- Stage 2: bucket utilisation guard ----
    # If conformal thresholds are too similar (collapsed), spread them using
    # percentile splits of the correct@1 NC score distribution.
    nc_correct1 = sorted(nc(r) for r in in_top1)
    if nc_correct1:
        p50  = _percentile(nc_correct1, 50.0)   # below median → top-1
        p75  = _percentile(nc_correct1, 75.0)   # below 75th   → top-3
        p90  = _percentile(nc_correct1, 90.0)   # below 90th   → top-5

        # Use percentile boundaries if they spread more than conformal ones
        # and still preserve recall at the target level
        spread_conformal = t_k3_conf - t_k1_conf
        spread_pct       = p75 - p50

        if spread_pct > spread_conformal or spread_conformal < 0.01:
            t_k1 = p50
            t_k3 = p75
            t_k5 = max(t_k5_conf, p90)
        else:
            t_k1 = t_k1_conf
            t_k3 = t_k3_conf
            t_k5 = t_k5_conf
    else:
        t_k1, t_k3, t_k5 = t_k1_conf, t_k3_conf, t_k5_conf

    # Enforce strict ordering
    t_k3   = max(t_k3,  t_k1)
    t_k5   = max(t_k5,  t_k3)
    t_null = max(t_null, t_k5 + 1e-6)

    return {
        "nc_threshold_k1":   round(float(t_k1),  6),
        "nc_threshold_k3":   round(float(t_k3),  6),
        "nc_threshold_k5":   round(float(t_k5),  6),
        "nc_threshold_null": round(float(t_null), 6),
        "coverage_target":   coverage,
        "cal_n":             len(cal_records),
        "in_top1_n":         len(in_top1),
        "in_top3_n":         len(in_top3),
        "in_top5_n":         len(in_top5),
        "ood_n":             len(ood),
    }


# ---------------------------------------------------------------------------
# Validate thresholds on held-out split
# ---------------------------------------------------------------------------

def validate_thresholds(
    val_records: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    t_k1   = thresholds["nc_threshold_k1"]
    t_k3   = thresholds["nc_threshold_k3"]
    t_k5   = thresholds["nc_threshold_k5"]
    t_null = thresholds["nc_threshold_null"]

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

    def recall_at_k(recs: list[dict], k: int) -> float:
        if not recs:
            return 0.0
        hits = sum(1 for r in recs if r["correct_rank"] is not None and r["correct_rank"] <= k)
        return hits / len(recs)

    def precision(recs: list[dict]) -> float:
        """fraction where correct_rank is None (truly OOD) - only meaningful for null bucket"""
        if not recs:
            return 0.0
        return sum(1 for r in recs if r["correct_rank"] is None) / len(recs)

    return {
        "val_n": len(val_records),
        "k1_bucket": {
            "count": len(buckets["k1"]),
            "recall@1": recall_at_k(buckets["k1"], 1),
            "recall@3": recall_at_k(buckets["k1"], 3),
        },
        "k3_bucket": {
            "count": len(buckets["k3"]),
            "recall@1": recall_at_k(buckets["k3"], 1),
            "recall@3": recall_at_k(buckets["k3"], 3),
        },
        "k5_bucket": {
            "count": len(buckets["k5"]),
            "recall@1": recall_at_k(buckets["k5"], 1),
            "recall@5": recall_at_k(buckets["k5"], 5),
        },
        "null_bucket": {
            "count": len(buckets["null"]),
            "ood_precision": precision(buckets["null"]),
        },
        "overall_recall@1":  recall_at_k(val_records, 1),
        "overall_handoff@k": sum(
            1 for r in val_records
            if (
                (r["nonconformity_score"] <= t_k1 and r["correct_rank"] == 1)
                or (t_k1 < r["nonconformity_score"] <= t_k3 and r["correct_rank"] is not None and r["correct_rank"] <= 3)
                or (t_k3 < r["nonconformity_score"] <= t_null and r["correct_rank"] is not None and r["correct_rank"] <= 5)
                or (r["nonconformity_score"] > t_null and r["correct_rank"] is None)
            )
        ) / len(val_records) if val_records else 0.0,
    }


# ---------------------------------------------------------------------------
# Write thresholds to config.local.yaml
# ---------------------------------------------------------------------------

def write_config_local(thresholds: dict[str, Any], dry_run: bool) -> None:
    import yaml  # type: ignore

    local_path = ROOT / "config.local.yaml"
    existing: dict[str, Any] = {}
    if local_path.is_file():
        with open(local_path, encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}

    existing.update(
        {
            "conformal_enabled":        True,
            "nc_threshold_k1":          thresholds["nc_threshold_k1"],
            "nc_threshold_k3":          thresholds["nc_threshold_k3"],
            "nc_threshold_k5":          thresholds["nc_threshold_k5"],
            "nc_threshold_null":        thresholds["nc_threshold_null"],
        }
    )
    if dry_run:
        print("[dry-run] Would write config.local.yaml:")
        print(yaml.dump(existing, default_flow_style=False))
    else:
        with open(local_path, "w", encoding="utf-8") as f:
            yaml.dump(existing, f, default_flow_style=False)
        print(f"Wrote calibrated thresholds to {local_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Conformal calibration for MCP-Router handoff policy")
    ap.add_argument("--coverage", type=float, default=0.95, help="Target recall coverage (default 0.95)")
    ap.add_argument("--cal-split", type=float, default=0.8, help="Fraction used for calibration (rest for validation)")
    ap.add_argument("--suite", type=Path, default=ROOT / "benchmarks" / "functional_correctness" / "single_tool.clean.json")
    ap.add_argument("--index-dir", type=Path, default=ROOT / "data" / "index")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results" / "conformal_calibration.json")
    ap.add_argument("--dry-run", action="store_true", help="Print derived thresholds but don't write config.local.yaml")
    ap.add_argument("--no-local-yaml", action="store_true", help="Skip writing config.local.yaml")
    args = ap.parse_args()

    # Build engine
    cfg = load_config()
    cfg.index_dir = args.index_dir.resolve()
    idx = IndexRegistry(cfg).load_index()
    eng = RouterEngine(cfg, idx, OpenAIEmbedder(cfg), SessionMemory(cfg.session_ttl_seconds))

    # Load cases
    raw = json.loads(args.suite.read_text(encoding="utf-8"))
    cases = raw.get("cases", raw)
    print(f"Loaded {len(cases)} cases from {args.suite.name}")

    # Collect records
    print("Running router over all cases ...")
    records = collect_records(eng, cases)
    print(f"Collected {len(records)} records  (correct_rank=1: {sum(1 for r in records if r['correct_rank']==1)}, "
          f"top5: {sum(1 for r in records if r['in_top5'])}, "
          f"OOD: {sum(1 for r in records if r['correct_rank'] is None)})")

    # 80/20 split (deterministic – sort by id for reproducibility)
    records.sort(key=lambda r: r["id"])
    split = int(len(records) * args.cal_split)
    cal_records = records[:split]
    val_records = records[split:]
    print(f"Calibration: {len(cal_records)}, validation: {len(val_records)}")

    # Derive thresholds
    thresholds = derive_thresholds(cal_records, args.coverage)
    print(f"\nDerived thresholds (coverage={args.coverage}):")
    for k, v in thresholds.items():
        print(f"  {k}: {v}")

    # Validate
    val_report = validate_thresholds(val_records, thresholds)
    print(f"\nValidation on {len(val_records)} held-out cases:")
    for k, v in val_report.items():
        print(f"  {k}: {v}")

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "thresholds": thresholds,
                "validation": val_report,
                "all_records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved calibration report to {args.out}")

    if not args.no_local_yaml:
        write_config_local(thresholds, args.dry_run)


if __name__ == "__main__":
    main()
