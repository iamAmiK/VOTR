#!/usr/bin/env python3
"""
Token-Budget Pareto Analysis
=============================
Compares accuracy vs. average tools-injected for a set of routing policies:

  fixed_k_1   — always inject exactly 1 tool (top-1 only)
  fixed_k_3   — always inject exactly 3 tools
  fixed_k_5   — always inject exactly 5 tools
  gap_adaptive — inject recommended_handoff_k from gap-threshold policy
  conformal_k  — inject recommended_handoff_k with conformal_enabled=True

For every policy we report:
  • accuracy          = fraction of queries where the correct tool is in the
                        injected set  (Recall@injected_k)
  • avg_injected_k    = mean number of tools injected per query
  • token_proxy       = avg_injected_k * AVG_TOKENS_PER_TOOL  (~50 tokens for
                        compressed one-liner format)
  • null_route_rate   = fraction of queries where router abstained (k=0)

Results are written to benchmarks/results/efficiency/token_pareto.json and
summarised as a markdown table on stdout.

Usage:
    python benchmarks/efficiency/run_token_pareto.py
    python benchmarks/efficiency/run_token_pareto.py \\
        --suite benchmarks/functional_correctness/single_tool.clean.json \\
        --index-dir data/index \\
        --out benchmarks/results/efficiency/token_pareto.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "benchmarks" / "results" / "efficiency"
sys.path.insert(0, str(ROOT / "src"))

from mcp_router.config import load_config  # noqa: E402
from mcp_router.registry.manager import IndexRegistry  # noqa: E402
from mcp_router.retrieval.embedder import OpenAIEmbedder  # noqa: E402
from mcp_router.retrieval.engine import RouteResponse, RouterEngine  # noqa: E402
from mcp_router.session.memory import SessionMemory  # noqa: E402

# Empirical mean tokens per compressed schema block (tiktoken cl100k_base on data/index).
# Regenerate: python benchmarks/efficiency/measure_schema_tokens.py --index-dir data/index
AVG_TOKENS_PER_TOOL: float = 26.25


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------

def make_engine(index_dir: Path, overrides: dict[str, Any] | None = None) -> RouterEngine:
    cfg = load_config()
    cfg.index_dir = index_dir.resolve()
    if overrides:
        for k, v in overrides.items():
            setattr(cfg, k, v)
    registry = IndexRegistry(cfg)
    index = registry.load_index()
    embedder = OpenAIEmbedder(cfg)
    sessions = SessionMemory(cfg.session_ttl_seconds)
    return RouterEngine(cfg, index, embedder, sessions)


# ---------------------------------------------------------------------------
# Case iteration
# ---------------------------------------------------------------------------

def iter_steps(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten single-tool, multi-hop, and multi-tool cases into individual steps."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        cid = str(case.get("id", ""))
        if "hops" in case:
            for hi, hop in enumerate(case["hops"]):
                rows.append({
                    "case_id": cid,
                    "step": hi,
                    "server_intent": hop["server_intent"],
                    "tool_intent": hop["tool_intent"],
                    "expected_tool_key": hop["expected_tool_key"],
                })
        elif "subtasks" in case:
            for hi, sub in enumerate(case["subtasks"]):
                rows.append({
                    "case_id": cid,
                    "step": hi,
                    "server_intent": sub["server_intent"],
                    "tool_intent": sub["tool_intent"],
                    "expected_tool_key": sub["expected_tool_key"],
                })
        else:
            rows.append({
                "case_id": cid,
                "step": 0,
                "server_intent": case["server_intent"],
                "tool_intent": case["tool_intent"],
                "expected_tool_key": case["expected_tool_key"],
            })
    return rows


# ---------------------------------------------------------------------------
# Route and collect raw observations
# ---------------------------------------------------------------------------

def collect_observations(
    engine: RouterEngine,
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run the engine over every step; capture full route metadata."""
    rows: list[dict[str, Any]] = []
    for step in steps:
        out: RouteResponse = engine.route(
            server_intent=step["server_intent"],
            tool_intent=step["tool_intent"],
            record_session=False,
        )
        predicted = [t.tool_key for t in out.tools]
        expected = step["expected_tool_key"]
        rows.append({
            **step,
            "predicted_keys": predicted,
            "recommended_handoff_k": int(out.recommended_handoff_k),
            "null_route": bool(out.null_route),
            "score_gap": float(out.score_gap),
            "confidence": out.confidence,
            "hit_at_1": expected in predicted[:1],
            "hit_at_3": expected in predicted[:3],
            "hit_at_5": expected in predicted[:5],
        })
    return rows


# ---------------------------------------------------------------------------
# Policy simulation (no re-embedding needed — post-process observations)
# ---------------------------------------------------------------------------

def simulate_fixed_k(obs: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Inject exactly k tools regardless of confidence."""
    n = len(obs)
    hits = sum(1 for r in obs if r["expected_tool_key"] in r["predicted_keys"][:k])
    return {
        "policy": f"fixed_k={k}",
        "accuracy": hits / n if n else 0.0,
        "avg_injected_k": float(k),
        "token_proxy": k * AVG_TOKENS_PER_TOOL,
        "null_route_rate": 0.0,
    }


def simulate_adaptive(obs: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Use recommended_handoff_k (already computed by the engine) as injection size."""
    n = len(obs)
    hits = 0
    total_k = 0
    null_count = 0
    for r in obs:
        k = int(r["recommended_handoff_k"])
        if r["null_route"]:
            null_count += 1
            k = 0
        hits += int(r["expected_tool_key"] in r["predicted_keys"][:k]) if k > 0 else 0
        total_k += k
    return {
        "policy": label,
        "accuracy": hits / n if n else 0.0,
        "avg_injected_k": total_k / n if n else 0.0,
        "token_proxy": (total_k / n if n else 0.0) * AVG_TOKENS_PER_TOOL,
        "null_route_rate": null_count / n if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def print_table(rows: list[dict[str, Any]]) -> None:
    header = "| Policy | Accuracy | Avg k | Tokens (proxy) | Null-route % |"
    sep    = "| --- | ---: | ---: | ---: | ---: |"
    print(header)
    print(sep)
    for r in rows:
        print(
            f"| {r['policy']} "
            f"| {r['accuracy']:.3f} "
            f"| {r['avg_injected_k']:.2f} "
            f"| {r['token_proxy']:.0f} "
            f"| {r['null_route_rate']*100:.1f}% |"
        )


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Token-Budget Pareto Analysis",
        "",
        "Accuracy = Recall@injected_k  "
        "(fraction of queries where correct tool is in the injected set).",
        f"Token proxy = avg_k × {AVG_TOKENS_PER_TOOL:.0f} tokens/tool (compressed schema format).",
        "",
        "| Policy | Accuracy | Avg k | Tokens (proxy) | Null-route % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['policy']} "
            f"| {r['accuracy']:.3f} "
            f"| {r['avg_injected_k']:.2f} "
            f"| {r['token_proxy']:.0f} "
            f"| {r['null_route_rate']*100:.1f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Token-budget Pareto analysis for MCP-Router routing policies")
    ap.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "benchmarks" / "functional_correctness" / "single_tool.clean.json",
        help="Path to benchmark suite JSON (cases array).",
    )
    ap.add_argument("--index-dir", type=Path, default=ROOT / "data" / "index")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "token_pareto.json")
    ap.add_argument(
        "--fixed-ks",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Fixed-k baselines to include.",
    )
    args = ap.parse_args()

    print(f"Loading suite: {args.suite.name}")
    raw = json.loads(args.suite.read_text(encoding="utf-8"))
    cases = raw.get("cases", raw)
    steps = iter_steps(cases)
    print(f"  {len(steps)} routing steps from {len(cases)} cases")

    # --- Gap-threshold engine (default config) ---
    print("\nRunning gap-threshold adaptive policy …")
    engine_gap = make_engine(args.index_dir)
    obs_gap = collect_observations(engine_gap, steps)

    # --- Conformal engine (conformal_enabled=True) ---
    print("Running conformal adaptive policy …")
    engine_conf = make_engine(args.index_dir, overrides={"conformal_enabled": True})
    obs_conf = collect_observations(engine_conf, steps)

    print("\n--- Results ---\n")
    policy_rows: list[dict[str, Any]] = []
    for k in args.fixed_ks:
        policy_rows.append(simulate_fixed_k(obs_gap, k))
    policy_rows.append(simulate_adaptive(obs_gap, "gap_adaptive"))
    policy_rows.append(simulate_adaptive(obs_conf, "conformal_adaptive"))

    print_table(policy_rows)

    # Write outputs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "suite": str(args.suite),
                "num_steps": len(steps),
                "policies": policy_rows,
                "observations_gap": obs_gap,
                "observations_conf": obs_conf,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path = args.out.with_suffix(".md")
    write_markdown(policy_rows, md_path)
    print(f"\nSaved JSON -> {args.out}")
    print(f"Saved table -> {md_path}")


if __name__ == "__main__":
    main()
