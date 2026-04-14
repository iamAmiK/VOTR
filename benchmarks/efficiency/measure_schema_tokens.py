#!/usr/bin/env python3
"""
Measure per-tool token counts for MCP-Zero-style vs VOTR compressed schemas.

Requires: pip install tiktoken (or pip install -e ".[eval]")

Example:
  python benchmarks/efficiency/measure_schema_tokens.py --index-dir data/index
  python benchmarks/efficiency/measure_schema_tokens.py --index-dir data/index --encoding o200k_base
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mcp_router.evaluation.tokens import (  # noqa: E402
    TiktokenNotInstalledError,
    compressed_schema_tokens,
    mcp_zero_schema_tokens,
)
from mcp_router.retrieval.tool_index import ToolIndex  # noqa: E402


def _pctile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure schema token counts over a built index.")
    ap.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT / "data" / "index",
        help="Directory containing meta.json and embeddings.",
    )
    ap.add_argument(
        "--encoding",
        default="cl100k_base",
        help="tiktoken encoding name (e.g. cl100k_base, o200k_base).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "efficiency" / "schema_token_stats.json",
        help="Write aggregate JSON here.",
    )
    args = ap.parse_args()

    try:
        index = ToolIndex.load(args.index_dir.resolve())
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    comp: list[float] = []
    mcpz: list[float] = []
    for tr in index.all_tool_rows():
        server, tool = index.get_tool_record(int(tr))
        comp.append(
            float(compressed_schema_tokens(server, tool, encoding_name=args.encoding))
        )
        mcpz.append(
            float(mcp_zero_schema_tokens(server, tool, encoding_name=args.encoding))
        )

    def stats(xs: list[float]) -> dict[str, float]:
        s = sorted(xs)
        return {
            "mean": float(statistics.mean(xs)),
            "median": float(statistics.median(xs)),
            "stdev": float(statistics.stdev(xs)) if len(xs) > 1 else 0.0,
            "p90": _pctile(s, 90.0),
            "min": float(min(xs)),
            "max": float(max(xs)),
        }

    payload = {
        "index_dir": str(args.index_dir.resolve()),
        "encoding": args.encoding,
        "tool_count": len(comp),
        "compressed_votr": stats(comp),
        "full_json_mcp_zero_style": stats(mcpz),
        "ratio_mean_mcp_zero_over_compressed": float(statistics.mean(mcpz) / statistics.mean(comp))
        if comp and statistics.mean(comp) > 0
        else 0.0,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TiktokenNotInstalledError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2) from e
