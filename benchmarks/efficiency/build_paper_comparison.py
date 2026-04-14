#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "benchmarks" / "results" / "efficiency"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_row(rows: list[dict], **match: str) -> dict:
    for row in rows:
        if all(row.get(key) == value for key, value in match.items()):
            return row
    raise KeyError(f"Could not find row matching {match}")


def pct(x: float) -> float:
    return round(x * 100.0, 2)


def main() -> None:
    schema_stats = load_json(RESULTS_DIR / "schema_token_stats.json")
    efficiency = load_json(RESULTS_DIR / "summary.json")
    ablations = load_json(ROOT / "benchmarks" / "results" / "baselines_ablations" / "summary.json")

    votr_eff = find_row(efficiency["summary_rows"], scope="single_tool.large")
    try:
        votr_acc = find_row(ablations["rows"], suite="single_tool.large", profile="full_stack")
    except KeyError:
        votr_acc = load_json(
            ROOT / "benchmarks" / "results" / "baselines_ablations" / "single_tool.large.full_stack.json"
        )["summary"]

    tool_count = int(schema_stats["tool_count"])
    mean_compact = float(schema_stats["compressed_votr"]["mean"])
    mean_mcp_zero_style = float(schema_stats["full_json_mcp_zero_style"]["mean"])

    full_catalog_mcp_zero_style = tool_count * mean_mcp_zero_style
    full_catalog_votr_compact = tool_count * mean_compact
    votr_avg_route_tokens = float(votr_eff["estimated_compressed_tokens"]["mean"])

    # MCP-Zero values are taken from the published paper / arXiv HTML:
    # https://arxiv.org/abs/2506.01056
    # https://arxiv.org/html/2506.01056v4
    mcp_zero = {
        "source": "MCP-Zero paper (arXiv:2506.01056v4)",
        "servers": 308,
        "tools": 2797,
        "top1_accuracy_pct": 95.2,
        "avg_prompt_tokens_full_single_turn_apibank": 111.0,
        "token_reduction_pct_full_single_turn_apibank": 98.24,
        "full_catalogue_tokens_mcp_tools": 248100.0,
        "latency_p50_ms": None,
        "features": {
            "hybrid_sparse_dense": False,
            "dynamic_registry": False,
            "adaptive_confidence_gated_k": False,
            "compressed_schema_format": False,
        },
        "notes": [
            "Published token and accuracy numbers come from MCP-Zero's reported experiments, not a rerun on the VOTR benchmark harness.",
            "No directly comparable end-to-end routing latency was located in the MCP-Zero paper.",
        ],
    }

    votr = {
        "source": "Local VOTR benchmark artifacts",
        "servers": 309,
        "tools": tool_count,
        "top1_accuracy_pct": pct(float(votr_acc["top1_accuracy"])),
        "handoff_accuracy_pct": pct(float(votr_acc["handoff_accuracy_at_recommended_k"])),
        "avg_prompt_tokens_single_tool_large": round(votr_avg_route_tokens, 2),
        "latency_p50_ms": round(float(votr_eff["latency_ms"]["p50"]), 2),
        "latency_p95_ms": round(float(votr_eff["latency_ms"]["p95"]), 2),
        "avg_handoff_k": round(float(votr_eff["avg_recommended_handoff_k"]), 2),
        "features": {
            "hybrid_sparse_dense": True,
            "dynamic_registry": True,
            "adaptive_confidence_gated_k": True,
            "compressed_schema_format": True,
        },
    }

    full_catalog = {
        "tool_count": tool_count,
        "full_catalog_mcp_zero_style_tokens_same_index": round(full_catalog_mcp_zero_style, 1),
        "full_catalog_votr_compact_tokens_same_index": round(full_catalog_votr_compact, 1),
        "votr_avg_route_tokens": round(votr_avg_route_tokens, 2),
        "reduction_vs_full_catalog_mcp_zero_style_pct": round(
            (1.0 - votr_avg_route_tokens / full_catalog_mcp_zero_style) * 100.0, 3
        ),
        "reduction_vs_full_catalog_votr_compact_pct": round(
            (1.0 - votr_avg_route_tokens / full_catalog_votr_compact) * 100.0, 3
        ),
    }

    deltas = {
        "top1_accuracy_gain_vs_mcp_zero_pct_points": round(
            votr["top1_accuracy_pct"] - mcp_zero["top1_accuracy_pct"], 2
        ),
        "server_count_delta": votr["servers"] - mcp_zero["servers"],
        "tool_count_delta": votr["tools"] - mcp_zero["tools"],
    }

    payload = {
        "mcp_zero": mcp_zero,
        "votr": votr,
        "full_catalog_savings": full_catalog,
        "deltas": deltas,
    }

    json_out = RESULTS_DIR / "paper_comparison.json"
    md_out = RESULTS_DIR / "paper_comparison.md"
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = "\n".join(
        [
            "# Paper Comparison Summary",
            "",
            "## MCP-Zero vs VOTR",
            "",
            "| Metric | MCP-Zero | VOTR |",
            "| --- | ---: | ---: |",
            f"| Servers | {mcp_zero['servers']} | {votr['servers']} |",
            f"| Tools | {mcp_zero['tools']} | {votr['tools']} |",
            f"| Top-1 accuracy | {mcp_zero['top1_accuracy_pct']:.1f}% | {votr['top1_accuracy_pct']:.1f}% |",
            f"| Accuracy gain | --- | +{deltas['top1_accuracy_gain_vs_mcp_zero_pct_points']:.1f} pp |",
            f"| Avg prompt / route tokens | {mcp_zero['avg_prompt_tokens_full_single_turn_apibank']:.1f}* | {votr['avg_prompt_tokens_single_tool_large']:.1f} |",
            f"| Latency p50 (ms) | n/r | {votr['latency_p50_ms']:.2f} |",
            f"| Latency p95 (ms) | n/r | {votr['latency_p95_ms']:.2f} |",
            f"| Hybrid sparse+dense retrieval | No | Yes |",
            f"| Dynamic registry | No | Yes |",
            f"| Confidence-gated adaptive k | No | Yes |",
            f"| Compressed schema format | No | Yes |",
            "",
            "* MCP-Zero token figure is the published full-collection single-turn APIBank result, not a rerun on the VOTR benchmark harness.",
            "",
            "## VOTR Full-Catalogue Savings",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            f"| Tool count | {tool_count} |",
            f"| Full-catalogue MCP-Zero-style tokens (same index) | {full_catalog['full_catalog_mcp_zero_style_tokens_same_index']:.1f} |",
            f"| Full-catalogue VOTR compact tokens (same index) | {full_catalog['full_catalog_votr_compact_tokens_same_index']:.1f} |",
            f"| VOTR average route tokens | {full_catalog['votr_avg_route_tokens']:.2f} |",
            f"| Reduction vs full-catalogue MCP-Zero-style injection | {full_catalog['reduction_vs_full_catalog_mcp_zero_style_pct']:.3f}% |",
            f"| Reduction vs full-catalogue VOTR compact injection | {full_catalog['reduction_vs_full_catalog_votr_compact_pct']:.3f}% |",
            "",
            "These numbers are derived from local benchmark artifacts and are ready to cite in the paper with an explicit note about which baseline metrics are published versus locally re-measured.",
            "",
        ]
    )
    md_out.write_text(md, encoding="utf-8")

    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
