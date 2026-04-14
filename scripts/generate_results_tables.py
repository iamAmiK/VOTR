#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "results"
LIVE = ROOT / "evaluation" / "results" / "livemcpbench"
EVAL = ROOT / "evaluation" / "results"
FULL_SWEEP = EVAL / "full_sweep"
OUT = ROOT / "evaluation" / "results" / "router_results_tables.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if i > 0 else "---" for i in range(len(headers))) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(out)


def safe_load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return load_json(path)


def suite_family(name: str) -> str:
    if name.startswith("single_tool"):
        return "single_tool"
    if name.startswith("multi_hop"):
        return "multi_hop"
    if name.startswith("multi_tool"):
        return "multi_tool"
    return "other"


def suite_scale(name: str) -> str:
    if ".small_" in name or ".small" in name:
        return "small"
    if ".medium_" in name or ".medium" in name:
        return "medium"
    if ".large" in name:
        return "large"
    return "other"


def functional_tables() -> list[str]:
    data = load_json(BENCH / "baselines_ablations" / "summary.json")
    rows = data["rows"]
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_suite.setdefault(row["suite"], []).append(row)

    sections = ["## Functional Correctness"]
    for suite in sorted(by_suite):
        suite_rows = sorted(by_suite[suite], key=lambda r: r["profile"])
        table_rows = []
        for row in suite_rows:
            table_rows.append(
                [
                    row["profile"],
                    row["top1_accuracy"],
                    row["top3_accuracy"],
                    row["top5_accuracy"],
                    row["handoff_accuracy_at_recommended_k"],
                    f"{row['case_metric_label']}={fmt(row['case_metric_value'])}" if row["case_metric_label"] else "",
                ]
            )
        sections.append(f"### {suite}")
        sections.append(
            md_table(
                ["Profile", "Top-1", "Top-3", "Top-5", "Handoff@k", "Case Metric"],
                table_rows,
            )
        )

    sections.append("## Functional Groups")
    for family in ("single_tool", "multi_hop", "multi_tool"):
        fam_rows = [
            row for row in rows if suite_family(row["suite"]) == family and row["profile"] == "full_stack"
        ]
        fam_rows.sort(key=lambda r: (suite_scale(r["suite"]), r["suite"]))
        sections.append(f"### {family}")
        sections.append(
            md_table(
                ["Suite", "Top-1", "Top-3", "Top-5", "Handoff@k", "Case Metric"],
                [
                    [
                        row["suite"],
                        row["top1_accuracy"],
                        row["top3_accuracy"],
                        row["top5_accuracy"],
                        row["handoff_accuracy_at_recommended_k"],
                        f"{row['case_metric_label']}={fmt(row['case_metric_value'])}" if row["case_metric_label"] else "",
                    ]
                    for row in fam_rows
                ],
            )
        )
    return sections


def confidence_tables() -> list[str]:
    data = load_json(BENCH / "confidence" / "summary.json")
    per_suite = data["per_suite"]
    sections = ["## Confidence Calibration"]
    for suite in sorted(per_suite):
        summary = per_suite[suite]["summary"]
        sections.append(f"### {suite}")
        sections.append(
            md_table(
                ["Items", "Top-1", "Handoff@k", "Avg Recommended k"],
                [[summary["num_items"], summary["top1_accuracy"], summary["handoff_accuracy_at_recommended_k"], summary["avg_recommended_handoff_k"]]],
            )
        )

    rows = []
    for suite, payload in per_suite.items():
        summary = payload["summary"]
        rows.append(
            [
                payload["scale"],
                suite,
                summary["top1_accuracy"],
                summary["handoff_accuracy_at_recommended_k"],
                summary["avg_recommended_handoff_k"],
            ]
        )
    rows.sort(key=lambda r: (r[0], r[1]))
    sections.append("## Confidence Groups")
    sections.append(md_table(["Scale", "Suite", "Top-1", "Handoff@k", "Avg Recommended k"], rows))
    return sections


def efficiency_tables() -> list[str]:
    data = load_json(BENCH / "efficiency" / "summary.json")
    per_suite = data["per_suite"]
    sections = ["## Efficiency"]
    for suite in sorted(per_suite):
        summary = per_suite[suite]["summary"]
        sections.append(f"### {suite}")
        sections.append(
            md_table(
                [
                    "Items",
                    "P50 ms",
                    "P95 ms",
                    "Mean ms",
                    "Returned Tools",
                    "Avg Recommended k",
                    "Mean Est. Tokens",
                ],
                [[
                    summary["num_items"],
                    summary["latency_ms"]["p50"],
                    summary["latency_ms"]["p95"],
                    summary["latency_ms"]["mean"],
                    summary["returned_tools_mean"],
                    summary["avg_recommended_handoff_k"],
                    summary["estimated_compressed_tokens"]["mean"],
                ]],
            )
        )

    rows = []
    for suite, payload in per_suite.items():
        summary = payload["summary"]
        rows.append(
            [
                payload["scale"],
                suite,
                summary["latency_ms"]["p50"],
                summary["latency_ms"]["p95"],
                summary["estimated_compressed_tokens"]["mean"],
            ]
        )
    rows.sort(key=lambda r: (r[0], r[1]))
    sections.append("## Efficiency Groups")
    sections.append(md_table(["Scale", "Suite", "P50 ms", "P95 ms", "Mean Est. Tokens"], rows))
    return sections


def overlap_tables() -> list[str]:
    data = load_json(BENCH / "overlap_aware" / "summary.json")
    rows = data["rows"]
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_suite.setdefault(row["suite"], []).append(row)

    sections = ["## Overlap-Aware Evaluation"]
    for suite in sorted(by_suite):
        suite_rows = sorted(by_suite[suite], key=lambda r: r["profile"])
        sections.append(f"### {suite}")
        sections.append(
            md_table(
                [
                    "Profile",
                    "Top-1",
                    "Top-3",
                    "Top-5",
                    "Handoff@k",
                    "Ambiguous Rate",
                    "Exact@1 On Ambiguous",
                    "Equiv@1 On Ambiguous",
                    "Equiv Top-1",
                ],
                [
                    [
                        row["profile"],
                        row["top1_accuracy"],
                        row["top3_accuracy"],
                        row["top5_accuracy"],
                        row["handoff_accuracy_at_recommended_k"],
                        row["ambiguous_rate"],
                        row["exact_top1_on_ambiguous"],
                        row["equivalence_top1_on_ambiguous"],
                        row["equivalence_top1_accuracy"],
                    ]
                    for row in suite_rows
                ],
            )
        )

    sections.append("## Overlap Groups")
    large_rows = [row for row in rows if suite_scale(row["suite"]) == "large"]
    large_rows.sort(key=lambda r: (r["suite"], r["profile"]))
    sections.append(
        md_table(
            ["Suite", "Profile", "Top-1", "Ambiguous Rate", "Exact@1 On Ambiguous", "Equiv@1 On Ambiguous"],
            [
                [
                    row["suite"],
                    row["profile"],
                    row["top1_accuracy"],
                    row["ambiguous_rate"],
                    row["exact_top1_on_ambiguous"],
                    row["equivalence_top1_on_ambiguous"],
                ]
                for row in large_rows
            ],
        )
    )
    return sections


def abstention_tables() -> list[str]:
    policy = safe_load_json(BENCH / "negative_controls.policy_cleaned.json")
    raw = safe_load_json(BENCH / "negative_controls.priority.json")
    threshold = safe_load_json(BENCH / "threshold_sensitivity.post_abstention.json")
    drift = safe_load_json(BENCH / "dynamic_registration.post_abstention.json")
    if not any((policy, raw, threshold, drift)):
        return []

    sections = ["## Abstention And Registration"]
    rows: list[list[Any]] = []
    if policy is not None:
        summary = policy["summary"]
        rows.append(
            [
                "Policy-cleaned unsupported intents",
                summary["num_cases"],
                summary["pass_rate"],
                summary["avg_recommended_handoff_k"],
                ", ".join(f"{k}:{v}" for k, v in summary["confidence_distribution"].items()),
            ]
        )
    if raw is not None:
        summary = raw["summary"]
        rows.append(
            [
                "Raw unsupported stress",
                summary["num_cases"],
                summary["pass_rate"],
                summary["avg_recommended_handoff_k"],
                ", ".join(f"{k}:{v}" for k, v in summary["confidence_distribution"].items()),
            ]
        )
    if rows:
        sections.append(
            md_table(
                ["Suite", "Cases", "Pass Rate", "Avg Recommended k", "Confidence Distribution"],
                rows,
            )
        )

    if threshold is not None:
        current = threshold["current_config"]
        recommended = threshold["recommended_from_dev"]
        sections.append("### Threshold Sensitivity")
        sections.append(
            md_table(
                ["Config", "High Gap", "Medium Gap", "Handoff@k", "Avg Recommended k"],
                [
                    [
                        "Current",
                        current["handoff_gap_high"],
                        current["handoff_gap_medium"],
                        current["test_metrics"]["handoff_accuracy"],
                        current["test_metrics"]["avg_recommended_k"],
                    ],
                    [
                        "Dev-recommended",
                        recommended["handoff_gap_high"],
                        recommended["handoff_gap_medium"],
                        recommended["test_metrics"]["handoff_accuracy"],
                        recommended["test_metrics"]["avg_recommended_k"],
                    ],
                ],
            )
        )

    if drift is not None:
        summary = drift["summary"]
        sections.append("### Dynamic Registration")
        sections.append(
            md_table(
                [
                    "Control Hit@1",
                    "Pre-register Miss@5 Rate",
                    "Post-register Hit@1 Rate",
                    "Controls Preserved@1 Rate",
                ],
                [[
                    summary["baseline_controls_hit_at_1"],
                    summary["before_all_miss_at_5_rate"],
                    summary["after_all_hit_at_1_rate"],
                    summary["controls_preserved_at_1_rate"],
                ]],
            )
        )
    return sections


def router_eval_tables() -> list[str]:
    if not FULL_SWEEP.is_dir():
        return []

    multihop_rows: list[list[Any]] = []
    confidence_rows: list[list[Any]] = []
    calibrate_rows: list[list[Any]] = []

    for path in sorted(FULL_SWEEP.glob("*.json")):
        data = load_json(path)
        name = path.stem
        if name.startswith("multihop."):
            summary = data["summary"]
            dataset = name.removeprefix("multihop.")
            multihop_rows.append(
                [
                    dataset,
                    summary["k"],
                    summary["mean_hop_acc_at_1"],
                    summary["mean_hop_acc_at_k"],
                    summary["chain_success_rate_at_k"],
                ]
            )
        elif name.startswith("confidence."):
            summary = data["summary"]
            dataset = name.removeprefix("confidence.")
            confidence_rows.append(
                [
                    dataset,
                    summary["num_hops"],
                    summary["overall_top1_accuracy"],
                    summary["overall_handoff_accuracy_at_recommended_k"],
                    ", ".join(f"{k}:{v['count']}" for k, v in summary["confidence_buckets"].items()),
                ]
            )
        elif name.startswith("calibrate."):
            summary = data["summary"]
            dataset = name.removeprefix("calibrate.")
            metrics = summary["metrics_at_recommended_thresholds"]
            calibrate_rows.append(
                [
                    dataset,
                    summary["recommended_thresholds"]["handoff_gap_high"],
                    summary["recommended_thresholds"]["handoff_gap_medium"],
                    metrics["handoff_accuracy"],
                    metrics["avg_recommended_k"],
                    metrics["constraints_ok"],
                ]
            )

    sections = ["## Router Eval Sweep"]
    if multihop_rows:
        sections.append("### Multi-hop Eval")
        sections.append(
            md_table(
                ["Dataset", "k", "Mean Hop@1", "Mean Hop@k", "Chain Success@k"],
                multihop_rows,
            )
        )
    if confidence_rows:
        sections.append("### Confidence Eval")
        sections.append(
            md_table(
                ["Dataset", "Hops", "Top-1", "Handoff@k", "Confidence Counts"],
                confidence_rows,
            )
        )
    if calibrate_rows:
        sections.append("### Handoff Calibration")
        sections.append(
            md_table(
                ["Dataset", "High Gap", "Medium Gap", "Handoff Accuracy", "Avg Recommended k", "Constraints OK"],
                calibrate_rows,
            )
        )
    return sections if len(sections) > 1 else []


def livemcpbench_tables() -> list[str]:
    exact = load_json(LIVE / "paper_aligned_exact_eval.json")["summary"]
    paper = load_json(LIVE / "paper_faithful_tool_to_agent_eval.json")["summary"]
    full95_task = load_json(LIVE / "full95_task_level_server_eval.json")["summary"]
    full95_recon = load_json(LIVE / "full95_reconstructed_stepwise_eval.json")["summary"]
    router_fmt = load_json(LIVE / "router_format_eval.json")
    router_policy = load_json(LIVE / "router_policy_payload_eval.json")["summary"]

    sections = ["## LiveMCPBench"]
    sections.append("### Individual LiveMCPBench Results")
    sections.append(
        md_table(
            ["Eval", "Queries", "Recall@1", "Recall@3", "Recall@5", "mAP@5", "nDCG@5"],
            [
                ["Exact subset server eval", exact["num_step_queries"], exact["recall"]["@1"], exact["recall"]["@3"], exact["recall"]["@5"], exact["mAP"]["@5"], exact["nDCG"]["@5"]],
                ["Paper-faithful tool-to-agent", paper["num_step_queries"], paper["recall"]["@1"], paper["recall"]["@3"], paper["recall"]["@5"], paper["mAP"]["@5"], paper["nDCG"]["@5"]],
                ["Full95 task-level", full95_task["num_queries"], full95_task["recall"]["@1"], full95_task["recall"]["@3"], full95_task["recall"]["@5"], full95_task["mAP"]["@5"], full95_task["nDCG"]["@5"]],
                ["Full95 reconstructed stepwise", full95_recon["num_queries"], full95_recon["recall"]["@1"], full95_recon["recall"]["@3"], full95_recon["recall"]["@5"], full95_recon["mAP"]["@5"], full95_recon["nDCG"]["@5"]],
                ["Router format exact subset", router_fmt["exact_step_subset"]["summary"]["num_queries"], router_fmt["exact_step_subset"]["summary"]["recall"]["@1"], router_fmt["exact_step_subset"]["summary"]["recall"]["@3"], router_fmt["exact_step_subset"]["summary"]["recall"]["@5"], router_fmt["exact_step_subset"]["summary"]["mAP"]["@5"], router_fmt["exact_step_subset"]["summary"]["nDCG"]["@5"]],
                ["Router format full95 task-level", router_fmt["full95_task_level"]["summary"]["num_queries"], router_fmt["full95_task_level"]["summary"]["recall"]["@1"], router_fmt["full95_task_level"]["summary"]["recall"]["@3"], router_fmt["full95_task_level"]["summary"]["recall"]["@5"], router_fmt["full95_task_level"]["summary"]["mAP"]["@5"], router_fmt["full95_task_level"]["summary"]["nDCG"]["@5"]],
                ["Router format full95 reconstructed", router_fmt["full95_reconstructed_stepwise"]["summary"]["num_queries"], router_fmt["full95_reconstructed_stepwise"]["summary"]["recall"]["@1"], router_fmt["full95_reconstructed_stepwise"]["summary"]["recall"]["@3"], router_fmt["full95_reconstructed_stepwise"]["summary"]["recall"]["@5"], router_fmt["full95_reconstructed_stepwise"]["summary"]["mAP"]["@5"], router_fmt["full95_reconstructed_stepwise"]["summary"]["nDCG"]["@5"]],
                ["Router policy payloads", router_policy["num_queries"], router_policy["recall"]["@1"], router_policy["recall"]["@3"], router_policy["recall"]["@5"], router_policy["mAP"]["@5"], router_policy["nDCG"]["@5"]],
            ],
        )
    )

    sections.append("### Grouped LiveMCPBench Results")
    sections.append(
        md_table(
            ["Group", "Eval", "Recall@1", "Recall@5", "mAP@5", "nDCG@5"],
            [
                ["Paper-aligned", "Exact subset server eval", exact["recall"]["@1"], exact["recall"]["@5"], exact["mAP"]["@5"], exact["nDCG"]["@5"]],
                ["Paper-aligned", "Paper-faithful tool-to-agent", paper["recall"]["@1"], paper["recall"]["@5"], paper["mAP"]["@5"], paper["nDCG"]["@5"]],
                ["Full95 extensions", "Task-level", full95_task["recall"]["@1"], full95_task["recall"]["@5"], full95_task["mAP"]["@5"], full95_task["nDCG"]["@5"]],
                ["Full95 extensions", "Reconstructed stepwise", full95_recon["recall"]["@1"], full95_recon["recall"]["@5"], full95_recon["mAP"]["@5"], full95_recon["nDCG"]["@5"]],
                ["Router-native", "Router format exact subset", router_fmt["exact_step_subset"]["summary"]["recall"]["@1"], router_fmt["exact_step_subset"]["summary"]["recall"]["@5"], router_fmt["exact_step_subset"]["summary"]["mAP"]["@5"], router_fmt["exact_step_subset"]["summary"]["nDCG"]["@5"]],
                ["Router-native", "Router format full95 task-level", router_fmt["full95_task_level"]["summary"]["recall"]["@1"], router_fmt["full95_task_level"]["summary"]["recall"]["@5"], router_fmt["full95_task_level"]["summary"]["mAP"]["@5"], router_fmt["full95_task_level"]["summary"]["nDCG"]["@5"]],
                ["Router-native", "Router format full95 reconstructed", router_fmt["full95_reconstructed_stepwise"]["summary"]["recall"]["@1"], router_fmt["full95_reconstructed_stepwise"]["summary"]["recall"]["@5"], router_fmt["full95_reconstructed_stepwise"]["summary"]["mAP"]["@5"], router_fmt["full95_reconstructed_stepwise"]["summary"]["nDCG"]["@5"]],
                ["Router-native", "Router policy payloads", router_policy["recall"]["@1"], router_policy["recall"]["@5"], router_policy["mAP"]["@5"], router_policy["nDCG"]["@5"]],
            ],
        )
    )
    return sections


def main() -> None:
    sections = ["# Router Results Tables", ""]
    sections.extend(functional_tables())
    sections.append("")
    sections.extend(confidence_tables())
    sections.append("")
    sections.extend(efficiency_tables())
    sections.append("")
    sections.extend(overlap_tables())
    sections.append("")
    sections.extend(abstention_tables())
    sections.append("")
    sections.extend(router_eval_tables())
    sections.append("")
    sections.extend(livemcpbench_tables())
    OUT.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote tables to {OUT}")


if __name__ == "__main__":
    main()
