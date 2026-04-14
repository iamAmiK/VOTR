#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "evaluation" / "results" / "livemcpbench"


def normalize_name(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def parse_numbered_items(text: str) -> list[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    items: list[str] = []
    current: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^\d+\.\s*(.*)$", line)
        if match:
            if current:
                items.append(" ".join(current).strip())
                current = []
            current.append(match.group(1).strip())
        else:
            if current:
                current.append(line)
            else:
                current = [line]
    if current:
        items.append(" ".join(current).strip())
    return items


def load_server_catalog(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for item in raw:
        config = item.get("config") or {}
        mcp_servers = config.get("mcpServers") or {}
        for server_name in mcp_servers.keys():
            out[server_name] = {
                "server_name": server_name,
                "display_name": item.get("name", server_name),
                "description": item.get("description", ""),
                "category": item.get("category", ""),
                "organization": item.get("organization", ""),
                "web": item.get("web", ""),
            }
    return out


def load_tool_catalog(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    tool_to_servers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    server_tools: dict[str, dict[str, Any]] = {}

    for item in raw:
        tool_blocks = item.get("tools") or {}
        for _, block in tool_blocks.items():
            server_name = block.get("server_name")
            if not server_name:
                continue
            server_tools.setdefault(
                server_name,
                {
                    "server_name": server_name,
                    "display_name": item.get("name", server_name),
                    "description": item.get("description", ""),
                    "tools": [],
                },
            )
            for tool in block.get("tools") or []:
                tool_name = tool.get("name")
                if not tool_name:
                    continue
                entry = {
                    "tool_name": tool_name,
                    "tool_description": tool.get("description", ""),
                    "server_name": server_name,
                    "server_display_name": item.get("name", server_name),
                }
                tool_to_servers[normalize_name(tool_name)].append(entry)
                server_tools[server_name]["tools"].append(entry)

    return tool_to_servers, server_tools


def prepare_dataset(
    annotations_path: Path,
    tool_catalog_path: Path,
    server_catalog_path: Path,
) -> dict[str, Any]:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    tool_to_servers, server_tools = load_tool_catalog(tool_catalog_path)
    server_catalog = load_server_catalog(server_catalog_path)

    tasks: list[dict[str, Any]] = []
    exact_step_subset: list[dict[str, Any]] = []

    stats = {
        "num_tasks": 0,
        "num_tasks_with_steps": 0,
        "num_tasks_with_tools": 0,
        "num_tasks_step_tool_count_equal": 0,
        "num_tasks_all_tools_uniquely_mapped": 0,
        "num_exact_step_records": 0,
        "num_tasks_in_exact_step_subset": 0,
        "unknown_tool_names": {},
        "ambiguous_tool_names": {},
    }

    unknown_tool_names: dict[str, int] = defaultdict(int)
    ambiguous_tool_names: dict[str, int] = defaultdict(int)

    for row in annotations:
        stats["num_tasks"] += 1
        meta = row.get("Annotator Metadata") or {}
        steps = parse_numbered_items(meta.get("Steps", ""))
        tools = parse_numbered_items(meta.get("Tools", ""))
        if steps:
            stats["num_tasks_with_steps"] += 1
        if tools:
            stats["num_tasks_with_tools"] += 1
        if len(steps) == len(tools) and steps:
            stats["num_tasks_step_tool_count_equal"] += 1

        resolved_tools = []
        all_tools_uniquely_mapped = True
        for tool_name in tools:
            matches = list(tool_to_servers.get(normalize_name(tool_name), []))
            if not matches:
                unknown_tool_names[tool_name] += 1
                all_tools_uniquely_mapped = False
            elif len({m["server_name"] for m in matches}) > 1:
                ambiguous_tool_names[tool_name] += 1
                all_tools_uniquely_mapped = False
            resolved_tools.append(
                {
                    "tool_name": tool_name,
                    "candidate_servers": sorted({m["server_name"] for m in matches}),
                    "matches": matches,
                }
            )

        if tools and all_tools_uniquely_mapped:
            stats["num_tasks_all_tools_uniquely_mapped"] += 1

        task_server_names = sorted(
            {
                server_name
                for tool in resolved_tools
                for server_name in tool["candidate_servers"]
            }
        )

        task_record = {
            "task_id": row.get("task_id"),
            "question": row.get("Question", ""),
            "category": row.get("category", ""),
            "answers": row.get("answers", ""),
            "file_name": row.get("file_name", ""),
            "steps": steps,
            "annotated_tools": resolved_tools,
            "task_gold_server_names": task_server_names,
        }
        tasks.append(task_record)

        if len(steps) == len(tools) and steps and all_tools_uniquely_mapped:
            exact_steps = []
            for idx, (step_text, tool_info) in enumerate(zip(steps, resolved_tools), start=1):
                server_name = tool_info["candidate_servers"][0]
                exact_steps.append(
                    {
                        "task_id": row.get("task_id"),
                        "question": row.get("Question", ""),
                        "step_index": idx,
                        "step_query": step_text,
                        "gold_tool_name": tool_info["tool_name"],
                        "gold_server_name": server_name,
                        "gold_server": server_catalog.get(server_name, server_tools.get(server_name, {})),
                    }
                )
            exact_step_subset.append(
                {
                    "task_id": row.get("task_id"),
                    "question": row.get("Question", ""),
                    "steps": exact_steps,
                }
            )
            stats["num_tasks_in_exact_step_subset"] += 1
            stats["num_exact_step_records"] += len(exact_steps)

    stats["unknown_tool_names"] = dict(sorted(unknown_tool_names.items()))
    stats["ambiguous_tool_names"] = dict(sorted(ambiguous_tool_names.items()))

    return {
        "source_annotations": str(annotations_path),
        "source_tool_catalog": str(tool_catalog_path),
        "source_server_catalog": str(server_catalog_path),
        "stats": stats,
        "tasks": tasks,
        "exact_step_subset": exact_step_subset,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare paper-aligned retrieval eval data from LiveMCPBench.")
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--tool-catalog", type=Path, required=True)
    ap.add_argument("--server-catalog", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    prepared = prepare_dataset(args.annotations, args.tool_catalog, args.server_catalog)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / "paper_aligned_prepared.json"
    exact_path = out_dir / "paper_aligned_exact_step_subset.json"
    stats_path = out_dir / "paper_aligned_stats.json"

    full_path.write_text(json.dumps(prepared, indent=2), encoding="utf-8")
    exact_path.write_text(json.dumps(prepared["exact_step_subset"], indent=2), encoding="utf-8")
    stats_path.write_text(json.dumps(prepared["stats"], indent=2), encoding="utf-8")

    print(json.dumps(prepared["stats"], indent=2))
    print(f"Saved full prepared dataset to: {full_path}")
    print(f"Saved exact-step subset to: {exact_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
