#!/usr/bin/env python3
"""
Regenerate tiered functional-correctness JSON so each scale has comparable
evaluation counts (route calls / hops / subtasks):

  small:  100 single-tool rows, or 100 total hops/subtasks for multi-* suites
  medium: 250
  large:  500 (single_tool.clean.json already has 500)

Multi-hop / multi-tool suites use fixed 5 hops (or subtasks) per case so
large = 100 cases × 5 = 500 step evaluations, etc.

Run from repo root:
  python benchmarks/functional_correctness/build_scaled_suites.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FC = ROOT / "benchmarks" / "functional_correctness"

SMALL_N = 100
MEDIUM_N = 250
LARGE_N = 500
CHAIN_LEN = 5


def tool_keys_from_meta(meta_path: Path) -> set[str]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for srv in meta["servers"]:
        name = srv["name"]
        for t in srv["tools"]:
            keys.add(f"{name}::{t['name']}")
    return keys


def load_cases(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = raw.get("cases", raw)
    if not isinstance(cases, list):
        raise SystemExit(f"Expected list of cases in {path}")
    return cases


def write_suite(path: Path, suite: str, description: str, cases: list[Any]) -> None:
    path.write_text(
        json.dumps({"suite": suite, "description": description, "cases": cases}, indent=2),
        encoding="utf-8",
    )


def expand_single_tool_cycle(pool: list[dict[str, Any]], target: int, id_prefix: str) -> list[dict[str, Any]]:
    if not pool:
        raise SystemExit(f"Empty pool for {id_prefix}")
    out: list[dict[str, Any]] = []
    for i in range(target):
        src = pool[i % len(pool)]
        out.append(
            {
                "id": f"{id_prefix}-{i + 1:04d}",
                "server_intent": src["server_intent"],
                "tool_intent": src["tool_intent"],
                "expected_tool_key": src["expected_tool_key"],
            }
        )
    return out


def hop_from_single(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "server_intent": c["server_intent"],
        "tool_intent": c["tool_intent"],
        "expected_tool_key": c["expected_tool_key"],
    }


def build_multi_hop(pool: list[dict[str, Any]], total_steps: int, suite: str, description: str) -> list[dict[str, Any]]:
    if total_steps % CHAIN_LEN:
        raise SystemExit(f"total_steps {total_steps} must be divisible by CHAIN_LEN {CHAIN_LEN}")
    n_chains = total_steps // CHAIN_LEN
    if not pool:
        raise SystemExit("empty pool for multi_hop")
    ext = pool * (1 + (n_chains * CHAIN_LEN) // len(pool))
    cases: list[dict[str, Any]] = []
    for i in range(n_chains):
        hops = [hop_from_single(ext[i * CHAIN_LEN + j]) for j in range(CHAIN_LEN)]
        cases.append(
            {
                "id": f"{suite.replace('.', '-')}-{i + 1:04d}",
                "description": f"Cross-application chain ({CHAIN_LEN} hops), case {i + 1}",
                "hops": hops,
            }
        )
    return cases


def build_multi_tool(pool: list[dict[str, Any]], total_steps: int, suite: str, description: str) -> list[dict[str, Any]]:
    if total_steps % CHAIN_LEN:
        raise SystemExit(f"total_steps {total_steps} must be divisible by CHAIN_LEN {CHAIN_LEN}")
    n_chains = total_steps // CHAIN_LEN
    if not pool:
        raise SystemExit("empty pool for multi_tool")
    ext = pool * (1 + (n_chains * CHAIN_LEN) // len(pool))
    cases: list[dict[str, Any]] = []
    for i in range(n_chains):
        chunk = [ext[i * CHAIN_LEN + j] for j in range(CHAIN_LEN)]
        subtasks = [hop_from_single(c) for c in chunk]
        parts = [f"({j + 1}) {c['tool_intent'][:100]}" for j, c in enumerate(chunk)]
        cases.append(
            {
                "id": f"{suite.replace('.', '-')}-{i + 1:04d}",
                "user_intent": f"Single user turn #{i + 1}: complete these intents in order: " + "; ".join(parts),
                "subtasks": subtasks,
            }
        )
    return cases


def main() -> None:
    clean_path = FC / "single_tool.clean.json"
    clean_pool = load_cases(clean_path)
    medium_keys = tool_keys_from_meta(ROOT / "data/index_medium_50/meta.json")
    medium_from_clean = [c for c in clean_pool if c["expected_tool_key"] in medium_keys]
    med_path = FC / "single_tool.medium_50.clean.json"
    if med_path.is_file():
        med_pool = load_cases(med_path)
    else:
        med_pool = medium_from_clean
        if len(med_pool) < 10:
            raise SystemExit("Need single_tool.medium_50.clean.json or a non-trivial medium_from_clean pool.")

    small_pool: list[dict[str, Any]] = []
    for name in (
        "single_tool.bloomberg.clean.json",
        "single_tool.github.clean.json",
        "single_tool.telegram.clean.json",
    ):
        small_pool.extend(load_cases(FC / name))

    # --- single-tool medium (250) from curated medium pool ---
    write_suite(
        FC / "single_tool.medium_250.clean.json",
        "single_tool.medium_250.clean",
        "Router-only single-tool benchmark for the medium 50-server catalog (250 evaluation queries).",
        expand_single_tool_cycle(med_pool, MEDIUM_N, "st-medium"),
    )

    # --- per-domain small single-tool → 100 each ---
    for stem, idp in (
        ("single_tool.bloomberg.clean", "st-bloomberg"),
        ("single_tool.github.clean", "st-github"),
        ("single_tool.telegram.clean", "st-telegram"),
    ):
        pool = load_cases(FC / f"{stem}.json")
        write_suite(
            FC / f"{stem}.json",
            stem,
            f"Router-only single-tool benchmark ({SMALL_N} evaluation queries), cycled from labeled catalog tools.",
            expand_single_tool_cycle(pool, SMALL_N, idp),
        )

    # --- multi-hop ---
    specs_hop = [
        ("multi_hop.small_100.cross_app", small_pool, SMALL_N, "Router-only multi-hop suite for the 3-server catalog (100 hop evaluations)."),
        ("multi_hop.medium_250.cross_app", med_pool, MEDIUM_N, "Router-only multi-hop suite for the medium 50-server catalog (250 hop evaluations)."),
        ("multi_hop.large.cross_app", clean_pool, LARGE_N, "Router-only multi-hop suite for the full router catalog (500 hop evaluations)."),
    ]
    for suite, pool, steps, desc in specs_hop:
        cases = build_multi_hop(pool, steps, suite, desc)
        write_suite(FC / f"{suite}.json", suite, desc, cases)

    # --- multi-tool ---
    specs_mt = [
        ("multi_tool.small_100.single_turn", small_pool, SMALL_N, "Router-only multi-target single-turn suite for the 3-server catalog (100 subtask evaluations)."),
        ("multi_tool.medium_250.single_turn", med_pool, MEDIUM_N, "Router-only multi-target single-turn suite for the medium catalog (250 subtask evaluations)."),
        ("multi_tool.large.single_turn", clean_pool, LARGE_N, "Router-only multi-target single-turn suite for the full catalog (500 subtask evaluations)."),
    ]
    for suite, pool, steps, desc in specs_mt:
        cases = build_multi_tool(pool, steps, suite, desc)
        write_suite(FC / f"{suite}.json", suite, desc, cases)

    # --- legacy unsized suite names: align with large (500 steps) ---
    write_suite(
        FC / "multi_hop.cross_app.json",
        "multi_hop.cross_app",
        "Router-only multi-hop benchmark across applications (500 hop evaluations; same scale as multi_hop.large).",
        build_multi_hop(clean_pool, LARGE_N, "multi_hop.cross_app", ""),
    )
    write_suite(
        FC / "multi_tool.single_turn.json",
        "multi_tool.single_turn",
        "Router-only multi-target single-turn benchmark (500 subtask evaluations; same scale as multi_tool.large).",
        build_multi_tool(clean_pool, LARGE_N, "multi_tool.single_turn", ""),
    )

    print("Wrote scaled suites under", FC)


if __name__ == "__main__":
    main()
