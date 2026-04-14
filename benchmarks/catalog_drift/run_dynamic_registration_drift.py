#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mcp_router.config import load_config
from mcp_router.registry.manager import IndexRegistry
from mcp_router.registry.schema import RegisteredServer
from mcp_router.retrieval.embedder import OpenAIEmbedder
from mcp_router.retrieval.engine import RouterEngine
from mcp_router.session.memory import SessionMemory


def make_engine(index_dir: Path) -> RouterEngine:
    cfg = load_config()
    cfg.index_dir = index_dir.resolve()
    registry = IndexRegistry(cfg)
    index = registry.load_index()
    embedder = OpenAIEmbedder(cfg)
    sessions = SessionMemory(cfg.session_ttl_seconds)
    return RouterEngine(cfg, index, embedder, sessions)


def route_probe(engine: RouterEngine, probe: dict[str, Any], session_prefix: str) -> dict[str, Any]:
    out = engine.route(
        server_intent=probe["server_intent"],
        tool_intent=probe["tool_intent"],
        session_id=f"{session_prefix}-{probe['id']}",
        record_session=False,
    )
    predicted = [tool.tool_key for tool in out.tools]
    expected = probe["expected_tool_key"]
    return {
        "id": probe["id"],
        "expected_tool_key": expected,
        "predicted_top5": predicted[:5],
        "hit_at_1": expected in predicted[:1],
        "hit_at_5": expected in predicted[:5],
        "confidence": out.confidence,
        "recommended_handoff_k": int(out.recommended_handoff_k),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark dynamic registration and catalog shift on a temporary index copy.")
    ap.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "benchmarks" / "catalog_drift" / "dynamic_registration_cases.json",
    )
    ap.add_argument("--index-dir", type=Path, default=ROOT / "data" / "index")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    raw = json.loads(args.cases.read_text(encoding="utf-8"))
    scenarios = raw["scenarios"]
    control_probes = raw["control_probes"]

    temp_root = Path(tempfile.mkdtemp(prefix="mcp-router-index-copy-"))
    temp_index = temp_root / "index"
    shutil.copytree(args.index_dir, temp_index)

    engine = make_engine(temp_index)
    baseline_controls = [route_probe(engine, probe, "control-baseline") for probe in control_probes]

    cfg = load_config()
    cfg.index_dir = temp_index
    registry = IndexRegistry(cfg)
    embedder = OpenAIEmbedder(cfg)

    scenario_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        before = [route_probe(engine, probe, f"{scenario['id']}-before") for probe in scenario["probes"]]
        registry.register_server(RegisteredServer.model_validate(scenario["server"]), embedder)
        engine = make_engine(temp_index)
        after = [route_probe(engine, probe, f"{scenario['id']}-after") for probe in scenario["probes"]]
        controls_after = [route_probe(engine, probe, f"{scenario['id']}-control-after") for probe in control_probes]
        scenario_rows.append(
            {
                "id": scenario["id"],
                "before": before,
                "after": after,
                "controls_after": controls_after,
                "before_all_miss_at_5": all(not row["hit_at_5"] for row in before),
                "after_all_hit_at_1": all(row["hit_at_1"] for row in after),
                "controls_preserved_at_1": all(row["hit_at_1"] for row in controls_after),
            }
        )

    summary = {
        "num_control_probes": len(control_probes),
        "baseline_controls_hit_at_1": (
            sum(int(row["hit_at_1"]) for row in baseline_controls) / len(baseline_controls)
            if baseline_controls
            else 0.0
        ),
        "num_scenarios": len(scenarios),
        "before_all_miss_at_5_rate": (
            sum(int(row["before_all_miss_at_5"]) for row in scenario_rows) / len(scenario_rows)
            if scenario_rows
            else 0.0
        ),
        "after_all_hit_at_1_rate": (
            sum(int(row["after_all_hit_at_1"]) for row in scenario_rows) / len(scenario_rows)
            if scenario_rows
            else 0.0
        ),
        "controls_preserved_at_1_rate": (
            sum(int(row["controls_preserved_at_1"]) for row in scenario_rows) / len(scenario_rows)
            if scenario_rows
            else 0.0
        ),
    }
    report = {
        "summary": summary,
        "baseline_controls": baseline_controls,
        "scenarios": scenario_rows,
    }

    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report to: {args.out}")


if __name__ == "__main__":
    main()
