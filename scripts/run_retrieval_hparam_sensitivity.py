#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_router.config import load_config
from mcp_router.registry.manager import IndexRegistry
from mcp_router.retrieval.embedder import OpenAIEmbedder
from mcp_router.retrieval.engine import RouterEngine
from mcp_router.session.memory import SessionMemory


def load_cases(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("cases", raw)


def make_engine(base_cfg, index, embedder, overrides: dict[str, Any]) -> RouterEngine:
    cfg = base_cfg.model_copy(deep=True)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return RouterEngine(cfg, index, embedder, SessionMemory(cfg.session_ttl_seconds))


def eval_single_tool(engine: RouterEngine, cases: list[dict[str, Any]]) -> dict[str, float]:
    n = len(cases)
    if n == 0:
        return {}
    t1 = t3 = t5 = hk = 0
    for case in cases:
        out = engine.route(case["server_intent"], case["tool_intent"], record_session=False)
        preds = [t.tool_key for t in out.tools]
        exp = case["expected_tool_key"]
        t1 += int(exp in preds[:1])
        t3 += int(exp in preds[:3])
        t5 += int(exp in preds[:5])
        rk = max(1, int(out.recommended_handoff_k))
        hk += int(exp in preds[:rk])
    return {
        "top1": t1 / n,
        "top3": t3 / n,
        "top5": t5 / n,
        "handoff": hk / n,
    }


def eval_multi_hop(engine: RouterEngine, cases: list[dict[str, Any]]) -> dict[str, float]:
    hops = 0
    t1 = t3 = t5 = hk = 0
    c1 = c3 = c5 = 0
    for case in cases:
        ok1 = ok3 = ok5 = True
        for hop in case.get("hops", []):
            out = engine.route(hop["server_intent"], hop["tool_intent"], record_session=False)
            preds = [t.tool_key for t in out.tools]
            exp = hop["expected_tool_key"]
            h1 = exp in preds[:1]
            h3 = exp in preds[:3]
            h5 = exp in preds[:5]
            rk = max(1, int(out.recommended_handoff_k))
            hh = exp in preds[:rk]
            t1 += int(h1)
            t3 += int(h3)
            t5 += int(h5)
            hk += int(hh)
            hops += 1
            ok1 = ok1 and h1
            ok3 = ok3 and h3
            ok5 = ok5 and h5
        c1 += int(ok1)
        c3 += int(ok3)
        c5 += int(ok5)
    nc = len(cases) if cases else 1
    nh = hops if hops else 1
    return {
        "hop_top1": t1 / nh,
        "hop_top3": t3 / nh,
        "hop_top5": t5 / nh,
        "hop_handoff": hk / nh,
        "chain_at1": c1 / nc,
        "chain_at3": c3 / nc,
        "chain_at5": c5 / nc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Small sensitivity sweep for retrieval hyperparameters.")
    ap.add_argument("--index-dir", type=Path, default=ROOT / "data" / "index")
    ap.add_argument("--single-suite", type=Path, default=ROOT / "benchmarks" / "functional_correctness" / "single_tool.clean.json")
    ap.add_argument("--multihop-suite", type=Path, default=ROOT / "benchmarks" / "functional_correctness" / "multi_hop.large.cross_app.json")
    ap.add_argument("--single-n", type=int, default=100)
    ap.add_argument("--multihop-n", type=int, default=20)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results" / "retrieval_hparam_sensitivity.json")
    args = ap.parse_args()

    base_cfg = load_config()
    base_cfg.index_dir = args.index_dir.resolve()
    index = IndexRegistry(base_cfg).load_index()
    embedder = OpenAIEmbedder(base_cfg)

    single_cases = load_cases(args.single_suite)[: args.single_n]
    mh_cases = load_cases(args.multihop_suite)[: args.multihop_n]

    experiments = [
        ("baseline", {}),
        ("rrf_k_30", {"rrf_k": 30}),
        ("rrf_k_90", {"rrf_k": 90}),
        ("splade_w_0.20", {"splade_rrf_weight": 0.20}),
        ("splade_w_0.50", {"splade_rrf_weight": 0.50}),
        ("field_bonus_0.0010", {"field_bonus_scale": 0.0010}),
        ("field_bonus_0.0025", {"field_bonus_scale": 0.0025}),
    ]

    rows: list[dict[str, Any]] = []
    for name, overrides in experiments:
        engine = make_engine(base_cfg, index, embedder, overrides)
        s = eval_single_tool(engine, single_cases)
        m = eval_multi_hop(engine, mh_cases)
        rows.append({"name": name, "overrides": overrides, "single": s, "multi_hop": m})
        print(f"[{name}] single_top1={s['top1']:.3f} multi_hop_top1={m['hop_top1']:.3f} chain@1={m['chain_at1']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "single_suite": str(args.single_suite),
                "multi_hop_suite": str(args.multihop_suite),
                "single_n": len(single_cases),
                "multi_hop_n_cases": len(mh_cases),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
