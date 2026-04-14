#!/usr/bin/env python3
"""
Build FAISS-free on-disk index (numpy) + meta.json from MCP-Zero-style
`mcp_tools_with_embedding.json` (array of servers with embeddings).

Usage:
  python scripts/build_index.py --input path/to/mcp_tools_with_embedding.json --output data/index
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ijson  # noqa: E402
import numpy as np  # noqa: E402

from mcp_router.registry.schema import RegisteredServer, RegisteredTool  # noqa: E402
from mcp_router.config import RouterConfig  # noqa: E402
from mcp_router.registry.manager import IndexRegistry  # noqa: E402


def iter_servers(fp):
    for item in ijson.items(fp, "item"):
        yield item


def build_arrays(servers_raw: list) -> tuple:
    servers: list[RegisteredServer] = []
    s_desc: list[list[float]] = []
    s_sum: list[list[float]] = []
    t_emb: list[list[float]] = []
    t_srv: list[int] = []
    t_loc: list[int] = []
    emb_dim: int | None = None

    for si, item in enumerate(servers_raw):
        name = item.get("name") or item.get("server_name") or "unknown"
        desc = item.get("description") or item.get("server_description") or ""
        summ = item.get("summary") or item.get("server_summary") or desc
        de = item.get("description_embedding")
        se = item.get("summary_embedding") or de
        if not de or not se:
            raise ValueError(f"Server {name} missing description/summary embeddings")
        emb_dim = len(de)

        tools_out: list[RegisteredTool] = []
        for li, t in enumerate(item.get("tools") or []):
            tn = t.get("name") or ""
            td = t.get("description") or ""
            tp = t.get("parameter") or t.get("parameters") or {}
            te = t.get("description_embedding")
            if not te:
                raise ValueError(f"Tool {tn} on {name} missing description_embedding")
            tools_out.append(
                RegisteredTool(name=tn, description=td, parameter=tp if isinstance(tp, dict) else {})
            )
            t_emb.append(te)
            t_srv.append(si)
            t_loc.append(li)

        servers.append(
            RegisteredServer(
                name=name,
                description=desc,
                summary=summ,
                tools=tools_out,
                source="mcp_tools_dataset",
            )
        )
        s_desc.append(de)
        s_sum.append(se)

    dim = emb_dim or 3072
    te_arr = (
        np.array(t_emb, dtype=np.float32)
        if t_emb
        else np.zeros((0, dim), dtype=np.float32)
    )

    return (
        servers,
        np.array(s_desc, dtype=np.float32),
        np.array(s_sum, dtype=np.float32),
        te_arr,
        np.array(t_srv, dtype=np.int64),
        np.array(t_loc, dtype=np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="mcp_tools_with_embedding.json")
    parser.add_argument("--output", type=Path, default=Path("data/index"))
    parser.add_argument("--max-servers", type=int, default=0, help="0 = all (dev: set e.g. 5)")
    args = parser.parse_args()

    cfg = RouterConfig(index_dir=args.output.resolve())

    collected = []
    with open(args.input, "rb") as fp:
        for i, item in enumerate(iter_servers(fp)):
            collected.append(item)
            if args.max_servers and i + 1 >= args.max_servers:
                break

    servers, sd, ss, te, ts, tl = build_arrays(collected)
    reg = IndexRegistry(cfg)
    reg.persist_index(servers, sd, ss, te, ts, tl)
    print(f"Wrote index for {len(servers)} servers, {te.shape[0]} tools -> {cfg.index_dir}")


if __name__ == "__main__":
    main()
