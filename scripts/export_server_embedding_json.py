#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def export_server(index_dir: Path, server_name: str) -> dict:
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    servers = meta["servers"]
    names = [s.get("name") for s in servers]
    if server_name not in names:
        raise SystemExit(f"Server not found in index: {server_name}")

    si = names.index(server_name)
    srv = servers[si]

    sd = np.load(index_dir / "server_description_embeddings.npy")
    ss = np.load(index_dir / "server_summary_embeddings.npy")
    te = np.load(index_dir / "tool_embeddings.npy")
    ts = np.load(index_dir / "tool_server_indices.npy")
    tl = np.load(index_dir / "tool_local_indices.npy")

    rows = np.where(ts == si)[0]
    rows = rows[np.argsort(tl[rows])]
    tools = srv.get("tools", [])
    if len(rows) != len(tools):
        raise SystemExit(f"Tool row mismatch for {server_name}: {len(rows)} vs {len(tools)}")

    out = {
        "name": srv.get("name", ""),
        "description": srv.get("description", ""),
        "summary": srv.get("summary", ""),
        "description_embedding": sd[si].tolist(),
        "summary_embedding": ss[si].tolist(),
        "tools": [],
    }
    for row, t in zip(rows, tools):
        out["tools"].append(
            {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameter": t.get("parameter", {}),
                "description_embedding": te[row].tolist(),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export one server from router index into MCP-Zero-style embedding JSON.")
    ap.add_argument("--index-dir", type=Path, default=Path("data/index"))
    ap.add_argument("--server", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    payload = export_server(args.index_dir, args.server)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {args.server} -> {args.out}")


if __name__ == "__main__":
    main()
