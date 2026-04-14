#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a subset of servers from router index into MCP-Zero-style embedding JSON.")
    ap.add_argument("--index-dir", type=Path, default=Path("data/index"))
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    meta = json.loads((args.index_dir / "meta.json").read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    wanted = manifest["servers"]

    servers = meta["servers"]
    names = [s.get("name") for s in servers]

    sd = np.load(args.index_dir / "server_description_embeddings.npy")
    ss = np.load(args.index_dir / "server_summary_embeddings.npy")
    te = np.load(args.index_dir / "tool_embeddings.npy")
    ts = np.load(args.index_dir / "tool_server_indices.npy")
    tl = np.load(args.index_dir / "tool_local_indices.npy")

    out = []
    missing = []
    for server_name in wanted:
        if server_name not in names:
            missing.append(server_name)
            continue
        si = names.index(server_name)
        srv = servers[si]
        rows = np.where(ts == si)[0]
        rows = rows[np.argsort(tl[rows])]
        tools = srv.get("tools", [])
        if len(rows) != len(tools):
            raise SystemExit(f"Tool row mismatch for {server_name}: {len(rows)} vs {len(tools)}")
        item = {
            "name": srv.get("name", ""),
            "description": srv.get("description", ""),
            "summary": srv.get("summary", ""),
            "description_embedding": sd[si].tolist(),
            "summary_embedding": ss[si].tolist(),
            "tools": [],
        }
        for row, t in zip(rows, tools):
            item["tools"].append(
                {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameter": t.get("parameter", {}),
                    "description_embedding": te[row].tolist(),
                }
            )
        out.append(item)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(out)} servers -> {args.out}")
    if missing:
        print("Missing servers:")
        for name in missing:
            print("-", name)


if __name__ == "__main__":
    main()
