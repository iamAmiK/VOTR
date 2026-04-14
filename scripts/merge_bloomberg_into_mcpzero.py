#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    base = Path(r"c:\Users\swaga\Downloads\Research Proj\MCP-Zero\MCP-tools\mcp_tools_with_embedding.json")
    backup = base.with_name("mcp_tools_with_embedding.pre_bloomberg_backup.json")
    idx = Path(r"c:\Users\swaga\Downloads\Research Proj\MCP-Router\data\index")

    meta = json.load(open(idx / "meta.json", encoding="utf-8"))
    servers = meta["servers"]
    names = [s.get("name") for s in servers]
    if "Bloomberg" not in names:
        raise SystemExit("Bloomberg server not found in router index")
    si = names.index("Bloomberg")
    srv = servers[si]

    sd = np.load(idx / "server_description_embeddings.npy")
    ss = np.load(idx / "server_summary_embeddings.npy")
    te = np.load(idx / "tool_embeddings.npy")
    ts = np.load(idx / "tool_server_indices.npy")
    tl = np.load(idx / "tool_local_indices.npy")

    rows = np.where(ts == si)[0]
    rows = rows[np.argsort(tl[rows])]
    tools = srv.get("tools", [])
    if len(rows) != len(tools):
        raise SystemExit(f"Tool row mismatch: {len(rows)} vs {len(tools)}")

    bloomberg = {
        "name": srv.get("name"),
        "description": srv.get("description", ""),
        "summary": srv.get("summary", ""),
        "description_embedding": sd[si].tolist(),
        "summary_embedding": ss[si].tolist(),
        "tools": [],
    }
    for row, t in zip(rows, tools):
        bloomberg["tools"].append(
            {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameter": t.get("parameter", {}),
                "description_embedding": te[row].tolist(),
            }
        )

    if not backup.exists():
        shutil.copy2(base, backup)
        print(f"Backup created: {backup}")
    else:
        print(f"Backup exists: {backup}")

    data = json.load(open(base, encoding="utf-8"))
    existing_names = [x.get("name") for x in data]
    if "Bloomberg" in existing_names:
        i = existing_names.index("Bloomberg")
        data[i] = bloomberg
        action = "updated existing Bloomberg entry"
    else:
        data.append(bloomberg)
        action = "appended Bloomberg entry"

    tmp = base.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    shutil.move(tmp, base)

    print(action)
    print(f"Final server count: {len(data)}")


if __name__ == "__main__":
    main()
