#!/usr/bin/env python3
"""
Optional helper: print the command to build an index from an MCP-Zero dataset file.

The dataset format is already compatible with `scripts/build_index.py`; this script
only documents paths and can symlink data for convenience.

Usage:
  python scripts/migrate_mcpzero.py --mcp-zero ../MCP-Zero
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mcp-zero", type=Path, default=Path("../MCP-Zero"))
    args = p.parse_args()
    root = args.mcp_zero.resolve()
    json_path = root / "MCP-tools" / "mcp_tools_with_embedding.json"
    if not json_path.is_file():
        print(f"Dataset not found: {json_path}")
        print("Download from MCP-Zero README (Google Drive) and place the file there.")
        return
    here = Path(__file__).resolve().parents[1]
    out = here / "data" / "index"
    print("Run:")
    print(
        f'  python "{here / "scripts" / "build_index.py"}" '
        f'--input "{json_path}" --output "{out}"'
    )


if __name__ == "__main__":
    main()
