#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer


def _tool_text(tool: dict[str, Any]) -> str:
    name = str(tool.get("name", "") or "")
    desc = str(tool.get("description", "") or "")
    return (desc or name).strip()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create MCP-Zero style embedding JSON from index meta.json using a local HF model."
    )
    ap.add_argument("--meta", type=Path, default=Path("data/index/meta.json"))
    ap.add_argument(
        "--model",
        type=str,
        default="google/embeddinggemma-300m",
        help="SentenceTransformer model id",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/gemma/mcp_tools_with_embedding.embeddinggemma-300m.json"),
    )
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    raw = json.loads(args.meta.read_text(encoding="utf-8"))
    servers: list[dict[str, Any]] = raw.get("servers", [])
    if not servers:
        raise SystemExit(f"No servers found in {args.meta}")

    model = SentenceTransformer(args.model)

    out_servers: list[dict[str, Any]] = []
    for i, server in enumerate(servers, start=1):
        s_name = str(server.get("name", "") or "")
        s_desc = str(server.get("description", "") or "")
        s_sum = str(server.get("summary", "") or s_desc)

        s_vecs = model.encode([s_desc or s_name, s_sum or s_desc or s_name], normalize_embeddings=True)
        s_desc_vec = s_vecs[0].tolist()
        s_sum_vec = s_vecs[1].tolist()

        tools = server.get("tools", []) or []
        tool_texts = [_tool_text(t) for t in tools]
        tool_vecs = model.encode(
            tool_texts if tool_texts else [""],
            normalize_embeddings=True,
            batch_size=args.batch_size,
            show_progress_bar=False,
        )

        out_tools: list[dict[str, Any]] = []
        for ti, tool in enumerate(tools):
            out_tools.append(
                {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameter": tool.get("parameter", {}) or {},
                    "description_embedding": tool_vecs[ti].tolist(),
                }
            )

        out_servers.append(
            {
                "name": s_name,
                "description": s_desc,
                "summary": s_sum,
                "description_embedding": s_desc_vec,
                "summary_embedding": s_sum_vec,
                "tools": out_tools,
            }
        )
        if i % 25 == 0:
            print(f"Embedded {i}/{len(servers)} servers")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_servers, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(out_servers)} servers to {args.out}")


if __name__ == "__main__":
    main()
