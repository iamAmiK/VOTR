#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests


def _tool_text(tool: dict[str, Any]) -> str:
    name = str(tool.get("name", "") or "")
    desc = str(tool.get("description", "") or "")
    return (desc or name or "unknown_tool").strip()


def _embed(api_key: str, model: str, texts: list[str], dimensions: int) -> list[list[float]]:
    payload = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
        "dimensions": dimensions,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers=headers,
        json=payload,
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenRouter embed failed: {resp.status_code} {resp.text}")

    response_data = resp.json()
    data = response_data.get("data") or []
    if not data:
        raise ValueError(f"OpenRouter embedding response had no data: {response_data}")
    return [list(item["embedding"]) for item in data]


def _embed_one_server(
    api_key: str,
    model: str,
    dimensions: int,
    batch_size: int,
    server: dict[str, Any],
) -> dict[str, Any]:
    s_name = str(server.get("name", "") or "")
    s_desc = str(server.get("description", "") or "")
    s_sum = str(server.get("summary", "") or s_desc)

    s_vecs = _embed(api_key, model, [s_desc or s_name, s_sum or s_desc or s_name], dimensions)
    s_desc_vec = s_vecs[0]
    s_sum_vec = s_vecs[1]

    tools = server.get("tools", []) or []
    tool_texts = [_tool_text(t) for t in tools]

    out_tools: list[dict[str, Any]] = []
    for j in range(0, len(tool_texts), batch_size):
        chunk_texts = tool_texts[j : j + batch_size]
        chunk_vecs = _embed(api_key, model, chunk_texts, dimensions)
        for k, vec in enumerate(chunk_vecs):
            tool = tools[j + k]
            out_tools.append(
                {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameter": tool.get("parameter", {}) or {},
                    "description_embedding": vec,
                }
            )

    return {
        "name": s_name,
        "description": s_desc,
        "summary": s_sum,
        "description_embedding": s_desc_vec,
        "summary_embedding": s_sum_vec,
        "tools": out_tools,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create MCP-Zero style embedding JSON from index meta.json using Qwen embeddings."
    )
    ap.add_argument("--meta", type=Path, default=Path("data/index/meta.json"))
    ap.add_argument(
        "--model",
        type=str,
        default="qwen/qwen3-embedding-8b",
        help="OpenRouter embedding model id",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/qwen_embedding_8b/mcp_tools_with_embedding.qwen3-embedding-8b.json"),
    )
    ap.add_argument("--dimensions", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--api-key-env", type=str, default="OPENROUTER_API_KEY")
    args = ap.parse_args()

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set")

    raw = json.loads(args.meta.read_text(encoding="utf-8"))
    servers: list[dict[str, Any]] = raw.get("servers", [])
    if not servers:
        raise SystemExit(f"No servers found in {args.meta}")

    out_servers: list[dict[str, Any]] = []
    for i, server in enumerate(servers, start=1):
        out_servers.append(_embed_one_server(api_key, args.model, args.dimensions, args.batch_size, server))
        if i % 25 == 0 or i == len(servers):
            print(f"Embedded {i}/{len(servers)} servers", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_servers, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(out_servers)} servers to {args.out}", flush=True)


if __name__ == "__main__":
    main()
