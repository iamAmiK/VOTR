#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_router.config import load_config  # noqa: E402
from mcp_router.retrieval.embedder import OpenAIEmbedder  # noqa: E402


def schema_to_parameter_map(schema: dict[str, Any]) -> dict[str, Any]:
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    out: dict[str, Any] = {}
    for name, spec in props.items():
        if isinstance(spec, dict):
            desc = spec.get("description", "")
            typ = spec.get("type", "any")
            suffix = " required" if name in required else ""
            out[name] = f"{typ}{suffix}: {desc}".strip(": ").strip()
        else:
            out[name] = str(spec)
    return out


def parameter_text(parameter: dict[str, Any]) -> str:
    if not parameter:
        return ""
    parts = [f"{name}: {value}" for name, value in parameter.items()]
    return " Parameters: " + "; ".join(parts)


def normalize_server_record(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_blocks = item.get("tools") or {}
    if not tool_blocks:
        raise ValueError(f"Server '{item.get('name', '<unknown>')}' has no tool blocks.")
    if len(tool_blocks) != 1:
        # The common case is one MCP server per item. If there are more, use the
        # first and keep the canonical key from the nested block.
        block = next(iter(tool_blocks.values()))
    else:
        block = next(iter(tool_blocks.values()))

    server_name = block.get("server_name")
    if not server_name:
        raise ValueError(f"Server '{item.get('name', '<unknown>')}' missing nested server_name.")

    tools = []
    for tool in block.get("tools") or []:
        parameter = schema_to_parameter_map(tool.get("inputSchema") or {})
        tools.append(
            {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameter": parameter,
            }
        )

    return server_name, {
        "name": server_name,
        "display_name": item.get("name", server_name),
        "organization": item.get("organization", ""),
        "category": item.get("category", ""),
        "description": item.get("description", ""),
        "summary": (
            f"{item.get('name', server_name)}. "
            f"Organization: {item.get('organization', 'unknown')}. "
            f"Category: {item.get('category', 'unknown')}. "
            f"{item.get('description', '')}"
        ).strip(),
        "tools": tools,
    }


def build_payload(tools_json: Path) -> list[dict[str, Any]]:
    raw = json.loads(tools_json.read_text(encoding="utf-8"))
    servers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        server_name, server = normalize_server_record(item)
        if server_name in seen:
            continue
        seen.add(server_name)
        servers.append(server)
    return servers


def embed_payload(servers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cfg = load_config()
    embedder = OpenAIEmbedder(cfg)

    server_desc_texts = [s["description"] or s["summary"] or s["name"] for s in servers]
    server_sum_texts = [s["summary"] or s["description"] or s["name"] for s in servers]
    server_desc_embeddings = embedder.embed_batch(server_desc_texts)
    server_sum_embeddings = embedder.embed_batch(server_sum_texts)

    tool_texts: list[str] = []
    tool_positions: list[tuple[int, int]] = []
    for si, server in enumerate(servers):
        for ti, tool in enumerate(server["tools"]):
            text = f"{tool['name']}. {tool['description']}.{parameter_text(tool['parameter'])}".strip()
            tool_texts.append(text)
            tool_positions.append((si, ti))

    tool_embeddings = embedder.embed_batch(tool_texts) if tool_texts else []

    for si, server in enumerate(servers):
        server["description_embedding"] = server_desc_embeddings[si]
        server["summary_embedding"] = server_sum_embeddings[si]

    for emb, (si, ti) in zip(tool_embeddings, tool_positions):
        servers[si]["tools"][ti]["description_embedding"] = emb

    return servers


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert LiveMCPBench tools.json into MCP-Zero-style embedding JSON.")
    ap.add_argument("--tools-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    payload = build_payload(args.tools_json)
    payload = embed_payload(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(payload)} embedded LiveMCPBench servers to {args.out}")


if __name__ == "__main__":
    main()
