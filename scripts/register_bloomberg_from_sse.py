#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


def _param_map_from_input_schema(schema: dict[str, Any]) -> dict[str, str]:
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = schema.get("required", []) if isinstance(schema, dict) else []
    out: dict[str, str] = {}
    for name, p in props.items():
        if not isinstance(p, dict):
            out[name] = "(any) "
            continue
        typ = p.get("type", "any")
        desc = p.get("description", "")
        optional = name not in required if isinstance(required, list) else True
        prefix = "(Optional, " if optional else "("
        out[name] = f"{prefix}{typ}) {desc}".strip()
    return out


async def fetch_tools(sse_url: str) -> list[dict[str, Any]]:
    async with sse_client(sse_url) as (read_stream, write_stream):
        async with ClientSession(read_stream=read_stream, write_stream=write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            out: list[dict[str, Any]] = []
            for t in tools.tools:
                name = getattr(t, "name", "")
                description = getattr(t, "description", "") or ""
                input_schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None) or {}
                out.append(
                    {
                        "name": name,
                        "description": description,
                        "parameter": _param_map_from_input_schema(input_schema),
                    }
                )
            return out


def register_with_router(router_url: str, server_name: str, server_description: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "server": {
            "name": server_name,
            "description": server_description,
            "summary": server_description,
            "tools": tools,
            "source": "mcp_live_sse",
        }
    }
    req = urllib.request.Request(
        f"{router_url.rstrip('/')}/register",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Discover Bloomberg MCP tools via SSE and register into router.")
    ap.add_argument("--sse-url", default="http://127.0.0.1:8000/sse")
    ap.add_argument("--router-url", default="http://127.0.0.1:8765")
    ap.add_argument("--server-name", default="Bloomberg")
    ap.add_argument("--server-description", default="Bloomberg BLPAPI financial data tools")
    args = ap.parse_args()

    tools = asyncio.run(fetch_tools(args.sse_url))
    print(f"Discovered tools: {len(tools)}")
    for t in tools:
        print("-", t["name"])
    result = register_with_router(
        router_url=args.router_url,
        server_name=args.server_name,
        server_description=args.server_description,
        tools=tools,
    )
    print("Router register response:", result)


if __name__ == "__main__":
    main()
