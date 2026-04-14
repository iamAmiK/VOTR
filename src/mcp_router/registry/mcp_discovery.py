"""
Discover tools from a live MCP server over stdio.

This is a minimal JSON-RPC client that speaks MCP framing:
- request/notification payloads encoded as JSON bytes
- prefixed by `Content-Length: <n>\\r\\n\\r\\n`
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any, Dict, List, Optional

import httpx

from mcp_router.registry.schema import RegisteredServer, RegisteredTool


class DiscoveryError(RuntimeError):
    pass


def _encode_message(obj: Dict[str, Any]) -> bytes:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _read_framed_message(stdout, timeout_seconds: float) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    header = b""
    while b"\r\n\r\n" not in header:
        if time.time() > deadline:
            raise DiscoveryError("Timed out waiting for MCP response header")
        chunk = stdout.read(1)
        if not chunk:
            raise DiscoveryError("MCP process ended while reading response header")
        header += chunk
    head, _ = header.split(b"\r\n\r\n", 1)
    length = None
    for line in head.decode("utf-8", errors="replace").split("\r\n"):
        low = line.lower()
        if low.startswith("content-length:"):
            length = int(line.split(":", 1)[1].strip())
            break
    if length is None:
        raise DiscoveryError("Missing Content-Length in MCP response")
    payload = stdout.read(length)
    if len(payload) != length:
        raise DiscoveryError("Incomplete MCP response body")
    try:
        return json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise DiscoveryError(f"Invalid JSON from MCP server: {exc}") from exc


def _rpc_call(proc: subprocess.Popen, req_id: int, method: str, params: Dict[str, Any], timeout_seconds: float) -> Dict[str, Any]:
    request = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    assert proc.stdin is not None
    proc.stdin.write(_encode_message(request))
    proc.stdin.flush()

    assert proc.stdout is not None
    deadline = time.time() + timeout_seconds
    while time.time() <= deadline:
        msg = _read_framed_message(proc.stdout, timeout_seconds=max(0.5, deadline - time.time()))
        # Skip notifications; return matching response
        if msg.get("id") != req_id:
            continue
        if "error" in msg:
            raise DiscoveryError(f"RPC error for `{method}`: {msg['error']}")
        return msg.get("result") or {}
    raise DiscoveryError(f"Timed out waiting for RPC result: {method}")


def discover_tools_via_stdio(
    command: str,
    args: Optional[List[str]] = None,
    timeout_seconds: float = 20.0,
) -> List[Dict[str, Any]]:
    """
    Spawn an MCP server via stdio and return `tools/list` entries.
    """
    cmd = [command] + (args or [])
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise DiscoveryError(f"Failed to spawn MCP command: {cmd}") from exc

    try:
        _rpc_call(
            proc,
            req_id=1,
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-router", "version": "0.1.0"},
            },
            timeout_seconds=timeout_seconds,
        )
        # notify initialized
        assert proc.stdin is not None
        proc.stdin.write(
            _encode_message({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        )
        proc.stdin.flush()

        res = _rpc_call(
            proc,
            req_id=2,
            method="tools/list",
            params={},
            timeout_seconds=timeout_seconds,
        )
        tools = res.get("tools") if isinstance(res, dict) else None
        if not isinstance(tools, list):
            raise DiscoveryError("MCP server returned invalid tools/list payload")
        return tools
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()


def _raise_if_rpc_error(msg: Dict[str, Any], method: str) -> None:
    if "error" in msg:
        raise DiscoveryError(f"RPC error for `{method}`: {msg['error']}")


def discover_tools_via_sse(
    url: str,
    timeout_seconds: float = 20.0,
) -> List[Dict[str, Any]]:
    """
    Connect to an MCP server endpoint over HTTP and request tools via JSON-RPC.

    Notes:
    - Different MCP deployments may expose different HTTP paths and wrappers.
    - This implementation sends plain JSON-RPC to the provided URL.
    """
    timeout = httpx.Timeout(timeout_seconds)
    with httpx.Client(timeout=timeout) as client:
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-router", "version": "0.1.0"},
            },
        }
        try:
            r = client.post(url, json=init_req)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise DiscoveryError(f"Failed initialize over HTTP at {url}: {exc}") from exc
        try:
            init_res = r.json()
        except ValueError as exc:
            raise DiscoveryError("Invalid JSON response for initialize") from exc
        if not isinstance(init_res, dict):
            raise DiscoveryError("Invalid initialize response payload")
        _raise_if_rpc_error(init_res, "initialize")

        # best-effort initialized notification
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        try:
            client.post(url, json=notif)
        except httpx.HTTPError:
            pass

        tools_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        try:
            r2 = client.post(url, json=tools_req)
            r2.raise_for_status()
        except httpx.HTTPError as exc:
            raise DiscoveryError(f"Failed tools/list over HTTP at {url}: {exc}") from exc
        try:
            tools_res = r2.json()
        except ValueError as exc:
            raise DiscoveryError("Invalid JSON response for tools/list") from exc
        if not isinstance(tools_res, dict):
            raise DiscoveryError("Invalid tools/list response payload")
        _raise_if_rpc_error(tools_res, "tools/list")
        result = tools_res.get("result") or {}
        if not isinstance(result, dict):
            raise DiscoveryError("Invalid `result` payload for tools/list")
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise DiscoveryError("HTTP MCP endpoint returned invalid tools/list payload")
        return tools


def tools_list_result_to_registered_server(
    server_name: str,
    server_description: str,
    tools_list: List[Dict[str, Any]],
) -> RegisteredServer:
    """Normalize MCP `tools/list` items toward RegisteredServer.tools shape."""
    tools: List[RegisteredTool] = []
    for t in tools_list:
        name = t.get("name", "")
        desc = ""
        raw = t.get("description")
        if isinstance(raw, str):
            desc = raw
        schema = t.get("inputSchema") or t.get("input_schema") or {}
        props = schema.get("properties") if isinstance(schema, dict) else {}
        param: Dict[str, Any] = {}
        if isinstance(props, dict):
            for pname, pinfo in props.items():
                if isinstance(pinfo, dict):
                    typ = pinfo.get("type", "any")
                    pdesc = pinfo.get("description", "")
                    req = schema.get("required") or []
                    opt = pname not in req if isinstance(req, list) else False
                    prefix = "(Optional, " if opt else "("
                    param[pname] = f"{prefix}{typ}) {pdesc}".strip()
        tools.append(RegisteredTool(name=name, description=desc, parameter=param))
    return RegisteredServer(
        name=server_name,
        description=server_description,
        summary=server_description,
        tools=tools,
        source="mcp_live",
    )
