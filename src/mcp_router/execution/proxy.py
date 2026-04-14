"""
Optional MCP execution proxy (not implemented).

The router returns ranked tool metadata; your agent stack should call MCP servers.
A future `ExecutionProxy` could maintain stdio/SSE connections and forward
`tools/call` JSON-RPC messages.
"""

from __future__ import annotations


class ExecutionProxy:
    def __init__(self) -> None:
        raise NotImplementedError("ExecutionProxy is intentionally not implemented in v0.1.")
