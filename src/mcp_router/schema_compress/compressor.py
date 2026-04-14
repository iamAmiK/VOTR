from __future__ import annotations

from typing import Any, Dict, List

from mcp_router.registry.schema import RegisteredTool


def _param_tokens(parameter: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    for name, spec in parameter.items():
        if isinstance(spec, str):
            low = spec.lower()
            optional = "optional" in low or "optional" in spec
            typ = "str"
            if "number" in low or "int" in low or "float" in low:
                typ = "num"
            if "bool" in low:
                typ = "bool"
            suffix = "?" if optional else ""
            parts.append(f"{name}: {typ}{suffix}")
        else:
            parts.append(f"{name}: any")
    return parts


def compress_tool_line(server_name: str, tool: RegisteredTool) -> str:
    params = _param_tokens(tool.parameter)
    param_str = ", ".join(params) if params else ""
    head = f"[server: {server_name}] {tool.name}"
    if param_str:
        head += f"({param_str})"
    return f"{head}\n  → {tool.description.strip() or '(no description)'}"
