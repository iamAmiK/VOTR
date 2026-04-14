"""
MCP-Zero-style tool strings for token-count comparisons (mirrors MCP-zero/reformatter.py).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from mcp_router.registry.schema import RegisteredTool


def _extract_parameter_type(param_desc: str) -> Tuple[str, str]:
    if not param_desc or not param_desc.startswith("("):
        return "string", param_desc

    try:
        type_end = param_desc.find(")")
        if type_end == -1:
            return "string", param_desc

        param_type = param_desc[1:type_end].strip().lower()
        description = param_desc[type_end + 1 :].strip()

        type_mapping = {
            "string": "string",
            "str": "string",
            "integer": "integer",
            "int": "integer",
            "number": "number",
            "float": "number",
            "boolean": "boolean",
            "bool": "boolean",
            "array": "array",
            "object": "object",
            "dict": "object",
        }

        mapped_type = type_mapping.get(param_type, "string")
        return mapped_type, description
    except Exception:
        return "string", param_desc


def _format_parameters(parameters: Dict[str, str]) -> Dict[str, Any]:
    if not parameters:
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {},
            "required": [],
        }

    properties: Dict[str, Any] = {}
    required: list[str] = []

    for param_name, param_desc in parameters.items():
        param_type, param_description = _extract_parameter_type(param_desc)
        properties[param_name] = {
            "description": param_description,
            "type": param_type,
        }
        if "optional" not in param_description.lower():
            required.append(param_name)

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
        "type": "object",
    }


def mcp_zero_function_block(server_name: str, tool: RegisteredTool) -> str:
    """Single-tool string as MCP-Zero injects (<function>...</function>)."""
    raw = tool.parameter or {}
    str_params: Dict[str, str] = {}
    for name, val in raw.items():
        str_params[name] = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)

    formatted_parameters = _format_parameters(str_params)
    function_obj = {
        "description": tool.description or "",
        "name": f"mcp_{server_name.lower()}_{tool.name.lower()}",
        "parameters": formatted_parameters,
    }
    return f"<function>{json.dumps(function_obj, ensure_ascii=False)}</function>"
