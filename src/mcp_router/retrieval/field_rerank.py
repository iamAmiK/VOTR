from __future__ import annotations

import re
from typing import Iterable

from mcp_router.registry.schema import RegisteredServer, RegisteredTool
from mcp_router.retrieval.field_scoring import FieldScoringWeights
from mcp_router.retrieval.query_fields import QueryFields


def _tokenize(text: str) -> set[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    text = re.sub(r"[_\-./:#]+", " ", text)
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _overlap_score(query: str, text: str) -> float:
    q = _tokenize(query)
    t = _tokenize(text)
    if not q or not t:
        return 0.0
    common = len(q & t)
    if common == 0:
        return 0.0
    return (2.0 * common) / (len(q) + len(t))


def _parameter_text(tool: RegisteredTool) -> str:
    parts: list[str] = []
    for key, value in (tool.parameter or {}).items():
        parts.append(str(key))
        if isinstance(value, (list, tuple)):
            parts.extend(str(v) for v in value)
        elif isinstance(value, dict):
            parts.extend(f"{k} {v}" for k, v in value.items())
        else:
            parts.append(str(value))
    return " ".join(parts)


def field_match_components(
    server: RegisteredServer,
    tool: RegisteredTool,
    query_fields: QueryFields,
) -> dict[str, float]:
    return {
        "server_name": _overlap_score(query_fields.server_query, server.name),
        "server_summary": _overlap_score(
            query_fields.server_query,
            f"{server.summary} {server.description}",
        ),
        "tool_name": _overlap_score(query_fields.action_query, tool.name),
        "tool_description": _overlap_score(query_fields.action_query, tool.description),
        "parameters": _overlap_score(query_fields.constraint_query, _parameter_text(tool)),
    }


def normalized_query_support(
    server: RegisteredServer,
    tool: RegisteredTool,
    query_fields: QueryFields,
    weights: FieldScoringWeights,
) -> float:
    parts = field_match_components(server, tool, query_fields)
    total_weight = (
        weights.server_name
        + weights.server_summary
        + weights.tool_name
        + weights.tool_description
        + weights.parameters
    )
    if total_weight <= 0:
        return 0.0
    return (
        (weights.server_name * parts["server_name"])
        + (weights.server_summary * parts["server_summary"])
        + (weights.tool_name * parts["tool_name"])
        + (weights.tool_description * parts["tool_description"])
        + (weights.parameters * parts["parameters"])
    ) / total_weight


def field_aware_bonus(
    server: RegisteredServer,
    tool: RegisteredTool,
    query_fields: QueryFields,
    weights: FieldScoringWeights,
) -> float:
    parts = field_match_components(server, tool, query_fields)
    bonus = 0.0
    bonus += weights.server_name * parts["server_name"]
    bonus += weights.server_summary * parts["server_summary"]
    bonus += weights.tool_name * parts["tool_name"]
    bonus += weights.tool_description * parts["tool_description"]
    bonus += weights.parameters * parts["parameters"]
    if query_fields.explicit_server_name and query_fields.explicit_server_name == server.name:
        bonus += weights.explicit_server_boost
    return bonus
