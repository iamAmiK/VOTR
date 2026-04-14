from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List

from mcp_router.registry.schema import RegisteredServer, RegisteredTool

_GENERIC_TOKENS = {
    "a",
    "an",
    "and",
    "api",
    "by",
    "for",
    "from",
    "get",
    "in",
    "into",
    "mcp",
    "of",
    "on",
    "operation",
    "operations",
    "server",
    "the",
    "to",
    "tool",
    "tools",
    "via",
    "with",
}


def _tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    text = re.sub(r"[_./:#-]+", " ", text)
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalized_tokens(text: str, *, drop: set[str] | None = None) -> list[str]:
    drop = drop or set()
    return [tok for tok in _tokenize(text) if tok not in _GENERIC_TOKENS and tok not in drop]


def _parameter_tokens(tool: RegisteredTool) -> list[str]:
    tokens: list[str] = []
    for key in (tool.parameter or {}).keys():
        tokens.extend(_tokenize(str(key)))
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens:
        if tok not in seen and tok not in _GENERIC_TOKENS:
            seen.add(tok)
            out.append(tok)
    return out


def capability_signature(server: RegisteredServer, tool: RegisteredTool) -> str:
    server_tokens = set(_tokenize(server.name))
    name_tokens = _normalized_tokens(tool.name, drop=server_tokens)
    desc_tokens = _normalized_tokens(tool.description, drop=server_tokens)
    param_tokens = _parameter_tokens(tool)

    core_tokens = name_tokens[:6]
    if len(core_tokens) < 2:
        for tok in desc_tokens:
            if tok not in core_tokens:
                core_tokens.append(tok)
            if len(core_tokens) >= 6:
                break

    if not core_tokens:
        core_tokens = _tokenize(tool.name)[:6]

    extra_tokens: list[str] = []
    for tok in param_tokens:
        if tok not in core_tokens:
            extra_tokens.append(tok)
        if len(extra_tokens) >= 4:
            break

    return " ".join(core_tokens + extra_tokens).strip()


def build_overlap_groups(servers: list[RegisteredServer]) -> tuple[list[str], Dict[str, list[int]]]:
    signatures: list[str] = []
    groups: dict[str, list[int]] = defaultdict(list)
    row = 0
    for server in servers:
        for tool in server.tools:
            sig = capability_signature(server, tool)
            signatures.append(sig)
            if sig:
                groups[sig].append(row)
            row += 1

    filtered: dict[str, list[int]] = {}
    for sig, rows in groups.items():
        if len(rows) < 2:
            continue
        filtered[sig] = rows
    return signatures, filtered
