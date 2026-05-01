from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from mcp_router.retrieval.tokenization import tokenize as _tokenize

_ACTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "in",
    "into",
    "me",
    "my",
    "of",
    "on",
    "please",
    "show",
    "the",
    "to",
    "with",
}

_CONSTRAINT_HINTS = {
    "account",
    "alias",
    "author",
    "branch",
    "bucket",
    "channel",
    "column",
    "comment",
    "commit",
    "cursor",
    "date",
    "dm",
    "document",
    "emoji",
    "field",
    "file",
    "filter",
    "folder",
    "id",
    "index",
    "issue",
    "key",
    "label",
    "main",
    "message",
    "name",
    "path",
    "private",
    "project",
    "pr",
    "pull",
    "query",
    "reaction",
    "record",
    "repo",
    "repository",
    "row",
    "schema",
    "sheet",
    "slack",
    "status",
    "table",
    "task",
    "ticket",
    "user",
}


@dataclass(frozen=True)
class QueryFields:
    server_query: str
    action_query: str
    constraint_query: str
    explicit_server_name: str | None


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _compact(text: str) -> str:
    """Lowercase and remove non-alphanumerics for loose alias matching."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _server_name_tokens(name: str) -> tuple[str, ...]:
    return tuple(tok for tok in re.findall(r"[a-z0-9]+", (name or "").lower()) if tok)


def _match_explicit_server(
    server_intent: str,
    tool_intent: str,
    known_server_names: Sequence[str],
) -> str | None:
    raw_haystack = _normalize_spaces(f"{server_intent} {tool_intent}").lower()
    compact_haystack = _compact(raw_haystack)
    candidates = sorted(
        {name.strip() for name in known_server_names if name.strip()},
        key=len,
        reverse=True,
    )
    token_sets = {name: set(_server_name_tokens(name)) for name in candidates}

    for name in candidates:
        compact_name = _compact(name)
        if len(compact_name) < 3:
            continue

        name_tokens = token_sets[name]
        if len(name_tokens) == 1:
            tok = next(iter(name_tokens))
            colliding = [
                other
                for other, other_tokens in token_sets.items()
                if other != name and tok in other_tokens
            ]
            if colliding:
                # Single-token matches like "Xero" are too ambiguous when
                # another server family also contains that token.
                continue

        needle = name.lower()
        if f" {needle} " in f" {raw_haystack} ":
            return name

        if compact_name in compact_haystack:
            return name

    return None

def decompose_query(
    server_intent: str,
    tool_intent: str,
    known_server_names: Sequence[str],
) -> QueryFields:
    server_query = _normalize_spaces(server_intent)
    tool_query = _normalize_spaces(tool_intent)
    explicit_server_name = _match_explicit_server(server_query, tool_query, known_server_names)

    tokens = _tokenize(tool_query)
    action_tokens = [tok for tok in tokens if tok not in _ACTION_STOPWORDS]
    if not action_tokens:
        action_tokens = tokens

    constraint_tokens: list[str] = []
    for tok in tokens:
        if any(ch.isdigit() for ch in tok) or tok.startswith(("#", "/", ".")):
            constraint_tokens.append(tok)
            continue
        if tok in _CONSTRAINT_HINTS:
            constraint_tokens.append(tok)

    seen: set[str] = set()
    deduped_constraints = []
    for tok in constraint_tokens:
        if tok not in seen:
            seen.add(tok)
            deduped_constraints.append(tok)

    return QueryFields(
        server_query=server_query,
        action_query=" ".join(action_tokens).strip() or tool_query,
        constraint_query=" ".join(deduped_constraints).strip(),
        explicit_server_name=explicit_server_name,
    )
