from __future__ import annotations

import re
from typing import Iterable


_BULK_WORDS = ("bulk", "batch", "multiple", "all", "many", "mass")
_SINGULAR_HINTS = (" a ", " an ", " one ", " single ", " specific ", " by id", " exact ", "task id", "file id", "user id")
_PLURAL_HINTS = (" all ", " multiple ", " many ", " list ", " batch ")


def _normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return f" {t} "


def is_singular_intent(tool_intent: str) -> bool:
    t = _normalize(tool_intent)
    return any(h in t for h in _SINGULAR_HINTS)


def is_plural_intent(tool_intent: str) -> bool:
    t = _normalize(tool_intent)
    return any(h in t for h in _PLURAL_HINTS)


def looks_bulk_tool(name: str, description: str) -> bool:
    blob = _normalize(f"{name} {description}")
    return any(w in blob for w in _BULK_WORDS)


def looks_destructive(tool_intent: str) -> bool:
    t = _normalize(tool_intent)
    destructive_roots: Iterable[str] = (" delete ", " remove ", " purge ", " clear ", " erase ")
    return any(r in t for r in destructive_roots)
