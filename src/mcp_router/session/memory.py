from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import DefaultDict, Set


class SessionMemory:
    """Track per-session tool history for analytics and future session-aware retrieval."""

    def __init__(self, ttl_seconds: int = 86400):
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._tools: DefaultDict[str, Set[str]] = defaultdict(set)
        self._last_seen: dict[str, float] = {}

    def _gc(self, now: float) -> None:
        if self._ttl <= 0:
            return
        dead = [sid for sid, t in self._last_seen.items() if now - t > self._ttl]
        for sid in dead:
            self._tools.pop(sid, None)
            self._last_seen.pop(sid, None)

    def add_tools(self, session_id: str | None, tool_keys: list[str]) -> None:
        if not session_id or not tool_keys:
            return
        with self._lock:
            now = time.time()
            self._gc(now)
            self._last_seen[session_id] = now
            self._tools[session_id].update(tool_keys)

    def get_tools(self, session_id: str | None) -> Set[str]:
        """Return all tools seen in this session (read-only view)."""
        if not session_id:
            return set()
        with self._lock:
            return set(self._tools.get(session_id, set()))

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._tools.pop(session_id, None)
            self._last_seen.pop(session_id, None)
