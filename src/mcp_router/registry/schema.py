from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


ToolParameterMap = Dict[str, Any]


class RegisteredTool(BaseModel):
    name: str
    description: str = ""
    parameter: ToolParameterMap = Field(default_factory=dict)


class RegisteredServer(BaseModel):
    """Canonical server record (matches MCP-Zero dataset shape + optional ids)."""

    name: str
    description: str = ""
    summary: str = ""
    tools: List[RegisteredTool] = Field(default_factory=list)
    source: str = "dataset"
    external_id: Optional[str] = None

    def tool_key(self, tool_name: str) -> str:
        return f"{self.name}::{tool_name}"
