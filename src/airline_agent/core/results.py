from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Stable result envelope returned by every tool execution."""

    success: bool
    tool_name: str
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    message: str
