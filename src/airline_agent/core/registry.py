from pydantic import BaseModel
from typing import Literal, Any, Callable
from dataclasses import dataclass

ToolFunction = Callable[[BaseModel], Any]

@dataclass(frozen=True)

class ToolSpec:
    """一个已注册工具的完整定义。"""

    name: str
    description: str
    args_schema: type[BaseModel]
    function: ToolFunction
    mutates_state: bool = False


class ToolRegistry:
    """一个工具注册表，允许注册和检索工具。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register_tool(
        self,
        spec: ToolSpec,
    ) -> None:
        """注册一个新工具。"""
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' is already registered.")
        self._tools[spec.name] = spec

    def get_tool(self, tool_name: str) -> ToolSpec | None:
        """检索已注册的工具定义。"""
        return self._tools.get(tool_name)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """返回所有已注册工具的定义列表。"""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.args_schema.model_json_schema(),
            }
            for spec in self._tools.values()
        ]
