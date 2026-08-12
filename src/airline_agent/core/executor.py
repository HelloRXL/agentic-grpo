from typing import Any

from pydantic import ValidationError

from .errors import AirlineBusinessError
from .registry import ToolRegistry
from .results import ToolResult


class ToolExecutor:
    """验证并执行已注册工具。"""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(
        self,
        tool_name: str,
        raw_arguments: dict[str, Any],
    ) -> ToolResult:
        spec = self._registry.get_tool(tool_name)

        if spec is None:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error="tool_not_found",
                message=f"Tool {tool_name} does not exist.",
            )

        try:
            validated_arguments = spec.args_schema.model_validate(
                raw_arguments
            )
        except ValidationError as error:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error="validation_error",
                message=str(error),
            )

        try:
            # Schema 已经完成验证，工具直接接收验证后的模型，避免重复解析嵌套字段。
            data = spec.function(validated_arguments)
        except AirlineBusinessError as error:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=error.code,
                message=error.message,
            )
        except Exception:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error="system_error",
                message="An unexpected error occurred while executing the tool.",
            )

        return ToolResult(
            success=True,
            tool_name=tool_name,
            data=data,
            message="Tool executed successfully.",
        )
