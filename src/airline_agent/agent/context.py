"""给 Agent 使用的航空工具 observation 视图。

环境和 RolloutStep 仍保存完整 ToolResult；这里只压缩发送给下一轮模型的文本，
避免大段重复的 JSON 占用 veRL 的 response token 预算。
"""

from __future__ import annotations

import json
from typing import Any

from ..core.results import ToolResult


def _project_value(value: Any, *, max_items: int = 12, max_string_length: int = 2000) -> tuple[Any, bool]:
    """递归保留结构并限制异常大的列表/字符串，始终返回合法 JSON。"""

    if isinstance(value, dict):
        projected: dict[str, Any] = {}
        truncated = False
        for key, item in value.items():
            projected_item, item_truncated = _project_value(
                item,
                max_items=max_items,
                max_string_length=max_string_length,
            )
            projected[str(key)] = projected_item
            truncated = truncated or item_truncated
        return projected, truncated
    if isinstance(value, list):
        projected_items = []
        truncated = len(value) > max_items
        for item in value[:max_items]:
            projected_item, item_truncated = _project_value(
                item,
                max_items=max_items,
                max_string_length=max_string_length,
            )
            projected_items.append(projected_item)
            truncated = truncated or item_truncated
        if truncated:
            projected_items.append({"_truncated_items": max(0, len(value) - max_items)})
        return projected_items, truncated
    if isinstance(value, str) and len(value) > max_string_length:
        return value[:max_string_length] + "...[truncated]", True
    return value, False


def project_tool_observation(observation: ToolResult) -> str:
    """构造 Agent 可见的紧凑 observation，不改变环境原始结果。"""

    payload, truncated = _project_value(observation.model_dump(mode="json"))
    assert isinstance(payload, dict)
    payload["projection"] = {
        "version": "airline-observation-v1",
        "truncated": truncated,
    }
    return (
        "Tool execution observation:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\nDecide the next action."
    )


__all__ = ["project_tool_observation"]
