from typing import Any, Literal

from pydantic import BaseModel, Field

from .actions import AgentAction
from .results import ToolResult


TerminationReason = Literal[
    "finished",
    "max_steps",
    "llm_error",
]


class RolloutStep(BaseModel):
    """Agent Loop 中的一步模型决策及执行结果。"""

    step_index: int = Field(ge=1)
    raw_model_output: str
    action: AgentAction | None = None
    observation: ToolResult | None = None # 工具调用结果
    user_reply: str | None = None # User Simulator 的回复
    parse_error: str | None = None # 解析错误信息
 

class RolloutRecord(BaseModel):
    """一条任务从开始到结束的完整交互轨迹。"""

    task_id: str
    user_request: str

    messages: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[RolloutStep] = Field(default_factory=list)

    initial_state: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )
    final_state: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )

    final_answer: str | None = None
    termination_reason: TerminationReason | None = None
    reward: float | None = None
