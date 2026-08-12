from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..domain.state_patches import InitialStatePatch


TaskSplit = Literal["train", "dev", "test", "base"]
TaskStatus = Literal["supported", "unsupported"]
RewardBasis = Literal["DB", "ACTION", "COMMUNICATE", "NL_ASSERTION"]


class UserScenarioView(BaseModel):
    """只提供给 User Simulator 的隐藏用户剧本。"""

    model_config = ConfigDict(extra="forbid")

    persona: str | None = None
    domain: str
    reason_for_call: str
    known_info: str | None = None
    unknown_info: str | None = None
    task_instructions: str


class ReferenceAction(BaseModel):
    """用于 Replay 推导目标状态的参考工具动作。"""

    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(ge=1)
    action_id: str
    original_tool_name: str
    tool_name: str
    arguments: dict[str, Any]
    info: str | None = None


class TaskSpec(BaseModel):
    """我们内部使用的、可追溯的任务定义。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    source_task_id: str
    source_version: str
    split: TaskSplit
    status: TaskStatus

    visible_request: str # 原始 reason_for_call，仅用于审计，不直接注入 Agent 对话。
    user_scenario: UserScenarioView  # 隐藏的用户剧本，提供给 User Simulator。
    reference_actions: list[ReferenceAction] = Field(default_factory=list) # Replay 推导目标状态的参考工具动作。
    communicate_info: list[str] = Field(default_factory=list)  # 必须向用户说明的内容
    nl_assertions: list[str] = Field(default_factory=list)   # 自然语言验收条件
    reward_basis: list[RewardBasis] = Field(default_factory=list)  # 任务采用哪些奖励依据

    database_path: str
    database_sha256: str
    # 训练专用状态变体在 base database 之上声明的类型安全 patch。既有 τ² task 为空。
    initial_state_patches: list[InitialStatePatch] = Field(default_factory=list)
    # patch 完成后的稳定数据库 hash；运行与 replay 都必须校验，防止状态漂移。
    initial_state_sha256: str | None = None
    conversion_warnings: list[str] = Field(default_factory=list)
    unsupported_reasons: list[str] = Field(default_factory=list) # 不支持的原因

    # 保存完整原始任务，便于审计、复现和发现语义漂移。
    source_payload: dict[str, Any]
