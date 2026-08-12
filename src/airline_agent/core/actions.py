from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentAction(BaseModel):
    """模型在 Agent Loop 中输出的一步动作。"""

    model_config = ConfigDict(extra="forbid")

    action_type: Literal["tool", "ask_user", "finish", "done"]
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    user_question: str | None = None
    final_answer: str | None = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> "AgentAction":
        if self.action_type == "tool":
            if not self.tool_name:
                raise ValueError("调用工具时必须提供 tool_name")
            if self.user_question is not None:
                raise ValueError("调用工具时不能提供 user_question")
            if self.final_answer is not None:
                raise ValueError("调用工具时不能提供 final_answer")

        if self.action_type == "ask_user":
            if not self.user_question:
                raise ValueError("询问用户时必须提供 user_question")
            if self.tool_name is not None:
                raise ValueError("询问用户时 tool_name 必须为 null")
            if self.arguments:
                raise ValueError("询问用户时 arguments 必须为空")
            if self.final_answer is not None:
                raise ValueError("询问用户时不能提供 final_answer")

        if self.action_type == "finish":
            if not self.final_answer:
                raise ValueError("回复用户时必须提供 final_answer")
            if self.tool_name is not None:
                raise ValueError("结束任务时 tool_name 必须为 null")
            if self.arguments:
                raise ValueError("结束任务时 arguments 必须为空")
            if self.user_question is not None:
                raise ValueError("回复用户时不能提供 user_question")

        if self.action_type == "done":
            if self.tool_name is not None or self.arguments:
                raise ValueError("结束任务时不能提供工具调用")
            if self.user_question is not None or self.final_answer is not None:
                raise ValueError("结束任务时不能提供用户消息")

        return self
