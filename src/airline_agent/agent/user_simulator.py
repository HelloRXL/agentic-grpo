"""用于验证多轮协议的确定性和模型驱动 User Simulator。"""

from typing import Any
from typing import TYPE_CHECKING, Protocol

from .prompts import build_user_system_prompt


STOP_TOKEN = "###STOP###"

if TYPE_CHECKING:
    from ..tasks.spec import TaskSpec


class UserSimulator(Protocol):
    """Agent 向用户追问时需要的最小接口。"""

    def reply(self, question: str) -> str:
        ...


class ScriptedUserSimulator:
    """按顺序返回预设回复，不使用 LLM，适合测试多轮状态机。"""

    def __init__(
        self,
        replies: list[str],
        fallback_reply: str | None = None,
    ) -> None:
        self._replies = replies.copy()
        self._next_reply_index = 0
        self._fallback_reply = fallback_reply
        self.questions: list[str] = []

    @classmethod
    def from_task(cls, task: "TaskSpec") -> "ScriptedUserSimulator":
        """从隐藏剧本创建确定性用户，首次以第一人称披露已知信息。"""

        replies = []
        if task.user_scenario.known_info:
            replies.append(_render_user_reply(task.user_scenario.known_info))
        return cls(
            replies=replies,
            fallback_reply=STOP_TOKEN,
        )

    def reply(self, question: str) -> str:
        self.questions.append(question)
        if self._next_reply_index >= len(self._replies):
            if self._fallback_reply is not None:
                return self._fallback_reply
            return STOP_TOKEN
        reply = self._replies[self._next_reply_index]
        self._next_reply_index += 1
        return reply


def _render_user_reply(hidden_info: str) -> str:
    """将给模拟器的二人称提示转换成给 Agent 的第一人称用户回复。"""

    replacements = (
        ("You are ", "I am "),
        ("Your ", "My "),
        ("You have ", "I have "),
        ("You want ", "I want "),
        ("You need ", "I need "),
        ("You don't ", "I don't "),
        ("You do not ", "I do not "),
    )
    reply = hidden_info
    for source, target in replacements:
        reply = reply.replace(source, target)
    return reply


class LLMUserSimulator:
    """让独立模型根据隐藏剧本回答 Agent 的追问。"""

    def __init__(self, llm_client: Any, task: "TaskSpec") -> None:
        self._llm_client = llm_client
        self._task = task
        self._history: list[dict[str, str]] = []

    def reply(self, question: str) -> str:
        # Chat API 视角：Agent 的问题是用户模型收到的 user 消息，
        # 用户模型生成的回复必须保留为 assistant 消息。
        self._history.append({"role": "user", "content": question})
        response = self._llm_client.think(self._build_messages())
        self._history.append({"role": "assistant", "content": response})
        return response

    def _build_messages(self) -> list[dict[str, str]]:
        scenario = self._task.user_scenario
        system = build_user_system_prompt(
            persona=scenario.persona,
            reason_for_call=scenario.reason_for_call,
            known_info=scenario.known_info,
            unknown_info=scenario.unknown_info,
            task_instructions=scenario.task_instructions,
        )
        return [{"role": "system", "content": system}, *self._history]

def is_stop_reply(reply: str) -> bool:
    return reply.strip() == STOP_TOKEN
