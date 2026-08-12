"""驱动 LLM、工具执行和 rollout 记录的最小 Agent Loop。"""

from copy import deepcopy
import json
from typing import Any, Callable

from pydantic import ValidationError

from ..core.actions import AgentAction
from ..core.executor import ToolExecutor
from ..core.llm_client import LLMClient
from ..core.registry import ToolRegistry
from ..core.rollout import RolloutRecord, RolloutStep, TerminationReason
from ..core.results import ToolResult
from ..domain.environment import AirlineEnvironment
from .prompts import build_agent_system_prompt
from .user_simulator import UserSimulator, is_stop_reply


class AgentLoop:
    """每轮让模型产生一个动作，并通过 Executor 与环境交互。"""

    def __init__(
        self,
        llm_client: LLMClient,
        environment: AirlineEnvironment,
        registry: ToolRegistry,
        executor: ToolExecutor,
        user_simulator: UserSimulator | None = None,
        max_steps: int = 15,
        event_handler: Callable[[str, dict[str, Any]], None] | None = None,
        observation_formatter: Callable[[ToolResult], str] | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._environment = environment
        self._registry = registry
        self._executor = executor
        self._user_simulator = user_simulator
        self._max_steps = max_steps
        self._event_handler = event_handler
        self._observation_formatter = observation_formatter

    def _emit(self, event: str, **payload: Any) -> None:
        """发送可选的实时展示事件，不参与 Agent 决策。"""

        if self._event_handler is not None:
            self._event_handler(event, payload)

    def _build_system_prompt(self) -> str:
        return build_agent_system_prompt(
            self._registry.get_tool_definitions()
        )

    def _snapshot(self) -> dict:
        return self._environment.snapshot().model_dump(mode="json")

    @staticmethod
    def _action_format_hint(raw_output: str) -> str:
        """根据原始 JSON 中的字段，给出对应的 Action 格式示例。"""

        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            payload = {}

        action_type = payload.get("action_type") if isinstance(payload, dict) else None
        if action_type == "tool" or (
            isinstance(payload, dict) and payload.get("tool_name") is not None
        ):
            return (
                'For a tool call use: '
                '{"action_type":"tool","tool_name":"<tool name>",'
                '"arguments":{},"final_answer":null}'
            )
        if action_type == "ask_user" or (
            isinstance(payload, dict) and payload.get("user_question") is not None
        ):
            return (
                'To ask the user use: '
                '{"action_type":"ask_user","tool_name":null,"arguments":{},'
                '"user_question":"<question>","final_answer":null}'
            )
        if action_type == "finish" or (
            isinstance(payload, dict) and payload.get("final_answer") is not None
        ):
            return (
                'To respond to the user use: '
                '{"action_type":"finish","tool_name":null,"arguments":{},'
                '"user_question":null,"final_answer":"<answer>"}'
            )
        return (
            'Valid action_type values are "tool", "ask_user", "finish", and "done". '
            'Choose the matching JSON format and return exactly one object.'
        )

    @staticmethod
    def _exception_detail(error: Exception) -> str:
        """保留空消息异常的类型，避免 rollout 只留下无信息的失败文本。"""

        message = str(error).strip()
        return f"{type(error).__name__}: {message}" if message else type(error).__name__

    def _finalize(
        self,
        rollout: RolloutRecord,
        messages: list[dict],
        termination_reason: TerminationReason,
    ) -> RolloutRecord:
        rollout.messages = deepcopy(messages)
        rollout.final_state = self._snapshot()
        rollout.termination_reason = termination_reason
        return rollout

    def run(
        self,
        task_id: str,
        initial_user_message: str | None = None,
    ) -> RolloutRecord:
        """Run one τ²-style conversation.

        A normal task starts with an empty conversation: the Agent greets first,
        and the User Simulator then reveals information from its hidden scenario.
        ``initial_user_message`` is kept for replaying a pre-existing conversation.
        """

        self._environment.reset()
        self._emit(
            "task_started",
            task_id=task_id,
            user_request=initial_user_message
            or "τ² 半双工模式：Agent 先问候，User Simulator 再回复",
        )
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
        ]
        if initial_user_message:
            messages.append({"role": "user", "content": initial_user_message})
        rollout = RolloutRecord(
            task_id=task_id,
            user_request=initial_user_message or "",
            initial_state=self._snapshot(),
        )
        has_received_user_message = initial_user_message is not None
        state_version = 0
        seen_tool_calls: set[tuple[int, str]] = set()

        for step_index in range(1, self._max_steps + 1):
            try:
                raw_output = self._llm_client.think(messages=messages)
            except Exception as error:
                detail = self._exception_detail(error)
                self._emit("llm_error", step_index=step_index, error=detail)
                rollout.steps.append(
                    RolloutStep(
                        step_index=step_index,
                        raw_model_output="",
                        parse_error=f"LLM 调用失败：{detail}",
                    )
                )
                return self._finalize(rollout, messages, "llm_error")

            messages.append({"role": "assistant", "content": raw_output})
            try:
                action = AgentAction.model_validate_json(raw_output)
            except ValidationError as error:
                self._emit(
                    "parse_error",
                    step_index=step_index,
                    raw_output=raw_output,
                    error=str(error),
                )
                rollout.steps.append(
                    RolloutStep(
                        step_index=step_index,
                        raw_model_output=raw_output,
                        parse_error=str(error),
                    )
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous Action JSON failed schema validation:\n"
                            f"{error}\n\n"
                            "Return exactly one corrected JSON object.\n"
                            f"{self._action_format_hint(raw_output)}"
                        ),
                    }
                )
                continue

            if not has_received_user_message and action.action_type != "ask_user":
                protocol_error = (
                    "Initial-turn protocol violation: no user message has been received. "
                    "The first action must be ask_user with a greeting in user_question; "
                    "do not use tool or finish."
                )
                self._emit(
                    "parse_error",
                    step_index=step_index,
                    raw_output=raw_output,
                    error=protocol_error,
                )
                rollout.steps.append(
                    RolloutStep(
                        step_index=step_index,
                        raw_model_output=raw_output,
                        action=action,
                        parse_error=protocol_error,
                    )
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"{protocol_error}\n"
                            "Return exactly one corrected ask_user JSON object."
                        ),
                    }
                )
                continue

            if action.action_type == "done":
                self._emit(
                    "agent_action",
                    step_index=step_index,
                    action=action.model_dump(mode="json"),
                )
                self._emit("finished", step_index=step_index, answer=rollout.final_answer)
                rollout.steps.append(
                    RolloutStep(
                        step_index=step_index,
                        raw_model_output=raw_output,
                        action=action,
                    )
                )
                return self._finalize(rollout, messages, "finished")

            if action.action_type == "finish":
                self._emit(
                    "agent_action",
                    step_index=step_index,
                    action=action.model_dump(mode="json"),
                )
                rollout.final_answer = action.final_answer
                if self._user_simulator is None:
                    self._emit("finished", step_index=step_index, answer=action.final_answer)
                    rollout.steps.append(
                        RolloutStep(step_index=step_index, raw_model_output=raw_output, action=action)
                    )
                    return self._finalize(rollout, messages, "finished")
                try:
                    user_reply = self._user_simulator.reply(action.final_answer or "")
                except Exception as error:
                    detail = self._exception_detail(error)
                    self._emit("user_error", step_index=step_index, error=detail)
                    rollout.steps.append(RolloutStep(step_index=step_index, raw_model_output=raw_output, action=action, parse_error=f"User Simulator 调用失败：{detail}"))
                    return self._finalize(rollout, messages, "llm_error")
                rollout.steps.append(RolloutStep(step_index=step_index, raw_model_output=raw_output, action=action, user_reply=user_reply))
                self._emit("user_reply", step_index=step_index, reply=user_reply)
                if is_stop_reply(user_reply):
                    self._emit("finished", step_index=step_index, answer=action.final_answer)
                    return self._finalize(rollout, messages, "finished")
                messages.append({"role": "user", "content": user_reply})
                has_received_user_message = True
                continue

            if action.action_type == "ask_user":
                self._emit(
                    "agent_action",
                    step_index=step_index,
                    action=action.model_dump(mode="json"),
                )
                if self._user_simulator is None:
                    rollout.steps.append(
                        RolloutStep(
                            step_index=step_index,
                            raw_model_output=raw_output,
                            action=action,
                            parse_error="当前 AgentLoop 未配置 User Simulator",
                        )
                    )
                    return self._finalize(rollout, messages, "llm_error")

                try:
                    user_reply = self._user_simulator.reply(action.user_question or "")
                except Exception as error:
                    detail = self._exception_detail(error)
                    self._emit("user_error", step_index=step_index, error=detail)
                    rollout.steps.append(
                        RolloutStep(
                            step_index=step_index,
                            raw_model_output=raw_output,
                            action=action,
                            parse_error=f"User Simulator 调用失败：{detail}",
                        )
                    )
                    return self._finalize(rollout, messages, "llm_error")

                rollout.steps.append(
                    RolloutStep(
                        step_index=step_index,
                        raw_model_output=raw_output,
                        action=action,
                        user_reply=user_reply,
                    )
                )
                self._emit("user_reply", step_index=step_index, reply=user_reply)
                messages.append({"role": "user", "content": user_reply})
                has_received_user_message = True
                continue

            self._emit(
                "agent_action",
                step_index=step_index,
                action=action.model_dump(mode="json"),
            )
            call_signature = json.dumps(
                {
                    "tool_name": action.tool_name,
                    "arguments": action.arguments,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            call_key = (state_version, call_signature)
            if call_key in seen_tool_calls:
                observation = ToolResult(
                    success=False,
                    tool_name=action.tool_name,
                    error="duplicate_tool_call",
                    message=(
                        "This exact tool call has already been executed without an intervening "
                        "state change. Use the previous observation and choose a different action."
                    ),
                )
            else:
                seen_tool_calls.add(call_key)
                observation = self._executor.execute(
                    tool_name=action.tool_name,
                    raw_arguments=action.arguments,
                )
                tool_spec = self._registry.get_tool(action.tool_name or "")
                if observation.success and tool_spec is not None and tool_spec.mutates_state:
                    state_version += 1
            rollout.steps.append(
                RolloutStep(
                    step_index=step_index,
                    raw_model_output=raw_output,
                    action=action,
                    observation=observation,
                )
            )
            self._emit(
                "tool_result",
                step_index=step_index,
                observation=observation.model_dump(mode="json"),
            )
            observation_text = (
                self._observation_formatter(observation)
                if self._observation_formatter is not None
                else (
                    "Tool execution observation:\n"
                    f"{observation.model_dump_json(indent=2)}\n"
                    "Decide the next action."
                )
            )
            messages.append({"role": "user", "content": observation_text})

        self._emit("max_steps", step_index=self._max_steps)
        return self._finalize(rollout, messages, "max_steps")
