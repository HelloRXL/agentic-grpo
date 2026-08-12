"""对 Agent 工具轨迹和自然语言结果进行校验。"""

import json
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .core.rollout import RolloutStep
from .core.llm_client import LLMClient
from .core.rollout import RolloutRecord
from .tasks.spec import ReferenceAction, TaskSpec


class ActionVerificationResult(BaseModel):
    """参考动作校验结果，供 Evaluator 和后续数据过滤使用。"""

    passed: bool
    matched_count: int = 0
    failures: list[str] = Field(default_factory=list)


AssertionStatus = Literal["satisfied", "violated", "unknown"]


class AssertionVerification(BaseModel):
    """LLM Judge 对单条自然语言断言的判断。"""

    model_config = ConfigDict(extra="forbid")

    assertion_id: int = Field(ge=0)
    status: AssertionStatus
    evidence_event_ids: list[str] = Field(default_factory=list)
    rationale: str


class CommunicationVerificationResult(BaseModel):
    """通信 Rubric 的结构化评测结果。"""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    assertions: list[AssertionVerification] = Field(default_factory=list)
    error: str | None = None


class CommunicationVerifier(Protocol):
    def verify(
        self,
        task: TaskSpec,
        rollout: RolloutRecord,
    ) -> CommunicationVerificationResult:
        ...


class ActionVerifier:
    """检查参考工具动作是否按顺序、按参数成功执行。"""

    def verify(
        self,
        reference_actions: list[ReferenceAction],
        rollout_steps: list[RolloutStep],
    ) -> ActionVerificationResult:
        expected_index = 0
        failures: list[str] = []
        expected_names = [action.tool_name for action in reference_actions]

        for step in rollout_steps:
            action = step.action
            if action is None or action.action_type != "tool" or not action.tool_name:
                continue

            if step.observation is None or not step.observation.success:
                if action.tool_name in expected_names:
                    failures.append(
                        f"step_{step.step_index}:tool_failed:{action.tool_name}"
                    )
                continue

            if expected_index >= len(reference_actions):
                continue

            expected = reference_actions[expected_index]
            if action.tool_name == expected.tool_name:
                if action.arguments != expected.arguments:
                    failures.append(
                        f"step_{step.step_index}:arguments_mismatch:"
                        f"expected={expected.arguments}:actual={action.arguments}"
                    )
                else:
                    expected_index += 1
                continue

            remaining_names = expected_names[expected_index + 1 :]
            if action.tool_name in remaining_names:
                failures.append(
                    f"step_{step.step_index}:out_of_order:"
                    f"expected={expected.tool_name}:actual={action.tool_name}"
                )

        for missing in reference_actions[expected_index:]:
            failures.append(f"missing_reference_action:{missing.action_id}")

        return ActionVerificationResult(
            passed=not failures and expected_index == len(reference_actions),
            matched_count=expected_index,
            failures=failures,
        )


class LLMCommunicationVerifier:
    """用独立 LLM 按冻结的自然语言断言评估通信质量。"""

    def __init__(self, llm_client: LLMClient, max_attempts: int = 2) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._llm_client = llm_client
        self._max_attempts = max_attempts

    def verify(
        self,
        task: TaskSpec,
        rollout: RolloutRecord,
    ) -> CommunicationVerificationResult:
        if not task.nl_assertions:
            return CommunicationVerificationResult(passed=True)

        messages = self._build_messages(task, rollout)
        last_error = "unknown judge error"
        for attempt in range(self._max_attempts):
            raw_output = ""
            try:
                raw_output = self._llm_client.think(messages)
                decision = self._parse_decision(raw_output)
                checked = self._validate_decision(task, rollout, decision)
                if checked.error is None:
                    return checked
                last_error = checked.error
            except Exception as error:
                last_error = f"communication_verifier_error:{error}"

            if attempt + 1 < self._max_attempts:
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw_output[-4000:]},
                    {
                        "role": "user",
                        "content": (
                            "Your previous evaluator output was invalid. "
                            f"Validation error: {last_error}. "
                            "Return one complete JSON object matching the requested schema. "
                            "Do not use Markdown and do not omit any assertion."
                        ),
                    },
                ]

        return CommunicationVerificationResult(passed=False, error=last_error)

    @staticmethod
    def _parse_decision(raw_output: str) -> CommunicationVerificationResult:
        """接受纯 JSON，并恢复常见的 Markdown JSON 外壳。"""

        text = raw_output.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline >= 0:
                text = text[first_newline + 1 :]
            if text.endswith("```"):
                text = text[:-3].rstrip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end < start:
                raise
            payload = json.loads(text[start : end + 1])
        return CommunicationVerificationResult.model_validate(payload)

    def _validate_decision(
        self,
        task: TaskSpec,
        rollout: RolloutRecord,
        decision: CommunicationVerificationResult,
    ) -> CommunicationVerificationResult:

        expected_ids = set(range(len(task.nl_assertions)))
        actual_ids = {item.assertion_id for item in decision.assertions}
        if actual_ids != expected_ids or len(decision.assertions) != len(expected_ids):
            return CommunicationVerificationResult(
                passed=False,
                assertions=decision.assertions,
                error=(
                    "communication_verifier_error:assertion ids must cover "
                    "all rubric items exactly once"
                ),
            )

        valid_event_ids = {
            event["event_id"] for event in self._trajectory_events(rollout)
        }
        evidence_error = self._validate_evidence(
            decision.assertions,
            valid_event_ids,
        )
        if evidence_error is not None:
            return CommunicationVerificationResult(
                passed=False,
                assertions=decision.assertions,
                error=f"communication_verifier_error:{evidence_error}",
            )

        passed = all(item.status == "satisfied" for item in decision.assertions)
        return decision.model_copy(update={"passed": passed, "error": None})

    @staticmethod
    def _build_messages(
        task: TaskSpec,
        rollout: RolloutRecord,
    ) -> list[dict[str, str]]:
        rubric = [
            {"assertion_id": index, "text": assertion}
            for index, assertion in enumerate(task.nl_assertions)
        ]
        events = LLMCommunicationVerifier._trajectory_events(rollout)

        system = (
            "You are a strict evaluator for an airline Agent trajectory. "
            "Evaluate only the rubric items using evidence in the trajectory. "
            "Do not infer hidden facts, reference actions, database state, or reward. "
            "Use status satisfied, violated, or unknown. "
            "Every assertion must cite one or more existing event IDs. "
            "A satisfied or violated assertion without evidence is invalid. "
            "Every assertion must cite event IDs when evidence exists. "
            "Cover every rubric assertion exactly once and use only the schema keys. "
            "Return exactly one JSON object and no Markdown."
        )
        user = json.dumps(
            {
                "task_id": task.task_id,
                "rubric": rubric,
                "trajectory_events": events,
                "final_answer": rollout.final_answer,
                "output_schema": {
                    "passed": "boolean (the runtime recomputes this)",
                    "assertions": [
                        {
                            "assertion_id": 0,
                            "status": "satisfied|violated|unknown",
                            "evidence_event_ids": ["step_1_agent"],
                            "rationale": "short evidence-based explanation",
                        }
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _trajectory_events(rollout: RolloutRecord) -> list[dict[str, Any]]:
        """将 rollout 转成 Judge 可引用的稳定事件 ID。"""

        events: list[dict[str, Any]] = []
        for step in rollout.steps:
            if step.action is not None:
                events.append(
                    {
                        "event_id": f"step_{step.step_index}_agent",
                        "type": "agent_action",
                        "content": step.action.model_dump(mode="json"),
                    }
                )
            if step.user_reply is not None:
                events.append(
                    {
                        "event_id": f"step_{step.step_index}_user",
                        "type": "user_message",
                        "content": step.user_reply,
                    }
                )
            if step.observation is not None:
                events.append(
                    {
                        "event_id": f"step_{step.step_index}_observation",
                        "type": "tool_observation",
                        "content": step.observation.model_dump(mode="json"),
                    }
                )
            if step.parse_error is not None:
                events.append(
                    {
                        "event_id": f"step_{step.step_index}_error",
                        "type": "runtime_error",
                        "content": step.parse_error,
                    }
                )
        if rollout.final_answer is not None:
            events.append(
                {
                    "event_id": "final_answer",
                    "type": "final_answer",
                    "content": rollout.final_answer,
                }
            )
        return events

    @staticmethod
    def _validate_evidence(
        assertions: list[AssertionVerification],
        valid_event_ids: set[str],
    ) -> str | None:
        for item in assertions:
            evidence_ids = item.evidence_event_ids
            if not evidence_ids:
                return f"assertion_{item.assertion_id}_missing_evidence"
            unknown_ids = sorted(set(evidence_ids) - valid_event_ids)
            if unknown_ids:
                return (
                    f"assertion_{item.assertion_id}_unknown_evidence_ids:"
                    f"{unknown_ids}"
                )
        return None
