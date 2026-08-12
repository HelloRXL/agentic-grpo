"""对 Airline Agent rollout 进行 Replay、验收和 reward 计算。"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .core.rollout import RolloutRecord
from .domain.runtime import create_airline_runtime
from .tasks.spec import TaskSpec
from .environment_reward import EnvironmentRewardResult, compute_environment_reward
from .verifier import (
    ActionVerifier,
    CommunicationVerificationResult,
    CommunicationVerifier,
)


class ReplayResult(BaseModel):
    """参考动作在干净环境中的重放结果。"""

    success: bool
    initial_state: dict[str, Any]
    final_state: dict[str, Any]
    failure_reason: str | None = None


class EvaluationResult(BaseModel):
    """统一环境奖励、通信审计与严格路径诊断。"""

    task_id: str
    environment_reward: EnvironmentRewardResult
    sft_accepted: bool
    judge_pass: bool
    full_task_success: bool
    strict_action_success: bool
    replay_success: bool
    initial_state_match: bool
    final_state_match: bool
    termination_ok: bool
    action_ok: bool
    actual_tools: list[str] = Field(default_factory=list)
    successful_tools: list[str] = Field(default_factory=list)
    action_failures: list[str] = Field(default_factory=list)
    communication_verification: CommunicationVerificationResult | None = None
    full_task_failure_reasons: list[str] = Field(default_factory=list)


class Evaluator:
    """在不修改 Agent rollout 的前提下生成目标状态和 reward。"""

    def __init__(
        self,
        database_path: Path,
        communication_verifier: CommunicationVerifier | None = None,
        reward_mode: str = "terminal_v4",
    ) -> None:
        self._database_path = database_path
        self._communication_verifier = communication_verifier
        self._reward_mode = reward_mode

    def replay_reference(self, task: TaskSpec) -> ReplayResult:
        """在全新环境中执行参考动作，得到目标最终状态。"""

        runtime = create_airline_runtime(
            self._database_path,
            initial_state_patches=task.initial_state_patches,
            expected_initial_state_sha256=task.initial_state_sha256,
        )
        initial_state = runtime.environment.snapshot().model_dump(mode="json")

        if task.status == "unsupported":
            return ReplayResult(
                success=False,
                initial_state=initial_state,
                final_state=initial_state,
                failure_reason="task_unsupported",
            )

        for action in task.reference_actions:
            result = runtime.executor.execute(action.tool_name, action.arguments)
            if not result.success:
                return ReplayResult(
                    success=False,
                    initial_state=initial_state,
                    final_state=runtime.environment.snapshot().model_dump(mode="json"),
                    failure_reason=(
                        f"reference_action_failed:{action.step_index}:"
                        f"{result.error}"
                    ),
                )

        return ReplayResult(
            success=True,
            initial_state=initial_state,
            final_state=runtime.environment.snapshot().model_dump(mode="json"),
        )

    def evaluate(self, task: TaskSpec, rollout: RolloutRecord) -> EvaluationResult:
        replay = self.replay_reference(task)
        actual_tools: list[str] = []
        successful_tools: list[str] = []
        for step in rollout.steps:
            if step.action is None or step.action.action_type != "tool":
                continue
            if step.action.tool_name is None:
                continue
            actual_tools.append(step.action.tool_name)
            if step.observation is not None and step.observation.success:
                successful_tools.append(step.action.tool_name)

        task_id_match = rollout.task_id == task.task_id
        initial_state_match = rollout.initial_state == replay.initial_state
        termination_ok = rollout.termination_reason == "finished"

        db_required = "DB" in task.reward_basis
        # 只有 reward_basis 明确包含 ACTION 时，参考动作才是完整任务成功条件。
        official_action_required = "ACTION" in task.reward_basis
        # 严格诊断额外检查参考动作，帮助分析工具选择和参数错误。
        strict_action_required = bool(task.reference_actions) or official_action_required
        judge_required = bool(task.nl_assertions)

        final_state_match = rollout.final_state == replay.final_state
        db_ok = not db_required or final_state_match

        action_verification = ActionVerifier().verify(
            task.reference_actions,
            rollout.steps,
        )
        action_ok = action_verification.passed

        judge_pass = True
        communication_verification: CommunicationVerificationResult | None = None
        if judge_required:
            if self._communication_verifier is None:
                judge_pass = bool(rollout.final_answer and rollout.final_answer.strip())
            else:
                communication_verification = self._communication_verifier.verify(
                    task,
                    rollout,
                )
                judge_pass = communication_verification.passed

        failure_reasons: list[str] = []
        if not replay.success:
            failure_reasons.append(replay.failure_reason or "replay_failed")
        if not task_id_match:
            failure_reasons.append("task_id_mismatch")
        if not initial_state_match:
            failure_reasons.append("initial_state_mismatch")
        if not termination_ok:
            failure_reasons.append("not_finished")
        if db_required and not final_state_match:
            failure_reasons.append("final_state_mismatch")
        if official_action_required and not action_ok:
            failure_reasons.append("required_tools_missing_or_out_of_order")
        environment_reward = compute_environment_reward(
            task,
            rollout,
            replay_success=replay.success,
            initial_state_match=initial_state_match,
            final_state_match=final_state_match,
            reward_mode=self._reward_mode,
        )
        if not replay.success and replay.failure_reason:
            # 环境 reward 保持 0，但把具体参考动作/Schema 失败原因带到 veRL 日志，
            # 避免只能看到笼统的 ``replay_failed``。
            environment_reward.reasons.append(replay.failure_reason)
        if environment_reward.missing_communicate_info:
            failure_reasons.append("required_communication_missing")
        full_task_success = (
            task_id_match
            and db_ok
            and environment_reward.success
            and (not official_action_required or action_ok)
        )
        strict_action_success = full_task_success and (
            not strict_action_required or action_ok
        )
        clean_sft_process = (
            environment_reward.invalid_action_count == 0
            and environment_reward.repeated_action_count == 0
        )
        sft_accepted = (
            environment_reward.success
            and clean_sft_process
            # 参考动作是诊断基准，不是只读/状态任务的唯一正确路径。只有
            # τ² 的 reward_basis 显式要求 ACTION 时，才把它作为 SFT 硬约束。
            and (not official_action_required or action_verification.passed)
            and communication_verification is not None
            and communication_verification.passed
        )

        return EvaluationResult(
            task_id=task.task_id,
            environment_reward=environment_reward,
            sft_accepted=sft_accepted,
            judge_pass=judge_pass,
            full_task_success=full_task_success,
            strict_action_success=strict_action_success,
            replay_success=replay.success,
            initial_state_match=initial_state_match,
            final_state_match=final_state_match,
            termination_ok=termination_ok,
            action_ok=action_ok,
            actual_tools=actual_tools,
            successful_tools=successful_tools,
            action_failures=action_verification.failures,
            communication_verification=communication_verification,
            full_task_failure_reasons=failure_reasons,
        )
