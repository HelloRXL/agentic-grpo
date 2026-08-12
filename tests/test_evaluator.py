from pathlib import Path

from airline_agent.core.actions import AgentAction
from airline_agent.core.results import ToolResult
from airline_agent.core.rollout import RolloutRecord, RolloutStep
from airline_agent.evaluator import Evaluator
from airline_agent.tasks.converter import convert_dataset
from airline_agent.verifier import CommunicationVerificationResult


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "reference-repos" / "tau2-bench-main" / "data" / "tau2" / "domains" / "airline"


def _tasks(split: str = "base"):
    return convert_dataset(
        DATA_DIR / "tasks.json",
        DATA_DIR / "split_tasks.json",
        DATA_DIR / "db.json",
    )[split]


class _PassingCommunicationVerifier:
    def verify(self, task, rollout):
        return CommunicationVerificationResult(passed=True)


def test_evaluator_replays_reference_and_accepts_equivalent_final_state() -> None:
    task = next(task for task in _tasks() if task.source_task_id == "8")
    evaluator = Evaluator(DATA_DIR / "db.json")
    replay = evaluator.replay_reference(task)

    steps = [
        RolloutStep(
            step_index=action.step_index,
            raw_model_output="{}",
            action=AgentAction(
                action_type="tool",
                tool_name=action.tool_name,
                arguments=action.arguments,
            ),
            observation=ToolResult(
                success=True,
                tool_name=action.tool_name,
                message="ok",
            ),
        )
        for action in task.reference_actions
    ]
    rollout = RolloutRecord(
        task_id=task.task_id,
        user_request=task.visible_request,
        steps=steps,
        initial_state=replay.initial_state,
        final_state=replay.final_state,
        final_answer="The reservation has been completed.",
        termination_reason="finished",
    )

    result = evaluator.evaluate(task, rollout)

    assert result.environment_reward.success is True
    assert result.environment_reward.reward == 1.0
    assert result.sft_accepted is False
    assert result.final_state_match is True


def test_evaluator_allows_reference_free_read_only_official_success() -> None:
    task = next(task for task in _tasks("test") if task.source_task_id == "2")
    evaluator = Evaluator(
        DATA_DIR / "db.json",
        communication_verifier=_PassingCommunicationVerifier(),
    )
    replay = evaluator.replay_reference(task)
    rollout = RolloutRecord(
        task_id=task.task_id,
        user_request="",
        initial_state=replay.initial_state,
        final_state=replay.final_state,
        final_answer="I can help with that.",
        termination_reason="finished",
    )

    result = evaluator.evaluate(task, rollout)

    assert result.environment_reward.success is True
    assert result.full_task_success is True
    assert result.strict_action_success is False
    assert result.sft_accepted is True
    assert "required_tools_missing_or_out_of_order" not in result.full_task_failure_reasons
    assert "missing_reference_action:2_0" in result.action_failures


def test_evaluator_reports_reference_argument_mismatch_only_as_strict_diagnostic() -> None:
    task = next(task for task in _tasks("test") if task.source_task_id == "2")
    evaluator = Evaluator(
        DATA_DIR / "db.json",
        communication_verifier=_PassingCommunicationVerifier(),
    )
    replay = evaluator.replay_reference(task)
    first_action = task.reference_actions[0]
    rollout = RolloutRecord(
        task_id=task.task_id,
        user_request="",
        steps=[
            RolloutStep(
                step_index=1,
                raw_model_output="{}",
                action=AgentAction(
                    action_type="tool",
                    tool_name=first_action.tool_name,
                    arguments={"user_id": "wrong_user"},
                ),
                observation=ToolResult(
                    success=True,
                    tool_name=first_action.tool_name,
                    message="ok",
                ),
            )
        ],
        initial_state=replay.initial_state,
        final_state=replay.final_state,
        final_answer="Done.",
        termination_reason="finished",
    )

    result = evaluator.evaluate(task, rollout)

    assert result.full_task_success is True
    assert result.environment_reward.success is True
    assert result.strict_action_success is False
    assert result.sft_accepted is True
    assert any("arguments_mismatch" in failure for failure in result.action_failures)


def test_evaluator_enforces_actions_when_reward_basis_requires_them() -> None:
    original = next(task for task in _tasks("test") if task.source_task_id == "2")
    task = original.model_copy(update={"reward_basis": ["DB", "COMMUNICATE", "ACTION"]})
    evaluator = Evaluator(
        DATA_DIR / "db.json",
        communication_verifier=_PassingCommunicationVerifier(),
    )
    replay = evaluator.replay_reference(task)
    rollout = RolloutRecord(
        task_id=task.task_id,
        user_request="",
        initial_state=replay.initial_state,
        final_state=replay.final_state,
        final_answer="Done.",
        termination_reason="finished",
    )

    result = evaluator.evaluate(task, rollout)

    assert result.full_task_success is False
    assert result.strict_action_success is False
    assert "required_tools_missing_or_out_of_order" in result.full_task_failure_reasons


def test_evaluator_checks_refusal_for_empty_reference_actions() -> None:
    task = next(task for task in _tasks() if task.source_task_id == "0")
    evaluator = Evaluator(DATA_DIR / "db.json")
    replay = evaluator.replay_reference(task)

    rollout = RolloutRecord(
        task_id=task.task_id,
        user_request=task.visible_request,
        initial_state=replay.initial_state,
        final_state=replay.final_state,
        final_answer="I cannot proceed with this cancellation under the policy.",
        termination_reason="finished",
    )

    result = evaluator.evaluate(task, rollout)

    assert result.environment_reward.success is True
    assert result.environment_reward.reward == 1.0


def test_evaluator_rejects_parse_error_trajectory_from_sft() -> None:
    task = next(task for task in _tasks() if task.source_task_id == "0")
    evaluator = Evaluator(
        DATA_DIR / "db.json",
        communication_verifier=_PassingCommunicationVerifier(),
    )
    replay = evaluator.replay_reference(task)
    rollout = RolloutRecord(
        task_id=task.task_id,
        user_request=task.visible_request,
        steps=[
            RolloutStep(
                step_index=1,
                raw_model_output="not JSON",
                parse_error="invalid JSON",
            )
        ],
        initial_state=replay.initial_state,
        final_state=replay.final_state,
        final_answer="I cannot proceed with this cancellation under the policy.",
        termination_reason="finished",
    )

    result = evaluator.evaluate(task, rollout)

    assert result.environment_reward.success is True
    assert result.environment_reward.invalid_action_count == 1
    assert result.sft_accepted is False


def test_evaluator_keeps_non_refusal_as_judge_diagnostic_not_official_reward() -> None:
    task = next(task for task in _tasks() if task.source_task_id == "0")
    evaluator = Evaluator(DATA_DIR / "db.json")
    replay = evaluator.replay_reference(task)

    rollout = RolloutRecord(
        task_id=task.task_id,
        user_request=task.visible_request,
        initial_state=replay.initial_state,
        final_state=replay.final_state,
        final_answer="Sure, I will cancel it for you.",
        termination_reason="finished",
    )

    result = evaluator.evaluate(task, rollout)

    assert result.environment_reward.success is True
    assert result.environment_reward.reward == 1.0
