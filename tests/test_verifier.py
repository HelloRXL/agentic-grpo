import json
from pathlib import Path

from airline_agent.core.llm_client import FakeLLMClient
from airline_agent.core.rollout import RolloutRecord
from airline_agent.evaluator import Evaluator
from airline_agent.tasks.converter import convert_dataset
from airline_agent.verifier import (
    CommunicationVerificationResult,
    LLMCommunicationVerifier,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (
    ROOT
    / "reference-repos"
    / "tau2-bench-main"
    / "data"
    / "tau2"
    / "domains"
    / "airline"
)


def _refusal_task():
    return next(
        task
        for task in convert_dataset(
            DATA_DIR / "tasks.json",
            DATA_DIR / "split_tasks.json",
            DATA_DIR / "db.json",
        )["base"]
        if task.source_task_id == "0"
    )


def test_llm_communication_verifier_returns_structured_rubric_result() -> None:
    task = _refusal_task()
    response = json.dumps(
        {
            "passed": False,
            "assertions": [
                {
                    "assertion_id": 0,
                    "status": "satisfied",
                    "evidence_event_ids": ["final_answer"],
                    "rationale": "The final answer refuses the cancellation.",
                }
            ],
        }
    )
    client = FakeLLMClient([response])
    verifier = LLMCommunicationVerifier(client)
    rollout = RolloutRecord(
        task_id=task.task_id,
        user_request="",
        final_answer="I cannot proceed with this cancellation.",
        termination_reason="finished",
    )

    result = verifier.verify(task, rollout)

    assert result.passed is True
    assert result.assertions[0].status == "satisfied"
    assert len(client.calls) == 1
    user_prompt = client.calls[0][1]["content"]
    assert "reference_actions" not in user_prompt
    assert "source_payload" not in user_prompt


def test_evaluator_uses_communication_verifier_for_reward() -> None:
    task = _refusal_task()
    replay = Evaluator(DATA_DIR / "db.json").replay_reference(task)
    rollout = RolloutRecord(
        task_id=task.task_id,
        user_request="",
        initial_state=replay.initial_state,
        final_state=replay.final_state,
        final_answer="I cannot proceed with this cancellation.",
        termination_reason="finished",
    )

    class FailingVerifier:
        def verify(self, task, rollout):
            return CommunicationVerificationResult(
                passed=False,
                error="judge_rejected_communication",
            )

    result = Evaluator(
        DATA_DIR / "db.json",
        communication_verifier=FailingVerifier(),
    ).evaluate(task, rollout)

    assert result.environment_reward.success is True
    assert result.full_task_success is True
        # LLM Judge 只影响语义审计与 SFT 过滤，不进入 τ2 官方成功或训练奖励。
    assert result.environment_reward.reward == 1.0
    assert result.sft_accepted is False
    assert result.judge_pass is False
    assert result.communication_verification.error == "judge_rejected_communication"


def test_llm_communication_verifier_rejects_duplicate_assertion_ids() -> None:
    task = _refusal_task()
    response = json.dumps(
        {
            "passed": True,
            "assertions": [
                {
                    "assertion_id": 0,
                    "status": "satisfied",
                    "evidence_event_ids": [],
                    "rationale": "First result.",
                },
                {
                    "assertion_id": 0,
                    "status": "satisfied",
                    "evidence_event_ids": [],
                    "rationale": "Duplicate result.",
                },
            ],
        }
    )
    result = LLMCommunicationVerifier(
        FakeLLMClient([response]), max_attempts=1
    ).verify(
        task,
        RolloutRecord(task_id=task.task_id, user_request=""),
    )

    assert result.passed is False
    assert result.error is not None
    assert "all rubric items exactly once" in result.error


def test_llm_communication_verifier_rejects_unknown_evidence_event() -> None:
    task = _refusal_task()
    response = json.dumps(
        {
            "passed": True,
            "assertions": [
                {
                    "assertion_id": 0,
                    "status": "satisfied",
                    "evidence_event_ids": ["step_99_agent"],
                    "rationale": "The trajectory proves the refusal.",
                }
            ],
        }
    )
    result = LLMCommunicationVerifier(
        FakeLLMClient([response]), max_attempts=1
    ).verify(
        task,
        RolloutRecord(task_id=task.task_id, user_request=""),
    )

    assert result.passed is False
    assert result.error is not None
    assert "unknown_evidence_ids" in result.error


def test_llm_communication_verifier_accepts_json_code_fence() -> None:
    task = _refusal_task()
    response = """```json
{"passed":false,"assertions":[{"assertion_id":0,"status":"satisfied","evidence_event_ids":["final_answer"],"rationale":"Refusal is explicit."}]}
```"""
    rollout = RolloutRecord(
        task_id=task.task_id,
        user_request="",
        final_answer="I cannot cancel this reservation.",
        termination_reason="finished",
    )

    result = LLMCommunicationVerifier(FakeLLMClient([response])).verify(task, rollout)

    assert result.passed is True


def test_llm_communication_verifier_retries_truncated_json() -> None:
    task = _refusal_task()
    valid = json.dumps(
        {
            "passed": False,
            "assertions": [
                {
                    "assertion_id": 0,
                    "status": "satisfied",
                    "evidence_event_ids": ["final_answer"],
                    "rationale": "Refusal is explicit.",
                }
            ],
        }
    )
    client = FakeLLMClient(['{"passed": false, "assertions": [', valid])
    rollout = RolloutRecord(
        task_id=task.task_id,
        user_request="",
        final_answer="I cannot cancel this reservation.",
        termination_reason="finished",
    )

    result = LLMCommunicationVerifier(client).verify(task, rollout)

    assert result.passed is True
    assert len(client.calls) == 2
    assert "previous evaluator output was invalid" in client.calls[1][-1]["content"]
