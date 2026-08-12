import json
from pathlib import Path

import pytest

from airline_agent.agent import run_task
from airline_agent.core.llm_client import FakeLLMClient
from airline_agent.domain.runtime import create_airline_runtime
from airline_agent.evaluator import Evaluator
from airline_agent.tasks.v9_state_variants import (
    build_v9_a_tasks,
    build_v9_b_tasks,
    build_v9_c_tasks,
)
from airline_agent.tasks.v8_variants import load_tasks


ROOT = Path(__file__).resolve().parents[1]
TRAIN_TASKS = ROOT / "data/tasks/train.jsonl"
DATABASE = ROOT / "../reference-repos/tau2-bench-main/data/tau2/domains/airline/db.json"


def _action(action_type: str, **fields: object) -> str:
    payload = {
        "action_type": action_type,
        "tool_name": None,
        "arguments": {},
        "user_question": None,
        "final_answer": None,
        **fields,
    }
    return json.dumps(payload)


def test_v9_a_builds_a_real_state_counterfactual_pair() -> None:
    tasks = build_v9_a_tasks(load_tasks(TRAIN_TASKS), DATABASE)
    uninsured, insured = tasks

    assert uninsured.initial_state_sha256 != insured.initial_state_sha256
    assert uninsured.initial_state_patches == []
    assert insured.initial_state_patches[0].insurance == "yes"
    assert uninsured.reference_actions == []
    assert insured.reference_actions[0].tool_name == "cancel_reservation"

    evaluator = Evaluator(DATABASE)
    uninsured_replay = evaluator.replay_reference(uninsured)
    insured_replay = evaluator.replay_reference(insured)
    assert uninsured_replay.success and insured_replay.success
    assert uninsured_replay.initial_state == uninsured_replay.final_state
    assert insured_replay.initial_state != insured_replay.final_state
    assert insured_replay.final_state["reservations"]["EHGLP3"]["status"] == "cancelled"


def test_v9_a_agent_and_evaluator_share_the_patched_runtime() -> None:
    insured = build_v9_a_tasks(load_tasks(TRAIN_TASKS), DATABASE)[1]
    greeting = _action("ask_user", user_question="Hello, how can I help?")
    cancel = _action(
        "tool",
        tool_name="cancel_reservation",
        arguments={"reservation_id": "EHGLP3"},
    )
    finish = _action("finish", final_answer="Your reservation has been cancelled.")

    result = run_task(
        insured,
        database_path=DATABASE,
        llm_client=FakeLLMClient([greeting, cancel, finish]),
    )

    assert result.evaluation.final_state_match is True
    assert result.evaluation.environment_reward.success is True
    assert result.rollout.initial_state["reservations"]["EHGLP3"]["insurance"] == "yes"


def test_runtime_rejects_a_declared_initial_state_hash_that_does_not_match() -> None:
    task = build_v9_a_tasks(load_tasks(TRAIN_TASKS), DATABASE)[1]
    with pytest.raises(ValueError, match="initial_state_sha256"):
        create_airline_runtime(
            DATABASE,
            initial_state_patches=task.initial_state_patches,
            expected_initial_state_sha256="not-the-patched-state",
        )


def test_v9_b_cabin_patch_flips_a_date_change_target() -> None:
    economy, basic_economy = build_v9_b_tasks(load_tasks(TRAIN_TASKS), DATABASE)
    parent = next(task for task in load_tasks(TRAIN_TASKS) if task.source_task_id == "33")
    evaluator = Evaluator(DATABASE)
    economy_replay = evaluator.replay_reference(economy)
    basic_replay = evaluator.replay_reference(basic_economy)

    assert economy.initial_state_sha256 != basic_economy.initial_state_sha256
    assert economy.visible_request == parent.visible_request
    assert economy.user_scenario == parent.user_scenario
    assert economy.reference_actions == parent.reference_actions
    assert economy.initial_state_patches == []
    assert basic_economy.initial_state_patches[0].cabin == "basic_economy"
    assert economy_replay.success and basic_replay.success
    assert economy_replay.initial_state != economy_replay.final_state
    assert basic_replay.initial_state == basic_replay.final_state
    assert economy_replay.final_state["reservations"]["HXDUBJ"]["cabin"] == "economy"
    assert basic_replay.initial_state["reservations"]["HXDUBJ"]["cabin"] == "basic_economy"


def test_v9_c_inherits_the_one_way_parent_scenario() -> None:
    business, basic_economy = build_v9_c_tasks(load_tasks(TRAIN_TASKS), DATABASE)
    parent = next(task for task in load_tasks(TRAIN_TASKS) if task.source_task_id == "15")
    evaluator = Evaluator(DATABASE)

    assert business.visible_request == parent.visible_request
    assert business.user_scenario == parent.user_scenario
    assert business.reference_actions == parent.reference_actions
    assert business.initial_state_sha256 != basic_economy.initial_state_sha256
    assert evaluator.replay_reference(business).success
    assert evaluator.replay_reference(basic_economy).success
