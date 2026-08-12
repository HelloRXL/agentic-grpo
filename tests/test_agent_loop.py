import json
from pathlib import Path
import re

from src.airline_agent.agent import AgentLoop, ScriptedUserSimulator
from src.airline_agent.core.llm_client import FakeLLMClient
from src.airline_agent.domain.runtime import create_airline_runtime
from src.airline_agent.evaluator import Evaluator
from src.airline_agent.tasks.converter import convert_dataset, file_sha256
from src.airline_agent.tasks.spec import ReferenceAction, TaskSpec, UserScenarioView


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "reference-repos" / "tau2-bench-main" / "data" / "tau2" / "domains" / "airline"


def test_clean_agent_loop_executes_tool_and_evaluator_scores_rollout() -> None:
    runtime = create_airline_runtime(DATA_DIR / "db.json")
    reservation_id = next(iter(runtime.environment.snapshot().reservations))
    responses = [
        json.dumps(
            {
                "action_type": "tool",
                "tool_name": "get_reservation_details",
                "arguments": {"reservation_id": reservation_id},
                "final_answer": None,
            }
        ),
        json.dumps(
            {
                "action_type": "finish",
                "tool_name": None,
                "arguments": {},
                "final_answer": "订单信息已查询。",
            },
            ensure_ascii=False,
        ),
    ]
    agent = AgentLoop(
        llm_client=FakeLLMClient(responses),
        environment=runtime.environment,
        registry=runtime.registry,
        executor=runtime.executor,
    )

    rollout = agent.run(
        task_id="smoke-query",
        initial_user_message=f"请查询订单 {reservation_id}。",
    )

    assert rollout.termination_reason == "finished"
    assert len(rollout.steps) == 2
    assert rollout.steps[0].observation is not None
    assert rollout.steps[0].observation.success is True
    assert rollout.initial_state == rollout.final_state

    task = TaskSpec(
        task_id="smoke-query",
        source_task_id="smoke",
        source_version="test",
        split="test",
        status="supported",
        visible_request=f"请查询订单 {reservation_id}。",
        user_scenario=UserScenarioView(
            domain="airline",
            reason_for_call=f"请查询订单 {reservation_id}。",
            task_instructions="",
        ),
        reference_actions=[
            ReferenceAction(
                step_index=1,
                action_id="smoke-1",
                original_tool_name="get_reservation_details",
                tool_name="get_reservation_details",
                arguments={"reservation_id": reservation_id},
            )
        ],
        reward_basis=["DB", "COMMUNICATE"],
        database_path=str(DATA_DIR / "db.json"),
        database_sha256=file_sha256(DATA_DIR / "db.json"),
        source_payload={},
    )
    result = Evaluator(DATA_DIR / "db.json").evaluate(task, rollout)
    assert result.environment_reward.success is True
    assert result.environment_reward.reward == 1.0


def test_agent_loop_blocks_exact_duplicate_tool_call_without_state_change() -> None:
    runtime = create_airline_runtime(DATA_DIR / "db.json")
    reservation_id = next(iter(runtime.environment.snapshot().reservations))
    tool_action = json.dumps(
        {
            "action_type": "tool",
            "tool_name": "get_reservation_details",
            "arguments": {"reservation_id": reservation_id},
            "final_answer": None,
        }
    )
    finish_action = json.dumps(
        {
            "action_type": "finish",
            "tool_name": None,
            "arguments": {},
            "final_answer": "I used the retrieved reservation details.",
        }
    )
    agent = AgentLoop(
        llm_client=FakeLLMClient([tool_action, tool_action, finish_action]),
        environment=runtime.environment,
        registry=runtime.registry,
        executor=runtime.executor,
    )

    rollout = agent.run(task_id="duplicate-tool", initial_user_message="Check my reservation.")

    assert rollout.termination_reason == "finished"
    assert rollout.steps[0].observation is not None
    assert rollout.steps[0].observation.success is True
    assert rollout.steps[1].observation is not None
    assert rollout.steps[1].observation.error == "duplicate_tool_call"


def test_agent_loop_asks_scripted_user_before_tool_call() -> None:
    runtime = create_airline_runtime(DATA_DIR / "db.json")
    user_id = next(iter(runtime.environment.snapshot().users))
    user_simulator = ScriptedUserSimulator([f"My user id is {user_id}."])
    responses = [
        json.dumps(
            {
                "action_type": "ask_user",
                "tool_name": None,
                "arguments": {},
                "user_question": "Please provide your user id.",
                "final_answer": None,
            }
        ),
        json.dumps(
            {
                "action_type": "tool",
                "tool_name": "get_user_details",
                "arguments": {"user_id": user_id},
                "final_answer": None,
            }
        ),
        json.dumps(
            {
                "action_type": "finish",
                "tool_name": None,
                "arguments": {},
                "final_answer": "I found your account.",
            }
        ),
    ]
    agent = AgentLoop(
        llm_client=FakeLLMClient(responses),
        environment=runtime.environment,
        registry=runtime.registry,
        executor=runtime.executor,
        user_simulator=user_simulator,
    )

    rollout = agent.run(task_id="smoke-user", initial_user_message="I need help.")

    assert rollout.termination_reason == "finished"
    assert len(rollout.steps) == 3
    assert rollout.steps[0].user_reply == f"My user id is {user_id}."
    assert rollout.steps[1].observation is not None
    assert rollout.steps[1].observation.success is True
    assert user_simulator.questions == [
        "Please provide your user id.",
        "I found your account.",
    ]


def test_scripted_user_reveals_task_known_info_without_leaking_instructions() -> None:
    task = next(
        task
        for task in convert_dataset(
            DATA_DIR / "tasks.json",
            DATA_DIR / "split_tasks.json",
            DATA_DIR / "db.json",
        )["base"]
        if task.source_task_id == "8"
    )
    runtime = create_airline_runtime(DATA_DIR / "db.json")
    user_simulator = ScriptedUserSimulator.from_task(task)
    responses = [
        json.dumps(
            {
                "action_type": "ask_user",
                "tool_name": None,
                "arguments": {},
                "user_question": "What is your user id?",
                "final_answer": None,
            }
        ),
        json.dumps(
            {
                "action_type": "finish",
                "tool_name": None,
                "arguments": {},
                "final_answer": "Thank you. I will check the available options.",
            }
        ),
    ]
    agent = AgentLoop(
        llm_client=FakeLLMClient(responses),
        environment=runtime.environment,
        registry=runtime.registry,
        executor=runtime.executor,
        user_simulator=user_simulator,
    )

    rollout = agent.run(task_id=task.task_id)
    all_agent_messages = "\n".join(
        message["content"] for message in rollout.messages if message["role"] != "system"
    )

    expected_reply = "My name is Sophia Silva.\n\nMy user id is sophia_silva_7557."
    assert rollout.steps[0].user_reply == expected_reply
    assert expected_reply in all_agent_messages
    assert task.user_scenario.known_info not in all_agent_messages
    assert task.user_scenario.task_instructions not in all_agent_messages


def test_tau2_start_does_not_leak_task_instruction_to_agent_context() -> None:
    runtime = create_airline_runtime(DATA_DIR / "db.json")
    fake_llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "action_type": "finish",
                    "tool_name": None,
                    "arguments": {},
                    "final_answer": "Hello.",
                }
            )
        ]
    )
    agent = AgentLoop(
        llm_client=fake_llm,
        environment=runtime.environment,
        registry=runtime.registry,
        executor=runtime.executor,
    )

    agent.run(task_id="tau2-start")

    first_call = fake_llm.calls[0]
    assert [message["role"] for message in first_call] == ["system"]
    assert re.search(r"[\u4e00-\u9fff]", first_call[0]["content"]) is None


def test_tau2_start_rejects_finish_until_a_user_message_arrives() -> None:
    runtime = create_airline_runtime(DATA_DIR / "db.json")
    fake_llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "action_type": "finish",
                    "tool_name": None,
                    "arguments": {},
                    "final_answer": "Hello! How can I help?",
                }
            ),
            json.dumps(
                {
                    "action_type": "ask_user",
                    "tool_name": None,
                    "arguments": {},
                    "user_question": "Hello! How can I help?",
                    "final_answer": None,
                }
            ),
            json.dumps(
                {
                    "action_type": "finish",
                    "tool_name": None,
                    "arguments": {},
                    "final_answer": "I can help with that.",
                }
            ),
        ]
    )
    agent = AgentLoop(
        llm_client=fake_llm,
        environment=runtime.environment,
        registry=runtime.registry,
        executor=runtime.executor,
        user_simulator=ScriptedUserSimulator(["I need help with my reservation."]),
    )

    rollout = agent.run(task_id="initial-turn-guard")

    assert rollout.termination_reason == "finished"
    assert rollout.steps[0].action is not None
    assert rollout.steps[0].action.action_type == "finish"
    assert rollout.steps[0].parse_error is not None
    assert rollout.steps[1].user_reply == "I need help with my reservation."
    assert "Initial-turn protocol violation" in fake_llm.calls[1][-1]["content"]


def test_finish_response_allows_a_follow_up_user_turn_before_stopping() -> None:
    runtime = create_airline_runtime(DATA_DIR / "db.json")
    fake_llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "action_type": "ask_user",
                    "tool_name": None,
                    "arguments": {},
                    "user_question": "Hello. How can I help?",
                    "final_answer": None,
                }
            ),
            json.dumps(
                {
                    "action_type": "finish",
                    "tool_name": None,
                    "arguments": {},
                    "final_answer": "That request is not eligible under the policy.",
                }
            ),
            json.dumps(
                {
                    "action_type": "finish",
                    "tool_name": None,
                    "arguments": {},
                    "final_answer": "The additional claim does not change the policy decision.",
                }
            ),
        ]
    )
    agent = AgentLoop(
        llm_client=fake_llm,
        environment=runtime.environment,
        registry=runtime.registry,
        executor=runtime.executor,
        user_simulator=ScriptedUserSimulator(
            ["Please cancel my reservation.", "I was told insurance was unnecessary."]
        ),
    )

    rollout = agent.run(task_id="follow-up-after-refusal")

    assert rollout.termination_reason == "finished"
    assert rollout.steps[1].user_reply == "I was told insurance was unnecessary."
    assert rollout.final_answer == "The additional claim does not change the policy decision."


def test_agent_loop_returns_specific_schema_feedback_after_invalid_action() -> None:
    runtime = create_airline_runtime(DATA_DIR / "db.json")
    fake_llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "action_type": "get_user_details",
                    "tool_name": "get_user_details",
                    "arguments": {"user_id": "noah_muller_9847"},
                    "final_answer": None,
                }
            ),
            json.dumps(
                {
                    "action_type": "finish",
                    "tool_name": None,
                    "arguments": {},
                    "final_answer": "Done.",
                }
            ),
        ]
    )
    agent = AgentLoop(
        llm_client=fake_llm,
        environment=runtime.environment,
        registry=runtime.registry,
        executor=runtime.executor,
    )

    agent.run(task_id="schema-feedback")

    feedback = fake_llm.calls[1][-1]
    assert feedback["role"] == "user"
    assert "Input should be 'tool', 'ask_user', 'finish' or 'done'" in feedback["content"]
    assert '"action_type":"tool"' in feedback["content"]


def test_agent_loop_uses_ask_user_hint_for_ask_user_schema_errors() -> None:
    runtime = create_airline_runtime(DATA_DIR / "db.json")
    fake_llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "action_type": "ask_user",
                    "tool_name": None,
                    "arguments": {},
                    "user_question": None,
                    "final_answer": None,
                }
            ),
            json.dumps(
                {
                    "action_type": "finish",
                    "tool_name": None,
                    "arguments": {},
                    "final_answer": "Done.",
                }
            ),
        ]
    )
    agent = AgentLoop(
        llm_client=fake_llm,
        environment=runtime.environment,
        registry=runtime.registry,
        executor=runtime.executor,
    )

    agent.run(task_id="ask-user-schema-feedback")

    feedback = fake_llm.calls[1][-1]["content"]
    assert '"action_type":"ask_user"' in feedback
    assert '"action_type":"tool"' not in feedback


def test_agent_loop_uses_finish_hint_for_finish_schema_errors() -> None:
    hint = AgentLoop._action_format_hint(
        json.dumps(
            {
                "action_type": "finish",
                "tool_name": None,
                "arguments": {},
                "final_answer": None,
            }
        )
    )

    assert '"action_type":"finish"' in hint
    assert '"action_type":"tool"' not in hint
