from airline_agent.core.actions import AgentAction
from airline_agent.core.results import ToolResult
from airline_agent.core.rollout import RolloutRecord, RolloutStep
from airline_agent.environment_reward import compute_environment_reward
from airline_agent.tasks.spec import TaskSpec


def _task(*, communicate_info=None, reference_actions=None) -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "task_id": "train-1",
            "source_task_id": "1",
            "source_version": "test",
            "split": "train",
            "status": "supported",
            "visible_request": "Check my reservation.",
            "user_scenario": {
                "domain": "airline",
                "reason_for_call": "Check my reservation.",
                "task_instructions": "Answer questions naturally.",
            },
            "reference_actions": reference_actions or [],
            "communicate_info": communicate_info or ["confirmation 123"],
            "reward_basis": ["DB", "COMMUNICATE"],
            "database_path": "db.json",
            "database_sha256": "abc",
            "source_payload": {},
        }
    )


def _rollout(*, answer="Your confirmation 123 is complete.", steps=None, termination="finished"):
    return RolloutRecord(
        task_id="train-1",
        user_request="",
        steps=steps or [],
        final_answer=answer,
        termination_reason=termination,
    )


def test_environment_reward_is_one_for_complete_official_outcome():
    result = compute_environment_reward(
        _task(), _rollout(), replay_success=True, initial_state_match=True,
        final_state_match=True,
    )

    assert result.success is True
    assert result.db_score == 1.0
    assert result.communicate_score == 1.0
    assert result.reward == 1.0


def test_environment_reward_decomposes_db_and_communication_failures():
    result = compute_environment_reward(
        _task(), _rollout(answer="The change is complete."),
        replay_success=True, initial_state_match=True, final_state_match=True,
    )

    assert result.success is False
    assert result.db_score == 1.0
    assert result.communicate_score == 0.0
    assert result.reward == 0.0
    assert result.training_reward == 0.0
    assert result.missing_communicate_info == ["confirmation 123"]


def test_environment_reward_checks_all_customer_facing_agent_messages():
    earlier_finish = AgentAction(
        action_type="finish",
        final_answer="Your confirmation 123 is complete.",
    )
    later_finish = AgentAction(
        action_type="finish",
        final_answer="Is there anything else I can help with?",
    )
    rollout = _rollout(
        answer=later_finish.final_answer,
        steps=[
            RolloutStep(step_index=1, raw_model_output="{}", action=earlier_finish),
            RolloutStep(step_index=2, raw_model_output="{}", action=later_finish),
        ],
    )

    result = compute_environment_reward(
        _task(), rollout, replay_success=True, initial_state_match=True,
        final_state_match=True,
    )

    assert result.communicate_score == 1.0
    assert result.success is True


def test_environment_reward_penalizes_exact_repeated_actions_but_not_paths():
    action = AgentAction(
        action_type="tool",
        tool_name="get_reservation_details",
        arguments={"reservation_id": "ABC123"},
    )
    repeated_steps = [
        RolloutStep(
            step_index=index,
            raw_model_output="{}",
            action=action,
            observation=ToolResult(success=True, tool_name=action.tool_name, message="ok"),
        )
        for index in (1, 2)
    ]
    result = compute_environment_reward(
        _task(), _rollout(steps=repeated_steps), replay_success=True,
        initial_state_match=True, final_state_match=False,
    )

    assert result.db_score == 0.0
    assert result.communicate_score == 1.0
    assert result.repeated_action_count == 1
    assert result.process_penalty == 0.1
    assert result.reward == 0.0
    assert result.training_reward == -0.08


def test_environment_reward_allows_a_query_again_after_a_successful_write():
    query = AgentAction(
        action_type="tool",
        tool_name="get_reservation_details",
        arguments={"reservation_id": "ABC123"},
    )
    write = AgentAction(
        action_type="tool",
        tool_name="update_reservation_baggages",
        arguments={"reservation_id": "ABC123", "total_baggages": 2, "nonfree_baggages": 0, "payment_id": "card"},
    )
    success = lambda action: ToolResult(success=True, tool_name=action.tool_name, message="ok")
    rollout = _rollout(steps=[
        RolloutStep(step_index=1, raw_model_output="{}", action=query, observation=success(query)),
        RolloutStep(step_index=2, raw_model_output="{}", action=write, observation=success(write)),
        RolloutStep(step_index=3, raw_model_output="{}", action=query, observation=success(query)),
    ])

    result = compute_environment_reward(
        _task(), rollout, replay_success=True,
        initial_state_match=True, final_state_match=True,
    )

    assert result.repeated_action_count == 0
    assert result.process_penalty == 0.0


def test_environment_reward_blocks_communication_credit_after_wrong_successful_write():
    action = AgentAction(
        action_type="tool",
        tool_name="update_reservation_flights",
        arguments={"reservation_id": "ABC123", "cabin": "economy", "flights": [], "payment_id": "card"},
    )
    rollout = _rollout(
        steps=[
            RolloutStep(
                step_index=1,
                raw_model_output="{}",
                action=action,
                observation=ToolResult(success=True, tool_name=action.tool_name, message="updated"),
            )
        ]
    )

    result = compute_environment_reward(
        _task(), rollout, replay_success=True,
        initial_state_match=True, final_state_match=False,
    )

    assert result.db_score == 0.0
    assert result.communicate_score == 1.0
    assert result.reward == 0.0
    assert "successful_write_final_state_mismatch" in result.reasons


def test_environment_reward_marks_llm_infrastructure_failure_invalid():
    result = compute_environment_reward(
        _task(), _rollout(termination="llm_error"), replay_success=True,
        initial_state_match=True, final_state_match=False,
    )

    assert result.valid is False
    assert result.reward == 0.0


def test_environment_reward_gives_no_training_reward_to_max_steps():
    result = compute_environment_reward(
        _task(), _rollout(termination="max_steps"), replay_success=True,
        initial_state_match=True, final_state_match=True,
    )

    assert result.valid is True
    assert result.success is False
    assert result.reward == 0.0


def test_environment_reward_gives_bounded_progress_credit_to_incomplete_path():
    query = AgentAction(
        action_type="tool",
        tool_name="get_reservation_details",
        arguments={"reservation_id": "ABC123"},
    )
    task = _task(reference_actions=[
        {
            "step_index": 1,
            "action_id": "query",
            "original_tool_name": "get_reservation_details",
            "tool_name": "get_reservation_details",
            "arguments": {"reservation_id": "ABC123"},
        },
        {
            "step_index": 2,
            "action_id": "cancel",
            "original_tool_name": "cancel_reservation",
            "tool_name": "cancel_reservation",
            "arguments": {"reservation_id": "ABC123"},
        },
    ])
    rollout = _rollout(
        termination="max_steps",
        steps=[
            RolloutStep(
                step_index=1,
                raw_model_output="{}",
                action=query,
                observation=ToolResult(
                    success=True,
                    tool_name=query.tool_name,
                    message="ok",
                ),
            )
        ],
    )

    result = compute_environment_reward(
        task,
        rollout,
        replay_success=True,
        initial_state_match=True,
        final_state_match=False,
    )

    assert result.success is False
    assert result.action_progress_score == 0.5
    assert result.progress_reward == 0.125
    assert result.terminal_reward == 0.0
    assert result.reward == 0.0
    assert result.training_reward == -0.3


def test_prm_lite_rewards_observation_to_action_data_chain():
    query = AgentAction(
        action_type="tool",
        tool_name="get_user_details",
        arguments={"user_id": "user-1"},
    )
    write = AgentAction(
        action_type="tool",
        tool_name="cancel_reservation",
        arguments={"reservation_id": "ABC123"},
    )
    rollout = _rollout(steps=[
        RolloutStep(
            step_index=1,
            raw_model_output="{}",
            action=query,
            observation=ToolResult(
                success=True,
                tool_name=query.tool_name,
                data={"user_id": "user-1", "reservation_id": "ABC123"},
                message="ok",
            ),
        ),
        RolloutStep(
            step_index=2,
            raw_model_output="{}",
            action=write,
            observation=ToolResult(
                success=True,
                tool_name=write.tool_name,
                message="cancelled",
            ),
        ),
    ])

    result = compute_environment_reward(
        _task(),
        rollout,
        replay_success=True,
        initial_state_match=True,
        final_state_match=True,
        reward_mode="prm_lite_v1",
    )

    assert result.process_quality_score == 0.045
    assert result.training_reward == 1.0135


def test_prm_lite_and_terminal_reward_modes_are_separate():
    terminal = compute_environment_reward(
        _task(), _rollout(), replay_success=True, initial_state_match=True,
        final_state_match=True, reward_mode="terminal_v4",
    )
    prm = compute_environment_reward(
        _task(), _rollout(), replay_success=True, initial_state_match=True,
        final_state_match=True, reward_mode="prm_lite_v1",
    )

    assert terminal.reward == prm.reward == 1.0
    assert terminal.training_reward == prm.training_reward == 1.0
