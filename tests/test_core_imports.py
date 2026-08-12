from airline_agent.core.actions import AgentAction
from airline_agent.core.llm_client import FakeLLMClient
from airline_agent.core.registry import ToolRegistry
from airline_agent.core.results import ToolResult
from airline_agent.core.rollout import RolloutRecord
from airline_agent.domain.models import Reservation


def test_clean_package_exposes_reusable_core() -> None:
    action = AgentAction(
        action_type="tool",
        tool_name="get_reservation_details",
        arguments={"reservation_id": "R001"},
    )

    assert action.tool_name == "get_reservation_details"
    assert isinstance(FakeLLMClient([]), FakeLLMClient)
    assert isinstance(ToolRegistry(), ToolRegistry)
    assert ToolResult(
        success=True,
        tool_name="example",
        message="ok",
    ).success is True
    assert RolloutRecord(task_id="task-1", user_request="查询订单").steps == []
    assert Reservation.model_fields["reservation_id"].is_required()
