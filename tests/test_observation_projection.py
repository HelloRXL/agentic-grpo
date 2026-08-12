import json

from src.airline_agent.agent.context import project_tool_observation
from src.airline_agent.core.results import ToolResult


def test_projection_keeps_tool_contract_and_returns_valid_json() -> None:
    result = ToolResult(
        success=True,
        tool_name="get_user_details",
        data={"user_id": "u1", "membership": "gold", "reservations": ["A1"]},
        error=None,
        message="ok",
    )

    text = project_tool_observation(result)
    payload = json.loads(text.split("\n", 1)[1].rsplit("\n", 1)[0])

    assert payload["data"]["user_id"] == "u1"
    assert payload["data"]["membership"] == "gold"
    assert payload["projection"]["truncated"] is False


def test_projection_marks_large_lists_instead_of_invalid_truncation() -> None:
    result = ToolResult(
        success=True,
        tool_name="search_direct_flight",
        data={"flights": [{"flight_number": str(index)} for index in range(20)]},
        error=None,
        message="ok",
    )

    text = project_tool_observation(result)
    payload = json.loads(text.split("\n", 1)[1].rsplit("\n", 1)[0])

    assert payload["projection"]["truncated"] is True
    assert payload["data"]["flights"][-1]["_truncated_items"] == 8
