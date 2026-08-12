"""将 Airline 环境、工具、注册表和执行器组装成可运行组件。"""

from dataclasses import dataclass
from pathlib import Path

from ..core.executor import ToolExecutor
from ..core.registry import ToolRegistry, ToolSpec
from .environment import AirlineEnvironment
from .models import AirlineDatabase
from .state_patches import (
    InitialStatePatch,
    apply_initial_state_patches,
    database_state_sha256,
)
from .tool_schemas import TOOL_ARGUMENT_SCHEMAS
from .tools import AirlineTools


@dataclass(frozen=True)
class AirlineRuntime:
    environment: AirlineEnvironment
    tools: AirlineTools
    registry: ToolRegistry
    executor: ToolExecutor


def create_airline_runtime(
    database_path: Path,
    *,
    initial_state_patches: tuple[InitialStatePatch, ...] | list[InitialStatePatch] = (),
    expected_initial_state_sha256: str | None = None,
) -> AirlineRuntime:
    """从 base DB 与受限 patch 构造 episode，并校验声明的初始状态。"""

    database = AirlineDatabase.model_validate_json(database_path.read_text(encoding="utf-8"))
    patched_database = apply_initial_state_patches(database, initial_state_patches)
    actual_hash = database_state_sha256(patched_database)
    if expected_initial_state_sha256 is not None and actual_hash != expected_initial_state_sha256:
        raise ValueError(
            "initial_state_sha256 不匹配："
            f"expected={expected_initial_state_sha256}, actual={actual_hash}"
        )
    environment = AirlineEnvironment(patched_database)
    tools = AirlineTools(environment)
    registry = ToolRegistry()

    registry.register_tool(
        ToolSpec(
            name="get_user_details",
            description="Retrieve a user's profile and reservation ID list.",
            args_schema=TOOL_ARGUMENT_SCHEMAS["get_user_details"],
            function=tools.get_user_details,
        )
    )
    registry.register_tool(
        ToolSpec(
            name="get_reservation_details",
            description="Retrieve the details of a specific reservation.",
            args_schema=TOOL_ARGUMENT_SCHEMAS["get_reservation_details"],
            function=tools.get_reservation_details,
        )
    )
    registry.register_tool(
        ToolSpec(
            name="list_all_airports",
            description="Look up IATA airport codes from full city names.",
            args_schema=TOOL_ARGUMENT_SCHEMAS["list_all_airports"],
            function=tools.list_all_airports,
        )
    )
    registry.register_tool(
        ToolSpec(
            name="search_direct_flight",
            description="Search available direct flights for a date and route.",
            args_schema=TOOL_ARGUMENT_SCHEMAS["search_direct_flight"],
            function=tools.search_direct_flight,
        )
    )
    registry.register_tool(
        ToolSpec(
            name="search_onestop_flight",
            description="Search available one-stop itineraries with connection times, seats, and total cabin prices.",
            args_schema=TOOL_ARGUMENT_SCHEMAS["search_onestop_flight"],
            function=tools.search_onestop_flight,
        )
    )
    registry.register_tool(
        ToolSpec(
            name="get_flight_status",
            description="Retrieve the exact operational status of a flight on a date.",
            args_schema=TOOL_ARGUMENT_SCHEMAS["get_flight_status"],
            function=tools.get_flight_status,
        )
    )
    registry.register_tool(
        ToolSpec(
            name="transfer_to_human_agents",
            description="Escalate an explicitly requested issue that cannot be handled by the registered tools; records no database change.",
            args_schema=TOOL_ARGUMENT_SCHEMAS["transfer_to_human_agents"],
            function=tools.transfer_to_human_agents,
        )
    )
    registry.register_tool(
        ToolSpec(
            name="book_reservation",
            description="Create a reservation from user, flights, passenger, and payment details.",
            args_schema=TOOL_ARGUMENT_SCHEMAS["book_reservation"],
            function=tools.book_reservation,
            mutates_state=True,
        )
    )
    registry.register_tool(
        ToolSpec(
            name="cancel_reservation",
            description="Cancel a reservation and record its refund transactions.",
            args_schema=TOOL_ARGUMENT_SCHEMAS["cancel_reservation"],
            function=tools.cancel_reservation,
            mutates_state=True,
        )
    )
    registry.register_tool(
        ToolSpec(
            name="update_reservation_flights",
            description="Replace a reservation's complete flight itinerary and cabin.",
            args_schema=TOOL_ARGUMENT_SCHEMAS["update_reservation_flights"],
            function=tools.update_reservation_flights,
            mutates_state=True,
        )
    )
    registry.register_tool(
        ToolSpec(
            name="update_reservation_baggages",
            description="Add checked bags to a reservation and charge newly added non-free bags.",
            args_schema=TOOL_ARGUMENT_SCHEMAS["update_reservation_baggages"],
            function=tools.update_reservation_baggages,
            mutates_state=True,
        )
    )

    return AirlineRuntime(
        environment=environment,
        tools=tools,
        registry=registry,
        executor=ToolExecutor(registry),
    )
