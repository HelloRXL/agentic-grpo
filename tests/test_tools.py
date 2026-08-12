import json
from pathlib import Path

import pytest

from airline_agent.core.errors import AirlineBusinessError
from airline_agent.domain.environment import AirlineEnvironment
from airline_agent.domain.runtime import create_airline_runtime
from airline_agent.domain.tool_schemas import (
    BookReservationArgs,
    CancelReservationArgs,
    GetReservationDetailsArgs,
    GetUserDetailsArgs,
    GetFlightStatusArgs,
    ListAllAirportsArgs,
    SearchDirectFlightArgs,
    SearchOnestopFlightArgs,
    TransferToHumanAgentsArgs,
    UpdateReservationBaggagesArgs,
    UpdateReservationFlightsArgs,
)
from airline_agent.domain.tools import AirlineTools


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "reference-repos" / "tau2-bench-main" / "data" / "tau2" / "domains" / "airline"


def _task_action(task_id: str, action_name: str) -> dict:
    tasks = json.loads((DATA_DIR / "tasks.json").read_text(encoding="utf-8"))
    task = next(task for task in tasks if task["id"] == task_id)
    action = next(
        action
        for action in task["evaluation_criteria"]["actions"]
        if action["name"] == action_name
    )
    return action["arguments"]


def test_core_tools_read_write_and_reset_state() -> None:
    environment = AirlineEnvironment.from_json(DATA_DIR / "db.json")
    tools = AirlineTools(environment)

    user_id = "mia_li_3668"
    user = tools.get_user_details(GetUserDetailsArgs(user_id=user_id))
    assert user["user_id"] == user_id

    reservation_id = user["reservations"][0]
    reservation = tools.get_reservation_details(
        GetReservationDetailsArgs(reservation_id=reservation_id)
    )
    assert reservation["reservation_id"] == reservation_id

    flights = tools.search_direct_flight(
        SearchDirectFlightArgs(
            origin="ORD",
            destination="PHL",
            date="2024-05-26",
        )
    )
    assert flights["count"] > 0

    cancelled = tools.cancel_reservation(
        CancelReservationArgs(reservation_id=reservation_id)
    )
    assert cancelled["status"] == "cancelled"

    with pytest.raises(AirlineBusinessError) as error:
        tools.cancel_reservation(CancelReservationArgs(reservation_id=reservation_id))
    assert error.value.code == "reservation_already_cancelled"

    environment.reset()
    restored = tools.get_reservation_details(
        GetReservationDetailsArgs(reservation_id=reservation_id)
    )
    assert restored["status"] is None


def test_booking_and_flight_update_use_official_task_arguments() -> None:
    environment = AirlineEnvironment.from_json(DATA_DIR / "db.json")
    tools = AirlineTools(environment)

    booking = tools.book_reservation(
        BookReservationArgs.model_validate(_task_action("8", "book_reservation"))
    )
    assert booking["reservation_id"] == "HATHAT"
    assert booking["status"] is None

    updated = tools.update_reservation_flights(
        UpdateReservationFlightsArgs.model_validate(
            _task_action("44", "update_reservation_flights")
        )
    )
    assert updated["cabin"] == "business"


def test_flight_downgrade_records_the_official_refund() -> None:
    environment = AirlineEnvironment.from_json(DATA_DIR / "db.json")
    tools = AirlineTools(environment)

    updated = tools.update_reservation_flights(
        UpdateReservationFlightsArgs.model_validate(
            _task_action("11", "update_reservation_flights")
        )
    )

    assert updated["cabin"] == "basic_economy"
    assert updated["payment_history"][-1] == {
        "payment_id": "gift_card_1642017",
        "amount": -5244,
    }
    user = tools.get_user_details(GetUserDetailsArgs(user_id=updated["user_id"]))
    assert user["payment_methods"]["gift_card_1642017"]["amount"] == 5372


def test_baggage_update_uses_official_arguments_and_rejects_removal() -> None:
    environment = AirlineEnvironment.from_json(DATA_DIR / "db.json")
    tools = AirlineTools(environment)

    updated = tools.update_reservation_baggages(
        UpdateReservationBaggagesArgs.model_validate(
            _task_action("21", "update_reservation_baggages")
        )
    )
    assert updated["total_baggages"] == 2
    assert updated["nonfree_baggages"] == 0

    charged = tools.update_reservation_baggages(
        UpdateReservationBaggagesArgs(
            reservation_id="OBUT9V",
            total_baggages=3,
            nonfree_baggages=1,
            payment_id="gift_card_6276644",
        )
    )
    assert charged["payment_history"][-1] == {
        "payment_id": "gift_card_6276644",
        "amount": 50,
    }

    with pytest.raises(AirlineBusinessError) as error:
        tools.update_reservation_baggages(
            UpdateReservationBaggagesArgs(
                reservation_id="OBUT9V",
                total_baggages=2,
                nonfree_baggages=0,
                payment_id="gift_card_6276644",
            )
        )
    assert error.value.code == "baggage_removal_not_allowed"


def test_runtime_executor_validates_then_calls_registered_tool() -> None:
    runtime = create_airline_runtime(DATA_DIR / "db.json")

    result = runtime.executor.execute(
        "search_direct_flight",
        {"origin": "ORD", "destination": "PHL", "date": "2024-05-26"},
    )

    assert result.success is True
    assert result.data["count"] > 0


def test_list_all_airports_returns_iata_mapping() -> None:
    environment = AirlineEnvironment.from_json(DATA_DIR / "db.json")
    tools = AirlineTools(environment)

    result = tools.list_all_airports(
        ListAllAirportsArgs(cities=["San Francisco", "New York"])
    )

    assert result["matches"] == [
        {"iata": "SFO", "city": "San Francisco"},
        {"iata": "JFK", "city": "New York"},
    ]
    assert result["unknown_cities"] == []


def test_onestop_search_and_flight_status_match_official_data() -> None:
    environment = AirlineEnvironment.from_json(DATA_DIR / "db.json")
    tools = AirlineTools(environment)

    itineraries = tools.search_onestop_flight(
        SearchOnestopFlightArgs(origin="JFK", destination="SEA", date="2024-05-20")
    )
    assert any(
        [flight["flight_number"] for flight in item["flights"]] == ["HAT136", "HAT039"]
        for item in itineraries["itineraries"]
    )
    assert itineraries["count"] <= 12
    assert itineraries["total_count"] >= itineraries["count"]

    status = tools.get_flight_status(
        GetFlightStatusArgs(flight_number="HAT039", date="2024-05-15")
    )
    assert status["status"] == "delayed"


def test_runtime_registers_airport_lookup_tool() -> None:
    runtime = create_airline_runtime(DATA_DIR / "db.json")

    result = runtime.executor.execute(
        "list_all_airports", {"cities": ["Newark", "Atlantis"]}
    )

    assert result.success is True
    assert result.data["matches"] == [{"iata": "EWR", "city": "Newark"}]
    assert result.data["unknown_cities"] == ["Atlantis"]


def test_transfer_to_human_agents_is_non_mutating_and_requires_a_summary() -> None:
    environment = AirlineEnvironment.from_json(DATA_DIR / "db.json")
    tools = AirlineTools(environment)
    before = environment.snapshot()

    result = tools.transfer_to_human_agents(
        TransferToHumanAgentsArgs(summary="Customer explicitly requests a human exception review.")
    )

    assert result == {
        "transferred": True,
        "summary": "Customer explicitly requests a human exception review.",
        "message": "Transfer successful. Send the required transfer notice to the customer.",
    }
    assert environment.snapshot() == before

    runtime = create_airline_runtime(DATA_DIR / "db.json")
    executed = runtime.executor.execute(
        "transfer_to_human_agents", {"summary": "Customer requests escalation."}
    )
    assert executed.success is True
