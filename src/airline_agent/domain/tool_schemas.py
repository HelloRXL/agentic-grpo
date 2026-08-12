"""Airline 工具共用的参数 Schema。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolArgs(BaseModel):
    """所有工具参数的共同配置：拒绝拼写错误或未支持的额外字段。"""

    model_config = ConfigDict(extra="forbid")


class GetUserDetailsArgs(ToolArgs):
    user_id: str = Field(min_length=1)


class GetReservationDetailsArgs(ToolArgs):
    reservation_id: str = Field(min_length=1)


class ListAllAirportsArgs(ToolArgs):
    """Look up airport codes from the full city names supplied by the user."""

    cities: list[str] = Field(
        min_length=1,
        description="Full city names from the user message, for example San Francisco or New York.",
    )


class SearchDirectFlightArgs(ToolArgs):
    origin: str = Field(pattern=r"^[A-Z]{3}$")
    destination: str = Field(pattern=r"^[A-Z]{3}$")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class SearchOnestopFlightArgs(SearchDirectFlightArgs):
    """Search one-stop itineraries between two airports on a travel date."""

    max_results: int = Field(default=12, ge=1, le=12)


class GetFlightStatusArgs(ToolArgs):
    flight_number: str = Field(min_length=1)
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class FlightSelection(ToolArgs):
    flight_number: str = Field(min_length=1)
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class PassengerInput(ToolArgs):
    first_name: str = Field(min_length=1)
    last_name: str = Field(min_length=1)
    dob: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class PaymentInput(ToolArgs):
    payment_id: str = Field(min_length=1)
    amount: int = Field(ge=0)


class BookReservationArgs(ToolArgs):
    user_id: str = Field(min_length=1)
    origin: str = Field(pattern=r"^[A-Z]{3}$")
    destination: str = Field(pattern=r"^[A-Z]{3}$")
    flight_type: Literal["one_way", "round_trip"]
    cabin: Literal["basic_economy", "economy", "business"]
    flights: list[FlightSelection] = Field(min_length=1)
    passengers: list[PassengerInput] = Field(min_length=1)
    payment_methods: list[PaymentInput] = Field(min_length=1)
    total_baggages: int = Field(ge=0)
    nonfree_baggages: int = Field(ge=0)
    insurance: Literal["yes", "no"]


class CancelReservationArgs(ToolArgs):
    reservation_id: str = Field(min_length=1)


class UpdateReservationFlightsArgs(ToolArgs):
    reservation_id: str = Field(min_length=1)
    cabin: Literal["basic_economy", "economy", "business"]
    flights: list[FlightSelection] = Field(min_length=1)
    payment_id: str = Field(min_length=1)


class UpdateReservationBaggagesArgs(ToolArgs):
    reservation_id: str = Field(min_length=1)
    total_baggages: int = Field(ge=0)
    nonfree_baggages: int = Field(ge=0)
    payment_id: str = Field(min_length=1)


class TransferToHumanAgentsArgs(ToolArgs):
    """Escalate an out-of-scope issue with an auditable customer summary."""

    summary: str = Field(min_length=1)


TOOL_ARGUMENT_SCHEMAS = {
    "get_user_details": GetUserDetailsArgs,
    "get_reservation_details": GetReservationDetailsArgs,
    "list_all_airports": ListAllAirportsArgs,
    "search_direct_flight": SearchDirectFlightArgs,
    "search_onestop_flight": SearchOnestopFlightArgs,
    "get_flight_status": GetFlightStatusArgs,
    "book_reservation": BookReservationArgs,
    "cancel_reservation": CancelReservationArgs,
    "update_reservation_flights": UpdateReservationFlightsArgs,
    "update_reservation_baggages": UpdateReservationBaggagesArgs,
    "transfer_to_human_agents": TransferToHumanAgentsArgs,
}
