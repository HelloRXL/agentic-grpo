import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from airline_agent.domain.models import AirlineDatabase, Passenger, Reservation


TAU2_AIRLINE_DB = (
    Path(__file__).resolve().parents[2]
    / "reference-repos"
    / "tau2-bench-main"
    / "data"
    / "tau2"
    / "domains"
    / "airline"
    / "db.json"
)


def test_all_tau2_reservations_match_clean_schema() -> None:
    database = json.loads(TAU2_AIRLINE_DB.read_text(encoding="utf-8"))
    reservations = [
        Reservation.model_validate(raw)
        for raw in database["reservations"].values()
    ]

    assert len(reservations) == 2000


def test_entire_tau2_database_matches_clean_schema() -> None:
    database = json.loads(TAU2_AIRLINE_DB.read_text(encoding="utf-8"))
    typed_database = AirlineDatabase.model_validate(database)

    assert typed_database.get_statistics() == {
        "num_flights": 300,
        "num_flight_dates": 9000,
        "num_users": 500,
        "num_reservations": 2000,
    }


def test_clean_reservation_uses_nested_models() -> None:
    database = json.loads(TAU2_AIRLINE_DB.read_text(encoding="utf-8"))
    reservation = Reservation.model_validate(
        next(iter(database["reservations"].values()))
    )

    assert isinstance(reservation.passengers[0], Passenger)


def test_clean_reservation_rejects_unknown_cabin() -> None:
    database = json.loads(TAU2_AIRLINE_DB.read_text(encoding="utf-8"))
    raw = next(iter(database["reservations"].values()))
    raw["cabin"] = "first_class"

    with pytest.raises(ValidationError):
        Reservation.model_validate(raw)
