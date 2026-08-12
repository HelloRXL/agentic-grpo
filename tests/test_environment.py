from pathlib import Path

from airline_agent.domain.environment import AirlineEnvironment


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "reference-repos" / "tau2-bench-main" / "data" / "tau2" / "domains" / "airline" / "db.json"


def test_environment_loads_searches_and_resets_without_reference_leaks() -> None:
    environment = AirlineEnvironment.from_json(DB_PATH)
    initial = environment.snapshot()

    user_id = next(iter(initial.users))
    user = environment.get_user(user_id)
    assert user is not None
    user.name.first_name = "changed-outside"
    assert environment.get_user(user_id).name.first_name != "changed-outside"

    reservation_id = next(iter(initial.reservations))
    reservation = environment.get_reservation(reservation_id)
    assert reservation is not None
    reservation.status = "cancelled"
    assert environment.get_reservation(reservation_id).status != "cancelled"

    results = environment.search_direct_flights("ORD", "PHL", "2024-05-26")
    assert results
    assert results[0][0].origin == "ORD"
    assert results[0][1].status == "available"

    environment.reset()
    assert environment.snapshot() == initial
