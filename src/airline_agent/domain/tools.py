"""Airline 工具的最小真实业务实现。"""

from copy import deepcopy
from datetime import date as Date
from datetime import timedelta
from typing import Any

from ..core.errors import AirlineBusinessError
from .environment import AirlineEnvironment
from .models import (
    AvailableFlightDate,
    Payment,
    Passenger,
    Reservation,
    ReservationFlight,
)
from .tool_schemas import (
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


class AirlineTools:
    """实现业务规则，但不负责解析 LLM 输出或选择工具。"""

    def __init__(self, environment: AirlineEnvironment) -> None:
        self._environment = environment

    def _user(self, user_id: str):
        user = self._environment.get_user(user_id)
        if user is None:
            raise AirlineBusinessError("user_not_found", f"User {user_id} does not exist.")
        return user

    def _reservation(self, reservation_id: str) -> Reservation:
        reservation = self._environment.get_reservation(reservation_id)
        if reservation is None:
            raise AirlineBusinessError(
                "reservation_not_found", f"Reservation {reservation_id} does not exist."
            )
        return reservation

    def _new_reservation_id(self) -> str:
        for reservation_id in ("HATHAT", "HATHAU", "HATHAV"):
            if self._environment.get_reservation(reservation_id) is None:
                return reservation_id
        raise AirlineBusinessError(
            "reservation_id_exhausted",
            "The reservation ID allocation limit for this task has been reached.",
        )

    def get_user_details(self, args: GetUserDetailsArgs) -> dict[str, Any]:
        return self._user(args.user_id).model_dump(mode="json")

    def get_reservation_details(
        self,
        args: GetReservationDetailsArgs,
    ) -> dict[str, Any]:
        return self._reservation(args.reservation_id).model_dump(mode="json")

    def list_all_airports(self, args: ListAllAirportsArgs) -> dict[str, Any]:
        """根据城市全称返回匹配的 IATA 代码，不向模型暴露完整机场表。"""

        airports = (
            ("SFO", "San Francisco"),
            ("JFK", "New York"),
            ("LAX", "Los Angeles"),
            ("ORD", "Chicago"),
            ("DFW", "Dallas"),
            ("DEN", "Denver"),
            ("SEA", "Seattle"),
            ("ATL", "Atlanta"),
            ("MIA", "Miami"),
            ("BOS", "Boston"),
            ("PHX", "Phoenix"),
            ("IAH", "Houston"),
            ("LAS", "Las Vegas"),
            ("MCO", "Orlando"),
            ("EWR", "Newark"),
            ("CLT", "Charlotte"),
            ("MSP", "Minneapolis"),
            ("DTW", "Detroit"),
            ("PHL", "Philadelphia"),
            ("LGA", "LaGuardia"),
        )
        requested = {city.strip().casefold() for city in args.cities if city.strip()}
        matches = [
            {"iata": iata, "city": city}
            for iata, city in airports
            if city.casefold() in requested
        ]
        matched_cities = {item["city"].casefold() for item in matches}
        unknown_cities = [
            city for city in args.cities if city.strip().casefold() not in matched_cities
        ]
        return {"matches": matches, "unknown_cities": unknown_cities}

    def search_direct_flight(
        self,
        args: SearchDirectFlightArgs,
    ) -> dict[str, Any]:
        results = self._environment.search_direct_flights(
            origin=args.origin,
            destination=args.destination,
            date=args.date,
        )
        flights = []
        for flight, flight_date in results:
            assert isinstance(flight_date, AvailableFlightDate)
            flights.append(
                {
                    "flight_number": flight.flight_number,
                    "origin": flight.origin,
                    "destination": flight.destination,
                    "date": args.date,
                    "scheduled_departure_time_est": flight.scheduled_departure_time_est,
                    "scheduled_arrival_time_est": flight.scheduled_arrival_time_est,
                    "available_seats": flight_date.available_seats,
                    "prices": flight_date.prices,
                }
            )
        return {"count": len(flights), "flights": flights}

    @staticmethod
    def _clock_seconds(value: str) -> int:
        """Convert τ²'s ``HH:MM:SS`` or ``HH:MM:SS+1`` time to seconds in a day."""

        clock = value.removesuffix("+1")
        hour, minute, second = (int(part) for part in clock.split(":"))
        return hour * 3600 + minute * 60 + second

    @staticmethod
    def _flight_payload(flight, flight_date, date_value: str) -> dict[str, Any]:
        assert isinstance(flight_date, AvailableFlightDate)
        return {
            "flight_number": flight.flight_number,
            "origin": flight.origin,
            "destination": flight.destination,
            "date": date_value,
            "scheduled_departure_time_est": flight.scheduled_departure_time_est,
            "scheduled_arrival_time_est": flight.scheduled_arrival_time_est,
            "available_seats": flight_date.available_seats,
            "prices": flight_date.prices,
        }

    def search_onestop_flight(
        self,
        args: SearchOnestopFlightArgs,
    ) -> dict[str, Any]:
        """Find available two-leg itineraries with a chronologically valid connection."""

        database = self._environment.snapshot()
        first_legs = [
            (flight, flight.dates.get(args.date))
            for flight in database.flights.values()
            if flight.origin == args.origin
        ]
        itineraries: list[dict[str, Any]] = []
        for first, first_status in first_legs:
            if not isinstance(first_status, AvailableFlightDate):
                continue
            arrival_next_day = first.scheduled_arrival_time_est.endswith("+1")
            second_date = (
                Date.fromisoformat(args.date) + timedelta(days=1 if arrival_next_day else 0)
            ).isoformat()
            first_arrival = self._clock_seconds(first.scheduled_arrival_time_est)
            for second in database.flights.values():
                if second.origin != first.destination or second.destination != args.destination:
                    continue
                second_status = second.dates.get(second_date)
                if not isinstance(second_status, AvailableFlightDate):
                    continue
                if self._clock_seconds(second.scheduled_departure_time_est) < first_arrival:
                    continue
                total_prices = {
                    cabin: first_status.prices[cabin] + second_status.prices[cabin]
                    for cabin in first_status.prices
                }
                available_seats = {
                    cabin: min(first_status.available_seats[cabin], second_status.available_seats[cabin])
                    for cabin in first_status.available_seats
                }
                itineraries.append(
                    {
                        "flights": [
                            self._flight_payload(first, first_status, args.date),
                            self._flight_payload(second, second_status, second_date),
                        ],
                        "layover_minutes": (
                            self._clock_seconds(second.scheduled_departure_time_est) - first_arrival
                        ) // 60,
                        "available_seats": available_seats,
                        "total_prices": total_prices,
                    }
                )
        itineraries.sort(
            key=lambda item: (
                item["total_prices"]["economy"],
                item["layover_minutes"],
                tuple(flight["flight_number"] for flight in item["flights"]),
            )
        )
        total_count = len(itineraries)
        returned = itineraries[: args.max_results]
        return {
            "count": len(returned),
            "total_count": total_count,
            "truncated": total_count > len(returned),
            "itineraries": returned,
        }

    def get_flight_status(self, args: GetFlightStatusArgs) -> dict[str, Any]:
        """Retrieve an existing flight's exact status rather than inferring it from dates."""

        flight = self._environment.get_flight(args.flight_number)
        if flight is None:
            raise AirlineBusinessError(
                "flight_not_found", f"Flight {args.flight_number} does not exist."
            )
        status = flight.dates.get(args.date)
        if status is None:
            raise AirlineBusinessError(
                "flight_date_not_found",
                f"Flight {args.flight_number} has no record on {args.date}.",
            )
        return {
            "flight_number": flight.flight_number,
            "origin": flight.origin,
            "destination": flight.destination,
            "date": args.date,
            **status.model_dump(mode="json"),
        }

    def transfer_to_human_agents(
        self,
        args: TransferToHumanAgentsArgs,
    ) -> dict[str, Any]:
        """Record a non-mutating escalation request; no human-side work is simulated."""

        return {
            "transferred": True,
            "summary": args.summary,
            "message": "Transfer successful. Send the required transfer notice to the customer.",
        }

    def _check_payment(
        self,
        user,
        payment_id: str,
        amount: int,
    ) -> None:
        payment_method = user.payment_methods.get(payment_id)
        if payment_method is None:
            raise AirlineBusinessError("payment_not_found", f"Payment method {payment_id} does not exist.")
        if payment_method.source in {"gift_card", "certificate"} and payment_method.amount < amount:
            raise AirlineBusinessError(
                "payment_insufficient",
                f"Payment method {payment_id} has insufficient balance.",
            )

    def book_reservation(self, args: BookReservationArgs) -> dict[str, Any]:
        user = self._user(args.user_id)
        working_user = user.model_copy(deep=True)
        working_flights = []
        total_price = 0

        for selected in args.flights:
            flight = self._environment.get_flight(selected.flight_number)
            if flight is None:
                raise AirlineBusinessError(
                    "flight_not_found", f"Flight {selected.flight_number} does not exist."
                )
            flight_date = flight.dates.get(selected.date)
            if not isinstance(flight_date, AvailableFlightDate):
                raise AirlineBusinessError(
                    "flight_not_available",
                    f"Flight {selected.flight_number} is unavailable on that date.",
                )
            seats = flight_date.available_seats[args.cabin]
            if seats < len(args.passengers):
                raise AirlineBusinessError(
                    "not_enough_seats",
                    f"Flight {selected.flight_number} does not have enough seats.",
                )
            price = flight_date.prices[args.cabin]
            working_flights.append(
                ReservationFlight(
                    flight_number=selected.flight_number,
                    date=selected.date,
                    origin=flight.origin,
                    destination=flight.destination,
                    price=price,
                )
            )
            total_price += price * len(args.passengers)

        if args.insurance == "yes":
            total_price += 30 * len(args.passengers)
        total_price += 50 * args.nonfree_baggages

        paid = sum(payment.amount for payment in args.payment_methods)
        if paid != total_price:
            raise AirlineBusinessError(
                "payment_total_mismatch",
                f"Payment total {paid} does not match reservation total {total_price}.",
            )
        for payment in args.payment_methods:
            self._check_payment(working_user, payment.payment_id, payment.amount)

        for payment in args.payment_methods:
            method = working_user.payment_methods[payment.payment_id]
            if method.source == "gift_card":
                method.amount -= payment.amount
            elif method.source == "certificate":
                working_user.payment_methods.pop(payment.payment_id)

        reservation = Reservation(
            reservation_id=self._new_reservation_id(),
            user_id=args.user_id,
            origin=args.origin,
            destination=args.destination,
            flight_type=args.flight_type,
            cabin=args.cabin,
            flights=working_flights,
            passengers=[
                Passenger.model_validate(passenger.model_dump())
                for passenger in args.passengers
            ],
            payment_history=[Payment.model_validate(payment.model_dump()) for payment in args.payment_methods],
            created_at="2024-05-15T15:00:00",
            total_baggages=args.total_baggages,
            nonfree_baggages=args.nonfree_baggages,
            insurance=args.insurance,
        )
        working_user.reservations.append(reservation.reservation_id)

        self._environment.save_user(working_user)
        self._environment.save_reservation(reservation)
        return reservation.model_dump(mode="json")

    def cancel_reservation(self, args: CancelReservationArgs) -> dict[str, Any]:
        reservation = self._reservation(args.reservation_id)
        if reservation.status == "cancelled":
            raise AirlineBusinessError(
                "reservation_already_cancelled",
                f"Reservation {args.reservation_id} is already cancelled.",
            )

        working = reservation.model_copy(deep=True)
        working.payment_history.extend(
            Payment(payment_id=payment.payment_id, amount=-payment.amount)
            for payment in reservation.payment_history
        )
        working.status = "cancelled"
        self._environment.save_reservation(working)
        return working.model_dump(mode="json")

    def update_reservation_baggages(
        self,
        args: UpdateReservationBaggagesArgs,
    ) -> dict[str, Any]:
        """Add checked bags and charge only newly added non-free bags."""

        reservation = self._reservation(args.reservation_id)
        if args.total_baggages < reservation.total_baggages:
            raise AirlineBusinessError(
                "baggage_removal_not_allowed",
                "Checked bags can be added but cannot be removed.",
            )
        if args.nonfree_baggages < reservation.nonfree_baggages:
            raise AirlineBusinessError(
                "nonfree_baggage_removal_not_allowed",
                "Non-free checked bags cannot be removed.",
            )
        if args.nonfree_baggages > args.total_baggages:
            raise AirlineBusinessError(
                "invalid_baggage_counts",
                "Non-free checked bags cannot exceed total checked bags.",
            )

        user = self._user(reservation.user_id)
        working_user = user.model_copy(deep=True)
        working = reservation.model_copy(deep=True)
        added_nonfree_bags = args.nonfree_baggages - reservation.nonfree_baggages
        price_delta = 50 * added_nonfree_bags
        self._check_payment(working_user, args.payment_id, price_delta)

        if price_delta:
            method = working_user.payment_methods[args.payment_id]
            if method.source == "certificate":
                raise AirlineBusinessError(
                    "certificate_not_allowed",
                    "A certificate cannot be used to update a reservation.",
                )
            if method.source == "gift_card":
                method.amount -= price_delta
            working.payment_history.append(
                Payment(payment_id=args.payment_id, amount=price_delta)
            )

        working.total_baggages = args.total_baggages
        working.nonfree_baggages = args.nonfree_baggages
        self._environment.save_user(working_user)
        self._environment.save_reservation(working)
        return working.model_dump(mode="json")

    def update_reservation_flights(
        self,
        args: UpdateReservationFlightsArgs,
    ) -> dict[str, Any]:
        reservation = self._reservation(args.reservation_id)
        user = self._user(reservation.user_id)
        working = reservation.model_copy(deep=True)
        total_price = 0
        new_flights: list[ReservationFlight] = []

        for selected in args.flights:
            existing = next(
                (
                    item
                    for item in reservation.flights
                    if item.flight_number == selected.flight_number
                    and item.date == selected.date
                    and args.cabin == reservation.cabin
                ),
                None,
            )
            if existing is not None:
                new_flights.append(existing)
                total_price += existing.price * len(reservation.passengers)
                continue

            flight = self._environment.get_flight(selected.flight_number)
            if flight is None:
                raise AirlineBusinessError(
                    "flight_not_found", f"Flight {selected.flight_number} does not exist."
                )
            flight_date = flight.dates.get(selected.date)
            if not isinstance(flight_date, AvailableFlightDate):
                raise AirlineBusinessError(
                    "flight_not_available",
                    f"Flight {selected.flight_number} is unavailable on that date.",
                )
            if flight_date.available_seats[args.cabin] < len(reservation.passengers):
                raise AirlineBusinessError(
                    "not_enough_seats",
                    f"Flight {selected.flight_number} does not have enough seats.",
                )
            price = flight_date.prices[args.cabin]
            new_flights.append(
                ReservationFlight(
                    flight_number=selected.flight_number,
                    date=selected.date,
                    origin=flight.origin,
                    destination=flight.destination,
                    price=price,
                )
            )
            total_price += price * len(reservation.passengers)

        old_price = sum(item.price for item in reservation.flights) * len(reservation.passengers)
        price_delta = total_price - old_price
        working_user = user.model_copy(deep=True)
        self._check_payment(working_user, args.payment_id, price_delta)
        if price_delta:
            method = working_user.payment_methods[args.payment_id]
            if method.source == "gift_card":
                # A negative delta is a refund and therefore restores gift-card balance.
                method.amount -= price_delta
            elif method.source == "certificate":
                raise AirlineBusinessError(
                    "certificate_not_allowed",
                    "A certificate cannot be used to modify a reservation.",
                )
            working.payment_history.append(Payment(payment_id=args.payment_id, amount=price_delta))

        working.flights = new_flights
        working.cabin = args.cabin
        self._environment.save_user(working_user)
        self._environment.save_reservation(working)
        return working.model_dump(mode="json")
