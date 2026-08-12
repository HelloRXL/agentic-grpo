from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


FlightType = Literal["round_trip", "one_way"]
CabinClass = Literal["business", "economy", "basic_economy"]
Insurance = Literal["yes", "no"]

MembershipLevel = Literal["gold", "silver", "regular"]


class Passenger(BaseModel):
    """订单中的一名乘客。"""

    first_name: str = Field(min_length=1)
    last_name: str = Field(min_length=1)
    dob: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="出生日期，格式为 YYYY-MM-DD。",
    )


class ReservationFlight(BaseModel):
    """订单中某一天的一段航班。"""

    flight_number: str = Field(min_length=1)
    origin: str = Field(min_length=3, max_length=3)
    destination: str = Field(min_length=3, max_length=3)
    date: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="航班日期，格式为 YYYY-MM-DD。",
    )
    price: int = Field(ge=0)


class Payment(BaseModel):
    """订单中的一条支付或退款记录。"""

    payment_id: str = Field(min_length=1)
    amount: int


class Name(BaseModel):
    first_name: str = Field(min_length=1)
    last_name: str = Field(min_length=1)


class Address(BaseModel):
    address1: str = Field(min_length=1)
    address2: str | None = None
    city: str = Field(min_length=1)
    country: str = Field(min_length=1)
    state: str = Field(min_length=1)
    zip: str = Field(min_length=1)


class CreditCard(BaseModel):
    source: Literal["credit_card"]
    id: str = Field(min_length=1)
    brand: str = Field(min_length=1)
    last_four: str = Field(min_length=4, max_length=4)


class GiftCard(BaseModel):
    source: Literal["gift_card"]
    id: str = Field(min_length=1)
    amount: float = Field(ge=0)


class Certificate(BaseModel):
    source: Literal["certificate"]
    id: str = Field(min_length=1)
    amount: float = Field(ge=0)


PaymentMethod = Annotated[
    Union[CreditCard, GiftCard, Certificate],
    Field(discriminator="source"),
]


class User(BaseModel):
    """航空系统中的用户及其保存的订单、乘客和支付方式。"""

    user_id: str = Field(min_length=1)
    name: Name
    address: Address
    email: str = Field(min_length=3)
    dob: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    payment_methods: dict[str, PaymentMethod]
    saved_passengers: list[Passenger]
    membership: MembershipLevel
    reservations: list[str]


class AvailableFlightDate(BaseModel):
    status: Literal["available"]
    available_seats: dict[CabinClass, int]
    prices: dict[CabinClass, int]


class OnTimeFlightDate(BaseModel):
    status: Literal["on time"]
    estimated_departure_time_est: str
    estimated_arrival_time_est: str


class FlyingFlightDate(BaseModel):
    status: Literal["flying"]
    actual_departure_time_est: str
    estimated_arrival_time_est: str


class LandedFlightDate(BaseModel):
    status: Literal["landed"]
    actual_departure_time_est: str
    actual_arrival_time_est: str


class CancelledFlightDate(BaseModel):
    status: Literal["cancelled"]


class DelayedFlightDate(BaseModel):
    status: Literal["delayed"]
    estimated_departure_time_est: str
    estimated_arrival_time_est: str


FlightDateStatus = Annotated[
    Union[
        AvailableFlightDate,
        OnTimeFlightDate,
        FlyingFlightDate,
        LandedFlightDate,
        CancelledFlightDate,
        DelayedFlightDate,
    ],
    Field(discriminator="status"),
]


class Flight(BaseModel):
    """航班的静态信息，以及每个日期对应的动态状态。"""

    flight_number: str = Field(min_length=1)
    origin: str = Field(min_length=3, max_length=3)
    destination: str = Field(min_length=3, max_length=3)
    scheduled_departure_time_est: str
    scheduled_arrival_time_est: str
    dates: dict[str, FlightDateStatus]


class Reservation(BaseModel):
    """A complete airline reservation compatible with tau2 Airline data."""

    reservation_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    origin: str = Field(min_length=3, max_length=3)
    destination: str = Field(min_length=3, max_length=3)
    flight_type: FlightType
    cabin: CabinClass
    flights: list[ReservationFlight]
    passengers: list[Passenger] = Field(min_length=1)
    payment_history: list[Payment]
    created_at: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$",
        description="Reservation creation time in ISO-8601 format.",
    )
    total_baggages: int = Field(ge=0)
    nonfree_baggages: int = Field(ge=0)
    insurance: Insurance
    status: Literal["cancelled"] | None = None


class AirlineDatabase(BaseModel):
    """一个航空环境 episode 的完整、类型化数据库快照。"""

    flights: dict[str, Flight]
    users: dict[str, User]
    reservations: dict[str, Reservation]

    def get_statistics(self) -> dict[str, int]:
        return {
            "num_flights": len(self.flights),
            "num_flight_dates": sum(
                len(flight.dates) for flight in self.flights.values()
            ),
            "num_users": len(self.users),
            "num_reservations": len(self.reservations),
        }
