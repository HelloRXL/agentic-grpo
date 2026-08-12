"""单个 Airline episode 的状态容器。"""

from copy import deepcopy
from pathlib import Path

from .models import (
    AirlineDatabase,
    Flight,
    FlightDateStatus,
    Reservation,
    User,
)


class AirlineEnvironment:
    """管理一个 episode 的数据库状态，不负责调用 LLM 或决定业务策略。"""

    def __init__(self, database: AirlineDatabase) -> None:
        self._initial_database = database.model_copy(deep=True)
        self._database = database.model_copy(deep=True)

    @classmethod
    def from_json(cls, path: Path) -> "AirlineEnvironment":
        """从官方或转换后的 db.json 创建环境。"""

        database = AirlineDatabase.model_validate_json(path.read_text(encoding="utf-8"))
        return cls(database)

    def reset(self) -> None:
        """恢复 episode 开始时的数据库快照。"""

        self._database = self._initial_database.model_copy(deep=True)

    def snapshot(self) -> AirlineDatabase:
        """返回隔离副本，防止调用方绕过工具修改内部状态。"""

        return self._database.model_copy(deep=True)

    def get_user(self, user_id: str) -> User | None:
        user = self._database.users.get(user_id)
        return user.model_copy(deep=True) if user is not None else None

    def get_reservation(self, reservation_id: str) -> Reservation | None:
        reservation = self._database.reservations.get(reservation_id)
        return reservation.model_copy(deep=True) if reservation is not None else None

    def get_flight(self, flight_number: str) -> Flight | None:
        flight = self._database.flights.get(flight_number)
        return flight.model_copy(deep=True) if flight is not None else None

    def save_user(self, user: User) -> None:
        """保存一个经过 Pydantic 验证的用户状态。"""

        self._database.users[user.user_id] = user.model_copy(deep=True)

    def save_reservation(self, reservation: Reservation) -> None:
        """保存一个经过 Pydantic 验证的订单状态。"""

        self._database.reservations[reservation.reservation_id] = (
            reservation.model_copy(deep=True)
        )

    def save_flight(self, flight: Flight) -> None:
        """保存一个经过 Pydantic 验证的航班状态。"""

        self._database.flights[flight.flight_number] = flight.model_copy(deep=True)

    def search_direct_flights(
        self,
        origin: str,
        destination: str,
        date: str,
    ) -> list[tuple[Flight, FlightDateStatus]]:
        """查找指定日期、航线和 available 状态的直达航班。"""

        results: list[tuple[Flight, FlightDateStatus]] = []
        for flight in self._database.flights.values():
            if flight.origin != origin or flight.destination != destination:
                continue
            flight_date = flight.dates.get(date)
            if flight_date is None or flight_date.status != "available":
                continue
            results.append((flight.model_copy(deep=True), deepcopy(flight_date)))
        return results
