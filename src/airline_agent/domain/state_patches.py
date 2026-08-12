"""训练专用状态变体的、类型受限的初始数据库 patch。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from .models import AirlineDatabase, CabinClass, Insurance


class SetReservationInsurancePatch(BaseModel):
    """仅允许修改已有订单的 insurance 字段，不能任意写入数据库。"""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["set_reservation_insurance"]
    reservation_id: str = Field(min_length=1)
    insurance: Insurance


class SetReservationCabinPatch(BaseModel):
    """仅允许修改已有订单的 cabin 字段，用于政策资格的反事实对。"""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["set_reservation_cabin"]
    reservation_id: str = Field(min_length=1)
    cabin: CabinClass


InitialStatePatch = Annotated[
    Union[SetReservationInsurancePatch, SetReservationCabinPatch],
    Field(discriminator="kind"),
]


def database_state_sha256(database: AirlineDatabase) -> str:
    """对类型化数据库做稳定哈希，作为 patch 后 episode 初态的契约。"""

    canonical = json.dumps(
        database.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def apply_initial_state_patches(
    database: AirlineDatabase,
    patches: Sequence[InitialStatePatch],
) -> AirlineDatabase:
    """返回独立的 patch 后数据库；拒绝未知记录和重复字段写入。"""

    patched = database.model_copy(deep=True)
    touched: set[tuple[str, str]] = set()
    for patch in patches:
        if isinstance(patch, SetReservationInsurancePatch):
            field_name, value = "insurance", patch.insurance
        elif isinstance(patch, SetReservationCabinPatch):
            field_name, value = "cabin", patch.cabin
        else:  # pragma: no cover - discriminated union ensures this at validation time.
            raise TypeError(f"不支持的初始状态 patch：{type(patch).__name__}")
        target = (patch.reservation_id, field_name)
        if target in touched:
            raise ValueError(f"重复修改初始状态字段：{target}")
        touched.add(target)
        reservation = patched.reservations.get(patch.reservation_id)
        if reservation is None:
            raise ValueError(f"初始状态 patch 的 reservation 不存在：{patch.reservation_id}")
        patched.reservations[patch.reservation_id] = reservation.model_copy(
            update={field_name: value}
        )
    # 通过完整 Pydantic round-trip 确保任何后续 patch 扩展都不能留下非法数据库。
    return AirlineDatabase.model_validate(patched.model_dump(mode="json"))
