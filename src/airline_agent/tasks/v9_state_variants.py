"""V9：用声明式定义构造可重放的初始状态 counterfactual 任务。"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from ..domain.models import AirlineDatabase
from ..domain.state_patches import (
    InitialStatePatch,
    SetReservationCabinPatch,
    SetReservationInsurancePatch,
    apply_initial_state_patches,
    database_state_sha256,
)
from .spec import ReferenceAction, TaskSpec
from .v8_variants import load_tasks, write_jsonl


V9_SOURCE_VERSION = "agentic-v9-state-counterfactual-2026-08-03"


@dataclass(frozen=True)
class StateVariantDefinition:
    """一行定义一个状态分支；新增任务只需添加定义，不复制构建器。"""

    suite: str
    pair: str
    parent_source_task_id: str
    slug: str
    # None 表示逐字继承父任务；状态 counterfactual 只能改初态，不能顺手改约束。
    reason_for_call: str | None
    task_instructions: str | None
    patches: tuple[InitialStatePatch, ...]
    # None=继承父任务 reference；空 tuple=目标是“不写入”。
    reference_actions: tuple[ReferenceAction, ...] | None
    nl_assertions: tuple[str, ...] | None
    expected_state_change: bool


def _action(action_id: str, reservation_id: str, arguments: dict) -> ReferenceAction:
    return ReferenceAction(
        step_index=1,
        action_id=action_id,
        original_tool_name="update_reservation_flights"
        if "flights" in arguments
        else "cancel_reservation",
        tool_name="update_reservation_flights"
        if "flights" in arguments
        else "cancel_reservation",
        arguments=arguments,
    )


_HEALTH_REQUEST = (
    "I need to cancel reservation EHGLP3 because a passenger has a medical emergency. I would like a full refund.",
    "Provide your user ID if asked. State clearly that the reason is a medical emergency. "
    "If the agent confirms the cancellation is eligible, explicitly confirm that you want to proceed. "
    "If it is not eligible, do not invent another reason; ask that the reservation remain unchanged.",
)
def _cancel_action() -> ReferenceAction:
    return _action("v9_cancel_health_insured_1", "EHGLP3", {"reservation_id": "EHGLP3"})


V9_DEFINITIONS: tuple[StateVariantDefinition, ...] = (
    StateVariantDefinition(
        "v9-a", "health-cancel", "0", "cancel-health-uninsured", *_HEALTH_REQUEST,
        (), (),
        ("Agent should refuse to cancel EHGLP3 because it does not have insurance.",), False,
    ),
    StateVariantDefinition(
        "v9-a", "health-cancel", "0", "cancel-health-insured", *_HEALTH_REQUEST,
        (SetReservationInsurancePatch(kind="set_reservation_insurance", reservation_id="EHGLP3", insurance="yes"),),
        (_cancel_action(),),
        ("Agent should cancel EHGLP3 after verifying the medical reason and insurance.",), True,
    ),
    StateVariantDefinition(
        "v9-b", "date-cabin-hxdubj", "33", "change-dates-economy", None, None,
        (), None, None, True,
    ),
    StateVariantDefinition(
        "v9-b", "date-cabin-hxdubj", "33", "change-dates-basic-economy", None, None,
        (SetReservationCabinPatch(kind="set_reservation_cabin", reservation_id="HXDUBJ", cabin="basic_economy"),),
        (),
        ("Agent should refuse the HXDUBJ date change because it is basic economy.",), False,
    ),
    StateVariantDefinition(
        "v9-c", "date-cabin-m05knl", "15", "change-dates-business", None, None,
        (), None, None, True,
    ),
    StateVariantDefinition(
        "v9-c", "date-cabin-m05knl", "15", "change-dates-basic-economy", None, None,
        (SetReservationCabinPatch(kind="set_reservation_cabin", reservation_id="M05KNL", cabin="basic_economy"),),
        (),
        ("Agent should refuse the M05KNL change because it is basic economy.",), False,
    ),
)


def _build_variant(parent: TaskSpec, definition: StateVariantDefinition, database: AirlineDatabase) -> TaskSpec:
    variant_id = f"tau2-airline-v9-{definition.parent_source_task_id}-{definition.slug}"
    patched_database = apply_initial_state_patches(database, definition.patches)
    state_hash = database_state_sha256(patched_database)
    payload = deepcopy(parent.source_payload)
    payload["agentic_variant"] = {
        "suite": definition.suite,
        "pair": definition.pair,
        "variant_id": variant_id,
        "parent_task_id": parent.task_id,
        "parent_source_task_id": parent.source_task_id,
        "variant_type": "state_counterfactual",
        "base_database_sha256": parent.database_sha256,
        "patches": [patch.model_dump(mode="json") for patch in definition.patches],
        "initial_state_sha256": state_hash,
        "expected_state_change": definition.expected_state_change,
    }
    scenario_update = {}
    if definition.reason_for_call is not None:
        scenario_update["reason_for_call"] = definition.reason_for_call
    if definition.task_instructions is not None:
        scenario_update["task_instructions"] = definition.task_instructions
    scenario = parent.user_scenario.model_copy(update=scenario_update)
    reference_actions = parent.reference_actions if definition.reference_actions is None else list(definition.reference_actions)
    nl_assertions = parent.nl_assertions if definition.nl_assertions is None else list(definition.nl_assertions)
    return parent.model_copy(deep=True, update={
        "task_id": variant_id,
        "source_version": V9_SOURCE_VERSION,
        "split": "train",
        "visible_request": definition.reason_for_call or parent.visible_request,
        "user_scenario": scenario,
        "reference_actions": reference_actions,
        "nl_assertions": nl_assertions,
        "initial_state_patches": list(definition.patches),
        "initial_state_sha256": state_hash,
        "source_payload": payload,
    })


def validate_v9_tasks(variants: list[TaskSpec], source_tasks: list[TaskSpec], database: AirlineDatabase, suite: str) -> None:
    definitions = [d for d in V9_DEFINITIONS if d.suite == suite]
    if len(variants) != len(definitions) or len({t.task_id for t in variants}) != len(variants):
        raise ValueError(f"{suite} 任务数量或 ID 重复")
    parents = {t.source_task_id: t for t in source_tasks}
    hashes_by_pair: dict[str, set[str]] = {}
    for task, definition in zip(variants, definitions, strict=True):
        parent = parents.get(definition.parent_source_task_id)
        if parent is None or parent.status != "supported" or parent.split != "train":
            raise ValueError(f"{task.task_id}: 父任务必须是 supported train")
        metadata = task.source_payload.get("agentic_variant")
        if not isinstance(metadata, dict) or metadata.get("suite") != suite:
            raise ValueError(f"{task.task_id}: provenance 不匹配")
        if task.database_sha256 != parent.database_sha256:
            raise ValueError(f"{task.task_id}: 不允许替换 base database")
        expected_hash = database_state_sha256(apply_initial_state_patches(database, definition.patches))
        if task.initial_state_sha256 != expected_hash:
            raise ValueError(f"{task.task_id}: initial state hash 不匹配")
        if bool(task.reference_actions) != definition.expected_state_change:
            raise ValueError(f"{task.task_id}: 目标动作与状态变化契约不匹配")
        hashes_by_pair.setdefault(definition.pair, set()).add(expected_hash)
    for pair, hashes in hashes_by_pair.items():
        if len(hashes) != 2:
            raise ValueError(f"{pair}: counterfactual pair 必须产生两个不同初始状态")


def build_v9_suite(source_tasks: list[TaskSpec], database_path: Path, suite: str) -> list[TaskSpec]:
    database = AirlineDatabase.model_validate_json(database_path.read_text(encoding="utf-8"))
    parents = {t.source_task_id: t for t in source_tasks}
    definitions = [d for d in V9_DEFINITIONS if d.suite == suite]
    variants = []
    for definition in definitions:
        parent = parents.get(definition.parent_source_task_id)
        if parent is None:
            raise ValueError(f"缺少父任务 {definition.parent_source_task_id}")
        variants.append(_build_variant(parent, definition, database))
    validate_v9_tasks(variants, source_tasks, database, suite)
    return variants


def build_v9_a_tasks(source_tasks: list[TaskSpec], database_path: Path) -> list[TaskSpec]:
    return build_v9_suite(source_tasks, database_path, "v9-a")


def build_v9_b_tasks(source_tasks: list[TaskSpec], database_path: Path) -> list[TaskSpec]:
    return build_v9_suite(source_tasks, database_path, "v9-b")


def build_v9_c_tasks(source_tasks: list[TaskSpec], database_path: Path) -> list[TaskSpec]:
    return build_v9_suite(source_tasks, database_path, "v9-c")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tasks", type=Path, default=Path("data/tasks/train.jsonl"))
    parser.add_argument("--database", type=Path, default=Path("../reference-repos/tau2-bench-main/data/tau2/domains/airline/db.json"))
    parser.add_argument("--suite", choices=("v9-a", "v9-b", "v9-c", "all"), default="all")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source_tasks = load_tasks(args.source_tasks)
    suites = ("v9-a", "v9-b", "v9-c") if args.suite == "all" else (args.suite,)
    variants = [task for suite in suites for task in build_v9_suite(source_tasks, args.database, suite)]
    output = args.output or Path("data/tasks/variants/v9_train.jsonl" if args.suite == "all" else f"data/tasks/variants/{args.suite.replace('-', '_')}_train.jsonl")
    if not args.check:
        write_jsonl(output, variants)
    print(json.dumps({"suites": suites, "variant_count": len(variants), "test_data_used": False, "output": str(output), "written": not args.check}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
