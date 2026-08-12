import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..domain.tool_schemas import TOOL_ARGUMENT_SCHEMAS
from .spec import ReferenceAction, TaskSpec, UserScenarioView


TOOL_NAME_MAP = {
    "get_user_details": "get_user_details",
    "get_reservation_details": "get_reservation_details",
    # 官方名称本身已经清晰，内部先保持一致，避免无必要的名称转换。
    "search_direct_flight": "search_direct_flight",
    "search_onestop_flight": "search_onestop_flight",
    "get_flight_status": "get_flight_status",
    "list_all_airports": "list_all_airports",
    "transfer_to_human_agents": "transfer_to_human_agents",
    "book_reservation": "book_reservation",
    "cancel_reservation": "cancel_reservation",
    "update_reservation_flights": "update_reservation_flights",
    "update_reservation_baggages": "update_reservation_baggages",
}


def file_sha256(path: Path) -> str:
    """计算输入文件哈希，用于记录数据版本。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scenario_view(raw_task: dict[str, Any]) -> UserScenarioView:
    scenario = raw_task.get("user_scenario") or {}
    raw_instructions = scenario.get("instructions") or {}

    if isinstance(raw_instructions, str): # 判断 instructions 是不是字符串
        return UserScenarioView(
            persona=scenario.get("persona"),
            domain="airline",
            reason_for_call=raw_instructions,
            task_instructions=raw_instructions,
        )

    return UserScenarioView(
        persona=scenario.get("persona"),
        domain=raw_instructions.get("domain", "airline"), # 默认领域为 "airline"
        reason_for_call=raw_instructions.get("reason_for_call", ""),
        known_info=raw_instructions.get("known_info"),
        unknown_info=raw_instructions.get("unknown_info"),
        task_instructions=raw_instructions.get("task_instructions", ""),
    )


def convert_task(
    raw_task: dict[str, Any],
    *,
    split: str,
    source_version: str,
    database_path: Path,
) -> TaskSpec:
    """将一条官方任务转换成内部 TaskSpec，不修改 raw_task。"""

    if split not in {"train", "dev", "test", "base"}:
        raise ValueError(f"不支持的任务分割：{split}")

    evaluation = raw_task.get("evaluation_criteria") or {}
    raw_actions = evaluation.get("actions") or []  
    reference_actions: list[ReferenceAction] = [] # Replay 推导目标状态的参考工具动作
    unsupported_reasons: list[str] = [] # 不支持的原因
    warnings: list[str] = [] 

    for step_index, raw_action in enumerate(raw_actions, start=1):
        original_name = raw_action["name"]
        mapped_name = TOOL_NAME_MAP.get(original_name)
        if mapped_name is None:
            unsupported_reasons.append(
                f"参考动作 {original_name} 不在第一版核心工具集合中"
            )
            mapped_name = original_name
        else:
            argument_schema = TOOL_ARGUMENT_SCHEMAS[mapped_name]
            try:
                argument_schema.model_validate(raw_action.get("arguments") or {})
            except ValidationError as error:
                unsupported_reasons.append(
                    f"参考动作 {original_name} 的参数不符合当前 Schema："
                    f"{error.errors()[0]['loc']}"
                )

        reference_actions.append(
            ReferenceAction(
                step_index=step_index,
                action_id=raw_action.get("action_id", f"{raw_task['id']}_{step_index}"),
                original_tool_name=original_name,
                tool_name=mapped_name,
                arguments=deepcopy(raw_action.get("arguments") or {}),
                info=raw_action.get("info"),
            )
        )

    if raw_task.get("initial_state") is not None:
        warnings.append(
            "官方任务包含 initial_state；当前转换器只记录来源，尚未执行初始化动作"
        )

    scenario = _scenario_view(raw_task)
    return TaskSpec(
        task_id=f"tau2-airline-{raw_task['id']}",
        source_task_id=str(raw_task["id"]),
        source_version=source_version,
        split=split,  # type: ignore[arg-type]
        status="unsupported" if unsupported_reasons else "supported",
        visible_request=scenario.reason_for_call,
        user_scenario=scenario,
        reference_actions=reference_actions,
        communicate_info=deepcopy(evaluation.get("communicate_info") or []),
        nl_assertions=deepcopy(evaluation.get("nl_assertions") or []),
        reward_basis=deepcopy(evaluation.get("reward_basis") or []),
        database_path=str(database_path),
        database_sha256=file_sha256(database_path),
        conversion_warnings=warnings,
        unsupported_reasons=unsupported_reasons,
        source_payload=deepcopy(raw_task),
    )


def convert_dataset(
    tasks_path: Path,
    split_path: Path,
    database_path: Path,
    *,
    source_version: str = "local-reference",
) -> dict[str, list[TaskSpec]]:
    """按官方 split 转换全部任务，返回不修改原始 JSON 的内部对象。"""

    raw_tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    raw_splits = json.loads(split_path.read_text(encoding="utf-8"))
    tasks_by_id = {
        str(task["id"]): task 
        for task in raw_tasks}
    converted: dict[str, list[TaskSpec]] = {}

    for split_name in ("train", "test", "base"):
        converted[split_name] = [
            convert_task(
                tasks_by_id[str(task_id)],
                split=split_name,
                source_version=source_version,
                database_path=database_path,
            )
            for task_id in raw_splits[split_name]
        ]

    return converted
