import json
from copy import deepcopy
from pathlib import Path

from airline_agent.tasks.converter import convert_dataset, convert_task


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "reference-repos" / "tau2-bench-main" / "data" / "tau2" / "domains" / "airline"


def test_converter_preserves_source_and_split_counts() -> None:
    raw_tasks = json.loads((DATA_DIR / "tasks.json").read_text(encoding="utf-8"))
    converted = convert_dataset(
        DATA_DIR / "tasks.json",
        DATA_DIR / "split_tasks.json",
        DATA_DIR / "db.json",
    )

    assert len(converted["train"]) == 30
    assert len(converted["test"]) == 20
    assert len(converted["base"]) == 50
    assert converted["train"][0].source_payload == raw_tasks[0]
    assert converted["train"][0].visible_request


def test_converter_preserves_supported_transfer_action_without_rewriting_it() -> None:
    converted = convert_dataset(
        DATA_DIR / "tasks.json",
        DATA_DIR / "split_tasks.json",
        DATA_DIR / "db.json",
    )
    task_13 = next(task for task in converted["base"] if task.source_task_id == "13")

    assert task_13.status == "supported"
    assert task_13.reference_actions[0].original_tool_name == "transfer_to_human_agents"
    assert task_13.reference_actions[0].tool_name == "transfer_to_human_agents"


def test_converter_separates_agent_request_from_user_script() -> None:
    converted = convert_dataset(
        DATA_DIR / "tasks.json",
        DATA_DIR / "split_tasks.json",
        DATA_DIR / "db.json",
    )
    task_8 = next(task for task in converted["base"] if task.source_task_id == "8")

    assert task_8.visible_request == task_8.user_scenario.reason_for_call
    assert task_8.user_scenario.task_instructions not in task_8.visible_request


def test_converter_reports_core_tool_coverage() -> None:
    converted = convert_dataset(
        DATA_DIR / "tasks.json",
        DATA_DIR / "split_tasks.json",
        DATA_DIR / "db.json",
    )

    assert sum(task.status == "supported" for task in converted["base"]) == 46
    assert sum(task.status == "supported" for task in converted["train"]) == 27
    assert sum(task.status == "supported" for task in converted["test"]) == 19


def test_converter_marks_invalid_arguments_as_unsupported() -> None:
    raw_tasks = json.loads((DATA_DIR / "tasks.json").read_text(encoding="utf-8"))
    raw_task = deepcopy(next(task for task in raw_tasks if task["id"] == "8"))
    raw_task["evaluation_criteria"]["actions"][2]["arguments"]["unexpected"] = True

    converted = convert_task(
        raw_task,
        split="train",
        source_version="test",
        database_path=DATA_DIR / "db.json",
    )

    assert converted.status == "unsupported"
    assert "参数不符合当前 Schema" in converted.unsupported_reasons[0]
