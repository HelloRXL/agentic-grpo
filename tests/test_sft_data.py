import json
from pathlib import Path

import pytest

from airline_agent.sft_data import (
    _has_clean_sft_process,
    build_sft_dataset,
    collect_sft_rows,
    split_rows_by_task,
)


ROOT = Path(__file__).resolve().parents[1]


def _record(task_id: str, accepted: bool = True) -> dict:
    return {
        "rollout": {
            "task_id": task_id,
            "termination_reason": "finished",
            "messages": [
                {"role": "system", "content": "tools"},
                {"role": "assistant", "content": '{"action_type":"finish"}'},
            ],
        },
        "evaluation": {"sft_accepted": accepted},
    }


def test_build_sft_dataset_only_keeps_accepted_train_records(tmp_path):
    tasks = ROOT / "data/tasks/train.jsonl"
    task_ids = [json.loads(line)["task_id"] for line in tasks.read_text().splitlines() if line]
    records = tmp_path / "records"
    records.mkdir()
    (records / "accepted.json").write_text(json.dumps(_record(task_ids[0])), encoding="utf-8")
    (records / "rejected.json").write_text(json.dumps(_record(task_ids[1], accepted=False)), encoding="utf-8")

    stats = build_sft_dataset(
        records_dir=records,
        tasks_path=tasks,
        output_dir=tmp_path / "sft",
        validation_ratio=0,
    )

    assert stats.accepted_records == 1
    assert stats.skipped_not_accepted == 1
    rows = [json.loads(line) for line in (tmp_path / "sft/train.jsonl").read_text().splitlines()]
    assert [row["task_id"] for row in rows] == [task_ids[0]]
    assert all(message["role"] != "tool" for message in rows[0]["messages"])


def test_sft_data_rejects_test_task_even_if_record_claims_acceptance(tmp_path):
    task = json.loads((ROOT / "data/tasks/test.jsonl").read_text().splitlines()[0])
    records = tmp_path / "records"
    records.mkdir()
    (records / "wrong-split.json").write_text(json.dumps(_record(task["task_id"])), encoding="utf-8")

    with pytest.raises(ValueError, match="不在指定 train 任务集"):
        collect_sft_rows(records, ROOT / "data/tasks/train.jsonl")


def test_sft_data_rechecks_stale_acceptance_when_a_parse_error_exists(tmp_path):
    task = json.loads((ROOT / "data/tasks/train.jsonl").read_text().splitlines()[0])
    record = _record(task["task_id"])
    record["rollout"]["steps"] = [{"parse_error": "invalid JSON", "action": None}]
    records = tmp_path / "records"
    records.mkdir()
    (records / "stale-accepted.json").write_text(json.dumps(record), encoding="utf-8")

    rows, total, skipped = collect_sft_rows(records, ROOT / "data/tasks/train.jsonl")

    assert rows == []
    assert total == 1
    assert skipped == 1


def test_clean_process_allows_requery_after_successful_write():
    query = {
        "action_type": "tool",
        "tool_name": "get_reservation_details",
        "arguments": {"reservation_id": "ABC123"},
    }
    write = {
        "action_type": "tool",
        "tool_name": "update_reservation_baggages",
        "arguments": {"reservation_id": "ABC123", "total_baggages": 2},
    }
    success = {"success": True}
    assert _has_clean_sft_process({"steps": [
        {"action": query, "observation": success},
        {"action": write, "observation": success},
        {"action": query, "observation": success},
    ]})


def test_split_rows_by_task_keeps_one_task_whole_in_validation():
    rows = [
        {"task_id": "a", "messages": []},
        {"task_id": "a", "messages": []},
        {"task_id": "b", "messages": []},
        {"task_id": "c", "messages": []},
    ]
    train_rows, validation_rows = split_rows_by_task(rows, validation_ratio=0.34, seed=42)

    train_ids = {row["task_id"] for row in train_rows}
    validation_ids = {row["task_id"] for row in validation_rows}
    assert train_ids.isdisjoint(validation_ids)
    assert validation_ids
