from pathlib import Path

from airline_agent.evaluator import Evaluator
from airline_agent.tasks.v8_variants import (
    V8_DEFINITIONS,
    build_v8_tasks,
    load_tasks,
    validate_v8_tasks,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_TASKS = ROOT / "data/tasks/train.jsonl"
DATABASE = ROOT / "../reference-repos/tau2-bench-main/data/tau2/domains/airline/db.json"


def test_v8_variants_are_train_only_and_preserve_replay_contract(tmp_path: Path) -> None:
    parents = load_tasks(TRAIN_TASKS)
    variants = build_v8_tasks(parents)

    assert len(variants) == len(V8_DEFINITIONS) == 12
    assert all(task.split == "train" and task.status == "supported" for task in variants)
    assert all(task.task_id.startswith("tau2-airline-v8-") for task in variants)
    assert all(task.source_payload["agentic_variant"]["suite"] == "v8" for task in variants)
    assert {task.source_task_id for task in variants} == {
        definition.source_task_id for definition in V8_DEFINITIONS
    }

    output = tmp_path / "v8_train.jsonl"
    write_jsonl(output, variants)
    assert len(load_tasks(output)) == 12


def test_v8_reference_replay_remains_executable() -> None:
    variants = build_v8_tasks(load_tasks(TRAIN_TASKS))
    evaluator = Evaluator(DATABASE.resolve())

    for task in variants:
        replay = evaluator.replay_reference(task)
        assert replay.success, task.task_id


def test_v8_validation_rejects_database_contract_drift() -> None:
    parents = load_tasks(TRAIN_TASKS)
    variants = build_v8_tasks(parents)
    drifted = variants[0].model_copy(update={"database_sha256": "not-the-parent"})

    try:
        validate_v8_tasks([drifted, *variants[1:]], parents)
    except ValueError as error:
        assert "替换数据库" in str(error)
    else:
        raise AssertionError("database drift must be rejected")
