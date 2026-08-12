"""从多批内部 Teacher rollout 构建可复现的 SFT 数据源清单。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent.prompts import build_agent_system_prompt
from .core.rollout import RolloutRecord
from .domain.runtime import create_airline_runtime
from .environment_reward import compute_environment_reward
from .sft_data import _has_clean_sft_process, _validated_messages, load_task_specs


@dataclass(frozen=True)
class InternalPoolStats:
    current_clean_records: int
    legacy_clean_records: int
    legacy_added_records: int
    selected_records: int
    unique_action_sequences: int
    covered_tasks: int
    refreshed_acceptances: int


def _load_task_specs_from_paths(paths: list[Path]) -> dict[str, Any]:
    """合并多个 train-only TaskSpec 文件，供官方任务与受控变体共用。"""

    tasks: dict[str, Any] = {}
    for path in paths:
        for task_id, task in load_task_specs(path).items():
            if task_id in tasks:
                raise ValueError(f"重复 task_id: {task_id} ({path})")
            tasks[task_id] = task
    return tasks


def _canonical_action_signature(rollout: dict[str, Any]) -> str:
    """忽略自然语言措辞，仅比较动作类型、工具名和参数。

    这与 2026-08-03 rollout audit 中的 ``canonical tool-action sequence`` 定义一致。
    当前批次内部不去重，以保留同策略的独立采样；只用它阻止旧批次重复补充。
    """

    actions: list[tuple[Any, Any, Any]] = []
    for step in rollout["steps"]:
        action = step.get("action")
        if not isinstance(action, dict):
            continue
        actions.append((action.get("action_type"), action.get("tool_name"), action.get("arguments")))
    return json.dumps(actions, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _acceptance_from_stored_evaluation(task: Any, record: dict[str, Any]) -> tuple[bool, bool]:
    """兼容 reward 语义升级后的历史记录，不重调 Judge 或模型。"""

    rollout_payload = record["rollout"]
    evaluation = record["evaluation"]
    if evaluation.get("sft_accepted"):
        return True, False
    # serialize_result 会移除完整 state；新 reward 只需下面三个已审计布尔量。
    rollout = RolloutRecord.model_validate({
        **rollout_payload,
        "initial_state": {},
        "final_state": {},
    })
    reward = compute_environment_reward(
        task,
        rollout,
        replay_success=bool(evaluation.get("replay_success")),
        initial_state_match=bool(evaluation.get("initial_state_match")),
        final_state_match=bool(evaluation.get("final_state_match")),
    )
    verification = evaluation.get("communication_verification")
    judge_pass = isinstance(verification, dict) and verification.get("passed") is True
    action_required = "ACTION" in task.reward_basis
    accepted = (
        rollout.termination_reason == "finished"
        and reward.success
        and reward.invalid_action_count == 0
        and reward.repeated_action_count == 0
        and (not action_required or bool(evaluation.get("action_ok")))
        and judge_pass
    )
    return accepted, accepted


def _load_clean_records(
    records_dirs: list[Path],
    tasks: dict[str, Any],
) -> list[tuple[str, Path, dict[str, Any], bool]]:
    records: list[tuple[str, Path, dict[str, Any], bool]] = []
    for records_dir in records_dirs:
        for source in sorted(records_dir.glob("*.json")):
            if source.name == "summary.json":
                continue
            record = json.loads(source.read_text(encoding="utf-8"))
            rollout = record.get("rollout")
            evaluation = record.get("evaluation")
            if not isinstance(rollout, dict) or not isinstance(evaluation, dict):
                raise ValueError(f"{source}: 缺少 rollout 或 evaluation")
            task = tasks.get(rollout.get("task_id"))
            if task is None or task.status != "supported":
                continue
            accepted, refreshed = _acceptance_from_stored_evaluation(task, record)
            if not accepted or not _has_clean_sft_process(rollout):
                continue
            records.append((records_dir.name, source, record, refreshed))
    return records


def _source_provenance(source: Path) -> dict[str, str | None]:
    """读取 rollout run 与模型标识；兼容早期没有 summary metadata 的批次。"""
    summary_path = source.parent / "summary.json"
    model: str | None = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        value = summary.get("agent_model")
        if isinstance(value, str) and value:
            model = value
    # .../outputs/<run>/teacher/train/<record>.json
    source_run = source.parents[2].name
    return {"source_run": source_run, "source_model": model}


def build_internal_teacher_pool(
    *,
    current_dirs: list[Path],
    legacy_dirs: list[Path],
    tasks_paths: list[Path],
    database_path: Path,
    output_dir: Path,
    output_name: str = "internal_teacher_v1",
    source_type: str = "internal_teacher",
) -> InternalPoolStats:
    if not output_name or Path(output_name).name != output_name:
        raise ValueError("output_name 必须是无路径的文件名 stem")
    tasks = _load_task_specs_from_paths(tasks_paths)
    current = _load_clean_records(current_dirs, tasks)
    legacy = _load_clean_records(legacy_dirs, tasks)
    selected = list(current)
    seen_signatures = {_canonical_action_signature(record["rollout"]) for _, _, record, _ in current}
    legacy_added = 0
    for item in legacy:
        signature = _canonical_action_signature(item[2]["rollout"])
        if signature in seen_signatures:
            continue
        selected.append(item)
        seen_signatures.add(signature)
        legacy_added += 1

    runtime = create_airline_runtime(database_path)
    current_system_prompt = build_agent_system_prompt(runtime.registry.get_tool_definitions())
    rows: list[dict[str, Any]] = []
    for source_batch, source, record, refreshed in selected:
        messages = _validated_messages(record, source)
        if messages[0]["role"] != "system":
            raise ValueError(f"{source}: rollout messages 必须以 system 开始")
        messages[0] = {"role": "system", "content": current_system_prompt}
        rollout = record["rollout"]
        provenance = _source_provenance(source)
        rows.append(
            {
                "task_id": rollout["task_id"],
                "source_type": source_type,
                "source_batch": source_batch,
                **provenance,
                "source_record": source.name,
                "sft_acceptance_refreshed": refreshed,
                "canonical_action_signature": _canonical_action_signature(rollout),
                "messages": messages,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{output_name}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    stats = InternalPoolStats(
        current_clean_records=len(current),
        legacy_clean_records=len(legacy),
        legacy_added_records=legacy_added,
        selected_records=len(rows),
        unique_action_sequences=len({_canonical_action_signature(item[2]["rollout"]) for item in selected}),
        covered_tasks=len({row["task_id"] for row in rows}),
        refreshed_acceptances=sum(item[3] for item in selected),
    )
    (output_dir / f"{output_name}.manifest.json").write_text(
        json.dumps(
            {
                **stats.__dict__,
                "selection_rule": "keep every clean current record; append a clean legacy record only when its canonical action signature is absent from all selected records",
                "system_prompt": "canonicalized_to_current_runtime",
                "task_split": "train_only",
                "tasks_paths": [str(path.resolve()) for path in tasks_paths],
                "current_dirs": [str(path.resolve()) for path in current_dirs],
                "legacy_dirs": [str(path.resolve()) for path in legacy_dirs],
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-dir", type=Path, action="append", required=True)
    parser.add_argument("--legacy-dir", type=Path, action="append")
    parser.add_argument("--tasks", type=Path, action="append")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft_sources"))
    parser.add_argument("--output-name", default="internal_teacher_v1")
    parser.add_argument("--source-type", default="internal_teacher")
    args = parser.parse_args()
    stats = build_internal_teacher_pool(
        current_dirs=args.current_dir,
        legacy_dirs=args.legacy_dir or [],
        tasks_paths=args.tasks or [Path("data/tasks/train.jsonl")],
        database_path=args.database,
        output_dir=args.output_dir,
        output_name=args.output_name,
        source_type=args.source_type,
    )
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
