"""合并已验收的内部与外部 SFT 源行，统一当前 prompt 并按任务划分训练/验证集。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .agent.prompts import build_agent_system_prompt
from .domain.runtime import create_airline_runtime
from .sft_data import _write_jsonl, load_task_specs, split_rows_by_task


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_task_specs_from_paths(paths: list[Path]) -> dict[str, Any]:
    """合并官方 train 与受控 train-only 变体，拒绝重复 task id。"""

    tasks: dict[str, Any] = {}
    for path in paths:
        for task_id, task in load_task_specs(path).items():
            if task_id in tasks:
                raise ValueError(f"重复 task_id: {task_id} ({path})")
            tasks[task_id] = task
    return tasks


def _signature(row: dict[str, Any]) -> str:
    existing = row.get("canonical_action_signature")
    if isinstance(existing, str) and existing:
        return existing
    actions = []
    for message in row["messages"]:
        if message.get("role") != "assistant":
            continue
        try:
            action = json.loads(message["content"])
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(action, dict) and "action_type" in action:
            actions.append((action.get("action_type"), action.get("tool_name"), action.get("arguments")))
    return json.dumps(actions, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _canonical_messages(messages: Any, system_prompt: str, source: Path) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{source}: 缺少 messages")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"{source}: 非对象 message")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"{source}: 非标准 chat message")
        normalized.append({"role": role, "content": content})
    if normalized[0]["role"] != "system":
        raise ValueError(f"{source}: messages 必须以 system 开始")
    if not any(message["role"] == "assistant" for message in normalized):
        raise ValueError(f"{source}: 没有 assistant 监督目标")
    normalized[0] = {"role": "system", "content": system_prompt}
    return normalized


def finalize_sft(
    *,
    source_paths: list[Path],
    tasks_paths: list[Path],
    database_path: Path,
    output_dir: Path,
    validation_ratio: float,
    seed: int,
    validation_task_ids: set[str] | None = None,
) -> dict[str, Any]:
    tasks = _load_task_specs_from_paths(tasks_paths)
    runtime = create_airline_runtime(database_path)
    system_prompt = build_agent_system_prompt(runtime.registry.get_tool_definitions())
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    source_counts: Counter[str] = Counter()
    for source_path in source_paths:
        for raw in _load_rows(source_path):
            task_id = raw.get("task_id")
            task = tasks.get(task_id)
            if task is None or task.status != "supported":
                raise ValueError(f"{source_path}: task_id={task_id!r} 不是 supported train task")
            messages = _canonical_messages(raw.get("messages"), system_prompt, source_path)
            signature = _signature(raw)
            if signature in seen:
                duplicates += 1
                continue
            seen.add(signature)
            row = {key: value for key, value in raw.items() if key != "messages"}
            row["canonical_action_signature"] = signature
            row["messages"] = messages
            selected.append(row)
            source_counts[str(raw.get("source_type", "unknown"))] += 1
    if not selected:
        raise ValueError("没有可写入最终 SFT 的样本")
    if validation_task_ids is None:
        train_rows, validation_rows = split_rows_by_task(selected, validation_ratio, seed)
        validation_strategy = "hashed_task_split"
    else:
        selected_task_ids = {row["task_id"] for row in selected}
        unknown = validation_task_ids - selected_task_ids
        if unknown:
            raise ValueError(f"验证任务不在当前 SFT 数据中: {sorted(unknown)}")
        train_rows = [row for row in selected if row["task_id"] not in validation_task_ids]
        validation_rows = [row for row in selected if row["task_id"] in validation_task_ids]
        validation_strategy = "explicit_frozen_task_ids"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "all_clean.jsonl", selected)
    _write_jsonl(output_dir / "train.jsonl", train_rows)
    _write_jsonl(output_dir / "validation.jsonl", validation_rows)
    manifest = {
        "source_paths": [str(path.resolve()) for path in source_paths],
        "tasks_paths": [str(path.resolve()) for path in tasks_paths],
        "input_rows": sum(len(_load_rows(path)) for path in source_paths),
        "selected_rows": len(selected),
        "deduplicated_rows": duplicates,
        "covered_tasks": len({row["task_id"] for row in selected}),
        "source_type_counts": dict(sorted(source_counts.items())),
        "system_prompt": "canonicalized_to_current_runtime",
        "split_unit": "task_id",
        "validation_ratio": validation_ratio,
        "seed": seed,
        "validation_strategy": validation_strategy,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "validation_task_ids": sorted({row["task_id"] for row in validation_rows}),
        "test_data_used": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--tasks", type=Path, action="append")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--validation-ratio", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-task-id", action="append")
    args = parser.parse_args()
    print(json.dumps(finalize_sft(
        source_paths=args.source,
        tasks_paths=args.tasks or [Path("data/tasks/train.jsonl")],
        database_path=args.database,
        output_dir=args.output_dir,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        validation_task_ids=set(args.validation_task_id) if args.validation_task_id else None,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
