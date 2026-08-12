#!/usr/bin/env python3
"""将内部 TaskSpec JSONL 导出为 veRL RLHFDataset 可读取的 parquet。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset

from airline_agent.agent.prompts import build_agent_system_prompt
from airline_agent.domain.runtime import create_airline_runtime
from airline_agent.tasks.spec import TaskSpec


def build_rows(
    tasks_paths: list[Path],
    user_prefix: str,
    max_steps: int,
    task_ids: set[str] | None = None,
) -> list[dict]:
    project_root = Path(__file__).resolve().parents[1]
    system_prompt: str | None = None
    rows: list[dict] = []
    seen_task_ids: set[str] = set()
    for tasks_path in tasks_paths:
        for line in tasks_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            task = TaskSpec.model_validate_json(line)
            if task.status != "supported":
                continue
            if task_ids is not None and task.task_id not in task_ids:
                continue
            if task.task_id in seen_task_ids:
                raise ValueError(f"重复的 task_id：{task.task_id}")
            seen_task_ids.add(task.task_id)
            if system_prompt is None:
                runtime = create_airline_runtime(
                    (project_root / task.database_path).resolve(),
                    initial_state_patches=task.initial_state_patches,
                    expected_initial_state_sha256=task.initial_state_sha256,
                )
                system_prompt = build_agent_system_prompt(
                    runtime.registry.get_tool_definitions()
                )
            rows.append(
                {
                    "data_source": "airline_tau2",
                    # 记录与 Adapter 实际使用的一致的初始 system prompt；
                    # 任务本身仍放在 extra_info 中，由 AgentLoop 恢复。
                    "prompt": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        }
                    ],
                    "ability": "tool_agent",
                    "reward_model": {"style": "environment", "ground_truth": task.task_id},
                    "extra_info": {
                        "task_id": task.task_id,
                        "user_prefix": user_prefix,
                        "max_steps": max_steps,
                        # Parquet/Arrow 会把嵌套 dict 推断成 Struct，并把不同工具
                        # 参数的缺失字段补成 None；extra="forbid" 的工具 Schema
                        # 会因此误判 Replay。序列化为 JSON 字符串可完整保留原始
                        # 参数边界，veRL adapter 会再还原成 TaskSpec。
                        "task_spec": task.model_dump_json(),
                    },
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        action="append",
        default=None,
        help="可重复传入多个 TaskSpec JSONL 文件；不同文件的 task_id 必须唯一。",
    )
    parser.add_argument("--output", type=Path, default=Path("data/verl/airline_train.parquet"))
    parser.add_argument("--user-prefix", default="USER")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--task-id", action="append", default=None)
    args = parser.parse_args()

    tasks_paths = args.tasks or [Path("data/tasks/train.jsonl")]
    rows = build_rows(
        tasks_paths,
        args.user_prefix,
        args.max_steps,
        task_ids=set(args.task_id) if args.task_id else None,
    )
    if not rows:
        raise SystemExit("没有 supported TaskSpec 可导出")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(args.output))
    manifest = {
        "source": [str(path) for path in tasks_paths],
        "output": str(args.output),
        "rows": len(rows),
        "task_ids": [row["extra_info"]["task_id"] for row in rows],
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
