"""在干净数据库上重放任务参考动作，审计七工具的真实业务覆盖率。"""

import argparse
import json
from pathlib import Path
from typing import Any

from ..domain.runtime import create_airline_runtime
from .spec import TaskSpec


def load_tasks(path: Path) -> list[TaskSpec]:
    """按 JSONL 逐行读取内部任务。"""

    return [
        TaskSpec.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_task(task: TaskSpec, database_path: Path) -> dict[str, Any]:
    """在该任务专属的干净环境中重放参考动作。"""

    if task.status == "unsupported":
        return {
            "task_id": task.task_id,
            "source_task_id": task.source_task_id,
            "status": "unsupported",
            "replay_success": False,
            "failures": [{"reason": reason} for reason in task.unsupported_reasons],
        }

    runtime = create_airline_runtime(database_path)
    failures: list[dict[str, Any]] = []
    for action in task.reference_actions:
        result = runtime.executor.execute(action.tool_name, action.arguments)
        if not result.success:
            failures.append(
                {
                    "step_index": action.step_index,
                    "tool_name": action.tool_name,
                    "error": result.error,
                    "message": result.message,
                }
            )
            break

    if failures:
        status = "replay_failed"
    elif not task.reference_actions:
        # 空 actions 常表示“正确拒绝”，还需要通信验收，不能直接当作完整通过。
        status = "needs_assertion"
    else:
        status = "replay_supported"

    return {
        "task_id": task.task_id,
        "source_task_id": task.source_task_id,
        "status": status,
        "replay_success": not failures,
        "reference_action_count": len(task.reference_actions),
        "failures": failures,
    }


def audit_directory(input_dir: Path, database_path: Path) -> dict[str, Any]:
    """审计 train、test、base 三个转换结果文件。"""

    report: dict[str, Any] = {"splits": {}}
    for split_name in ("train", "test", "base"):
        tasks = load_tasks(input_dir / f"{split_name}.jsonl")
        records = [audit_task(task, database_path) for task in tasks]
        report["splits"][split_name] = {
            "total": len(records),
            "replay_supported": sum(r["status"] == "replay_supported" for r in records),
            "replay_failed": sum(r["status"] == "replay_failed" for r in records),
            "needs_assertion": sum(r["status"] == "needs_assertion" for r in records),
            "unsupported": sum(r["status"] == "unsupported" for r in records),
            "tasks": records,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_directory(args.input_dir, args.database)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for split_name, summary in report["splits"].items():
        print(
            f"{split_name}: total={summary['total']}, "
            f"replay_supported={summary['replay_supported']}, "
            f"replay_failed={summary['replay_failed']}, "
            f"needs_assertion={summary['needs_assertion']}, "
            f"unsupported={summary['unsupported']}"
        )


if __name__ == "__main__":
    main()
