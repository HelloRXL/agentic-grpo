"""将 tau2 Airline 原始任务转换为可审计的内部 JSONL 数据集。"""

import argparse
import json
from pathlib import Path

from .converter import convert_dataset, file_sha256
from .spec import TaskSpec


def write_jsonl(path: Path, tasks: list[TaskSpec]) -> None:
    """每行写入一条 Pydantic TaskSpec，便于后续流式读取训练数据。"""

    lines = [task.model_dump_json() for task in tasks]
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def write_pretty_json(path: Path, tasks: list[TaskSpec]) -> None:
    """以缩进后的 JSON 数组保存，主要用于人工阅读和审计。"""

    payload = [task.model_dump(mode="json") for task in tasks]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数，路径全部显式传入，避免依赖当前工作目录。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="官方 tasks.json")
    parser.add_argument("--splits", type=Path, required=True, help="官方 split_tasks.json")
    parser.add_argument("--database", type=Path, required=True, help="官方 db.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="转换结果目录")
    parser.add_argument(
        "--source-version",
        default="local-tau2-reference",
        help="写入每条任务的来源版本标识",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="额外输出缩进后的 JSON 数组，供阅读和审计，不用于流式训练",
    )
    return parser


def main() -> None:
    """执行转换并写入数据集和可复现性 manifest。"""

    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    converted = convert_dataset(
        tasks_path=args.tasks,
        split_path=args.splits,
        database_path=args.database,
        source_version=args.source_version,
    )

    split_summary: dict[str, dict[str, int]] = {}
    for split_name, tasks in converted.items():
        write_jsonl(args.output_dir / f"{split_name}.jsonl", tasks)
        if args.pretty:
            write_pretty_json(args.output_dir / f"{split_name}.json", tasks)
        split_summary[split_name] = {
            "total": len(tasks),
            "supported": sum(task.status == "supported" for task in tasks),
            "unsupported": sum(task.status == "unsupported" for task in tasks),
        }

    manifest = {
        "source_version": args.source_version,
        "inputs": {
            "tasks": {"path": str(args.tasks), "sha256": file_sha256(args.tasks)},
            "splits": {"path": str(args.splits), "sha256": file_sha256(args.splits)},
            "database": {
                "path": str(args.database),
                "sha256": file_sha256(args.database),
            },
        },
        "splits": split_summary,
    }
    manifest_path = args.output_dir / "conversion_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for split_name, summary in split_summary.items():
        print(
            f"{split_name}: total={summary['total']}, "
            f"supported={summary['supported']}, "
            f"unsupported={summary['unsupported']}"
        )
    print(f"转换完成：{args.output_dir}")


if __name__ == "__main__":
    main()
