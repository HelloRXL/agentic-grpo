"""对已有 rollout JSON 重新执行通信 Judge，不重新采集轨迹。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .core.llm_client import OpenAICompatibleLLMClient, load_dotenv
from .core.rollout import RolloutRecord
from .sft_data import _has_clean_sft_process, load_task_specs
from .verifier import LLMCommunicationVerifier


def _project_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def judge_rollouts(
    *,
    records_dir: Path,
    tasks_paths: list[Path],
    output_dir: Path,
    judge_prefix: str = "JUDGE",
    max_attempts: int = 2,
) -> dict[str, Any]:
    """复审 records_dir 中的记录，并把结果写入一个新的目录。"""

    tasks: dict[str, Any] = {}
    for tasks_path in tasks_paths:
        for task_id, task in load_task_specs(tasks_path).items():
            if task_id in tasks:
                raise ValueError(f"重复 task_id: {task_id}")
            tasks[task_id] = task

    client = OpenAICompatibleLLMClient.from_env(judge_prefix)
    verifier = LLMCommunicationVerifier(client, max_attempts=max_attempts)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    sources = sorted(
        path for path in records_dir.glob("*.json") if path.name != "summary.json"
    )
    for source in sources:
        record = json.loads(source.read_text(encoding="utf-8"))
        rollout_payload = record.get("rollout")
        if not isinstance(rollout_payload, dict):
            counts["invalid_rollout"] += 1
            continue
        task = tasks.get(rollout_payload.get("task_id"))
        if task is None or task.status != "supported":
            counts["task_not_supported"] += 1
            continue

        rollout = RolloutRecord.model_validate(rollout_payload)
        communication = verifier.verify(task, rollout)
        evaluation = dict(record.get("evaluation") or {})
        reward = dict(evaluation.get("environment_reward") or {})
        action_required = "ACTION" in task.reward_basis
        clean = _has_clean_sft_process(rollout_payload)
        sft_accepted = bool(
            reward.get("success")
            and rollout.termination_reason == "finished"
            and clean
            and (not action_required or evaluation.get("action_ok") is True)
            and communication.passed
        )
        evaluation["communication_verification"] = communication.model_dump(mode="json")
        evaluation["judge_pass"] = communication.passed
        evaluation["sft_accepted"] = sft_accepted
        updated = {**record, "evaluation": evaluation}
        (output_dir / source.name).write_text(
            json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rows.append(updated)
        counts["judge_pass" if communication.passed else "judge_fail"] += 1
        if sft_accepted:
            counts["sft_accepted"] += 1
        print(
            json.dumps(
                {
                    "file": source.name,
                    "task_id": rollout.task_id,
                    "judge_pass": communication.passed,
                    "sft_accepted": sft_accepted,
                    "error": communication.error,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "records_dir": str(records_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "tasks_paths": [str(path.resolve()) for path in tasks_paths],
        "judge_prefix": judge_prefix,
        "input_records": len(sources),
        "processed_records": len(rows),
        "counts": dict(sorted(counts.items())),
        "max_attempts": max_attempts,
        "source_is_not_overwritten": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-prefix", default="JUDGE")
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    summary = judge_rollouts(
        records_dir=_project_path(project_root, args.records_dir),
        tasks_paths=[_project_path(project_root, path) for path in args.tasks],
        output_dir=_project_path(project_root, args.output_dir),
        judge_prefix=args.judge_prefix,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
