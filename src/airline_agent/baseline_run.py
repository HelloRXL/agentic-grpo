"""批量运行冻结的 train/test Baseline，并保存每条轨迹与汇总指标。"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .agent import LLMUserSimulator, run_task
from .agent.console_display import LiveRolloutDisplay
from .core.llm_client import OpenAICompatibleLLMClient, load_dotenv
from .real_run import serialize_result
from .tasks.spec import TaskSpec
from .verifier import LLMCommunicationVerifier


def load_tasks(path: Path) -> list[TaskSpec]:
    """读取一个转换后的 JSONL split。"""

    return [
        TaskSpec.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_supported_tasks(
    tasks: list[TaskSpec], task_ids: set[str] | None = None
) -> list[TaskSpec]:
    """只运行当前能力审计支持、且可选地在指定集合中的任务。"""

    selected = [task for task in tasks if task.status == "supported"]
    if task_ids is None:
        return selected
    available_ids = {task.task_id for task in tasks}
    unknown = task_ids - available_ids
    if unknown:
        raise ValueError(f"指定 task_id 不在当前 split：{sorted(unknown)}")
    return [task for task in selected if task.task_id in task_ids]


def build_summary(records: list[dict], samples_per_task: int = 1) -> dict:
    """汇总统一环境奖励、通信审计与严格路径诊断。"""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for index, record in enumerate(records):
        task_id = record["rollout"].get("task_id", f"record-{index}")
        grouped[task_id].append(record)
    total = len(grouped)
    first_records = [items[0] for items in grouped.values()]

    def pass_at_k(predicate, k: int) -> float:
        return sum(
            any(predicate(item["evaluation"]) for item in items[:k])
            for items in grouped.values()
        ) / total if total else 0.0

    environment_successes = sum(
        record["evaluation"]["environment_reward"]["success"]
        for record in first_records
    )
    full_task_successes = sum(
        record["evaluation"]["full_task_success"] for record in first_records
    )
    strict_successes = sum(
        record["evaluation"]["strict_action_success"] for record in first_records
    )
    sft_successes = sum(record["evaluation"]["sft_accepted"] for record in first_records)
    environment_rewards = [
        record["evaluation"]["environment_reward"]["reward"] for record in records
    ]
    finished = sum(record["rollout"]["termination_reason"] == "finished" for record in first_records)
    step_counts = [len(record["rollout"].get("steps", [])) for record in records]
    tool_calls = sum(len(record["evaluation"].get("actual_tools", [])) for record in records)
    successful_tool_calls = sum(len(record["evaluation"].get("successful_tools", [])) for record in records)
    judge_passes = sum(record["evaluation"]["judge_pass"] for record in records)
    full_task_failure_counts: dict[str, int] = {}
    environment_failure_counts: dict[str, int] = {}
    for record in records:
        for reason in record["evaluation"]["full_task_failure_reasons"]:
            full_task_failure_counts[reason] = full_task_failure_counts.get(reason, 0) + 1
        for reason in record["evaluation"]["environment_reward"]["reasons"]:
            environment_failure_counts[reason] = environment_failure_counts.get(reason, 0) + 1
    return {
        "total_tasks": total,
        "environment_successful_tasks": environment_successes,
        "environment_pass_at_1": environment_successes / total if total else 0.0,
        "environment_pass_at_4": pass_at_k(
            lambda evaluation: evaluation["environment_reward"]["success"], 4
        ),
        "full_task_successful_tasks": full_task_successes,
        "full_task_pass_at_1": full_task_successes / total if total else 0.0,
        "full_task_pass_at_4": pass_at_k(
            lambda evaluation: evaluation["full_task_success"], 4
        ),
        "strict_successful_tasks": strict_successes,
        "strict_pass_at_1": strict_successes / total if total else 0.0,
        "strict_pass_at_4": pass_at_k(
            lambda evaluation: evaluation["strict_action_success"], 4
        ),
        "sft_accepted_tasks": sft_successes,
        "sft_acceptance_rate": sft_successes / total if total else 0.0,
        "mean_environment_reward": (
            sum(environment_rewards) / len(records) if records else 0.0
        ),
        "finished_tasks": finished,
        "finished_rate": finished / total if total else 0.0,
        "mean_steps": sum(step_counts) / len(step_counts) if step_counts else 0.0,
        "tool_success_rate": successful_tool_calls / tool_calls if tool_calls else 0.0,
        "judge_pass_rate": judge_passes / len(records) if records else 0.0,
        "sampled_rollouts": len(records),
        "samples_per_task": samples_per_task,
        "full_task_failure_counts": full_task_failure_counts,
        "environment_failure_counts": environment_failure_counts,
    }


def run_split(
    *,
    tasks_path: Path,
    output_dir: Path,
    agent_prefix: str,
    user_prefix: str,
    verifier_prefix: str,
    max_steps: int,
    samples_per_task: int,
    task_ids: set[str] | None = None,
) -> dict:
    """顺序运行一个 split；顺序执行可避免 API 限流并保证日志可读。"""

    project_root = Path(__file__).resolve().parents[2]
    source_tasks = load_tasks(tasks_path)
    tasks = select_supported_tasks(source_tasks, task_ids)
    if not tasks:
        raise ValueError("筛选后没有可运行的 supported task")
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_client = OpenAICompatibleLLMClient.from_env(agent_prefix)
    user_client = OpenAICompatibleLLMClient.from_env(user_prefix)
    verifier_client = OpenAICompatibleLLMClient.from_env(verifier_prefix)
    records: list[dict] = []

    for index, task in enumerate(tasks, start=1):
        for sample_index in range(samples_per_task):
            suffix = "" if samples_per_task == 1 else f"-s{sample_index + 1}"
            print(f"\n===== Baseline {tasks_path.stem}: {index}/{len(tasks)} sample {sample_index + 1}/{samples_per_task} | {task.task_id} =====")
            display = LiveRolloutDisplay()
            result = run_task(
                task,
                database_path=(project_root / task.database_path).resolve(),
                llm_client=agent_client,
                user_simulator=LLMUserSimulator(user_client, task),
                max_steps=max_steps,
                event_handler=display.handle,
                communication_verifier=LLMCommunicationVerifier(verifier_client),
            )
            record = serialize_result(result)
            records.append(record)
            output_path = output_dir / f"{index - 1:03d}-{task.task_id}{suffix}.json"
            output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            display.show_summary(evaluation=record["evaluation"], output_path=str(output_path))

    summary = build_summary(records, samples_per_task=samples_per_task)
    summary["tasks_path"] = str(tasks_path)
    summary["source_tasks"] = len(source_tasks)
    summary["skipped_unsupported_tasks"] = len(source_tasks) - len(tasks)
    summary["agent_prefix"] = agent_prefix
    summary["user_prefix"] = user_prefix
    summary["verifier_prefix"] = verifier_prefix
    # 轨迹文件不应携带 API key；但用于后续 SFT provenance 的模型标识必须落盘。
    summary["agent_model"] = agent_client.config.model
    summary["user_model"] = user_client.config.model
    summary["verifier_model"] = verifier_client.config.model
    summary["max_steps"] = max_steps
    summary["selected_task_ids"] = sorted(task_ids) if task_ids is not None else None
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n===== Baseline summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="all",
        help="默认依次运行 train 和冻结 test。",
    )
    parser.add_argument("--agent-prefix", default="POLICY", choices=("POLICY", "TEACHER"))
    parser.add_argument("--user-prefix", default="USER")
    parser.add_argument(
        "--verifier-prefix",
        default="JUDGE",
        choices=("JUDGE", "TEACHER", "USER"),
    )
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help=(
            "可选的自定义 TaskSpec JSONL（仅 train 语义）；用于训练专用变体，"
            "不会改变冻结 test。"
        ),
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=None,
        help="只运行指定任务；可重复传入，如 --task-id 7 --task-id tau2-airline-39",
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/baseline"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    selected_task_ids = None
    if args.task_id:
        selected_task_ids = {
            task_id if task_id.startswith("tau2-airline-") else f"tau2-airline-{task_id}"
            for task_id in args.task_id
        }
    if args.tasks is not None and args.split not in {"train", "all"}:
        raise ValueError("--tasks 仅能与 --split train（或默认 all）一起使用")
    # 自定义任务集只有训练用途，故即使默认 --split all 也只运行一次，绝不把
    # 训练变体误当作冻结 test。
    splits = ("train",) if args.tasks is not None else (
        ("train", "test") if args.split == "all" else (args.split,)
    )
    for split in splits:
        tasks_path = (
            (project_root / args.tasks).resolve()
            if args.tasks is not None
            else project_root / "data" / "tasks" / f"{split}.jsonl"
        )
        run_split(
            tasks_path=tasks_path,
            output_dir=project_root / args.output_root / args.agent_prefix.lower() / split,
            agent_prefix=args.agent_prefix,
            user_prefix=args.user_prefix,
            verifier_prefix=args.verifier_prefix,
            max_steps=args.max_steps,
            samples_per_task=args.samples_per_task,
            task_ids=selected_task_ids,
        )


if __name__ == "__main__":
    main()
