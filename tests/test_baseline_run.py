from pathlib import Path

from airline_agent.baseline_run import build_summary, load_tasks, select_supported_tasks


ROOT = Path(__file__).resolve().parents[1]


def _record(*, finished, environment_success, full_task_success, reward, reasons):
    return {
        "rollout": {"termination_reason": "finished" if finished else "max_steps"},
        "evaluation": {
            "environment_reward": {
                "success": environment_success,
                "reward": reward,
                "reasons": reasons,
            },
            "full_task_success": full_task_success,
            "strict_action_success": full_task_success,
            "judge_pass": full_task_success,
            "sft_accepted": full_task_success,
            "full_task_failure_reasons": reasons,
            "actual_tools": [],
            "successful_tools": [],
        },
    }


def test_build_summary_separates_environment_and_full_task_success():
    summary = build_summary(
        [
            _record(
                finished=True, environment_success=True, full_task_success=True,
                reward=1.0, reasons=[],
            ),
            _record(
                finished=False, environment_success=False, full_task_success=False,
                reward=0.15, reasons=["not_finished"],
            ),
        ]
    )

    assert summary["environment_pass_at_1"] == 0.5
    assert summary["full_task_pass_at_1"] == 0.5
    assert summary["mean_environment_reward"] == 0.575
    assert summary["judge_pass_rate"] == 0.5
    assert summary["environment_failure_counts"] == {"not_finished": 1}


def test_baseline_uses_only_supported_train_and_test_tasks():
    train = select_supported_tasks(load_tasks(ROOT / "data/tasks/train.jsonl"))
    test = select_supported_tasks(load_tasks(ROOT / "data/tasks/test.jsonl"))

    assert len(train) == 27
    assert len(test) == 19


def test_baseline_can_select_a_supported_task_subset():
    train = load_tasks(ROOT / "data/tasks/train.jsonl")

    selected = select_supported_tasks(train, {"tau2-airline-7", "tau2-airline-39"})

    assert [task.task_id for task in selected] == ["tau2-airline-7", "tau2-airline-39"]
