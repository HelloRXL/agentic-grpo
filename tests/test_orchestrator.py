import json
from pathlib import Path

from src.airline_agent.agent import run_task
from src.airline_agent.core.llm_client import FakeLLMClient
from src.airline_agent.tasks.converter import convert_dataset


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "reference-repos" / "tau2-bench-main" / "data" / "tau2" / "domains" / "airline"


def test_orchestrator_runs_official_refusal_task_to_reward() -> None:
    task = next(
        task
        for task in convert_dataset(
            DATA_DIR / "tasks.json",
            DATA_DIR / "split_tasks.json",
            DATA_DIR / "db.json",
        )["base"]
        if task.source_task_id == "0"
    )
    greeting = json.dumps(
        {
            "action_type": "ask_user",
            "tool_name": None,
            "arguments": {},
            "user_question": "Hello. How can I help?",
            "final_answer": None,
        }
    )
    finish = json.dumps(
        {
            "action_type": "finish",
            "tool_name": None,
            "arguments": {},
            "final_answer": "I cannot proceed with this cancellation under the policy.",
        }
    )

    result = run_task(
        task,
        database_path=DATA_DIR / "db.json",
        llm_client=FakeLLMClient([greeting, finish]),
    )

    assert result.rollout.task_id == task.task_id
    assert result.rollout.reward == 1.0
    assert result.evaluation.environment_reward.success is True
    assert result.evaluation.final_state_match is True
