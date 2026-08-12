"""把 TaskSpec、User Simulator、AgentLoop 和 Evaluator 串成一次任务运行。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..core.llm_client import LLMClient
from ..core.rollout import RolloutRecord
from ..evaluator import EvaluationResult, Evaluator
from ..tasks.spec import TaskSpec
from ..verifier import CommunicationVerifier
from .loop import AgentLoop
from .user_simulator import ScriptedUserSimulator, UserSimulator
from ..domain.runtime import create_airline_runtime


@dataclass(frozen=True)
class OrchestrationResult:
    """一次任务运行同时保留原始 rollout 和详细评测结果。"""

    rollout: RolloutRecord
    evaluation: EvaluationResult


def run_task(
    task: TaskSpec,
    *,
    database_path: Path,
    llm_client: LLMClient,
    user_simulator: UserSimulator | None = None,
    max_steps: int = 15,
    event_handler: Callable[[str, dict[str, Any]], None] | None = None,
    communication_verifier: CommunicationVerifier | None = None,
    observation_formatter: Callable[[Any], str] | None = None,
    reward_mode: str = "terminal_v4",
) -> OrchestrationResult:
    """运行一条可支持任务并立即计算 reward。"""

    if task.status == "unsupported":
        raise ValueError(
            f"任务 {task.task_id} 依赖当前版本未实现的能力，不能进入七工具运行链路"
        )

    runtime = create_airline_runtime(
        database_path,
        initial_state_patches=task.initial_state_patches,
        expected_initial_state_sha256=task.initial_state_sha256,
    )
    simulator = user_simulator or ScriptedUserSimulator.from_task(task)
    agent = AgentLoop(
        llm_client=llm_client,
        environment=runtime.environment,
        registry=runtime.registry,
        executor=runtime.executor,
        user_simulator=simulator,
        max_steps=max_steps,
        event_handler=event_handler,
        observation_formatter=observation_formatter,
    )
    rollout = agent.run(
        task_id=task.task_id,
    )
    evaluation = Evaluator(
        database_path,
        communication_verifier=communication_verifier,
        reward_mode=reward_mode,
    ).evaluate(task, rollout)
    rollout.reward = evaluation.environment_reward.reward
    return OrchestrationResult(rollout=rollout, evaluation=evaluation)
