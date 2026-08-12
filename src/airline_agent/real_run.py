"""用真实 OpenAI 兼容模型运行一条转换后的 Airline 任务。"""

import argparse
import hashlib
import json
from pathlib import Path

from .agent import LLMUserSimulator, run_task
from .agent.console_display import LiveRolloutDisplay
from .core.llm_client import OpenAICompatibleLLMClient, load_dotenv
from .tasks.spec import TaskSpec
from .verifier import LLMCommunicationVerifier


def _state_hash(state: dict) -> str:
    """为调试和训练记录生成稳定的数据库状态指纹。"""

    canonical = json.dumps(
        state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_task(path: Path, index: int) -> TaskSpec:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        return TaskSpec.model_validate_json(lines[index])
    except IndexError as exc:
        raise ValueError(f"任务索引 {index} 超出文件范围（共 {len(lines)} 条）") from exc


def serialize_result(result: object) -> dict:
    """将单条运行结果转换为可持久化的 rollout 和评测记录。"""

    rollout = result.rollout
    evaluation = result.evaluation
    rollout_payload = rollout.model_dump(mode="json")
    # 完整数据库仍保留在内存中供 Evaluator 比较；训练文件只保留指纹。
    rollout_payload["initial_state_hash"] = _state_hash(rollout.initial_state)
    rollout_payload["final_state_hash"] = _state_hash(rollout.final_state)
    rollout_payload.pop("initial_state", None)
    rollout_payload.pop("final_state", None)
    return {
        "rollout": rollout_payload,
        "evaluation": evaluation.model_dump(mode="json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks/test.jsonl"))
    parser.add_argument("--index", type=int, default=0, help="JSONL 中的任务行号，从 0 开始")
    parser.add_argument(
        "--agent-prefix",
        default="POLICY",
        choices=("POLICY", "TEACHER"),
        help="POLICY=本地 vLLM，TEACHER=Modelink 教师模型",
    )
    parser.add_argument("--user-prefix", default="USER")
    parser.add_argument(
        "--verifier-prefix",
        default="JUDGE",
        choices=("JUDGE", "TEACHER", "USER"),
        help="用于轨迹通信评估的独立模型配置前缀",
    )
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    task = _load_task(args.tasks, args.index)
    database_path = (project_root / task.database_path).resolve()
    agent_client = OpenAICompatibleLLMClient.from_env(args.agent_prefix)
    user_client = OpenAICompatibleLLMClient.from_env(args.user_prefix)
    verifier_client = OpenAICompatibleLLMClient.from_env(args.verifier_prefix)
    display = LiveRolloutDisplay()
    result = run_task(
        task,
        database_path=database_path,
        llm_client=agent_client,
        user_simulator=LLMUserSimulator(user_client, task),
        max_steps=args.max_steps,
        event_handler=display.handle,
        communication_verifier=LLMCommunicationVerifier(verifier_client),
    )
    payload = serialize_result(result)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    display.show_summary(
        evaluation=result.evaluation.model_dump(mode="json"),
        output_path=str(args.output) if args.output is not None else None,
    )


if __name__ == "__main__":
    main()
