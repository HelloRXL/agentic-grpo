"""清洗 τ² Airline 公共评测轨迹，产出待 Judge 审核的外部 SFT 候选集。

τ² 发布的是评测日志而不是训练集。本模块只取 train split 中官方判定成功的
轨迹，将原生 function call 转为本项目的严格 Action JSON，并在当前 runtime
重新执行每一个工具调用。这里不调用 LLM Judge：通信质量审核是下一阶段，避免
把 API 波动和可复现的数据转换混在一起。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent.prompts import build_agent_system_prompt
from .core.actions import AgentAction
from .core.rollout import RolloutRecord, RolloutStep
from .domain.runtime import AirlineRuntime, create_airline_runtime
from .environment_reward import compute_environment_reward
from .tasks.spec import TaskSpec
from .verifier import ActionVerifier


TRANSFER_STOP = "###TRANSFER###"
CONVERTER_VERSION = "tau2-external-replay-v1"


@dataclass(frozen=True)
class ImportStats:
    source_trajectories: int
    candidates: int
    rejected: int
    rejection_reasons: dict[str, int]


def _action_json(action: AgentAction) -> str:
    return json.dumps(action.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))


def _state_hash(state: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tool_observation(result: Any) -> str:
    return "Tool execution observation:\n" + json.dumps(
        result.model_dump(mode="json"), ensure_ascii=False, indent=2
    ) + "\nDecide the next action."


def _is_question(text: str) -> bool:
    return text.rstrip().endswith(("?", "？"))


def _make_text_action(text: str, *, first_assistant_turn: bool) -> AgentAction:
    if first_assistant_turn or _is_question(text):
        return AgentAction(
            action_type="ask_user",
            user_question=text,
        )
    return AgentAction(action_type="finish", final_answer=text)


def _load_tasks(path: Path) -> dict[str, TaskSpec]:
    tasks = [
        TaskSpec.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(task.split != "train" for task in tasks):
        raise ValueError(f"外部 SFT 只能使用 train 任务：{path}")
    return {task.task_id.removeprefix("tau2-airline-"): task for task in tasks}


def _source_files(results_dir: Path) -> list[Path]:
    files = sorted(results_dir.glob("*_airline_*_4trials.json"))
    if not files:
        raise ValueError(f"未找到 Airline 4-trials 结果文件：{results_dir}")
    return files


def _convert_one(
    *,
    simulation: dict[str, Any],
    task: TaskSpec,
    runtime: AirlineRuntime,
    reference_success: bool,
    reference_initial_state: dict[str, Any],
    reference_final_state: dict[str, Any],
    provenance: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """返回 current-runtime replay 通过的候选 rollout；否则给出稳定拒绝原因。"""

    runtime.environment.reset()
    tool_definitions = runtime.registry.get_tool_definitions()
    allowed_tools = {definition["name"] for definition in tool_definitions}
    system_prompt = build_agent_system_prompt(tool_definitions)
    initial_state = runtime.environment.snapshot().model_dump(mode="json")
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    steps: list[RolloutStep] = []
    seen_actions: set[str] = set()
    first_assistant_turn = True
    final_answer: str | None = None

    raw_messages = simulation.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return None, "missing_messages"

    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            return None, "invalid_message"
        role = raw_message.get("role")
        if role == "tool":
            # Old observation must never enter the new prompt; the tool is replayed below.
            continue
        if role == "user":
            content = raw_message.get("content")
            if not isinstance(content, str):
                return None, "invalid_user_content"
            if content.strip() == TRANSFER_STOP:
                continue
            messages.append({"role": "user", "content": content})
            continue
        if role != "assistant":
            return None, f"unsupported_role:{role}"

        tool_calls = raw_message.get("tool_calls")
        content = raw_message.get("content")
        if tool_calls:
            if not isinstance(tool_calls, list) or len(tool_calls) != 1:
                return None, "tool_call_count_not_one"
            tool_call = tool_calls[0]
            if not isinstance(tool_call, dict):
                return None, "invalid_tool_call"
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments")
            if not isinstance(tool_name, str) or not isinstance(arguments, dict):
                return None, "invalid_tool_call_fields"
            if tool_name not in allowed_tools:
                return None, f"unsupported_tool:{tool_name}"
            action = AgentAction(action_type="tool", tool_name=tool_name, arguments=arguments)
            signature = _action_json(action)
            if signature in seen_actions:
                return None, "repeated_action"
            seen_actions.add(signature)
            result = runtime.executor.execute(tool_name, arguments)
            if not result.success:
                return None, f"replay_tool_failed:{tool_name}"
            messages.append({"role": "assistant", "content": signature})
            messages.append({"role": "user", "content": _tool_observation(result)})
            steps.append(
                RolloutStep(
                    step_index=len(steps) + 1,
                    raw_model_output=signature,
                    action=action,
                    observation=result,
                )
            )
            first_assistant_turn = False
            continue
        if not isinstance(content, str) or not content.strip():
            return None, "empty_assistant_content"
        action = _make_text_action(content, first_assistant_turn=first_assistant_turn)
        encoded = _action_json(action)
        messages.append({"role": "assistant", "content": encoded})
        steps.append(RolloutStep(step_index=len(steps) + 1, raw_model_output=encoded, action=action))
        first_assistant_turn = False
        if action.action_type == "finish":
            final_answer = action.final_answer

    if first_assistant_turn:
        return None, "no_assistant_turn"
    done = AgentAction(action_type="done")
    encoded_done = _action_json(done)
    messages.append({"role": "assistant", "content": encoded_done})
    steps.append(RolloutStep(step_index=len(steps) + 1, raw_model_output=encoded_done, action=done))
    rollout = RolloutRecord(
        task_id=task.task_id,
        user_request="",
        messages=messages,
        steps=steps,
        initial_state=initial_state,
        final_state=runtime.environment.snapshot().model_dump(mode="json"),
        final_answer=final_answer,
        termination_reason="finished",
    )
    initial_state_match = initial_state == reference_initial_state
    final_state_match = rollout.final_state == reference_final_state
    action_ok = ActionVerifier().verify(task.reference_actions, steps).passed
    environment_reward = compute_environment_reward(
        task,
        rollout,
        replay_success=reference_success,
        initial_state_match=initial_state_match,
        final_state_match=final_state_match,
    )
    if not reference_success:
        return None, "reference_replay_failed"
    if not initial_state_match:
        return None, "initial_state_mismatch"
    if "DB" in task.reward_basis and not final_state_match:
        return None, "final_state_mismatch"
    if "ACTION" in task.reward_basis and not action_ok:
        return None, "action_constraint_failed"
    if not environment_reward.success:
        return None, "environment_reward_failed"
    rollout_payload = rollout.model_dump(mode="json")
    # 训练只需要对话；完整 DB snapshot 每条约数 MB，会让外部数据不必要地膨胀。
    rollout_payload["initial_state_hash"] = _state_hash(rollout_payload.pop("initial_state"))
    rollout_payload["final_state_hash"] = _state_hash(rollout_payload.pop("final_state"))
    return {
        "rollout": rollout_payload,
        "external_provenance": provenance,
        "deterministic_audit": {
            "converter_version": CONVERTER_VERSION,
            "source_official_reward": simulation["reward_info"]["reward"],
            "current_environment_success": environment_reward.success,
            "current_final_state_match": final_state_match,
            "current_action_ok": action_ok,
            "status": "replay_passed_pending_judge",
        },
    }, None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def import_external_tau2(
    *,
    results_dir: Path,
    tasks_path: Path,
    database_path: Path,
    output_dir: Path,
) -> ImportStats:
    tasks = _load_tasks(tasks_path)
    reference_runtime = create_airline_runtime(database_path)
    reference_cache: dict[str, tuple[bool, dict[str, Any], dict[str, Any]]] = {}
    for task_id, task in tasks.items():
        if task.status != "supported":
            continue
        reference_runtime.environment.reset()
        reference_initial = reference_runtime.environment.snapshot().model_dump(mode="json")
        reference_success = True
        for action in task.reference_actions:
            if not reference_runtime.executor.execute(action.tool_name, action.arguments).success:
                reference_success = False
                break
        reference_cache[task_id] = (
            reference_success,
            reference_initial,
            reference_runtime.environment.snapshot().model_dump(mode="json"),
        )
    runtime = create_airline_runtime(database_path)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    total = 0
    source_manifest: list[dict[str, str]] = []
    for source in _source_files(results_dir):
        payload = json.loads(source.read_text(encoding="utf-8"))
        content_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        source_manifest.append({"path": str(source.resolve()), "sha256": content_hash})
        simulations = payload.get("simulations") if isinstance(payload, dict) else None
        if not isinstance(simulations, list):
            raise ValueError(f"{source}: 缺少 simulations")
        for simulation in simulations:
            if not isinstance(simulation, dict):
                continue
            task_raw_id = simulation.get("task_id")
            task = tasks.get(str(task_raw_id))
            if task is None or task.status != "supported":
                continue
            reward_info = simulation.get("reward_info")
            if not isinstance(reward_info, dict) or reward_info.get("reward") != 1:
                continue
            total += 1
            provenance = {
                "source": "tau2_public_benchmark_log",
                "license": "MIT (Sierra Research τ²-bench)",
                "source_file": source.name,
                "source_file_sha256": content_hash,
                "source_simulation_id": simulation.get("id"),
                "source_model": payload.get("info", {}).get("agent_model") if isinstance(payload.get("info"), dict) else None,
                "source_task_id": str(task_raw_id),
                "source_trial": simulation.get("trial"),
            }
            converted, reason = _convert_one(
                simulation=simulation,
                task=task,
                runtime=runtime,
                reference_success=reference_cache[task.task_id.removeprefix("tau2-airline-")][0],
                reference_initial_state=reference_cache[task.task_id.removeprefix("tau2-airline-")][1],
                reference_final_state=reference_cache[task.task_id.removeprefix("tau2-airline-")][2],
                provenance=provenance,
            )
            if converted is not None:
                candidates.append(converted)
                continue
            assert reason is not None
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            rejected.append({"external_provenance": provenance, "rejection_reason": reason})

    _write_jsonl(output_dir / "candidates_pending_judge.jsonl", candidates)
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    manifest = {
        "converter_version": CONVERTER_VERSION,
        "purpose": "train-only public τ² benchmark trajectories converted and replayed in current runtime; pending independent communication Judge",
        "source_files": source_manifest,
        "source_successful_supported_train_trajectories": total,
        "replay_passed_candidates": len(candidates),
        "rejected": len(rejected),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "test_data_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ImportStats(total, len(candidates), len(rejected), dict(sorted(reason_counts.items())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks/train.jsonl"))
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/external_tau2"))
    args = parser.parse_args()
    stats = import_external_tau2(
        results_dir=args.results_dir,
        tasks_path=args.tasks,
        database_path=args.database,
        output_dir=args.output_dir,
    )
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
