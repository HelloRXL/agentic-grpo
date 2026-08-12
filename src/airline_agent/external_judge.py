"""对已通过 current-runtime replay 的 τ² 外部候选做当前通信 Judge，并生成 SFT 源行。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .core.llm_client import OpenAICompatibleLLMClient, load_dotenv
from .core.rollout import RolloutRecord
from .sft_data import _has_clean_sft_process, load_task_specs
from .tasks.spec import TaskSpec
from .verifier import ActionVerifier, LLMCommunicationVerifier


def _action_signature(rollout: dict[str, Any]) -> str:
    actions = []
    for step in rollout.get("steps", []):
        action = step.get("action") if isinstance(step, dict) else None
        if isinstance(action, dict):
            actions.append((action.get("action_type"), action.get("tool_name"), action.get("arguments")))
    return json.dumps(actions, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source_key(candidate: dict[str, Any]) -> str:
    provenance = candidate.get("external_provenance", {})
    return "|".join(
        str(provenance.get(key, ""))
        for key in ("source_file", "source_simulation_id", "source_trial")
    )


def _load_done(path: Path, *, retry_reasons: set[str] | None = None) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("rejection_reason") in (retry_reasons or set()):
            continue
        done.add(_source_key(row))
    return done


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sft_row(candidate: dict[str, Any]) -> dict[str, Any]:
    rollout = candidate["rollout"]
    provenance = candidate["external_provenance"]
    return {
        "task_id": rollout["task_id"],
        "source_type": "tau2_public_external",
        "source_model": provenance.get("source_model"),
        "source_file": provenance.get("source_file"),
        "source_file_sha256": provenance.get("source_file_sha256"),
        "source_simulation_id": provenance.get("source_simulation_id"),
        "source_trial": provenance.get("source_trial"),
        "canonical_action_signature": _action_signature(rollout),
        "messages": rollout["messages"],
    }


def judge_external_candidates(
    *,
    candidates_path: Path,
    tasks_path: Path,
    output_dir: Path,
    judge_prefix: str,
    retry_rejected_reasons: set[str] | None = None,
) -> dict[str, Any]:
    tasks = load_task_specs(tasks_path)
    client = OpenAICompatibleLLMClient.from_env(judge_prefix)
    judge = LLMCommunicationVerifier(client)
    accepted_path = output_dir / "external_tau2_judged_accepted.jsonl"
    rejected_path = output_dir / "external_tau2_judged_rejected.jsonl"
    sft_path = output_dir / "external_tau2_judged_sft.jsonl"
    done = _load_done(accepted_path) | _load_done(
        rejected_path,
        retry_reasons=retry_rejected_reasons,
    )
    counts: Counter[str] = Counter()
    accepted = 0
    rejected = 0
    total = 0

    for candidate in _load_candidates(candidates_path):
        key = _source_key(candidate)
        if key in done:
            continue
        total += 1
        provenance = candidate.get("external_provenance", {})
        rollout_payload = candidate.get("rollout")
        task_id = rollout_payload.get("task_id") if isinstance(rollout_payload, dict) else None
        task = tasks.get(task_id)
        reason: str | None = None
        communication = None
        action_result = None
        if task is None or task.status != "supported":
            reason = "task_not_supported"
        elif not isinstance(rollout_payload, dict):
            reason = "invalid_rollout"
        elif not _has_clean_sft_process(rollout_payload):
            reason = "unclean_process"
        elif not candidate.get("deterministic_audit", {}).get("current_environment_success"):
            reason = "deterministic_environment_failed"
        elif not candidate.get("deterministic_audit", {}).get("current_final_state_match"):
            reason = "deterministic_final_state_mismatch"
        else:
            rollout = RolloutRecord.model_validate(rollout_payload)
            action_result = ActionVerifier().verify(task.reference_actions, rollout.steps)
            # 同一终态和用户沟通可以由不同查询顺序得到；仅当官方任务本身
            # 把 ACTION 放进 reward_basis 时，才将参考工具序列作为硬约束。
            if "ACTION" in task.reward_basis and not action_result.passed:
                reason = "strict_action_failed"
            else:
                communication = judge.verify(task, rollout)
                if not communication.passed:
                    reason = "judge_failed"

        if reason is None:
            record = {
                **candidate,
                "judge": communication.model_dump(mode="json") if communication else None,
                "reference_action_diagnostic": action_result.model_dump(mode="json"),
                "status": "accepted_for_sft",
            }
            _append_jsonl(accepted_path, record)
            _append_jsonl(sft_path, _sft_row(candidate))
            accepted += 1
            counts["accepted"] += 1
        else:
            record = {
                "external_provenance": provenance,
                "rejection_reason": reason,
                "judge": communication.model_dump(mode="json") if communication else None,
                "reference_action_diagnostic": action_result.model_dump(mode="json") if action_result else None,
            }
            _append_jsonl(rejected_path, record)
            rejected += 1
            counts[reason] += 1
        done.add(key)
        print(json.dumps({"processed": total, "accepted": accepted, "rejected": rejected, "last": key, "reason": reason}, ensure_ascii=False), flush=True)

    manifest = {
        "source_candidates": str(candidates_path.resolve()),
        "tasks_path": str(tasks_path.resolve()),
        "judge_prefix": judge_prefix,
        "input_candidates": len(_load_candidates(candidates_path)),
        "processed_this_run": total,
        "accepted_for_sft": accepted,
        "rejected": rejected,
        "counts": dict(sorted(counts.items())),
        "retry_rejected_reasons": sorted(retry_rejected_reasons or []),
        "resume_supported": True,
        "test_data_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "external_tau2_judged_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft_sources"))
    parser.add_argument("--judge-prefix", default="JUDGE")
    parser.add_argument(
        "--retry-rejected-reason",
        action="append",
        default=[],
        help="重新审查此前具有该 rejection_reason 的候选；用于修正规则，不重跑其余已完成项。",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    print(json.dumps(judge_external_candidates(
        candidates_path=args.candidates,
        tasks_path=args.tasks,
        output_dir=project_root / args.output_dir,
        judge_prefix=args.judge_prefix,
        retry_rejected_reasons=set(args.retry_rejected_reason),
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
