"""τ2 Airline 官方组件分解出的确定性 GRPO 训练奖励。"""

import json

from pydantic import BaseModel, Field

from .core.rollout import RolloutRecord
from .tasks.spec import TaskSpec
from .verifier import ActionVerifier


class EnvironmentRewardResult(BaseModel):
    """训练标量、τ2 官方组件与确定性过程诊断。"""

    reward: float
    training_reward: float = 0.0
    valid: bool
    success: bool
    db_score: float
    communicate_score: float
    action_progress_score: float = 0.0
    progress_reward: float = 0.0
    terminal_reward: float = 0.0
    process_penalty: float
    training_process_penalty: float = 0.0
    process_quality_score: float = 0.0
    invalid_action_count: int = 0
    repeated_action_count: int = 0
    missing_communicate_info: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


INVALID_ACTION_PENALTY = 0.05
REPEATED_ACTION_PENALTY = 0.10
MAX_PROCESS_PENALTY = 0.20
MAX_PROGRESS_REWARD = 0.25

# Reward V4 将官方 τ² 终局分与训练过程罚分分开记录，便于同时比较评测结果和
# 训练信号。
TRAINING_INVALID_ACTION_PENALTY = 0.10
TRAINING_REPEATED_ACTION_PENALTY = 0.08
MAX_TRAINING_PROCESS_PENALTY = 0.70
INCOMPLETE_BASE_REWARD = -0.30
WRONG_STATE_BASE_REWARD = -0.50
PRM_LITE_WEIGHT = 0.30
PRM_LITE_MIN_REWARD = -0.70
PRM_LITE_MAX_REWARD = 1.15

REWARD_MODES = frozenset({"terminal_v4", "prm_lite_v1"})
READ_TOOLS = frozenset(
    {
        "list_all_airports",
        "get_user_details",
        "get_reservation_details",
        "get_flight_status",
        "search_direct_flight",
        "search_onestop_flight",
    }
)
ESCALATION_TOOLS = frozenset({"transfer_to_human_agents"})
ENTITY_ARGUMENTS = frozenset(
    {"reservation_id", "user_id", "payment_id", "flight_number"}
)
PLACEHOLDER_VALUES = frozenset(
    {"bad", "unknown", "placeholder", "previous_reservation", "my_trip"}
)

# 这些工具能直接改变订单数据库。对写入任务，只有话术正确而最终状态错误不能
# 作为 GRPO 的正向部分奖励，否则策略可能学会“先做错误写入、再给出看似正确的
# 解释”。查询、搜索和转人工不在此集合中。
STATE_CHANGING_TOOLS = frozenset(
    {
        "book_reservation",
        "cancel_reservation",
        "update_reservation_baggages",
        "update_reservation_flights",
    }
)


def _empty_result(reasons: list[str]) -> EnvironmentRewardResult:
    return EnvironmentRewardResult(
        reward=0.0,
        training_reward=0.0,
        valid=False,
        success=False,
        db_score=0.0,
        communicate_score=0.0,
        action_progress_score=0.0,
        progress_reward=0.0,
        terminal_reward=0.0,
        process_penalty=0.0,
        training_process_penalty=0.0,
        process_quality_score=0.0,
        reasons=reasons,
    )


def _customer_facing_text(rollout: RolloutRecord) -> str:
    """汇总实际已发送给客户的 Agent 文本，而非只保留最后一次 finish。"""

    messages: list[str] = []
    for step in rollout.steps:
        action = step.action
        if action is None:
            continue
        if action.action_type == "ask_user" and action.user_question:
            messages.append(action.user_question)
        elif action.action_type == "finish" and action.final_answer:
            messages.append(action.final_answer)
    # 兼容旧记录和没有结构化步骤的单轮 rollout。
    if not messages and rollout.final_answer:
        messages.append(rollout.final_answer)
    return "\n".join(messages)


def _communicate_score(task: TaskSpec, rollout: RolloutRecord) -> tuple[float, list[str]]:
    """在完整 customer-facing 对话上执行 τ² ``communicate_info`` 子串检查。"""

    if "COMMUNICATE" not in task.reward_basis:
        return 1.0, []
    answer = _customer_facing_text(rollout).lower().replace(",", "")
    missing = [
        info
        for info in task.communicate_info
        if info.lower() not in answer
    ]
    return (1.0 if not missing else 0.0), missing


def _entity_values(value: object, key: str | None = None) -> set[str]:
    """从工具结果或参数中提取可被后续工具复用的业务实体。"""

    entities: set[str] = set()
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            entities.update(_entity_values(item_value, str(item_key)))
    elif isinstance(value, list):
        for item in value:
            entities.update(_entity_values(item, key))
    elif key in ENTITY_ARGUMENTS and isinstance(value, (str, int)):
        entities.add(str(value))
    return entities


def _has_placeholder(arguments: dict[str, object]) -> bool:
    for value in arguments.values():
        if isinstance(value, str) and value.strip().lower() in PLACEHOLDER_VALUES:
            return True
        if isinstance(value, dict) and _has_placeholder(value):
            return True
        if isinstance(value, list) and any(
            (isinstance(item, dict) and _has_placeholder(item))
            or (
                isinstance(item, str)
                and item.strip().lower() in PLACEHOLDER_VALUES
            )
            for item in value
        ):
            return True
    return False


def _process_quality_score(rollout: RolloutRecord) -> float:
    """PRM-Lite：对可验证的工具过程按步打分，再做轨迹级长度修正。"""

    tool_steps = [
        step
        for step in rollout.steps
        if step.action is not None and step.action.action_type == "tool"
    ]
    if not tool_steps:
        return 0.0

    scores: list[float] = []
    seen_signatures: set[str] = set()
    seen_reads: set[str] = set()
    observed_entities: set[str] = set()
    previous_failed = False
    previous_signature: str | None = None

    for step in tool_steps:
        action = step.action
        assert action is not None
        tool_name = action.tool_name or ""
        arguments = action.arguments
        signature = json.dumps(
            {"tool_name": tool_name, "arguments": arguments},
            ensure_ascii=True,
            sort_keys=True,
        )
        succeeded = step.observation is not None and step.observation.success
        score = 0.0

        if _has_placeholder(arguments):
            score -= 0.05 if tool_name in STATE_CHANGING_TOOLS else 0.03
        if signature in seen_signatures:
            score -= 0.03
        if not succeeded:
            score -= 0.05
        if previous_failed:
            score += -0.04 if signature == previous_signature else 0.05

        if tool_name in ESCALATION_TOOLS:
            score -= 0.05 if seen_reads else 0.10
        if tool_name in READ_TOOLS and tool_name not in seen_reads:
            score += 0.01
            seen_reads.add(tool_name)

        if _entity_values(arguments) & observed_entities:
            score += 0.08 if tool_name in STATE_CHANGING_TOOLS else 0.04

        if succeeded and step.observation is not None:
            observed_entities.update(_entity_values(step.observation.data))
        seen_signatures.add(signature)
        previous_failed = not succeeded
        previous_signature = signature
        scores.append(score)

    process_score = sum(scores) / len(scores)
    if len(seen_reads) >= 3:
        process_score += 0.01
    if len(tool_steps) > 8:
        process_score -= 0.01 * (len(tool_steps) - 8)
    return round(max(-0.5, min(0.5, process_score)), 6)


def compute_environment_reward(
    task: TaskSpec,
    rollout: RolloutRecord,
    *,
    replay_success: bool,
    initial_state_match: bool,
    final_state_match: bool,
    reward_mode: str = "terminal_v4",
) -> EnvironmentRewardResult:
    """计算训练 reward，同时保留 τ2 官方成功条件。

    官方成功是 DB 与 COMMUNICATE 的乘积；训练标量在此终局门控上加入确定性过程罚分。
    reference action 只用于诊断，不参与主奖励，避免把一条合法替代路径误判为错误。
    """

    if reward_mode not in REWARD_MODES:
        raise ValueError(
            f"未知 reward_mode={reward_mode!r}，可选值：{sorted(REWARD_MODES)}"
        )

    reasons: list[str] = []
    if task.status != "supported":
        reasons.append("task_unsupported")
    if not replay_success:
        reasons.append("replay_failed")
    if not initial_state_match:
        reasons.append("initial_state_mismatch")
    if rollout.termination_reason == "llm_error":
        reasons.append("rollout_infrastructure_error")
    if reasons:
        return _empty_result(reasons)

    db_score = (
        float(final_state_match)
        if "DB" in task.reward_basis
        else 1.0
    )
    communicate_score, missing_communicate_info = _communicate_score(
        task, rollout
    )
    action_verification = ActionVerifier().verify(
        task.reference_actions,
        rollout.steps,
    )
    action_progress_score = (
        action_verification.matched_count / len(task.reference_actions)
        if task.reference_actions
        else 0.0
    )
    progress_reward = round(MAX_PROGRESS_REWARD * action_progress_score, 6)

    tool_steps = [
        step
        for step in rollout.steps
        if step.action is not None and step.action.action_type == "tool"
    ]
    failed_tool_count = sum(
        step.observation is None or not step.observation.success
        for step in tool_steps
    )
    parse_error_count = sum(step.parse_error is not None for step in rollout.steps)
    invalid_action_count = failed_tool_count + parse_error_count

    # 与 AgentLoop 的运行时约束保持一致：只有“同一数据库版本下的同一工具调用”
    # 才是无效重复。多阶段任务在一次成功写入后重新查询预约是合法的；finish/ask_user
    # 也可能是用户新增请求后的正常交互，不能按字符串完全相同就扣分。
    repeated_action_count = 0
    state_version = 0
    seen_tool_calls: set[tuple[int, str]] = set()
    for step in rollout.steps:
        action = step.action
        if action is None or action.action_type != "tool":
            continue
        call_signature = json.dumps(
            {"tool_name": action.tool_name, "arguments": action.arguments},
            ensure_ascii=True,
            sort_keys=True,
        )
        call_key = (state_version, call_signature)
        if call_key in seen_tool_calls:
            repeated_action_count += 1
        else:
            seen_tool_calls.add(call_key)
        if (
            step.observation is not None
            and step.observation.success
            and action.tool_name in STATE_CHANGING_TOOLS
        ):
            state_version += 1

    process_penalty = min(
        MAX_PROCESS_PENALTY,
        invalid_action_count * INVALID_ACTION_PENALTY
        + repeated_action_count * REPEATED_ACTION_PENALTY,
    )
    training_process_penalty = min(
        MAX_TRAINING_PROCESS_PENALTY,
        invalid_action_count * TRAINING_INVALID_ACTION_PENALTY
        + repeated_action_count * TRAINING_REPEATED_ACTION_PENALTY,
    )
    process_quality_score = _process_quality_score(rollout)
    if invalid_action_count:
        reasons.append("invalid_tool_or_action_format")
    if repeated_action_count:
        reasons.append("repeated_action")
    if not final_state_match and "DB" in task.reward_basis:
        reasons.append("final_state_mismatch")
    if missing_communicate_info:
        reasons.append("required_communication_missing")

    if rollout.termination_reason != "finished":
        reasons.append("not_finished")
        if reward_mode == "prm_lite_v1":
            training_reward = max(
                PRM_LITE_MIN_REWARD,
                INCOMPLETE_BASE_REWARD + PRM_LITE_WEIGHT * process_quality_score,
            )
        else:
            training_reward = max(
                -MAX_TRAINING_PROCESS_PENALTY,
                INCOMPLETE_BASE_REWARD - training_process_penalty,
            )
        training_reward = round(training_reward, 6)
        return EnvironmentRewardResult(
            reward=0.0,
            training_reward=training_reward,
            valid=True,
            success=False,
            db_score=db_score,
            communicate_score=communicate_score,
            action_progress_score=action_progress_score,
            progress_reward=progress_reward,
            terminal_reward=0.0,
            process_penalty=process_penalty,
            training_process_penalty=training_process_penalty,
            process_quality_score=process_quality_score,
            invalid_action_count=invalid_action_count,
            repeated_action_count=repeated_action_count,
            missing_communicate_info=missing_communicate_info,
            reasons=reasons,
        )

    successful_write_with_wrong_state = (
        db_score == 0.0
        and any(
            step.action is not None
            and step.action.action_type == "tool"
            and step.action.tool_name in STATE_CHANGING_TOOLS
            and step.observation is not None
            and step.observation.success
            for step in rollout.steps
        )
    )
    if successful_write_with_wrong_state:
        reasons.append("successful_write_final_state_mismatch")
        base_reward = 0.0
    else:
        active_scores = []
        if "DB" in task.reward_basis:
            active_scores.append(db_score)
        if "COMMUNICATE" in task.reward_basis:
            active_scores.append(communicate_score)
        # τ² 官方 evaluator 对多个 reward basis 使用乘法门控；没有启用的
        # 组件不参与乘积。这样 DB 正确但话术缺失仍是正式失败，而不是 0.5 成功。
        base_reward = 1.0
        for score in active_scores:
            base_reward *= score
    reward = round(base_reward, 6)
    # 写入工具成功但最终状态错误时，进度匹配不能成为正向捷径；这是 Reward V4
    # 的硬门控。reference action progress 仍写入日志供诊断，但不进入主训练分。
    if successful_write_with_wrong_state:
        shaped_reward = WRONG_STATE_BASE_REWARD
    else:
        shaped_reward = base_reward
    if reward_mode == "prm_lite_v1":
        training_reward = max(
            PRM_LITE_MIN_REWARD,
            min(
                PRM_LITE_MAX_REWARD,
                shaped_reward + PRM_LITE_WEIGHT * process_quality_score,
            ),
        )
    else:
        training_reward = max(
            -MAX_TRAINING_PROCESS_PENALTY,
            min(1.0, shaped_reward - training_process_penalty),
        )
    training_reward = round(training_reward, 6)
    success = bool(db_score == 1.0 and communicate_score == 1.0)
    return EnvironmentRewardResult(
        reward=reward,
        training_reward=training_reward,
        valid=True,
        success=success,
        db_score=db_score,
        communicate_score=communicate_score,
        action_progress_score=action_progress_score,
        progress_reward=progress_reward,
        terminal_reward=base_reward,
        process_penalty=process_penalty,
        training_process_penalty=training_process_penalty,
        process_quality_score=process_quality_score,
        invalid_action_count=invalid_action_count,
        repeated_action_count=repeated_action_count,
        missing_communicate_info=missing_communicate_info,
        reasons=reasons,
    )
