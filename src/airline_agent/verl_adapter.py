"""Airline Agent 与 veRL Agent Loop 的最小适配层。

业务逻辑仍由现有 ``AgentLoop``、Airline runtime 和 deterministic reward 负责。
本文件只处理 veRL 的两个边界：

1. 用 veRL ``server_manager.generate`` 生成 policy Action；
2. 把多轮生成结果整理成 ``AgentLoopOutput``，并用 response_mask 区分
   assistant Action token（1）和用户/工具上下文 token（0）。

veRL 是可选依赖。未安装 veRL 时，本模块仍可被导入，方便在训练环境之外运行单测。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import torch

from .agent import LLMUserSimulator, run_task
from .agent.context import project_tool_observation
from .chat_template import (
    CHAT_TEMPLATE_KWARGS,
    TEMPLATE_MODE,
    chat_template_sha256,
    install_chat_template,
    load_chat_template,
)
from .core.llm_client import OpenAICompatibleLLMClient, load_dotenv
from .tasks.spec import TaskSpec


MAX_ACTION_TOKENS = 512
_SFT_PROTOCOL_FILE = "airline_sft_protocol.json"


def _progress_preview(value: Any, limit: int = 120) -> str:
    """将 rollout 事件压缩成一行终端进度，不把完整对话和 observation 刷屏。"""

    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else f"{text[:limit]}..."


def _rollout_progress(task_id: str, max_steps: int):
    """返回 AgentLoop 事件回调，用于 veRL 异步 rollout 的可观测进度。"""

    started_at = time.monotonic()

    def handle(event: str, payload: dict[str, Any]) -> None:
        step = payload.get("step_index", "-")
        prefix = f"[veRL rollout {task_id} step {step}/{max_steps}]"
        if event == "task_started":
            print(f"[veRL rollout {task_id}] 开始", flush=True)
        elif event == "agent_action":
            action = payload.get("action") or {}
            action_type = action.get("action_type", "unknown")
            detail = action.get("tool_name") or action.get("user_question") or ""
            print(f"{prefix} Action={action_type} {_progress_preview(detail)}", flush=True)
        elif event == "user_reply":
            print(f"{prefix} User 回复：{_progress_preview(payload.get('reply', ''))}", flush=True)
        elif event == "parse_error":
            print(f"{prefix} Action 解析失败：{_progress_preview(payload.get('error', ''))}", flush=True)
        elif event == "llm_error":
            print(f"{prefix} 模型调用失败：{_progress_preview(payload.get('error', ''))}", flush=True)
        elif event == "user_error":
            print(f"{prefix} User 模拟器调用失败：{_progress_preview(payload.get('error', ''))}", flush=True)
        elif event == "finished":
            elapsed = time.monotonic() - started_at
            print(f"[veRL rollout {task_id}] 结束，耗时 {elapsed:.1f}s", flush=True)

    return handle


try:  # pragma: no cover - 真实 veRL 环境才会执行
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )

    VERL_AVAILABLE = True
except ImportError:  # pragma: no cover - clean 开发环境故意不安装 veRL
    VERL_AVAILABLE = False

    class AgentLoopBase:  # type: ignore[no-redef]
        """仅供导入和静态测试使用的占位基类。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    @dataclass
    class AgentLoopMetrics:  # type: ignore[no-redef]
        generate_sequences: float = 0.0
        tool_calls: float = 0.0
        compute_score: float = 0.0
        num_preempted: int = -1

    @dataclass
    class AgentLoopOutput:  # type: ignore[no-redef]
        prompt_ids: list[int]
        response_ids: list[int]
        response_mask: list[int]
        response_logprobs: list[float] | None = None
        reward_score: float | None = None
        num_turns: int = 0
        metrics: Any = None
        extra_fields: dict[str, Any] = field(default_factory=dict)

    def register(_name: str):  # type: ignore[misc]
        def decorator(cls):
            return cls

        return decorator


def _token_ids(value: Any) -> list[int]:
    """兼容 transformers 新旧版本的 chat-template 返回值。"""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = value["input_ids"]
        if hasattr(value, "tolist"):
            value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    """用统一模板生成严格前缀可拼接的多轮 policy prompt token。"""

    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    value = tokenizer.apply_chat_template(
        messages,
        **dict(chat_template_kwargs or CHAT_TEMPLATE_KWARGS),
        **kwargs,
    )
    return _token_ids(value)


def _patch_tokenizer_pad(tokenizer: Any) -> None:
    """兼容 veRL 0.8 对 tokenizer.pad 的 Tensor 返回值假设。"""

    if getattr(tokenizer, "_airline_pad_returns_tensor", False):
        return
    original_pad = tokenizer.pad

    def tensorizing_pad(*args: Any, **kwargs: Any) -> Any:
        padded = original_pad(*args, **kwargs)
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            value = padded.get(key) if hasattr(padded, "get") else None
            if isinstance(value, list):
                padded[key] = torch.tensor(value, dtype=torch.long)
        return padded

    tokenizer.pad = tensorizing_pad
    tokenizer._airline_pad_returns_tensor = True


def _require_prefix_preserving_checkpoint(model_path: str | Path) -> None:
    """拒绝使用模板协议不一致的 SFT checkpoint 做多轮 veRL GRPO。"""

    protocol_path = Path(model_path) / _SFT_PROTOCOL_FILE
    if not protocol_path.is_file():
        raise RuntimeError(
            "Airline veRL GRPO requires a template-aligned SFT checkpoint: "
            f"missing {protocol_path}. Train SFT with the project template, "
            "then run scripts/prepare_verl_model.py on that merged checkpoint."
        )
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read SFT protocol file {protocol_path}: {error}") from error
    template = load_chat_template()
    if protocol.get("chat_template_mode") != TEMPLATE_MODE:
        raise RuntimeError(
            "Airline veRL GRPO requires the project prefix-preserving template; "
            f"got {protocol.get('chat_template_mode')!r} in {protocol_path}."
        )
    expected_hash = chat_template_sha256(template)
    if protocol.get("chat_template_sha256") != expected_hash:
        raise RuntimeError(
            "Airline chat template hash mismatch between checkpoint and source: "
            f"checkpoint={protocol.get('chat_template_sha256')!r}, source={expected_hash!r}"
        )


@dataclass
class _TokenTrace:
    """将多轮 prompt 和生成结果拼接成 veRL 所需的 response mask。"""

    prompt_ids: list[int] | None = None
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    full_ids: list[int] = field(default_factory=list)

    def add_generation(
        self,
        prompt_ids: list[int],
        completion_ids: list[int],
        logprobs: list[float] | None,
    ) -> None:
        if self.prompt_ids is None:
            self.prompt_ids = list(prompt_ids)
            self.full_ids = list(prompt_ids)
        else:
            if self.full_ids and prompt_ids[: len(self.full_ids)] == self.full_ids:
                context_ids = prompt_ids[len(self.full_ids) :]
            else:
                # rollout 与训练 forward 的条件状态必须是同一段 token 历史；
                # chat template 重写历史时不能继续使用这条轨迹训练。
                raise RuntimeError(
                    "Airline chat template is not prefix-preserving; "
                    "rejecting this rollout because old and new policy states differ"
                )
            self.response_ids.extend(context_ids)
            self.response_mask.extend([0] * len(context_ids))
            self.response_logprobs.extend([0.0] * len(context_ids))
            self.full_ids.extend(context_ids)

        self.response_ids.extend(completion_ids)
        self.response_mask.extend([1] * len(completion_ids))
        if logprobs is None:
            raise RuntimeError("Airline rollout server did not return token log-probs")
        values = list(logprobs)
        if len(values) != len(completion_ids):
            raise RuntimeError(
                "Airline rollout token/log-prob length mismatch: "
                f"tokens={len(completion_ids)}, logprobs={len(values)}"
            )
        if any(value is None for value in values):
            raise RuntimeError("Airline rollout returned a missing token log-prob")
        self.response_logprobs.extend(float(value) for value in values)
        self.full_ids.extend(completion_ids)


class _VeRLPolicyClient:
    """把 veRL 异步 token 服务包装为现有 AgentLoop 的同步 LLMClient。"""

    def __init__(
        self,
        server_manager: Any,
        tokenizer: Any,
        event_loop: asyncio.AbstractEventLoop,
        sampling_params: dict[str, Any],
        initial_prompt_tokens: int,
        max_model_len: int,
        chat_template_kwargs: dict[str, Any],
        max_action_tokens: int = MAX_ACTION_TOKENS,
    ) -> None:
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.event_loop = event_loop
        self.sampling_params = dict(sampling_params)
        self.initial_prompt_tokens = int(initial_prompt_tokens)
        self.max_model_len = int(max_model_len)
        self.chat_template_kwargs = dict(chat_template_kwargs)
        self.max_action_tokens = int(max_action_tokens)
        self.request_id = uuid4().hex
        self.trace = _TokenTrace()
        self.last_error: str | None = None

    def think(self, messages: list[dict[str, Any]]) -> str:
        prompt_ids = _apply_chat_template(
            self.tokenizer,
            messages,
            self.chat_template_kwargs,
        )
        if self.trace.prompt_ids is None and len(prompt_ids) > self.initial_prompt_tokens:
            self.last_error = (
                "Airline initial prompt exceeds rollout.prompt_length: "
                f"{len(prompt_ids)} > {self.initial_prompt_tokens} tokens"
            )
            raise RuntimeError(self.last_error)
        if len(prompt_ids) + self.max_action_tokens > self.max_model_len:
            self.last_error = (
                "Airline prompt plus Action budget exceeds max_model_len: "
                f"{len(prompt_ids)} + {self.max_action_tokens} > {self.max_model_len}"
            )
            raise RuntimeError(self.last_error)
        last_stop_reason = "unknown"
        for attempt in range(2):
            # veRL 的 vLLM server 会原地 pop max_tokens/logprobs；每轮必须传副本，
            # 否则同一条多轮轨迹的后续请求会继承被修改过的参数。
            sampling_params = dict(self.sampling_params)
            sampling_params["max_tokens"] = self.max_action_tokens
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                stop_token_ids = list(sampling_params.get("stop_token_ids") or [])
                sampling_params["stop_token_ids"] = list(
                    {int(eos_token_id), *[int(token) for token in stop_token_ids]}
                )
            future = asyncio.run_coroutine_threadsafe(
                self.server_manager.generate(
                    request_id=self.request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                ),
                self.event_loop,
            )
            try:
                generated = future.result()
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
                raise
            completion_ids = [int(item) for item in generated.token_ids]
            last_stop_reason = str(getattr(generated, "stop_reason", "unknown"))
            if completion_ids:
                logprobs = getattr(generated, "log_probs", None)
                self.trace.add_generation(prompt_ids, completion_ids, logprobs)
                text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                if len(completion_ids) >= self.max_action_tokens:
                    # 达到长度上限通常表示本轮没有完成 Action JSON。这里仍返回已有文本，
                    # 由现有 AgentLoop 记录 parse_error 并给予低奖励；因为这些 token
                    # 已有真实 rollout log-prob，不应使整个 GRPO group 直接失败。
                    preview = text[:240].replace("\n", " ")
                    self.last_error = (
                        "Airline Action reached max_action_tokens "
                        f"({self.max_action_tokens}, stop_reason={last_stop_reason}, "
                        f"preview={preview!r})"
                    )
                return text
            if attempt == 0:
                time.sleep(0.2)

        self.last_error = (
            "veRL vLLM rollout returned no response token "
            f"(stop_reason={last_stop_reason}, prompt_tokens={len(prompt_ids)})"
        )
        raise RuntimeError(
            self.last_error
        )


def _task_from_kwargs(kwargs: dict[str, Any]) -> TaskSpec:
    """从 veRL dataset 的 extra_info 中恢复一条 TaskSpec。"""

    candidate = kwargs.get("task_spec")
    if candidate is None:
        candidate = (kwargs.get("extra_info") or {}).get("task_spec")
    if isinstance(candidate, TaskSpec):
        return candidate
    if isinstance(candidate, str):
        return TaskSpec.model_validate_json(candidate)
    if isinstance(candidate, dict):
        return TaskSpec.model_validate(candidate)
    raise ValueError(
        "Airline veRL rollout 缺少 task_spec；请在 parquet 的 extra_info 中保存完整 TaskSpec"
    )


def _validate_response_trace_length(trace: _TokenTrace, response_length: int) -> None:
    """在 veRL padding 前检查多轮 trace 是否超过配置预算。"""

    actual_length = len(trace.response_ids)
    if actual_length > response_length:
        raise ValueError(
            "Airline 多轮轨迹包含 "
            f"{actual_length} 个 response token，超过 rollout.response_length={response_length}。"
            "请降低 parquet extra_info.max_steps，或成对提高 response_length 和 actor/reference "
            "token budget；不要静默截断，否则最终 reward 会和保留的动作 token 错配。"
        )


@register("airline_json_agent")
class AirlineVeRLAgentLoop(AgentLoopBase):
    """使用现有 JSON Action 协议的 veRL 多轮 AgentLoop。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _patch_tokenizer_pad(self.tokenizer)
        install_chat_template(self.tokenizer)
        _require_prefix_preserving_checkpoint(self.config.actor_rollout_ref.model.path)

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        if not VERL_AVAILABLE:  # pragma: no cover - 由真实训练环境触发
            raise RuntimeError("AirlineVeRLAgentLoop 需要安装 verl==0.8.0")

        task = _task_from_kwargs(kwargs)
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")
        user_prefix = str(
            (kwargs.get("extra_info") or {}).get("user_prefix", "USER")
        )
        user_client = OpenAICompatibleLLMClient.from_env(user_prefix)
        # 只记录 endpoint/model/重试配置，不记录 API key；用于定位 Ray worker
        # 是否继承了旧环境变量，以及区分模型上游错误与本地 rollout 问题。
        print(
            f"[veRL rollout {task.task_id}] User model={user_client.config.model} "
            f"endpoint={user_client.config.base_url} retries={user_client.config.max_retries}",
            flush=True,
        )
        max_model_len = int(self.rollout_config.max_model_len or 0)
        response_length = int(self.rollout_config.response_length)
        chat_template_kwargs = dict(self.apply_chat_template_kwargs or CHAT_TEMPLATE_KWARGS)
        if chat_template_kwargs != CHAT_TEMPLATE_KWARGS:
            raise RuntimeError(
                "Airline veRL requires chat_template_kwargs="
                f"{CHAT_TEMPLATE_KWARGS!r}; got {chat_template_kwargs!r}"
            )
        if max_model_len <= response_length:
            raise RuntimeError(
                "veRL 上下文配置无效：max_model_len 必须大于 response_length，"
                f"当前为 {max_model_len} <= {response_length}。"
            )
        policy_client = _VeRLPolicyClient(
            self.server_manager,
            self.tokenizer,
            asyncio.get_running_loop(),
            sampling_params,
            initial_prompt_tokens=int(self.rollout_config.prompt_length),
            max_model_len=max_model_len,
            chat_template_kwargs=chat_template_kwargs,
        )

        max_steps = int((kwargs.get("extra_info") or {}).get("max_steps", 30))
        result = await asyncio.to_thread(
            run_task,
            task,
            database_path=(project_root / task.database_path).resolve(),
            llm_client=policy_client,
            user_simulator=LLMUserSimulator(user_client, task),
            max_steps=max_steps,
            event_handler=_rollout_progress(task.task_id, max_steps),
            communication_verifier=None,
            observation_formatter=project_tool_observation,
            reward_mode=os.environ.get(
                "VERL_AIRLINE_REWARD_MODE", "prm_lite_v1"
            ),
        )
        trace = policy_client.trace
        if not trace.response_ids:
            raise RuntimeError(
                "Airline veRL rollout did not produce any policy Action token: "
                f"{policy_client.last_error or 'unknown policy failure'}"
            )
        if not (
            len(trace.response_ids)
            == len(trace.response_mask)
            == len(trace.response_logprobs)
        ):
            raise RuntimeError("veRL token trace 的 response ids、mask 和 log-prob 长度不一致")
        _validate_response_trace_length(
            trace,
            response_length=response_length,
        )
        output = AgentLoopOutput(
            prompt_ids=trace.prompt_ids or [],
            response_ids=trace.response_ids,
            response_mask=trace.response_mask,
            response_logprobs=trace.response_logprobs,
            reward_score=float(result.evaluation.environment_reward.training_reward),
            num_turns=len(result.rollout.steps),
            metrics=AgentLoopMetrics(),
            extra_fields={
                "task_id": task.task_id,
                "termination_reason": result.rollout.termination_reason,
                "environment_success": result.evaluation.environment_reward.success,
                "rollout_summary": {
                    "task_id": task.task_id,
                    "steps": len(result.rollout.steps),
                    "final_answer": result.rollout.final_answer,
                    "termination_reason": result.rollout.termination_reason,
                    "environment_reward": result.evaluation.environment_reward.model_dump(
                        mode="json"
                    ),
                    "failure_reasons": result.evaluation.full_task_failure_reasons,
                },
            },
        )
        print(
            f"[veRL rollout {task.task_id}] training_reward={output.reward_score:.3f} "
            f"official_reward={result.evaluation.environment_reward.reward:.3f} "
            f"db={result.evaluation.environment_reward.db_score:.3f} "
            f"communicate={result.evaluation.environment_reward.communicate_score:.3f} "
            f"terminal_reward={result.evaluation.environment_reward.terminal_reward:.3f} "
            f"action_progress={result.evaluation.environment_reward.action_progress_score:.3f} "
            f"progress_reward={result.evaluation.environment_reward.progress_reward:.3f} "
            f"penalty={result.evaluation.environment_reward.process_penalty:.3f} "
            f"training_penalty={result.evaluation.environment_reward.training_process_penalty:.3f} "
            f"process_quality={result.evaluation.environment_reward.process_quality_score:.3f} "
            f"turns={output.num_turns} response_tokens={len(trace.response_ids)} "
            f"termination={result.rollout.termination_reason} "
            f"reasons={result.evaluation.environment_reward.reasons}",
            flush=True,
        )
        return output


__all__ = ["AirlineVeRLAgentLoop", "VERL_AVAILABLE"]
