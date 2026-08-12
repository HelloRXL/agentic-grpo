"""最小、可审计的 Airline GRPO 训练入口。

每条轨迹仍由现有 ``AgentLoop`` 逐轮运行；GRPO 只对模型生成的 Action
token 计算策略损失，用户回复和工具观察只作为上下文，不作为训练目标。
第一版故意不引入异步、分布式或 Judge 内循环，便于 smoke 和面试解释。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from tqdm import tqdm

from .agent import LLMUserSimulator, run_task
from .chat_template import CHAT_TEMPLATE_KWARGS, install_chat_template
from .core.llm_client import load_dotenv, OpenAICompatibleLLMClient
from .real_run import serialize_result
from .tasks.spec import TaskSpec
from .verifier import LLMCommunicationVerifier


@dataclass
class ActionSpan:
    """一次 assistant Action 的 prompt、token 和旧策略 log-prob。"""

    prefix_ids: list[int]
    completion_ids: list[int]
    old_logprobs: list[float]


class LocalPolicyClient:
    """用本地 HuggingFace policy 逐轮生成 Action。"""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        record_spans: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.record_spans = record_spans
        self.spans: list[ActionSpan] = []

    def reset(self) -> None:
        self.spans.clear()

    def _tokenize(self, messages: list[dict[str, Any]]) -> torch.Tensor:
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        ids = self.tokenizer.apply_chat_template(
            messages,
            **CHAT_TEMPLATE_KWARGS,
            **kwargs,
        )
        # Transformers 5.x 返回 BatchEncoding，旧版本可能直接返回 Tensor。
        # GRPO 后续需要的是模型输入的 input_ids Tensor，而非整个编码对象。
        input_ids = ids if isinstance(ids, torch.Tensor) else ids["input_ids"]
        return input_ids.to(self.device)

    @torch.no_grad()
    def think(self, messages: list[dict[str, Any]]) -> str:
        was_training = self.model.training
        self.model.eval()
        prompt_ids = self._tokenize(messages)
        prompt_len = prompt_ids.shape[1]
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": False,
        }
        if self.temperature > 0:
            kwargs.update({"temperature": self.temperature, "top_p": self.top_p})
        generated = self.model.generate(prompt_ids, **kwargs)
        completion_ids = generated[:, prompt_len:]
        text = self.tokenizer.decode(completion_ids[0], skip_special_tokens=True)

        if self.record_spans:
            # 用 rollout 时同一个 policy 重新计算 sampled token 的 log-prob。
            full_ids = generated
            completion_length = completion_ids.shape[1]
            outputs = self.model(
                input_ids=full_ids[:, :-1], logits_to_keep=completion_length
            )
            target = completion_ids
            logprobs = F.log_softmax(outputs.logits, dim=-1).gather(
                -1, target.unsqueeze(-1)
            ).squeeze(-1)
            sampled_logprobs = logprobs[0]
            self.spans.append(
                ActionSpan(
                    prefix_ids=prompt_ids[0].tolist(),
                    completion_ids=completion_ids[0].tolist(),
                    old_logprobs=sampled_logprobs.float().cpu().tolist(),
                )
            )
        if was_training:
            self.model.train()
        return text


def _load_tasks(path: Path) -> list[TaskSpec]:
    return [
        TaskSpec.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _select_tasks(tasks: list[TaskSpec], task_ids: list[str] | None) -> list[TaskSpec]:
    supported = [task for task in tasks if task.status == "supported"]
    if task_ids is None:
        return supported
    known = {task.task_id for task in tasks}
    unknown = set(task_ids) - known
    if unknown:
        raise ValueError(f"任务不存在：{sorted(unknown)}")
    selected = [task for task in supported if task.task_id in set(task_ids)]
    if not selected:
        raise ValueError("筛选后没有 supported task")
    return selected


def _load_model(model_path: Path, device: torch.device) -> tuple[Any, Any, Any]:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    install_chat_template(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = True
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        # rollout 在 eval 模式计算 old log-prob；关闭 LoRA dropout，保证训练
        # forward 与 rollout 使用同一份 policy 分布。
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    reference = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return model, reference, tokenizer


def _span_logprobs(model: Any, span: ActionSpan) -> torch.Tensor:
    device = next(model.parameters()).device
    ids = torch.tensor(
        [span.prefix_ids + span.completion_ids], device=device, dtype=torch.long
    )
    completion_length = len(span.completion_ids)
    outputs = model(input_ids=ids[:, :-1], logits_to_keep=completion_length)
    targets = ids[:, -completion_length:]
    all_logprobs = F.log_softmax(outputs.logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    return all_logprobs[0]


def _trajectory_loss(
    model: Any,
    reference: Any,
    span: ActionSpan,
    advantage: float,
    *,
    clip_epsilon: float,
    beta: float,
) -> torch.Tensor:
    new_logprobs = _span_logprobs(model, span)
    with torch.no_grad():
        ref_logprobs = _span_logprobs(reference, span)
        old_logprobs = torch.tensor(
            span.old_logprobs, device=new_logprobs.device, dtype=new_logprobs.dtype
        )
    # 只对 completion_ids（即 assistant Action）计算；prefix 不在损失中。
    ratio = torch.exp(new_logprobs - old_logprobs)
    adv = torch.full_like(ratio, float(advantage))
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_objective = torch.minimum(ratio * adv, clipped * adv)
    # sampled-token KL 的稳定近似，参考模型是冻结的 SFT V2。
    approx_kl = torch.exp(ref_logprobs - new_logprobs) - (ref_logprobs - new_logprobs) - 1.0
    return -policy_objective.mean() + beta * approx_kl.mean()


def _set_gradient_checkpointing(model: Any, enabled: bool) -> None:
    """只在反向传播阶段保存激活显存，生成阶段仍使用 KV cache。"""

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    if enabled:
        base_model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        base_model.gradient_checkpointing_disable()
        base_model.config.use_cache = True


def train(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    if not torch.cuda.is_available():
        raise SystemExit("GRPO 需要 CUDA；当前没有可见 GPU")
    device = torch.device("cuda")
    model, reference, tokenizer = _load_model(args.model, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    user_client = OpenAICompatibleLLMClient.from_env(args.user_prefix)
    tasks = _select_tasks(_load_tasks(args.tasks), args.task_id)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    database_root = project_root
    training_started_at = time.monotonic()

    for update in range(args.updates):
        samples: list[dict[str, Any]] = []
        grouped: dict[str, list[dict[str, Any]]] = {}
        rollout_total = len(tasks) * args.group_size
        rollout_started_at = time.monotonic()
        with tqdm(
            total=rollout_total,
            desc=f"Update {update + 1}/{args.updates} rollout",
            unit="trajectory",
        ) as rollout_bar:
            for task in tasks:
                grouped[task.task_id] = []
                for sample_index in range(args.group_size):
                    policy_client = LocalPolicyClient(
                        model,
                        tokenizer,
                        device,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                    )
                    result = run_task(
                        task,
                        database_path=(database_root / task.database_path).resolve(),
                        llm_client=policy_client,
                        user_simulator=LLMUserSimulator(user_client, task),
                        max_steps=args.max_steps,
                        communication_verifier=None,
                    )
                    item = {
                        "task_id": task.task_id,
                        "sample_index": sample_index,
                        "reward": result.evaluation.environment_reward.training_reward,
                        "spans": policy_client.spans,
                        "record": serialize_result(result),
                    }
                    grouped[task.task_id].append(item)
                    samples.append(item)
                    rollout_bar.update(1)
                    rollout_bar.set_postfix(
                        task=task.task_id,
                        sample=sample_index + 1,
                        reward=f"{item['reward']:.3f}",
                    )
        rollout_seconds = time.monotonic() - rollout_started_at
        advantages: dict[int, float] = {}
        for items in grouped.values():
            rewards = torch.tensor([item["reward"] for item in items], dtype=torch.float32)
            mean = rewards.mean().item()
            std = rewards.std(unbiased=False).item()
            for item, reward in zip(items, rewards.tolist(), strict=True):
                advantages[id(item)] = (reward - mean) / (std + 1e-4) if std > 1e-6 else 0.0

        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        span_count = 0
        _set_gradient_checkpointing(model, enabled=True)
        total_spans = sum(len(item["spans"]) for item in samples)
        loss_started_at = time.monotonic()
        with tqdm(
            total=total_spans,
            desc=f"Update {update + 1}/{args.updates} GRPO loss",
            unit="action",
        ) as loss_bar:
            for item in samples:
                # 先在轨迹内平均各 Action span，再在轨迹之间平均，避免长对话
                # 因 span 更多而获得更大的 GRPO 梯度权重。
                trajectory_span_count = max(1, len(item["spans"]))
                for span in item["spans"]:
                    loss = _trajectory_loss(
                        model,
                        reference,
                        span,
                        advantages[id(item)],
                        clip_epsilon=args.clip_epsilon,
                        beta=args.beta,
                    ) / max(1, len(samples)) / trajectory_span_count
                    loss.backward()
                    loss_value = float(loss.detach().cpu())
                    losses.append(loss_value)
                    span_count += 1
                    loss_bar.update(1)
                    loss_bar.set_postfix(loss=f"{loss_value:.4f}")
        loss_seconds = time.monotonic() - loss_started_at
        if span_count:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        _set_gradient_checkpointing(model, enabled=False)
        for index, item in enumerate(samples):
            record = item["record"]
            record["grpo"] = {
                "update": update,
                "advantage": advantages[id(item)],
                "action_span_count": len(item["spans"]),
            }
            (output_dir / f"update-{update:03d}-{index:04d}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        summary = {
            "update": update,
            "tasks": [task.task_id for task in tasks],
            "group_size": args.group_size,
            "rollouts": len(samples),
            "action_spans": span_count,
            "mean_reward": sum(item["reward"] for item in samples) / max(1, len(samples)),
            "mean_loss": sum(losses) / max(1, len(losses)),
            "zero_variance_groups": sum(
                len({item["reward"] for item in items}) == 1 for items in grouped.values()
            ),
            # 记录墙钟时间，后续与 veRL 的异步 rollout 做同条件吞吐对比。
            "rollout_seconds": round(rollout_seconds, 3),
            "loss_seconds": round(loss_seconds, 3),
            "total_elapsed_seconds": round(time.monotonic() - training_started_at, 3),
            "rollouts_per_second": round(len(samples) / rollout_seconds, 6)
            if rollout_seconds > 0
            else 0.0,
            "action_spans_per_second": round(span_count / loss_seconds, 6)
            if loss_seconds > 0
            else 0.0,
        }
        (output_dir / f"update-{update:03d}-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        model.save_pretrained(output_dir / f"checkpoint-{update:03d}")
        tokenizer.save_pretrained(output_dir / f"checkpoint-{update:03d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks/train.jsonl"))
    parser.add_argument("--task-id", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models/grpo-v1"))
    parser.add_argument("--user-prefix", default="USER")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
