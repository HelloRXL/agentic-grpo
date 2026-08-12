"""将多轮 Action 对话显式转换为 assistant-only SFT labels。

不依赖 tokenizer 是否在 chat template 中实现 ``{% generation %}``。每个 assistant
回合的标签边界由同一 chat template 分别渲染前缀和完整回合后得到，因此训练格式与实际
AgentLoop 使用的 Qwen chat template 保持一致。
"""

from __future__ import annotations

from typing import Any


IGNORE_INDEX = -100


def _render_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    chat_template_kwargs: dict[str, Any],
) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_dict": False,
    }
    token_ids = tokenizer.apply_chat_template(
        messages,
        **chat_template_kwargs,
        **kwargs,
    )
    return list(token_ids)


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def build_assistant_labeled_examples(
    messages: list[dict[str, str]],
    tokenizer: Any,
    *,
    max_length: int,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, list[int]]]:
    """将每个 assistant Action 展开为一条预分词、assistant-only 样本。

    逐回合构造 ``prompt + target action``，使每一步 label 的条件前缀与 SFT 时
    选择的 chat template 一致。用于 veRL 多轮 GRPO 的 checkpoint 必须传入
    ``chat_template_kwargs`` 必须与 rollout 使用的模板协议一致；当前 Airline
    模板使用 ``{"enable_thinking": False}``。
    样本超过预算时明确失败，防止静默截断工具调用或 ``done`` 动作。
    """
    if not messages or messages[0].get("role") != "system":
        raise ValueError("SFT messages 必须以 system 开始")
    template_kwargs = dict(chat_template_kwargs or {})
    assistant_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("SFT messages 缺少 assistant 监督目标")

    examples: list[dict[str, list[int]]] = []
    for index in assistant_indices:
        prefix_ids = _render_ids(
            tokenizer,
            messages[:index],
            add_generation_prompt=True,
            chat_template_kwargs=template_kwargs,
        )
        through_assistant_ids = _render_ids(
            tokenizer,
            messages[: index + 1],
            add_generation_prompt=False,
            chat_template_kwargs=template_kwargs,
        )
        start = _common_prefix_length(prefix_ids, through_assistant_ids)
        end = len(through_assistant_ids)
        if start >= end:
            raise ValueError(f"cannot locate assistant label span at message {index}")
        completion_ids = through_assistant_ids[start:end]
        input_ids = [*prefix_ids, *completion_ids]
        if len(input_ids) > max_length:
            raise ValueError(
                f"assistant message {index} tokenized length {len(input_ids)} "
                f"exceeds max_length={max_length}"
            )
        labels = [IGNORE_INDEX] * len(prefix_ids) + completion_ids
        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
    return examples
