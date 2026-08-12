"""在 veRL 注册适用于多轮 Action mask 的 LATA advantage。"""

from collections import defaultdict
import math
from typing import Any

import numpy as np
import torch
from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("grpo_lata")
def compute_grpo_lata_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """组内标准化后，对早期 Action token 加权并按 ``sqrt(L)`` 归一化。"""

    scores = token_level_rewards.sum(dim=-1)
    grouped: dict[object, list[torch.Tensor]] = defaultdict(list)
    for sample_index, group_id in enumerate(index):
        grouped[group_id].append(scores[sample_index])

    means: dict[object, torch.Tensor] = {}
    stds: dict[object, torch.Tensor] = {}
    for group_id, values in grouped.items():
        stacked = torch.stack(values)
        means[group_id] = stacked.mean()
        stds[group_id] = stacked.std() if len(values) > 1 else stacked.new_tensor(1.0)

    normalized = scores.clone()
    for sample_index, group_id in enumerate(index):
        normalized[sample_index] -= means[group_id]
        if norm_adv_by_std_in_grpo:
            normalized[sample_index] /= stds[group_id] + epsilon

    alpha = 1.05
    if config is not None:
        turn_discount = config.get("turn_discount", {})
        alpha = float(turn_discount.get("alpha", alpha))
    if alpha <= 1.0:
        raise ValueError(f"LATA alpha 必须大于 1，当前为 {alpha}")

    mask = response_mask.to(torch.float64)
    lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    # Observation token 的 mask 为 0；按 Action token 自身的次序计算位置，避免
    # 长 Observation 间隔扭曲 turn discount。
    action_positions = torch.cumsum(mask, dim=1) - 1.0
    exponents = lengths - 1.0 - action_positions
    log_weights = exponents * math.log(alpha)
    log_weights = log_weights.masked_fill(mask == 0, -torch.inf)
    max_log_weights = log_weights.max(dim=1, keepdim=True).values
    stable_weights = torch.exp(log_weights - max_log_weights) * mask
    weights = stable_weights * lengths / stable_weights.sum(
        dim=1, keepdim=True
    ).clamp(min=epsilon)

    advantages = (
        normalized.unsqueeze(-1)
        * weights.to(normalized.dtype)
        * response_mask
        / torch.sqrt(lengths).to(normalized.dtype)
    )
    return advantages, advantages

