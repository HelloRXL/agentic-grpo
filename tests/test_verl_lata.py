import numpy as np
import torch

from airline_agent.verl_lata import compute_grpo_lata_advantage


def test_lata_masks_context_and_weights_early_action_tokens_more():
    rewards = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    action_mask = torch.tensor([
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0, 1.0],
    ])

    advantages, returns = compute_grpo_lata_advantage(
        token_level_rewards=rewards,
        response_mask=action_mask,
        index=np.array(["task-1", "task-1"]),
        config={"turn_discount": {"alpha": 1.05}},
    )

    assert torch.isfinite(advantages).all()
    assert torch.equal(advantages, returns)
    assert torch.all(advantages[:, [1, 3]] == 0)
    assert advantages[0, 0] > advantages[0, 2] > advantages[0, 4] > 0
    assert advantages[1, 0] < advantages[1, 2] < advantages[1, 4] < 0
