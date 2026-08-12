import torch
from types import SimpleNamespace

from airline_agent.grpo_train import ActionSpan, LocalPolicyClient, _span_logprobs


class _BatchEncodingTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]])}


def test_local_policy_tokenize_extracts_input_ids_from_batch_encoding():
    client = LocalPolicyClient(
        model=object(),
        tokenizer=_BatchEncodingTokenizer(),
        device=torch.device("cpu"),
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=8,
    )

    token_ids = client._tokenize([{"role": "user", "content": "hello"}])

    assert torch.equal(token_ids, torch.tensor([[1, 2, 3]]))


class _LogprobModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.seen_input_ids = None
        self.seen_logits_to_keep = None

    def forward(self, input_ids, logits_to_keep):
        self.seen_input_ids = input_ids.detach().clone()
        self.seen_logits_to_keep = logits_to_keep
        logits = torch.zeros((1, 2, 6), dtype=torch.float32)
        logits[0, 0, 4] = 2.0
        logits[0, 1, 5] = 3.0
        return SimpleNamespace(logits=logits)


def test_span_logprobs_aligns_previous_tokens_and_keeps_action_logits():
    model = _LogprobModel()
    span = ActionSpan(prefix_ids=[1, 2, 3], completion_ids=[4, 5], old_logprobs=[])

    values = _span_logprobs(model, span)

    assert torch.equal(model.seen_input_ids, torch.tensor([[1, 2, 3, 4]]))
    assert model.seen_logits_to_keep == 2
    assert values.shape == (2,)
