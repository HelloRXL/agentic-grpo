import json
from pathlib import Path

import pytest

from airline_agent.verl_adapter import (
    _TokenTrace,
    _apply_chat_template,
    _require_prefix_preserving_checkpoint,
    _task_from_kwargs,
    _validate_response_trace_length,
)
from airline_agent.chat_template import (
    CHAT_TEMPLATE_KWARGS,
    TEMPLATE_MODE,
    chat_template_sha256,
    load_chat_template,
)


def test_token_trace_masks_context_and_action_separately() -> None:
    trace = _TokenTrace()
    trace.add_generation([1, 2], [3, 4], [-0.1, -0.2])
    trace.add_generation([1, 2, 3, 4, 5], [6], [-0.3])

    assert trace.prompt_ids == [1, 2]
    assert trace.response_ids == [3, 4, 5, 6]
    assert trace.response_mask == [1, 1, 0, 1]
    assert trace.response_logprobs == [-0.1, -0.2, 0.0, -0.3]


def test_token_trace_rejects_unreliable_rollout_metadata() -> None:
    trace = _TokenTrace()
    trace.add_generation([1, 2], [3], [-0.1])

    with pytest.raises(RuntimeError, match="did not return token log-probs"):
        trace.add_generation([1, 2, 3], [4], None)

    with pytest.raises(RuntimeError, match="token/log-prob length mismatch"):
        _TokenTrace().add_generation([1, 2], [3, 4], [-0.1])

    with pytest.raises(RuntimeError, match="prefix-preserving"):
        trace.add_generation([1, 9], [4], [-0.2])


def test_response_trace_limit_fails_before_verl_padding() -> None:
    trace = _TokenTrace(response_ids=[1, 2, 3])

    with pytest.raises(ValueError, match="response_length=2"):
        _validate_response_trace_length(trace, response_length=2)


def test_rollout_template_uses_project_nonthinking_protocol() -> None:
    class Tokenizer:
        def apply_chat_template(self, _messages, **kwargs):
            self.kwargs = kwargs
            return [1, 2]

    tokenizer = Tokenizer()
    assert _apply_chat_template(tokenizer, [{"role": "system", "content": "x"}]) == [1, 2]
    assert tokenizer.kwargs["enable_thinking"] is False


def test_verl_requires_matching_template_protocol(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="missing"):
        _require_prefix_preserving_checkpoint(tmp_path)

    (tmp_path / "airline_sft_protocol.json").write_text(
        json.dumps({"chat_template_mode": "wrong"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="prefix-preserving template"):
        _require_prefix_preserving_checkpoint(tmp_path)

    template = load_chat_template()
    (tmp_path / "airline_sft_protocol.json").write_text(
        json.dumps(
            {
                "chat_template_mode": TEMPLATE_MODE,
                "chat_template_sha256": chat_template_sha256(template),
            }
        ),
        encoding="utf-8",
    )
    _require_prefix_preserving_checkpoint(tmp_path)


@pytest.mark.skipif(
    not Path("/data/raoxinlong/model_cache/Qwen3-1.7B").is_dir(),
    reason="本地 Qwen3-1.7B 模型不可用",
)
def test_real_qwen3_template_is_prefix_preserving_for_three_turns() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/data/raoxinlong/model_cache/Qwen3-1.7B",
        trust_remote_code=True,
    )
    tokenizer.chat_template = load_chat_template()

    def render(messages: list[dict[str, str]]) -> list[int]:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS,
        )

    system = {"role": "system", "content": "Airline policy"}
    user_one = {"role": "user", "content": "I need help."}
    action_one = '{"action_type":"ask_user","tool_name":null,"arguments":{},"user_question":"Please provide your booking id.","final_answer":null}'
    user_two = {"role": "user", "content": "Here is the booking id."}
    action_two = '{"action_type":"finish","tool_name":null,"arguments":{},"user_question":null,"final_answer":"Done."}'
    json.loads(action_one)
    json.loads(action_two)

    prompt_one = render([system, user_one])
    action_one_ids = tokenizer.encode(action_one, add_special_tokens=False)
    prompt_two = render(
        [system, user_one, {"role": "assistant", "content": action_one}, user_two]
    )
    full_one = prompt_one + action_one_ids
    assert prompt_two[: len(full_one)] == full_one
    assert "<think>\n\n</think>" in tokenizer.decode(prompt_two)

    action_two_ids = tokenizer.encode(action_two, add_special_tokens=False)
    prompt_three = render(
        [
            system,
            user_one,
            {"role": "assistant", "content": action_one},
            user_two,
            {"role": "assistant", "content": action_two},
            {"role": "user", "content": "Thank you."},
        ]
    )
    full_two = prompt_two + action_two_ids
    assert prompt_three[: len(full_two)] == full_two


def test_task_spec_can_be_recovered_from_extra_info() -> None:
    task = _task_from_kwargs(
        {
            "extra_info": {
                "task_spec": {
                    "task_id": "task-1",
                    "source_task_id": "1",
                    "source_version": "test",
                    "split": "train",
                    "status": "supported",
                    "visible_request": "test",
                    "user_scenario": {
                        "domain": "airline",
                        "reason_for_call": "test",
                        "task_instructions": "test",
                    },
                    "database_path": "db.json",
                    "database_sha256": "hash",
                    "source_payload": {},
                }
            }
        }
    )
    assert task.task_id == "task-1"
