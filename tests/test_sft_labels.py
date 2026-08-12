from airline_agent.sft_labels import IGNORE_INDEX, build_assistant_labeled_examples


class _TemplateTokenizer:
    """最小 prefix-preserving chat template，用于验证 label 边界而非模型 token。"""

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict=False,
        **kwargs,
    ):
        assert tokenize is True
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return [ord(char) for char in rendered]


def test_build_assistant_labeled_examples_masks_non_assistant_tokens() -> None:
    examples = build_assistant_labeled_examples(
        [
            {"role": "system", "content": "policy"},
            {"role": "assistant", "content": "action_one"},
            {"role": "user", "content": "observation"},
            {"role": "assistant", "content": "action_two"},
        ],
        _TemplateTokenizer(),
        max_length=1000,
    )

    assert len(examples) == 2
    for example in examples:
        labels = example["labels"]
        assert any(label == IGNORE_INDEX for label in labels)
        assert any(label != IGNORE_INDEX for label in labels)
        assert all(
            label == IGNORE_INDEX or label == token
            for token, label in zip(example["input_ids"], labels)
        )
    assert "action_one" in "".join(chr(token) for token in examples[0]["labels"] if token != IGNORE_INDEX)
    assert "action_two" in "".join(chr(token) for token in examples[1]["labels"] if token != IGNORE_INDEX)
