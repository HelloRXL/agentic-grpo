"""Airline JSON Agent 的统一 Qwen3 chat-template 协议。"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any


TEMPLATE_MODE = "qwen3_nonthinking_prefix_preserving_v1"
TEMPLATE_FILE = "qwen3_nonthinking_prefix_preserving.jinja"
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def default_chat_template_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / TEMPLATE_FILE


def load_chat_template(path: Path | None = None) -> str:
    template_path = path or default_chat_template_path()
    template = template_path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"chat template 为空：{template_path}")
    return template


def install_chat_template(tokenizer: Any, path: Path | None = None) -> str:
    """安装统一模板，并返回用于 checkpoint 协议校验的模板文本。"""

    template = load_chat_template(path)
    tokenizer.chat_template = template
    return template


def chat_template_sha256(template: str) -> str:
    return sha256(template.encode("utf-8")).hexdigest()


def template_protocol(template: str) -> dict[str, Any]:
    return {
        "chat_template_mode": TEMPLATE_MODE,
        "chat_template_sha256": chat_template_sha256(template),
        "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS),
        "label_mode": "assistant_only_per_action",
    }
