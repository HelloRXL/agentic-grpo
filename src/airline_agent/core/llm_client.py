"""模型调用接口：测试 Fake 模型和真实 OpenAI 兼容 HTTP 客户端。"""

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Protocol
from http.client import RemoteDisconnected
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_LAST_REQUEST_BY_BASE_URL: dict[str, float] = {}


class LLMClient(Protocol):
    """Agent Loop 所依赖的最小模型接口。"""

    def think(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        ...


class FakeLLMClient:
    """按顺序返回预设响应的测试模型。"""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses.copy()
        self._next_response_index = 0
        self.calls: list[list[dict[str, Any]]] = []

    def think(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        self.calls.append(deepcopy(messages))

        if self._next_response_index >= len(self._responses):
            raise RuntimeError("Fake LLM 没有更多预设响应")

        response = self._responses[self._next_response_index]
        self._next_response_index += 1
        return response


@dataclass(frozen=True)
class ChatClientConfig:
    """一次 Chat Completions 调用所需的最小配置。"""

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_seconds: float = 120.0
    enable_thinking: bool = False
    reasoning_effort: str = ""
    max_retries: int = 4
    retry_delay_seconds: float = 5.0
    request_interval_seconds: float = 1.0
    provider: str = "auto"
    json_mode: bool = False

    @classmethod
    def from_env(cls, prefix: str) -> "ChatClientConfig":
        """从 ``PREFIX_BASE_URL/API_KEY/MODEL`` 读取配置。"""

        normalized = prefix.upper()
        fallback = "TEACHER" if normalized == "JUDGE" else None

        def read(name: str, default: str = "") -> str:
            value = os.getenv(f"{normalized}_{name}", "").strip()
            if not value and fallback is not None:
                value = os.getenv(f"{fallback}_{name}", "").strip()
            return value or default

        values = {
            "base_url": read("BASE_URL"),
            "api_key": read("API_KEY"),
            "model": read("MODEL"),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(
                f"{normalized} 配置缺少: {', '.join(missing)}；请检查 .env"
            )
        return cls(
            **values,
            temperature=float(os.getenv(f"{normalized}_TEMPERATURE", "0")),
            max_tokens=int(
                os.getenv(
                    f"{normalized}_MAX_TOKENS",
                    "2048" if normalized == "JUDGE" else os.getenv("MAX_TOKENS", "1024"),
                )
            ),
            timeout_seconds=float(
                os.getenv(f"{normalized}_TIMEOUT_SECONDS", os.getenv("TIMEOUT_SECONDS", "120"))
            ),
            enable_thinking=os.getenv(
                f"{normalized}_ENABLE_THINKING",
                os.getenv("ENABLE_THINKING", "false"),
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            reasoning_effort=read("REASONING_EFFORT"),
            max_retries=int(
                os.getenv(
                    f"{normalized}_MAX_RETRIES",
                    os.getenv("MAX_RETRIES", "2"),
                )
            ),
            retry_delay_seconds=float(
                os.getenv(
                    f"{normalized}_RETRY_DELAY_SECONDS",
                    os.getenv("RETRY_DELAY_SECONDS", "2"),
                )
            ),
            request_interval_seconds=float(
                os.getenv(
                    f"{normalized}_REQUEST_INTERVAL_SECONDS",
                    os.getenv("REQUEST_INTERVAL_SECONDS", "1"),
                )
            ),
            provider=read("PROVIDER", "auto").lower(),
            json_mode=read(
                "JSON_MODE",
                "true" if normalized in {"POLICY", "TEACHER", "JUDGE"} else "false",
            ).lower() in {"1", "true", "yes", "on"},
        )


def load_dotenv(path: str | Path = ".env") -> None:
    """读取简单的 KEY=VALUE 文件，不覆盖已有环境变量。"""

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


class OpenAICompatibleLLMClient:
    """调用 Modelink、vLLM 等 OpenAI 兼容服务的最小客户端。"""

    def __init__(self, config: ChatClientConfig) -> None:
        self.config = config

    @classmethod
    def from_env(cls, prefix: str) -> "OpenAICompatibleLLMClient":
        return cls(ChatClientConfig.from_env(prefix))

    def _is_local_vllm(self) -> bool:
        if self.config.provider != "auto":
            return self.config.provider in {"vllm", "local-vllm"}
        hostname = (urlparse(self.config.base_url).hostname or "").lower()
        return hostname in {"127.0.0.1", "localhost", "::1"}

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """只向各服务发送其支持的扩展字段。"""

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        normalized_model = self.config.model.casefold()
        if "deepseek" in normalized_model:
            payload["thinking"] = {
                "type": "enabled" if self.config.enable_thinking else "disabled",
            }
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self._is_local_vllm():
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.config.enable_thinking,
            }
        return payload

    def think(self, messages: list[dict[str, Any]]) -> str:
        payload = self._build_payload(messages)
        request = Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "agentic-grpo-airline/0.1",
            },
            method="POST",
        )
        for attempt in range(self.config.max_retries + 1):
            raw_response = ""
            try:
                base_url = self.config.base_url.rstrip("/")
                last_request_at = _LAST_REQUEST_BY_BASE_URL.get(base_url)
                if last_request_at is not None:
                    elapsed = time.monotonic() - last_request_at
                    remaining = self.config.request_interval_seconds - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
                _LAST_REQUEST_BY_BASE_URL[base_url] = time.monotonic()
                with urlopen(request, timeout=self.config.timeout_seconds) as response:
                    raw_response = response.read().decode("utf-8")
                body = json.loads(raw_response)
                break
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                detail = raw_response.strip()[:500]
                detail = detail or "<empty response body>"
                if detail.casefold().startswith(("<!doctype html", "<html")):
                    detail = (
                        "<HTML document instead of an API JSON response; "
                        "check the API base URL and gateway routing>"
                    )
                if attempt >= self.config.max_retries:
                    raise RuntimeError(
                        f"模型服务返回无效 JSON: {detail}"
                    ) from exc
                time.sleep(self.config.retry_delay_seconds * (2**attempt))
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
                if not retryable or attempt >= self.config.max_retries:
                    raise RuntimeError(f"模型服务 HTTP {exc.code}: {detail}") from exc
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 0.0
                except ValueError:
                    delay = 0.0
                time.sleep(delay or self.config.retry_delay_seconds * (2**attempt))
            except (URLError, TimeoutError, RemoteDisconnected) as exc:
                if attempt >= self.config.max_retries:
                    detail = getattr(exc, "reason", str(exc))
                    raise RuntimeError(f"模型服务连接失败: {detail}") from exc
                time.sleep(self.config.retry_delay_seconds * (2**attempt))

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("模型服务返回缺少 choices[0].message.content") from exc
        if not isinstance(content, str):
            raise RuntimeError("模型服务返回的 content 不是字符串")
        return content
