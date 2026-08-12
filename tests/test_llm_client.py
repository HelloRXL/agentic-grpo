import json

from airline_agent.core import ChatClientConfig, OpenAICompatibleLLMClient, load_dotenv
from airline_agent.core import llm_client as llm_module


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": "模型回复"}}]}
        ).encode("utf-8")


class _RawResponse(_FakeResponse):
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self):
        return self._body


def test_openai_compatible_client_builds_chat_request(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)
    client = OpenAICompatibleLLMClient(
        ChatClientConfig(
            base_url="https://example.test/v1/",
            api_key="secret",
            model="teacher-model",
        )
    )

    assert client.think([{"role": "user", "content": "你好"}]) == "模型回复"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["auth"] == "Bearer secret"
    assert captured["payload"]["model"] == "teacher-model"
    assert captured["payload"]["messages"][0]["content"] == "你好"
    assert "thinking" not in captured["payload"]
    assert "chat_template_kwargs" not in captured["payload"]


def test_remote_deepseek_uses_thinking_without_vllm_fields(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["user_agent"] = request.get_header("User-agent")
        return _FakeResponse()

    monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)
    client = OpenAICompatibleLLMClient(
        ChatClientConfig(
            base_url="https://opencode.ai/zen/go/v1",
            api_key="secret",
            model="deepseek-v4-flash",
            request_interval_seconds=0,
        )
    )

    client.think([{"role": "user", "content": "hello"}])

    assert captured["payload"]["thinking"] == {"type": "disabled"}
    assert "chat_template_kwargs" not in captured["payload"]
    assert captured["user_agent"] == "agentic-grpo-airline/0.1"


def test_openai_reasoning_effort_is_forwarded(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)
    client = OpenAICompatibleLLMClient(
        ChatClientConfig(
            base_url="https://xiaoqian.art",
            api_key="secret",
            model="gpt-5.6-luna",
            reasoning_effort="none",
            request_interval_seconds=0,
        )
    )

    client.think([{"role": "user", "content": "hello"}])

    assert captured["payload"]["reasoning_effort"] == "none"
    assert "thinking" not in captured["payload"]
    assert "chat_template_kwargs" not in captured["payload"]


def test_json_mode_is_forwarded_when_enabled(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)
    client = OpenAICompatibleLLMClient(
        ChatClientConfig(
            base_url="https://xiaoqian.art",
            api_key="secret",
            model="gpt-5.6-luna",
            json_mode=True,
            request_interval_seconds=0,
        )
    )

    client.think([{"role": "user", "content": "hello"}])

    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_client_retries_empty_or_invalid_json_response(monkeypatch):
    responses = iter(
        [
            _RawResponse(b""),
            _RawResponse(b'{"choices":[{"message":{"content":"recovered"}}]}'),
        ]
    )
    monkeypatch.setattr(llm_module, "urlopen", lambda _request, timeout: next(responses))
    monkeypatch.setattr(llm_module.time, "sleep", lambda _delay: None)
    client = OpenAICompatibleLLMClient(
        ChatClientConfig(
            base_url="https://xiaoqian.art",
            api_key="secret",
            model="gpt-5.6-luna",
            max_retries=1,
            retry_delay_seconds=0,
            request_interval_seconds=0,
        )
    )

    assert client.think([{"role": "user", "content": "hello"}]) == "recovered"


def test_client_reports_html_as_gateway_routing_error(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "urlopen",
        lambda _request, timeout: _RawResponse(b"<!doctype html><html>site</html>"),
    )
    client = OpenAICompatibleLLMClient(
        ChatClientConfig(
            base_url="https://xiaoqian.art",
            api_key="secret",
            model="gpt-5.6-luna",
            max_retries=0,
            request_interval_seconds=0,
        )
    )

    try:
        client.think([{"role": "user", "content": "hello"}])
    except RuntimeError as error:
        assert "HTML document instead of an API JSON response" in str(error)
    else:
        raise AssertionError("Expected an invalid-JSON response error")


def test_local_vllm_uses_chat_template_kwargs(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)
    client = OpenAICompatibleLLMClient(
        ChatClientConfig(
            base_url="http://127.0.0.1:8001/v1",
            api_key="EMPTY",
            model="qwen3-1.7b",
            request_interval_seconds=0,
        )
    )

    client.think([{"role": "user", "content": "hello"}])

    assert captured["payload"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert "thinking" not in captured["payload"]


def test_judge_config_falls_back_to_teacher_credentials(monkeypatch):
    for name in ("BASE_URL", "API_KEY", "MODEL", "REASONING_EFFORT"):
        monkeypatch.delenv(f"JUDGE_{name}", raising=False)
    monkeypatch.setenv("TEACHER_BASE_URL", "https://teacher.test/v1")
    monkeypatch.setenv("TEACHER_API_KEY", "teacher-secret")
    monkeypatch.setenv("TEACHER_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("TEACHER_REASONING_EFFORT", "none")

    config = ChatClientConfig.from_env("JUDGE")

    assert config.base_url == "https://teacher.test/v1"
    assert config.api_key == "teacher-secret"
    assert config.model == "deepseek-v4-flash"
    assert config.reasoning_effort == "none"
    assert config.max_tokens == 2048


def test_load_dotenv_does_not_override_existing_value(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("NEW_VALUE=from-file\nEXISTING=from-file\n", encoding="utf-8")
    monkeypatch.setenv("EXISTING", "from-process")

    load_dotenv(env_path)

    assert __import__("os").environ["NEW_VALUE"] == "from-file"
    assert __import__("os").environ["EXISTING"] == "from-process"
