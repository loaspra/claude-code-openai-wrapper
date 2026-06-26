import pytest
from fastapi.testclient import TestClient
import sys
import types

fake_sdk = types.ModuleType("claude_agent_sdk")


class FakeClaudeAgentOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


async def fake_query(*args, **kwargs):
    return
    yield


fake_sdk.ClaudeAgentOptions = FakeClaudeAgentOptions
fake_sdk.query = fake_query
sys.modules.setdefault("claude_agent_sdk", fake_sdk)

from src import main


async def _mock_tool_call_completion(*args, **kwargs):
    yield {
        "subtype": "success",
        "result": '<tool_call>{"tool_calls":[{"id":"call_weather","name":"get_weather","arguments":{"city":"London"}}]}</tool_call>',
    }


async def _allow_api_key(*args, **kwargs):
    return True


@pytest.fixture(autouse=True)
def mock_auth_and_claude(monkeypatch):
    monkeypatch.setattr(main, "validate_claude_code_auth", lambda: (True, {"method": "claude_cli"}))
    monkeypatch.setattr(main, "verify_api_key", _allow_api_key)
    monkeypatch.setattr(main.claude_cli, "run_completion", _mock_tool_call_completion)


def test_openai_chat_completions_returns_tool_calls():
    client = TestClient(main.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["id"] == "call_weather"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert choice["message"]["tool_calls"][0]["function"]["arguments"] == '{"city":"London"}'


def test_openai_chat_completions_accepts_tool_result_messages():
    client = TestClient(main.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"London"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_weather",
                    "name": "get_weather",
                    "content": "sunny",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {"type": "object"}},
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"


def test_openai_chat_completions_reasoning_header_overrides_body(monkeypatch):
    captured_kwargs = []

    async def mock_completion(*args, **kwargs):
        captured_kwargs.append(kwargs)
        yield {"subtype": "success", "result": "ok"}

    monkeypatch.setattr(main.claude_cli, "run_completion", mock_completion)

    client = TestClient(main.app)
    response = client.post(
        "/v1/chat/completions",
        headers={"X-Claude-Max-Thinking-Tokens": "3000"},
        json={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "think"}],
            "reasoning_effort": "high",
        },
    )

    assert response.status_code == 200
    assert captured_kwargs[0]["max_thinking_tokens"] == 3000


def test_anthropic_messages_returns_tool_use_blocks():
    client = TestClient(main.app)
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["stop_reason"] == "tool_use"
    assert payload["content"][0]["type"] == "tool_use"
    assert payload["content"][0]["id"] == "call_weather"
    assert payload["content"][0]["name"] == "get_weather"
    assert payload["content"][0]["input"] == {"city": "London"}
