# Claude Code OpenAI Proxy

Small OpenAI/Anthropic-compatible HTTP proxy backed by Claude Code credentials.

## Scope

Kept endpoints:

- `POST /v1/chat/completions`
- `POST /v1/messages`
- `GET /v1/models`
- `GET /v1/auth/status`
- `GET /health`
- `GET /version`
- `GET /`

Supported credential sources:

- `ANTHROPIC_API_KEY` injected through the environment.
- Existing Claude Code credentials mounted into the runtime environment.

No Bedrock, Vertex, MCP management, server-side sessions, or Claude Code built-in tool-management API is included.

## Quick Start

```bash
poetry install
export ANTHROPIC_API_KEY=sk-ant-...
poetry run claude-wrapper
```

Or run with Uvicorn:

```bash
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## PydanticAI Tool Calling

The proxy accepts native provider tool payloads used by PydanticAI:

- OpenAI-compatible `tools`, `tool_choice`, assistant `tool_calls`, and `role: "tool"` result messages.
- Anthropic-compatible `tools`, `tool_choice`, `tool_use`, and `tool_result` content blocks.

Custom tools are executed by the client/framework, not by this proxy. The proxy converts tool definitions into a strict Claude prompt protocol and converts Claude tool-call protocol output back into provider-native wire shapes.

## Example OpenAI-Compatible Tool Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "What is the weather in London?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

## Configuration

Environment variables:

- `ANTHROPIC_API_KEY`: optional when Claude Code credentials are mounted; enables live model discovery.
- `API_KEY`: optional bearer token required by clients calling the proxy.
- `PORT`: default `8000`.
- `CLAUDE_WRAPPER_HOST`: default `0.0.0.0`.
- `CLAUDE_CWD`: optional Claude working directory; otherwise an isolated temp directory is used.
- `MAX_TIMEOUT`: Claude SDK timeout in milliseconds, default `600000`.
- `CORS_ORIGINS`: JSON list, default `["*"]`.
- `DEFAULT_MODEL`: optional default model override.
- `CLAUDE_MODELS_OVERRIDE`: optional comma-separated advertised model list.
- `RATE_LIMIT_ENABLED`: default `true`; rate limiting is disabled automatically if `slowapi` is unavailable.

## Tests

```bash
python -m pytest -q
```

The test suite uses mocked Claude SDK behavior for tool-call compatibility tests, so it does not require live Claude credentials.
