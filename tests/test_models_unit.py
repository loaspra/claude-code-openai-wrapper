import pytest

from src.models import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicTextBlock,
    AnthropicTool,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
    ChatCompletionRequest,
    Message,
    OpenAITool,
    OpenAIToolCall,
)


def test_openai_message_accepts_tool_role():
    msg = Message(role="tool", tool_call_id="call_1", name="lookup", content="42")
    assert msg.role == "tool"
    assert msg.tool_call_id == "call_1"


def test_openai_request_accepts_pydanticai_tools():
    request = ChatCompletionRequest(
        messages=[Message(role="user", content="lookup")],
        tools=[
            OpenAITool(
                function={
                    "name": "lookup",
                    "description": "Lookup a value",
                    "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
                }
            )
        ],
        tool_choice="auto",
    )
    assert request.tools[0].function.name == "lookup"
    assert request.tool_choice == "auto"


def test_openai_assistant_accepts_tool_calls():
    msg = Message(
        role="assistant",
        content=None,
        tool_calls=[OpenAIToolCall(function={"name": "lookup", "arguments": '{"key":"x"}'})],
    )
    assert msg.tool_calls[0].function.name == "lookup"


def test_openai_rejects_multiple_choices():
    with pytest.raises(ValueError):
        ChatCompletionRequest(messages=[Message(role="user", content="hi")], n=2)


@pytest.mark.parametrize(
    ("effort", "tokens"),
    [
        ("none", 0),
        ("minimal", 1024),
        ("low", 2048),
        ("medium", 10000),
        ("high", 16384),
        ("xhigh", 32768),
    ],
)
def test_openai_reasoning_effort_maps_to_max_thinking_tokens(effort, tokens):
    request = ChatCompletionRequest(
        messages=[Message(role="user", content="think")],
        reasoning_effort=effort,
    )
    assert request.to_claude_options()["max_thinking_tokens"] == tokens


def test_openai_reasoning_effort_beats_legacy_max_tokens_mapping():
    request = ChatCompletionRequest(
        messages=[Message(role="user", content="think")],
        max_tokens=123,
        reasoning_effort="high",
    )
    assert request.to_claude_options()["max_thinking_tokens"] == 16384


def test_openai_nested_reasoning_effort_beats_flat_reasoning_effort():
    request = ChatCompletionRequest(
        messages=[Message(role="user", content="think")],
        reasoning_effort="low",
        reasoning={"effort": "medium"},
    )
    assert request.to_claude_options()["max_thinking_tokens"] == 10000


def test_openai_nested_reasoning_token_budget_beats_reasoning_effort():
    request = ChatCompletionRequest(
        messages=[Message(role="user", content="think")],
        reasoning_effort="high",
        reasoning={"max_thinking_tokens": 7777, "effort": "low"},
    )
    assert request.to_claude_options()["max_thinking_tokens"] == 7777


def test_openai_rejects_invalid_reasoning_effort():
    with pytest.raises(ValueError):
        ChatCompletionRequest(
            messages=[Message(role="user", content="think")],
            reasoning_effort="extreme",
        )


def test_anthropic_request_accepts_tools_and_tool_results():
    request = AnthropicMessagesRequest(
        model="claude-sonnet-4-6",
        messages=[
            AnthropicMessage(role="user", content="lookup"),
            AnthropicMessage(
                role="assistant",
                content=[AnthropicToolUseBlock(id="toolu_1", name="lookup", input={"key": "x"})],
            ),
            AnthropicMessage(
                role="user",
                content=[AnthropicToolResultBlock(tool_use_id="toolu_1", content="42")],
            ),
        ],
        tools=[AnthropicTool(name="lookup", input_schema={"type": "object"})],
        tool_choice="auto",
    )
    assert request.tools[0].name == "lookup"
    assert request.messages[1].content[0].type == "tool_use"
    assert request.messages[2].content[0].type == "tool_result"


def test_anthropic_text_blocks_convert_to_openai_messages():
    request = AnthropicMessagesRequest(
        model="claude-sonnet-4-6",
        messages=[AnthropicMessage(role="user", content=[AnthropicTextBlock(text="Hello")])],
    )
    messages = request.to_openai_messages()
    assert messages[0].role == "user"
    assert messages[0].content == "Hello"
