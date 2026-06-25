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
