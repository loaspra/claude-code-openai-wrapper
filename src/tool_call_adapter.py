import json
import re
from typing import Any, Dict, List, Optional

from src.models import (
    AnthropicTool,
    AnthropicToolUseBlock,
    Message,
    OpenAITool,
    OpenAIToolCall,
    OpenAIToolCallFunction,
)

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"


def build_openai_tool_prompt(
    tools: Optional[List[OpenAITool]], tool_choice: Any = None
) -> Optional[str]:
    if not tools:
        return None

    tool_specs = [tool.model_dump(exclude_none=True) for tool in tools]
    return _build_tool_prompt(tool_specs, tool_choice)


def build_anthropic_tool_prompt(
    tools: Optional[List[AnthropicTool]], tool_choice: Any = None
) -> Optional[str]:
    if not tools:
        return None

    tool_specs = [tool.model_dump(exclude_none=True) for tool in tools]
    return _build_tool_prompt(tool_specs, tool_choice)


def _build_tool_prompt(tool_specs: List[Dict[str, Any]], tool_choice: Any) -> str:
    return (
        "You may call one or more custom client tools. These tools are executed by the "
        "API client, not by Claude Code.\n"
        "If a tool is needed, respond with only a JSON object wrapped in "
        f"{TOOL_CALL_START} and {TOOL_CALL_END}. Do not include prose.\n"
        "The JSON shape must be: "
        '{"tool_calls":[{"name":"tool_name","arguments":{"arg":"value"}}]}.\n'
        "If no tool is needed, answer normally.\n"
        f"tool_choice: {json.dumps(tool_choice)}\n"
        f"tools: {json.dumps(tool_specs, separators=(',', ':'))}"
    )


def append_openai_tool_messages(messages: List[Message]) -> str:
    parts = []
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            parts.append(
                "Assistant requested tool calls: "
                + json.dumps([call.model_dump() for call in message.tool_calls])
            )
        elif message.role == "tool":
            parts.append(
                "Tool result "
                + json.dumps(
                    {
                        "tool_call_id": message.tool_call_id,
                        "name": message.name,
                        "content": message.content or "",
                    }
                )
            )
    return "\n\n".join(parts)


def parse_openai_tool_calls(text: str) -> List[OpenAIToolCall]:
    payload = _extract_tool_payload(text)
    if not payload:
        return []

    calls = payload.get("tool_calls")
    if not isinstance(calls, list):
        return []

    result = []
    for call in calls:
        if not isinstance(call, dict) or not call.get("name"):
            continue
        arguments = call.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, separators=(",", ":"))
        data = {"function": OpenAIToolCallFunction(name=call["name"], arguments=arguments)}
        if call.get("id"):
            data["id"] = call["id"]
        result.append(OpenAIToolCall(**data))
    return result


def parse_anthropic_tool_uses(text: str) -> List[AnthropicToolUseBlock]:
    payload = _extract_tool_payload(text)
    if not payload:
        return []

    calls = payload.get("tool_calls")
    if not isinstance(calls, list):
        return []

    result = []
    for call in calls:
        if not isinstance(call, dict) or not call.get("name"):
            continue
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        data = {"name": call["name"], "input": arguments}
        if call.get("id"):
            data["id"] = call["id"]
        result.append(AnthropicToolUseBlock(**data))
    return result


def _extract_tool_payload(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    match = re.search(
        rf"{re.escape(TOOL_CALL_START)}\s*(.*?)\s*{re.escape(TOOL_CALL_END)}",
        text,
        flags=re.DOTALL,
    )
    raw = match.group(1) if match else text.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
