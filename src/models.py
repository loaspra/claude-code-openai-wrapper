from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


# Resolve the default model lazily (avoids circular imports). If the operator
# set DEFAULT_MODEL via env var, honor it; otherwise prefer the live-resolved
# latest Sonnet (set at startup by main.resolve_default_model), falling back
# to the static constant when resolution hasn't happened yet.
def get_default_model():
    from src import constants

    if constants.DEFAULT_MODEL_ENV:
        return constants.DEFAULT_MODEL_ENV
    return constants.RESOLVED_DEFAULT_MODEL or constants.DEFAULT_MODEL_FALLBACK


class ContentPart(BaseModel):
    """Content part for multimodal messages (OpenAI format)."""

    type: Literal["text"]
    text: str


class OpenAIToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class OpenAITool(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIToolFunction


class OpenAIToolCallFunction(BaseModel):
    name: str
    arguments: str


class OpenAIToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    type: Literal["function"] = "function"
    function: OpenAIToolCallFunction


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[ContentPart]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None

    @model_validator(mode="after")
    def normalize_content(self):
        """Convert array content to string for Claude Code compatibility."""
        if isinstance(self.content, list):
            # Extract text from content parts and concatenate
            text_parts = []
            for part in self.content:
                if isinstance(part, ContentPart) and part.type == "text":
                    text_parts.append(part.text)
                elif isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))

            # Join all text parts with newlines
            self.content = "\n".join(text_parts) if text_parts else ""

        return self


class StreamOptions(BaseModel):
    """Options for streaming responses."""

    include_usage: bool = Field(
        default=False, description="Include usage information in the final streaming chunk"
    )


class ChatCompletionRequest(BaseModel):
    model: str = Field(default_factory=get_default_model)
    messages: List[Message]
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens to generate in the completion (OpenAI standard)"
    )
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Optional[Union[Literal["auto", "none", "required"], Dict[str, Any]]] = None
    stream_options: Optional[StreamOptions] = Field(
        default=None, description="Options for streaming responses"
    )

    @field_validator("n")
    @classmethod
    def validate_n(cls, v):
        if v > 1:
            raise ValueError(
                "Claude Code SDK does not support multiple choices (n > 1). Only single response generation is supported."
            )
        return v

    def log_parameter_info(self):
        """Log information about parameter handling."""
        info_messages = []
        warnings = []

        if self.temperature != 1.0:
            info_messages.append(
                f"temperature={self.temperature} will be applied via system prompt (best-effort)"
            )

        if self.top_p != 1.0:
            info_messages.append(
                f"top_p={self.top_p} will be applied via system prompt (best-effort)"
            )

        if self.max_tokens is not None or self.max_completion_tokens is not None:
            max_val = self.max_completion_tokens or self.max_tokens
            info_messages.append(
                f"max_tokens={max_val} will be mapped to max_thinking_tokens (best-effort)"
            )

        if self.presence_penalty != 0:
            warnings.append(
                f"presence_penalty={self.presence_penalty} is not supported and will be ignored"
            )

        if self.frequency_penalty != 0:
            warnings.append(
                f"frequency_penalty={self.frequency_penalty} is not supported and will be ignored"
            )

        if self.logit_bias:
            warnings.append("logit_bias is not supported and will be ignored")

        if self.stop:
            warnings.append("stop sequences are not supported and will be ignored")

        for msg in info_messages:
            logger.info(f"OpenAI API compatibility: {msg}")

        for warning in warnings:
            logger.warning(f"OpenAI API compatibility: {warning}")

    def get_sampling_instructions(self) -> Optional[str]:
        """
        Generate sampling instructions based on temperature and top_p.

        Returns system prompt text to approximate the requested sampling behavior.
        """
        instructions = []

        if self.temperature is not None and self.temperature != 1.0:
            if self.temperature < 0.3:
                instructions.append(
                    "Be highly focused and deterministic in your responses. Choose the most likely and predictable options."
                )
            elif self.temperature < 0.7:
                instructions.append(
                    "Be somewhat focused and consistent in your responses, preferring reliable and expected solutions."
                )
            elif self.temperature > 1.5:
                instructions.append(
                    "Be highly creative and exploratory in your responses. Consider unusual and diverse approaches."
                )
            elif self.temperature > 1.0:
                instructions.append(
                    "Be creative and varied in your responses, exploring different approaches and possibilities."
                )

        if self.top_p is not None and self.top_p < 1.0:
            if self.top_p < 0.5:
                instructions.append(
                    "Focus on the most probable and mainstream solutions, avoiding less likely alternatives."
                )
            elif self.top_p < 0.9:
                instructions.append(
                    "Prefer well-established and common approaches over unusual ones."
                )

        return " ".join(instructions) if instructions else None

    def to_claude_options(self) -> Dict[str, Any]:
        """Convert OpenAI request parameters to Claude Code SDK options."""
        # Log parameter handling information
        self.log_parameter_info()

        options = {}

        # Direct mappings
        if self.model:
            options["model"] = self.model

        # Map max_tokens to max_thinking_tokens (best effort)
        max_token_value = self.max_completion_tokens or self.max_tokens
        if max_token_value is not None:
            # Claude SDK doesn't have exact token limiting, but we can try max_thinking_tokens
            # This is approximate and may not work as expected
            options["max_thinking_tokens"] = max_token_value
            logger.info(
                f"Mapped max_tokens={max_token_value} to max_thinking_tokens (approximate behavior)"
            )

        # Use user field for session identification if provided
        if self.user:
            # Could be used for analytics/logging or session tracking
            logger.info(f"Request from user: {self.user}")

        return options


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls", "null"]] = (
        None
    )


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class StreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[Literal["stop", "length", "content_filter", "null"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str
    choices: List[StreamChoice]
    usage: Optional[Usage] = Field(
        default=None,
        description="Usage information (only in final chunk when stream_options.include_usage=true)",
    )
    system_fingerprint: Optional[str] = None


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ============================================================================
# Anthropic API Compatible Models (for /v1/messages endpoint)
# ============================================================================


class AnthropicTextBlock(BaseModel):
    """Anthropic text content block."""

    type: Literal["text"] = "text"
    text: str


class AnthropicToolUseBlock(BaseModel):
    """Anthropic tool_use content block."""

    type: Literal["tool_use"] = "tool_use"
    id: str = Field(default_factory=lambda: f"toolu_{uuid.uuid4().hex[:24]}")
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)


class AnthropicToolResultBlock(BaseModel):
    """Anthropic tool_result content block."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[AnthropicTextBlock]]
    is_error: Optional[bool] = None


AnthropicContentBlock = Union[AnthropicTextBlock, AnthropicToolUseBlock, AnthropicToolResultBlock]


class AnthropicTool(BaseModel):
    """Anthropic Messages API tool definition."""

    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class AnthropicMessage(BaseModel):
    """Anthropic message format."""

    role: Literal["user", "assistant"]
    content: Union[str, List[AnthropicContentBlock]]


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request format."""

    model: str
    messages: List[AnthropicMessage]
    max_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    system: Optional[Union[str, List[Any]]] = Field(default=None, description="System prompt")

    @model_validator(mode="after")
    def normalize_system(self):
        if isinstance(self.system, list):
            parts = [
                b.get("text", "") if isinstance(b, dict) else (b.text if hasattr(b, "text") else "")
                for b in self.system
                if (isinstance(b, dict) and b.get("type") == "text")
                or (hasattr(b, "type") and b.type == "text")
            ]
            self.system = "\n".join(parts) if parts else None
        return self

    temperature: Optional[float] = Field(default=1.0, ge=0, le=1)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=0)
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[Union[Literal["auto", "any"], Dict[str, Any]]] = None

    def to_openai_messages(self) -> List[Message]:
        """Convert Anthropic messages to OpenAI format."""
        result = []
        for msg in self.messages:
            content = msg.content
            if isinstance(content, list):
                # Extract text from content blocks
                text_parts = [
                    block.text for block in content if isinstance(block, AnthropicTextBlock)
                ]
                content = "\n".join(text_parts)
            result.append(Message(role=msg.role, content=content))
        return result


class AnthropicUsage(BaseModel):
    """Anthropic usage information."""

    input_tokens: int
    output_tokens: int


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response format."""

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[Union[AnthropicTextBlock, AnthropicToolUseBlock]]
    model: str
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = (
        "end_turn"
    )
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage
