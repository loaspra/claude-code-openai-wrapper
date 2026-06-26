import os
import json
import asyncio
import logging
import time
import uuid
from typing import Optional, AsyncGenerator, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
import httpx
from dotenv import load_dotenv

from src.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Choice,
    Message,
    Usage,
    StreamChoice,
    # Anthropic API compatible models
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicTextBlock,
    AnthropicUsage,
)
from src.claude_cli import ClaudeCodeCLI
from src.message_adapter import MessageAdapter
from src.auth import verify_api_key, security, validate_claude_code_auth, get_claude_code_auth_info
from src.parameter_validator import ParameterValidator, CompatibilityReporter
from src.landing import render_landing_page
from src.tool_call_adapter import (
    append_openai_tool_messages,
    build_anthropic_tool_prompt,
    build_openai_tool_prompt,
    parse_anthropic_tool_uses,
    parse_openai_tool_calls,
)
from src.rate_limiter import (
    limiter,
    rate_limit_exceeded_handler,
    rate_limit_endpoint,
)
from datetime import datetime, timezone

from src import constants
from src.constants import (
    ANTHROPIC_MODELS_URL,
    ANTHROPIC_VERSION,
    CLAUDE_MODELS,
    DEFAULT_MODEL_FALLBACK,
    MODEL_LIST_CACHE_TTL_SECONDS,
    MODEL_LIST_ERROR_TTL_SECONDS,
    MODEL_LIST_REQUEST_TIMEOUT_SECONDS,
)

# Load environment variables
load_dotenv()

# Configure logging based on debug mode
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "yes", "on")
VERBOSE = os.getenv("VERBOSE", "false").lower() in ("true", "1", "yes", "on")

# Set logging level based on debug/verbose mode
log_level = logging.DEBUG if (DEBUG_MODE or VERBOSE) else logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Best-effort cache for Anthropic's live Models API.  The static constants remain
# the fallback so /v1/models keeps working for Claude CLI credentials, local
# development, and transient Anthropic API outages.
_model_list_cache: Dict[str, Any] = {"expires_at": 0.0, "models": None}
# Serializes cache refreshes so concurrent /v1/models requests at TTL expiry
# don't all stampede the upstream Anthropic API.
_model_list_lock = asyncio.Lock()


def _iso_to_unix(value: Any) -> Optional[int]:
    """Convert an Anthropic ISO-8601 'created_at' string to a unix timestamp."""
    if not isinstance(value, str):
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _openai_model_from_anthropic(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Anthropic ModelInfo object to OpenAI-compatible model metadata."""
    created = _iso_to_unix(model_info.get("created_at"))
    model: Dict[str, Any] = {
        "id": model_info["id"],
        "object": "model",
        "created": created if created is not None else int(datetime.now(timezone.utc).timestamp()),
        "owned_by": "anthropic",
    }

    # Preserve useful Anthropic metadata for clients that want it.  OpenAI clients
    # ignore unknown keys, and the existing id/object/owned_by shape is retained.
    for key in (
        "display_name",
        "created_at",
        "max_input_tokens",
        "max_tokens",
        "capabilities",
        "type",
    ):
        if key in model_info:
            model[key] = model_info[key]

    return model


def _fallback_model_payload() -> List[Dict[str, Any]]:
    now = int(datetime.now(timezone.utc).timestamp())
    return [
        {"id": model_id, "object": "model", "created": now, "owned_by": "anthropic"}
        for model_id in CLAUDE_MODELS
    ]


async def _fetch_anthropic_models() -> Optional[List[Dict[str, Any]]]:
    """Fetch all available models from Anthropic, returning None on fallback-worthy errors."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    headers = {
        "anthropic-version": ANTHROPIC_VERSION,
        "x-api-key": api_key,
    }
    beta_header = os.getenv("ANTHROPIC_BETA") or os.getenv("ANTHROPIC_BETA_HEADER")
    if beta_header:
        headers["anthropic-beta"] = beta_header

    params: Dict[str, Any] = {"limit": 1000}
    models: List[Dict[str, Any]] = []

    try:
        async with httpx.AsyncClient(timeout=MODEL_LIST_REQUEST_TIMEOUT_SECONDS) as client:
            while True:
                response = await client.get(ANTHROPIC_MODELS_URL, headers=headers, params=params)
                response.raise_for_status()
                payload = response.json()
                models.extend(
                    _openai_model_from_anthropic(model)
                    for model in payload.get("data", [])
                    if model.get("id")
                )

                if not payload.get("has_more") or not payload.get("last_id"):
                    break
                params["after_id"] = payload["last_id"]
    except Exception as exc:  # noqa: BLE001 - endpoint should degrade gracefully
        logger.warning("Failed to fetch Anthropic model list, using fallback: %s", exc)
        return None

    return models or None


async def get_available_models() -> List[Dict[str, Any]]:
    """Return live Anthropic models when possible, with cached static fallback."""
    if os.getenv("CLAUDE_MODELS_OVERRIDE", "").strip():
        return _fallback_model_payload()

    now = time.time()
    cached_models = _model_list_cache.get("models")
    if cached_models and now < float(_model_list_cache.get("expires_at", 0)):
        return cached_models

    async with _model_list_lock:
        # Recheck inside the lock so the first waiter populates the cache and
        # subsequent waiters return without re-fetching.
        now = time.time()
        cached_models = _model_list_cache.get("models")
        if cached_models and now < float(_model_list_cache.get("expires_at", 0)):
            return cached_models

        live_models = await _fetch_anthropic_models()
        if live_models:
            _model_list_cache.update(
                {"models": live_models, "expires_at": now + MODEL_LIST_CACHE_TTL_SECONDS}
            )
            return live_models

        fallback_models = _fallback_model_payload()
        # Use a short TTL on failure so transient outages don't suppress live
        # discovery for the full MODEL_LIST_CACHE_TTL_SECONDS window.
        _model_list_cache.update(
            {"models": fallback_models, "expires_at": now + MODEL_LIST_ERROR_TTL_SECONDS}
        )
        return fallback_models


def _pick_latest_sonnet(models: List[Dict[str, Any]]) -> Optional[str]:
    """Return the id of the newest Sonnet model in `models`, or None."""
    sonnets = [m for m in models if isinstance(m.get("id"), str) and "sonnet" in m["id"].lower()]
    if not sonnets:
        return None
    # Prefer Anthropic-provided created_at; fall back to the int `created` we set,
    # then to id-sort (date-suffixed ids sort correctly newest-last).
    sonnets.sort(
        key=lambda m: (
            _iso_to_unix(m.get("created_at")) or m.get("created") or 0,
            m["id"],
        )
    )
    return sonnets[-1]["id"]


async def resolve_default_model() -> Optional[str]:
    """Pick the latest Sonnet from /v1/models and store it as the default.

    Skipped when the operator pinned DEFAULT_MODEL via env var, or when no
    ANTHROPIC_API_KEY is configured (live discovery is the only auth-aware
    path; Claude CLI credential users get the static
    DEFAULT_MODEL_FALLBACK).
    """
    if constants.DEFAULT_MODEL_ENV:
        return constants.DEFAULT_MODEL_ENV

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.info(
            "Live model discovery disabled (no ANTHROPIC_API_KEY); " "using fallback default %s",
            DEFAULT_MODEL_FALLBACK,
        )
        return None

    try:
        models = await get_available_models()
    except Exception as exc:  # noqa: BLE001 - startup should never abort on this
        logger.warning("Could not resolve default model from /v1/models: %s", exc)
        return None

    latest = _pick_latest_sonnet(models)
    if latest:
        constants.RESOLVED_DEFAULT_MODEL = latest
        logger.info("Resolved default model from Anthropic Models API: %s", latest)
        return latest

    logger.info(
        "No Sonnet model found in /v1/models response; using fallback %s",
        DEFAULT_MODEL_FALLBACK,
    )
    return None


# Initialize Claude CLI
claude_cli = ClaudeCodeCLI(
    timeout=int(os.getenv("MAX_TIMEOUT", "600000")), cwd=os.getenv("CLAUDE_CWD")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Verify Claude Code authentication and CLI on startup."""
    logger.info("Verifying Claude Code authentication and CLI...")

    # Validate authentication first
    auth_valid, auth_info = validate_claude_code_auth()

    if not auth_valid:
        logger.error("❌ Claude Code authentication failed!")
        for error in auth_info.get("errors", []):
            logger.error(f"  - {error}")
        logger.warning("Authentication setup guide:")
        logger.warning("  1. Set ANTHROPIC_API_KEY, or")
        logger.warning("  2. Mount existing Claude Code credentials into the runtime")
    else:
        logger.info(f"✅ Claude Code authentication validated: {auth_info['method']}")

    # Verify Claude Agent SDK with timeout for graceful degradation
    try:
        logger.info("Testing Claude Agent SDK connection...")
        # Use asyncio.wait_for to enforce timeout (30 seconds)
        cli_verified = await asyncio.wait_for(claude_cli.verify_cli(), timeout=30.0)

        if cli_verified:
            logger.info("✅ Claude Agent SDK verified successfully")
        else:
            logger.warning("⚠️  Claude Agent SDK verification returned False")
            logger.warning("The server will start, but requests may fail.")
    except asyncio.TimeoutError:
        logger.warning("⚠️  Claude Agent SDK verification timed out (30s)")
        logger.warning("This may indicate network issues or SDK configuration problems.")
        logger.warning("The server will start, but first request may be slow.")
    except Exception as e:
        logger.error(f"⚠️  Claude Agent SDK verification failed: {e}")
        logger.warning("The server will start, but requests may fail.")
        logger.warning("Check that Claude Code CLI is properly installed and authenticated.")

    # Log debug information if debug mode is enabled
    if DEBUG_MODE or VERBOSE:
        logger.debug("🔧 Debug mode enabled - Enhanced logging active")
        logger.debug("🔧 Environment variables:")
        logger.debug(f"   DEBUG_MODE: {DEBUG_MODE}")
        logger.debug(f"   VERBOSE: {VERBOSE}")
        logger.debug(f"   PORT: {os.getenv('PORT', '8000')}")
        cors_origins_val = os.getenv("CORS_ORIGINS", '["*"]')
        logger.debug(f"   CORS_ORIGINS: {cors_origins_val}")
        logger.debug(f"   MAX_TIMEOUT: {os.getenv('MAX_TIMEOUT', '600000')}")
        logger.debug(f"   CLAUDE_CWD: {os.getenv('CLAUDE_CWD', 'Not set')}")
        logger.debug("🔧 Available endpoints:")
        logger.debug("   POST /v1/chat/completions - Main chat endpoint")
        logger.debug("   GET  /v1/models - List available models")
        logger.debug("   GET  /v1/auth/status - Authentication status")
        logger.debug("   GET  /health - Health check")
        logger.debug(f"🔧 API Key protection: {'Enabled' if os.getenv('API_KEY') else 'Disabled'}")

    # Resolve the default model from the live Anthropic Models API so /v1/chat
    # uses the latest Sonnet without a code change. Best-effort: any failure
    # leaves the static fallback in place.
    try:
        await resolve_default_model()
    except Exception as e:
        logger.warning(f"Default model resolution skipped: {e}")

    yield


# Create FastAPI app
app = FastAPI(
    title="Claude Code OpenAI API Wrapper",
    description="OpenAI-compatible API for Claude Code",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
cors_origins = json.loads(os.getenv("CORS_ORIGINS", '["*"]'))
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting error handler
if limiter:
    app.state.limiter = limiter
    app.add_exception_handler(429, rate_limit_exceeded_handler)

# Security configuration
MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", str(10 * 1024 * 1024)))  # 10MB default

# Add middleware
from starlette.middleware.base import BaseHTTPMiddleware


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request for audit trails."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Limit request body size to prevent DoS attacks."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_SIZE} bytes.",
                        "type": "request_too_large",
                        "code": 413,
                    }
                },
            )
        return await call_next(request)


# Add security middleware (order matters - first added = last executed)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(RequestSizeLimitMiddleware)


class DebugLoggingMiddleware(BaseHTTPMiddleware):
    """ASGI-compliant middleware for logging request/response details when debug mode is enabled."""

    async def dispatch(self, request: Request, call_next):
        # Get request ID for correlation
        request_id = getattr(request.state, "request_id", "unknown")

        if not (DEBUG_MODE or VERBOSE):
            return await call_next(request)

        # Log request details
        start_time = asyncio.get_event_loop().time()

        # Log basic request info with request ID for correlation
        logger.debug(f"🔍 [{request_id}] Incoming request: {request.method} {request.url}")
        logger.debug(f"🔍 [{request_id}] Headers: {dict(request.headers)}")

        # For POST requests, try to log body (but don't break if we can't)
        body_logged = False
        if request.method == "POST" and request.url.path.startswith("/v1/"):
            try:
                # Only attempt to read body if it's reasonable size and content-type
                content_length = request.headers.get("content-length")
                if content_length and int(content_length) < 100000:  # Less than 100KB
                    body = await request.body()
                    if body:
                        try:
                            import json as json_lib

                            parsed_body = json_lib.loads(body.decode())
                            logger.debug(
                                f"🔍 Request body: {json_lib.dumps(parsed_body, indent=2)}"
                            )
                            body_logged = True
                        except:
                            logger.debug(f"🔍 Request body (raw): {body.decode()[:500]}...")
                            body_logged = True
            except Exception as e:
                logger.debug(f"🔍 Could not read request body: {e}")

        if not body_logged and request.method == "POST":
            logger.debug("🔍 Request body: [not logged - streaming or large payload]")

        # Process the request
        try:
            response = await call_next(request)

            # Log response details
            end_time = asyncio.get_event_loop().time()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds

            logger.debug(f"🔍 Response: {response.status_code} in {duration:.2f}ms")

            return response

        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            duration = (end_time - start_time) * 1000

            logger.debug(f"🔍 Request failed after {duration:.2f}ms: {e}")
            raise


# Add the debug middleware
app.add_middleware(DebugLoggingMiddleware)


# Custom exception handler for 422 validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with detailed debugging information."""

    # Log the validation error details
    logger.error(f"❌ Request validation failed for {request.method} {request.url}")
    logger.error(f"❌ Validation errors: {exc.errors()}")

    # Create detailed error response
    error_details = []
    for error in exc.errors():
        location = " -> ".join(str(loc) for loc in error.get("loc", []))
        error_details.append(
            {
                "field": location,
                "message": error.get("msg", "Unknown validation error"),
                "type": error.get("type", "validation_error"),
                "input": error.get("input"),
            }
        )

    # If debug mode is enabled, include the raw request body
    debug_info = {}
    if DEBUG_MODE or VERBOSE:
        try:
            body = await request.body()
            if body:
                debug_info["raw_request_body"] = body.decode()
        except:
            debug_info["raw_request_body"] = "Could not read request body"

    error_response = {
        "error": {
            "message": "Request validation failed - the request body doesn't match the expected format",
            "type": "validation_error",
            "code": "invalid_request_error",
            "details": error_details,
            "help": {
                "common_issues": [
                    "Missing required fields (model, messages)",
                    "Invalid field types (e.g. messages should be an array)",
                    "Invalid role values (must be 'system', 'user', or 'assistant')",
                    "Invalid parameter ranges (e.g. temperature must be 0-2)",
                ],
                "debug_tip": "Set DEBUG_MODE=true or VERBOSE=true environment variable for more detailed logging",
            },
        }
    }

    # Add debug info if available
    if debug_info:
        error_response["error"]["debug"] = debug_info

    return JSONResponse(status_code=422, content=error_response)


async def generate_streaming_response(
    request: ChatCompletionRequest, request_id: str, claude_headers: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[str, None]:
    """Generate SSE formatted streaming response."""
    try:
        # Convert messages to prompt
        prompt, system_prompt = MessageAdapter.messages_to_prompt(request.messages)

        # Add sampling instructions from temperature/top_p if present
        sampling_instructions = request.get_sampling_instructions()
        if sampling_instructions:
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{sampling_instructions}"
            else:
                system_prompt = sampling_instructions
            logger.debug(f"Added sampling instructions: {sampling_instructions}")

        tool_prompt = build_openai_tool_prompt(request.tools, request.tool_choice)
        if tool_prompt:
            system_prompt = f"{system_prompt}\n\n{tool_prompt}" if system_prompt else tool_prompt

        tool_context = append_openai_tool_messages(request.messages)
        if tool_context:
            prompt = f"{prompt}\n\n{tool_context}"

        # Filter content for unsupported features
        prompt = MessageAdapter.filter_content(prompt)
        if system_prompt:
            system_prompt = MessageAdapter.filter_content(system_prompt)

        # Get Claude Agent SDK options from request
        claude_options = request.to_claude_options()

        # Merge with Claude-specific headers if provided
        if claude_headers:
            claude_options.update(claude_headers)

        # Validate model
        if claude_options.get("model"):
            ParameterValidator.validate_model(claude_options["model"])

        if "max_turns" not in claude_options:
            claude_options["max_turns"] = 3 if request.tools else 1

        # Run Claude Code
        chunks_buffer = []
        role_sent = False  # Track if we've sent the initial role chunk
        content_sent = False  # Track if we've sent any content

        async for chunk in claude_cli.run_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            model=claude_options.get("model"),
            max_turns=claude_options.get("max_turns", 10),
            allowed_tools=claude_options.get("allowed_tools"),
            disallowed_tools=claude_options.get("disallowed_tools"),
            permission_mode=claude_options.get("permission_mode"),
            max_thinking_tokens=claude_options.get("max_thinking_tokens"),
            stream=True,
        ):
            chunks_buffer.append(chunk)

            # Check if we have an assistant message
            # Handle both old format (type/message structure) and new format (direct content)
            content = None
            if chunk.get("type") == "assistant" and "message" in chunk:
                # Old format: {"type": "assistant", "message": {"content": [...]}}
                message = chunk["message"]
                if isinstance(message, dict) and "content" in message:
                    content = message["content"]
            elif "content" in chunk and isinstance(chunk["content"], list):
                # New format: {"content": [TextBlock(...)]}  (converted AssistantMessage)
                content = chunk["content"]

            if content is not None:
                # Send initial role chunk if we haven't already
                if not role_sent:
                    initial_chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        model=request.model,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta={"role": "assistant", "content": ""},
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {initial_chunk.model_dump_json()}\n\n"
                    role_sent = True

                # Handle content blocks
                if isinstance(content, list):
                    for block in content:
                        # Handle TextBlock objects from Claude Agent SDK
                        if hasattr(block, "text"):
                            raw_text = block.text
                        # Handle dictionary format for backward compatibility
                        elif isinstance(block, dict) and block.get("type") == "text":
                            raw_text = block.get("text", "")
                        else:
                            continue

                        # Filter out tool usage and thinking blocks
                        filtered_text = MessageAdapter.filter_content(raw_text)

                        if filtered_text and not filtered_text.isspace():
                            # Create streaming chunk
                            stream_chunk = ChatCompletionStreamResponse(
                                id=request_id,
                                model=request.model,
                                choices=[
                                    StreamChoice(
                                        index=0,
                                        delta={"content": filtered_text},
                                        finish_reason=None,
                                    )
                                ],
                            )

                            yield f"data: {stream_chunk.model_dump_json()}\n\n"
                            content_sent = True

                elif isinstance(content, str):
                    # Filter out tool usage and thinking blocks
                    filtered_content = MessageAdapter.filter_content(content)

                    if filtered_content and not filtered_content.isspace():
                        # Create streaming chunk
                        stream_chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            model=request.model,
                            choices=[
                                StreamChoice(
                                    index=0, delta={"content": filtered_content}, finish_reason=None
                                )
                            ],
                        )

                        yield f"data: {stream_chunk.model_dump_json()}\n\n"
                        content_sent = True

        # Handle case where no role was sent (send at least role chunk)
        if not role_sent:
            # Send role chunk with empty content if we never got any assistant messages
            initial_chunk = ChatCompletionStreamResponse(
                id=request_id,
                model=request.model,
                choices=[
                    StreamChoice(
                        index=0, delta={"role": "assistant", "content": ""}, finish_reason=None
                    )
                ],
            )
            yield f"data: {initial_chunk.model_dump_json()}\n\n"
            role_sent = True

        # If we sent role but no content, send a minimal response
        if role_sent and not content_sent:
            fallback_chunk = ChatCompletionStreamResponse(
                id=request_id,
                model=request.model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta={"content": "I'm unable to provide a response at the moment."},
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {fallback_chunk.model_dump_json()}\n\n"

        # Extract assistant response from all chunks
        assistant_content = None
        if chunks_buffer:
            assistant_content = claude_cli.parse_claude_message(chunks_buffer)

        # Prepare usage data if requested
        usage_data = None
        if request.stream_options and request.stream_options.include_usage:
            # Estimate token usage based on prompt and completion
            completion_text = assistant_content or ""
            token_usage = claude_cli.estimate_token_usage(prompt, completion_text, request.model)
            usage_data = Usage(
                prompt_tokens=token_usage["prompt_tokens"],
                completion_tokens=token_usage["completion_tokens"],
                total_tokens=token_usage["total_tokens"],
            )
            logger.debug(f"Estimated usage: {usage_data}")

        # Send final chunk with finish reason and optionally usage data
        final_chunk = ChatCompletionStreamResponse(
            id=request_id,
            model=request.model,
            choices=[StreamChoice(index=0, delta={}, finish_reason="stop")],
            usage=usage_data,
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = {"error": {"message": str(e), "type": "streaming_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"


@app.post("/v1/chat/completions")
@rate_limit_endpoint("chat")
async def chat_completions(
    request_body: ChatCompletionRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """OpenAI-compatible chat completions endpoint."""
    # Check FastAPI API key if configured
    await verify_api_key(request, credentials)

    # Validate Claude Code authentication
    auth_valid, auth_info = validate_claude_code_auth()

    if not auth_valid:
        error_detail = {
            "message": "Claude Code authentication failed",
            "errors": auth_info.get("errors", []),
            "method": auth_info.get("method", "none"),
            "help": "Check /v1/auth/status for detailed authentication information",
        }
        raise HTTPException(status_code=503, detail=error_detail)

    try:
        request_id = f"chatcmpl-{os.urandom(8).hex()}"

        # Extract Claude-specific parameters from headers
        claude_headers = ParameterValidator.extract_claude_headers(dict(request.headers))

        # Log compatibility info
        if logger.isEnabledFor(logging.DEBUG):
            compatibility_report = CompatibilityReporter.generate_compatibility_report(request_body)
            logger.debug(f"Compatibility report: {compatibility_report}")

        if request_body.stream:
            # Return streaming response
            return StreamingResponse(
                generate_streaming_response(request_body, request_id, claude_headers),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Non-streaming response
            logger.info(f"Chat completion: total_messages={len(request_body.messages)}")

            # Convert messages to prompt
            prompt, system_prompt = MessageAdapter.messages_to_prompt(request_body.messages)

            # Add sampling instructions from temperature/top_p if present
            sampling_instructions = request_body.get_sampling_instructions()
            if sampling_instructions:
                if system_prompt:
                    system_prompt = f"{system_prompt}\n\n{sampling_instructions}"
                else:
                    system_prompt = sampling_instructions
                logger.debug(f"Added sampling instructions: {sampling_instructions}")

            tool_prompt = build_openai_tool_prompt(request_body.tools, request_body.tool_choice)
            if tool_prompt:
                system_prompt = (
                    f"{system_prompt}\n\n{tool_prompt}" if system_prompt else tool_prompt
                )

            tool_context = append_openai_tool_messages(request_body.messages)
            if tool_context:
                prompt = f"{prompt}\n\n{tool_context}"

            # Filter content
            prompt = MessageAdapter.filter_content(prompt)
            if system_prompt:
                system_prompt = MessageAdapter.filter_content(system_prompt)

            # Get Claude Agent SDK options from request
            claude_options = request_body.to_claude_options()

            # Merge with Claude-specific headers
            if claude_headers:
                claude_options.update(claude_headers)

            # Validate model
            if claude_options.get("model"):
                ParameterValidator.validate_model(claude_options["model"])

            if "max_turns" not in claude_options:
                claude_options["max_turns"] = 3 if request_body.tools else 1

            # Collect all chunks
            chunks = []
            async for chunk in claude_cli.run_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                model=claude_options.get("model"),
                max_turns=claude_options.get("max_turns", 10),
                allowed_tools=claude_options.get("allowed_tools"),
                disallowed_tools=claude_options.get("disallowed_tools"),
                permission_mode=claude_options.get("permission_mode"),
                max_thinking_tokens=claude_options.get("max_thinking_tokens"),
                stream=False,
            ):
                chunks.append(chunk)

            # Extract assistant message
            raw_assistant_content = claude_cli.parse_claude_message(chunks)

            if not raw_assistant_content:
                raise HTTPException(status_code=500, detail="No response from Claude Code")

            # Filter out tool usage and thinking blocks
            assistant_content = MessageAdapter.filter_content(raw_assistant_content)

            tool_calls = (
                parse_openai_tool_calls(raw_assistant_content) if request_body.tools else []
            )
            if tool_calls:
                assistant_content = None

            # Estimate tokens (rough approximation)
            prompt_tokens = MessageAdapter.estimate_tokens(prompt)
            completion_tokens = MessageAdapter.estimate_tokens(
                assistant_content or raw_assistant_content
            )

            # Create response
            response = ChatCompletionResponse(
                id=request_id,
                model=request_body.model,
                choices=[
                    Choice(
                        index=0,
                        message=Message(
                            role="assistant",
                            content=assistant_content,
                            tool_calls=tool_calls or None,
                        ),
                        finish_reason="tool_calls" if tool_calls else "stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/messages")
@rate_limit_endpoint("chat")
async def anthropic_messages(
    request_body: AnthropicMessagesRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Anthropic Messages API compatible endpoint.

    This endpoint provides compatibility with the native Anthropic SDK,
    allowing tools like VC to use this wrapper via the VC_API_BASE setting.
    """
    # Check FastAPI API key if configured
    await verify_api_key(request, credentials)

    # Validate Claude Code authentication
    auth_valid, auth_info = validate_claude_code_auth()

    if not auth_valid:
        error_detail = {
            "message": "Claude Code authentication failed",
            "errors": auth_info.get("errors", []),
            "method": auth_info.get("method", "none"),
            "help": "Check /v1/auth/status for detailed authentication information",
        }
        raise HTTPException(status_code=503, detail=error_detail)

    try:
        logger.info(f"Anthropic Messages API request: model={request_body.model}")

        # Convert Anthropic messages to internal format
        messages = request_body.to_openai_messages()

        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            if msg.role == "user":
                prompt_parts.append(msg.content)
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt = "\n\n".join(prompt_parts)
        system_prompt = request_body.system

        tool_prompt = build_anthropic_tool_prompt(request_body.tools, request_body.tool_choice)
        if tool_prompt:
            system_prompt = f"{system_prompt}\n\n{tool_prompt}" if system_prompt else tool_prompt

        tool_results = []
        for msg in request_body.messages:
            if isinstance(msg.content, list):
                for block in msg.content:
                    if getattr(block, "type", None) == "tool_result":
                        tool_results.append(block.model_dump(exclude_none=True))
        if tool_results:
            prompt = f"{prompt}\n\nTool results: {json.dumps(tool_results)}"

        # Filter content
        prompt = MessageAdapter.filter_content(prompt)
        if system_prompt:
            system_prompt = MessageAdapter.filter_content(system_prompt)

        # Run Claude Code. Custom client tools are represented in the prompt and
        # executed by the caller (for example PydanticAI), not by this proxy.
        chunks = []
        async for chunk in claude_cli.run_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            model=request_body.model,
            max_turns=3 if request_body.tools else 1,
            stream=False,
        ):
            chunks.append(chunk)

        # Extract assistant message
        raw_assistant_content = claude_cli.parse_claude_message(chunks)

        if not raw_assistant_content:
            raise HTTPException(status_code=500, detail="No response from Claude Code")

        # Filter out tool usage and thinking blocks
        assistant_content = MessageAdapter.filter_content(raw_assistant_content)
        tool_uses = parse_anthropic_tool_uses(raw_assistant_content) if request_body.tools else []
        response_content = tool_uses or [AnthropicTextBlock(text=assistant_content)]

        # Estimate tokens
        prompt_tokens = MessageAdapter.estimate_tokens(prompt)
        completion_tokens = MessageAdapter.estimate_tokens(assistant_content)

        # Create Anthropic-format response
        response = AnthropicMessagesResponse(
            model=request_body.model,
            content=response_content,
            stop_reason="tool_use" if tool_uses else "end_turn",
            usage=AnthropicUsage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            ),
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anthropic Messages API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """List available models, preferring Anthropic's live Models API when configured."""
    # Check FastAPI API key if configured
    await verify_api_key(request, credentials)

    return {"object": "list", "data": await get_available_models()}


@app.post("/v1/compatibility")
async def check_compatibility(request_body: ChatCompletionRequest):
    """Check OpenAI API compatibility for a request."""
    report = CompatibilityReporter.generate_compatibility_report(request_body)
    return {
        "compatibility_report": report,
        "claude_agent_sdk_options": {
            "supported": [
                "model",
                "system_prompt",
                "max_turns",
                "allowed_tools",
                "disallowed_tools",
                "permission_mode",
                "max_thinking_tokens",
                "reasoning_effort",
                "reasoning",
                "continue_conversation",
                "resume",
                "cwd",
            ],
            "custom_headers": [
                "X-Claude-Max-Turns",
                "X-Claude-Allowed-Tools",
                "X-Claude-Disallowed-Tools",
                "X-Claude-Permission-Mode",
                "X-Claude-Max-Thinking-Tokens",
            ],
        },
    }


@app.get("/health")
@rate_limit_endpoint("health")
async def health_check(request: Request):
    """Health check endpoint."""
    return {"status": "healthy", "service": "claude-code-openai-wrapper"}


@app.get("/version")
@rate_limit_endpoint("health")
async def version_info(request: Request):
    """Version information endpoint."""
    from src import __version__

    return {
        "version": __version__,
        "service": "claude-code-openai-wrapper",
        "api_version": "v1",
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page with API documentation."""
    from src import __version__

    auth_info = get_claude_code_auth_info()
    auth_method = auth_info.get("method", "unknown")
    auth_valid = auth_info.get("status", {}).get("valid", False)
    return HTMLResponse(content=render_landing_page(__version__, auth_method, auth_valid))


@app.get("/v1/auth/status")
@rate_limit_endpoint("auth")
async def get_auth_status(request: Request):
    """Get Claude Code authentication status."""
    from src.auth import auth_manager

    auth_info = get_claude_code_auth_info()
    active_api_key = auth_manager.get_api_key()

    return {
        "claude_code_auth": auth_info,
        "server_info": {
            "api_key_required": bool(active_api_key),
            "api_key_source": ("environment" if os.getenv("API_KEY") else "none"),
            "version": "1.0.0",
        },
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Format HTTP exceptions as OpenAI-style errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {"message": exc.detail, "type": "api_error", "code": str(exc.status_code)}
        },
    )


def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    import socket

    for port in range(start_port, start_port + max_attempts):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            result = sock.connect_ex(("127.0.0.1", port))
            if result != 0:  # Port is available
                return port
        except Exception:
            return port
        finally:
            sock.close()

    raise RuntimeError(
        f"No available ports found in range {start_port}-{start_port + max_attempts - 1}"
    )


def run_server(port: int = None, host: str = None):
    """Run the server - used as Poetry script entry point."""
    import uvicorn

    # Priority: CLI arg > ENV var > default
    if port is None:
        port = int(os.getenv("PORT", "8000"))
    if host is None:
        # Default to 0.0.0.0 for container/development use (configurable via CLAUDE_WRAPPER_HOST env)
        host = os.getenv("CLAUDE_WRAPPER_HOST", "0.0.0.0")  # nosec B104
    preferred_port = port

    try:
        # Try the preferred port first
        # Binding to 0.0.0.0 is intentional for container/development use
        uvicorn.run(app, host=host, port=preferred_port)  # nosec B104
    except OSError as e:
        if "Address already in use" in str(e) or e.errno == 48:
            logger.warning(f"Port {preferred_port} is already in use. Finding alternative port...")
            try:
                available_port = find_available_port(preferred_port + 1)
                logger.info(f"Starting server on alternative port {available_port}")
                print(f"\n🚀 Server starting on http://localhost:{available_port}")
                print(f"📝 Update your client base_url to: http://localhost:{available_port}/v1")
                # Binding to 0.0.0.0 is intentional for container/development use
                uvicorn.run(app, host=host, port=available_port)  # nosec B104
            except RuntimeError as port_error:
                logger.error(f"Could not find available port: {port_error}")
                print(f"\n❌ Error: {port_error}")
                print("💡 Try setting a specific port with: PORT=9000 poetry run claude-wrapper")
                raise
        else:
            raise


if __name__ == "__main__":
    import sys

    # Simple CLI argument parsing for port
    port = None
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            print(f"Using port from command line: {port}")
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}. Using default.")

    run_server(port)
