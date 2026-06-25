import os
import logging
from typing import Optional, Dict, Any, Tuple
from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


class ClaudeCodeAuthManager:
    """Manages authentication for Claude Code SDK integration."""

    def __init__(self):
        self.env_api_key = os.getenv("API_KEY")  # Environment API key
        self.auth_method = self._detect_auth_method()
        self.auth_status = self._validate_auth_method()

    def get_api_key(self):
        """Get the configured proxy API key, if any."""
        return self.env_api_key

    def _detect_auth_method(self) -> str:
        """Detect supported Claude Code authentication.

        The proxy supports either an injected ANTHROPIC_API_KEY or existing
        Claude Code credentials mounted into the runtime environment.
        """
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "claude_cli"

    def _validate_auth_method(self) -> Dict[str, Any]:
        """Validate the detected authentication method."""
        method = self.auth_method
        status = {"method": method, "valid": False, "errors": [], "config": {}}

        if method == "anthropic":
            status.update(self._validate_anthropic_auth())
        elif method == "claude_cli":
            status.update(self._validate_claude_cli_auth())
        else:
            status["errors"].append("No Claude Code authentication method configured")

        return status

    def _validate_anthropic_auth(self) -> Dict[str, Any]:
        """Validate Anthropic API key authentication."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return {
                "valid": False,
                "errors": ["ANTHROPIC_API_KEY environment variable not set"],
                "config": {},
            }

        if len(api_key) < 10:  # Basic sanity check
            return {
                "valid": False,
                "errors": ["ANTHROPIC_API_KEY appears to be invalid (too short)"],
                "config": {},
            }

        return {
            "valid": True,
            "errors": [],
            "config": {"api_key_present": True, "api_key_length": len(api_key)},
        }

    def _validate_claude_cli_auth(self) -> Dict[str, Any]:
        """Validate that Claude Code CLI is already authenticated."""
        # For CLI authentication, we assume it's valid and let the SDK handle auth
        # The actual validation will happen when we try to use the SDK
        return {
            "valid": True,
            "errors": [],
            "config": {
                "method": "Claude Code CLI authentication",
                "note": "Using existing Claude Code CLI authentication",
            },
        }

    def get_claude_code_env_vars(self) -> Dict[str, str]:
        """Get environment variables needed for Claude Code SDK."""
        env_vars = {}

        if self.auth_method == "anthropic":
            if os.getenv("ANTHROPIC_API_KEY"):
                env_vars["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

        elif self.auth_method == "claude_cli":
            # For CLI auth, don't set any environment variables
            # Let Claude Code SDK use the existing CLI authentication
            pass

        return env_vars


# Initialize the auth manager
auth_manager = ClaudeCodeAuthManager()

# HTTP Bearer security scheme (for FastAPI endpoint protection)
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials] = None
):
    """
    Verify API key if one is configured for FastAPI endpoint protection.
    This is separate from Claude Code authentication.
    """
    # Get the active API key (environment or runtime-generated)
    active_api_key = auth_manager.get_api_key()

    # If no API key is configured, allow all requests
    if not active_api_key:
        return True

    # Get credentials from Authorization header
    if credentials is None:
        credentials = await security(request)

    # Check if credentials were provided
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify the API key
    if credentials.credentials != active_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


def validate_claude_code_auth() -> Tuple[bool, Dict[str, Any]]:
    """
    Validate Claude Code authentication and return status.
    Returns (is_valid, status_info)
    """
    status = auth_manager.auth_status

    if not status["valid"]:
        logger.error(f"Claude Code authentication failed: {status['errors']}")
        return False, status

    logger.info(f"Claude Code authentication validated: {status['method']}")
    return True, status


def get_claude_code_auth_info() -> Dict[str, Any]:
    """Get Claude Code authentication information for diagnostics."""
    return {
        "method": auth_manager.auth_method,
        "status": auth_manager.auth_status,
        "environment_variables": list(auth_manager.get_claude_code_env_vars().keys()),
    }
