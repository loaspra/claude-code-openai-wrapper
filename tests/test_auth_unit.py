import importlib
import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request


def reload_auth(env):
    with patch.dict(os.environ, env, clear=True):
        import src.auth

        return importlib.reload(src.auth)


def test_anthropic_api_key_is_preferred():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-key"}, clear=True):
        import src.auth

        auth = importlib.reload(src.auth)
        assert auth.auth_manager.auth_method == "anthropic"
        assert auth.auth_manager.auth_status["valid"] is True
        assert auth.auth_manager.get_claude_code_env_vars() == {
            "ANTHROPIC_API_KEY": "sk-ant-test-key"
        }


def test_claude_cli_credentials_are_default():
    auth = reload_auth({})
    assert auth.auth_manager.auth_method == "claude_cli"
    assert auth.auth_manager.auth_status["valid"] is True
    assert auth.auth_manager.get_claude_code_env_vars() == {}


def test_invalid_short_anthropic_key_fails_validation():
    auth = reload_auth({"ANTHROPIC_API_KEY": "short"})
    assert auth.auth_manager.auth_method == "anthropic"
    assert auth.auth_manager.auth_status["valid"] is False


def test_proxy_api_key_can_protect_endpoints():
    auth = reload_auth({"API_KEY": "proxy-key"})
    assert auth.auth_manager.get_api_key() == "proxy-key"


@pytest.mark.asyncio
async def test_verify_api_key_accepts_x_api_key_header():
    auth = reload_auth({"API_KEY": "proxy-key"})
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"x-api-key", b"proxy-key")],
        }
    )
    assert await auth.verify_api_key(request) is True


@pytest.mark.asyncio
async def test_verify_api_key_rejects_wrong_x_api_key_header():
    auth = reload_auth({"API_KEY": "proxy-key"})
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"x-api-key", b"wrong")],
        }
    )
    with pytest.raises(HTTPException):
        await auth.verify_api_key(request)
