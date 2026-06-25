import os
from unittest.mock import patch

from fastapi import Request

from src import rate_limiter


def test_create_rate_limiter_returns_none_when_disabled():
    with patch.dict(os.environ, {"RATE_LIMIT_ENABLED": "false"}):
        assert rate_limiter.create_rate_limiter() is None


def test_create_rate_limiter_is_optional_when_slowapi_missing():
    if rate_limiter.Limiter is None:
        assert rate_limiter.create_rate_limiter() is None
    else:
        with patch.dict(os.environ, {"RATE_LIMIT_ENABLED": "true"}):
            assert rate_limiter.create_rate_limiter() is not None


def test_rate_limit_decorator_is_noop_without_limiter(monkeypatch):
    monkeypatch.setattr(rate_limiter, "limiter", None)

    def endpoint():
        return "ok"

    assert rate_limiter.rate_limit_endpoint("chat")(endpoint) is endpoint


def test_rate_limit_exceeded_handler_returns_json():
    response = rate_limiter.rate_limit_exceeded_handler(Request({"type": "http"}), Exception())
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"
