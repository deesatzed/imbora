"""Tests for src/llm/client.py retry logic using httpx.MockTransport.

httpx.MockTransport is NOT a mock library — it is httpx's own transport
replacement mechanism (like FastAPI's TestClient). It replaces the network
layer while exercising the real _request_with_retry code path.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from src.core.config import LLMConfig
from src.core.exceptions import AuthenticationError, LLMError, ModelNotFoundError
from src.llm.client import LLMMessage, LLMResponse, OpenRouterClient


def _ok_response(content: str = "Hello", model: str = "test/model", tokens: int = 42) -> dict:
    """Build a valid OpenRouter chat completion response body."""
    return {
        "choices": [{"message": {"content": content}}],
        "model": model,
        "usage": {"total_tokens": tokens},
    }


def _make_transport(handler) -> httpx.MockTransport:
    """Wrap a handler function as an httpx MockTransport."""
    return httpx.MockTransport(handler)


def _client_with_transport(transport: httpx.MockTransport, **kwargs) -> OpenRouterClient:
    """Create an OpenRouterClient with a custom transport (bypasses real network)."""
    config = LLMConfig(
        provider_retries=2,
        provider_backoff_seconds=0.01,
        timeout_seconds=5,
        **kwargs,
    )
    client = OpenRouterClient(config=config, api_key="test-key-123")
    # Replace the httpx.Client with one using our transport
    client._client = httpx.Client(transport=transport, timeout=httpx.Timeout(5))
    return client


class TestRetrySuccess:
    def test_complete_success(self):
        """200 OK → LLMResponse."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_response())

        client = _client_with_transport(_make_transport(handler))
        messages = [LLMMessage(role="user", content="Hi")]
        resp = client.complete(messages, model="test/model")

        assert isinstance(resp, LLMResponse)
        assert resp.content == "Hello"
        assert resp.model == "test/model"
        assert resp.tokens_used == 42
        client.close()


class TestRetryErrors:
    def test_complete_401_raises_auth_error(self):
        """401 → AuthenticationError (no retry)."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "unauthorized"})

        client = _client_with_transport(_make_transport(handler))
        messages = [LLMMessage(role="user", content="Hi")]
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            client.complete(messages, model="test/model")
        client.close()

    def test_complete_404_raises_model_not_found(self):
        """404 → ModelNotFoundError (no retry)."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "not found"})

        client = _client_with_transport(_make_transport(handler))
        messages = [LLMMessage(role="user", content="Hi")]
        with pytest.raises(ModelNotFoundError, match="Model not found"):
            client.complete(messages, model="ghost/model")
        client.close()

    def test_complete_429_retries_with_backoff(self):
        """429, 429, 200 → success after retries."""
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] <= 2:
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(200, json=_ok_response())

        client = _client_with_transport(_make_transport(handler))
        messages = [LLMMessage(role="user", content="Hi")]
        resp = client.complete(messages, model="test/model")

        assert resp.content == "Hello"
        assert attempts["count"] == 3
        client.close()

    def test_complete_500_retries(self):
        """500, 200 → success after retry."""
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return httpx.Response(500, json={"error": "server error"})
            return httpx.Response(200, json=_ok_response())

        client = _client_with_transport(_make_transport(handler))
        messages = [LLMMessage(role="user", content="Hi")]
        resp = client.complete(messages, model="test/model")

        assert resp.content == "Hello"
        assert attempts["count"] == 2
        client.close()

    def test_complete_timeout_retries(self):
        """TimeoutException, 200 → success after retry."""
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise httpx.TimeoutException("connect timeout")
            return httpx.Response(200, json=_ok_response())

        client = _client_with_transport(_make_transport(handler))
        messages = [LLMMessage(role="user", content="Hi")]
        resp = client.complete(messages, model="test/model")

        assert resp.content == "Hello"
        assert attempts["count"] == 2
        client.close()

    def test_complete_exhausted_retries_raises(self):
        """500, 500, 500 → LLMError after exhausting retries."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "server error"})

        client = _client_with_transport(_make_transport(handler))
        messages = [LLMMessage(role="user", content="Hi")]
        with pytest.raises(LLMError, match="Request failed after"):
            client.complete(messages, model="test/model")
        client.close()


class TestJsonParsing:
    def test_complete_json_strips_fences(self):
        """Markdown-fenced JSON → parsed dict."""
        fenced_json = '```json\n{"status": "ok", "count": 42}\n```'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_response(content=fenced_json))

        client = _client_with_transport(_make_transport(handler))
        messages = [LLMMessage(role="user", content="Give me JSON")]
        result = client.complete_json(messages, model="test/model")

        assert result == {"status": "ok", "count": 42}
        client.close()


class TestFallbackChain:
    def test_complete_with_fallback_tries_chain(self):
        """Model 1 fails, model 2 succeeds."""
        attempts = {"models": []}

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            model = body["model"]
            attempts["models"].append(model)
            if model == "primary/model":
                return httpx.Response(500, json={"error": "down"})
            return httpx.Response(200, json=_ok_response(model=model))

        client = _client_with_transport(_make_transport(handler))
        messages = [LLMMessage(role="user", content="Hi")]
        resp = client.complete_with_fallback(
            messages, models=["primary/model", "fallback/model"]
        )

        assert resp.model == "fallback/model"
        assert "primary/model" in attempts["models"]
        client.close()

    def test_complete_with_fallback_auth_error_stops(self):
        """AuthenticationError is not retried across fallback chain."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "unauthorized"})

        client = _client_with_transport(_make_transport(handler))
        messages = [LLMMessage(role="user", content="Hi")]
        with pytest.raises(AuthenticationError):
            client.complete_with_fallback(
                messages, models=["primary/model", "fallback/model"]
            )
        client.close()


class TestClientLifecycle:
    def test_close_cleans_up(self):
        """close() → client._client is None."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_response())

        client = _client_with_transport(_make_transport(handler))
        assert client._client is not None
        client.close()
        assert client._client is None


class TestModelCooldownFailover:
    def test_failing_model_enters_cooldown_and_is_skipped(self):
        """Consecutive failures trigger cooldown so fallback model is preferred."""
        attempts = {"primary/model": 0, "fallback/model": 0}
        config = LLMConfig(
            provider_retries=0,
            provider_backoff_seconds=0.01,
            timeout_seconds=5,
            model_failure_threshold=2,
            model_cooldown_seconds=30,
        )
        client = OpenRouterClient(config=config, api_key="test-key-123")

        def fake_complete(messages, model, temperature=None, max_tokens=None, response_format=None):
            del messages, temperature, max_tokens, response_format
            attempts[model] += 1
            if model == "primary/model":
                raise LLMError("simulated provider outage")
            return LLMResponse(content="ok", model=model, tokens_used=10)

        client.complete = fake_complete  # type: ignore[method-assign]
        messages = [LLMMessage(role="user", content="Hi")]

        # First call: primary fails once, fallback succeeds.
        first = client.complete_with_fallback(messages, models=["primary/model", "fallback/model"])
        assert first.model == "fallback/model"

        # Second call: primary fails again -> enters cooldown, fallback succeeds.
        second = client.complete_with_fallback(messages, models=["primary/model", "fallback/model"])
        assert second.model == "fallback/model"
        state = client.get_model_failover_state()
        assert state["primary/model"]["cooldown_remaining_seconds"] > 0

        # Third call should skip primary due to cooldown (no new primary attempt).
        third = client.complete_with_fallback(messages, models=["primary/model", "fallback/model"])
        assert third.model == "fallback/model"
        assert attempts["primary/model"] == 2
        assert attempts["fallback/model"] == 3
