"""Tests for src/llm/client.py — OpenRouter client.

Requires OPENROUTER_API_KEY env var. Uses real API calls.
"""

import os

import pytest

from src.core.config import LLMConfig
from src.core.exceptions import AuthenticationError, LLMError, ResponseParseError
from src.llm.client import LLMMessage, LLMResponse, OpenRouterClient, _backoff_delay, _parse_json_response

from tests.conftest import requires_openrouter


class TestLLMMessage:
    def test_to_dict(self):
        msg = LLMMessage(role="user", content="Hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Hello"}


class TestLLMResponse:
    def test_creation(self):
        resp = LLMResponse(content="Hello", model="test/model", tokens_used=42)
        assert resp.content == "Hello"
        assert resp.model == "test/model"
        assert resp.tokens_used == 42
        assert resp.raw == {}


class TestBackoffDelay:
    def test_exponential(self):
        assert _backoff_delay(0) == 2
        assert _backoff_delay(1) == 4
        assert _backoff_delay(2) == 8

    def test_capped_at_60(self):
        assert _backoff_delay(10) == 60


class TestParseJsonResponse:
    def test_plain_json(self):
        result = _parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_fences(self):
        text = '```json\n{"key": "value"}\n```'
        result = _parse_json_response(text)
        assert result == {"key": "value"}

    def test_json_with_plain_fences(self):
        text = '```\n{"key": "value"}\n```'
        result = _parse_json_response(text)
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(ResponseParseError, match="Failed to parse JSON"):
            _parse_json_response("not json at all")


class TestOpenRouterClientInit:
    def test_default_config(self):
        client = OpenRouterClient()
        assert "openrouter.ai" in client.base_url
        assert client.config.default_temperature == 0.3
        assert client.config.model_failure_threshold == 2
        assert client.config.model_cooldown_seconds == 90

    def test_custom_config(self):
        config = LLMConfig(default_temperature=0.7, timeout_seconds=60)
        client = OpenRouterClient(config=config)
        assert client.config.default_temperature == 0.7
        assert client.config.timeout_seconds == 60

    def test_no_api_key_raises_on_call(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        client = OpenRouterClient(api_key="")
        messages = [LLMMessage(role="user", content="test")]
        with pytest.raises(AuthenticationError, match="OPENROUTER_API_KEY not set"):
            client.complete(messages, model="test/model")


class TestOpenRouterClientFallback:
    def test_fallback_succeeds_on_second_model(self, monkeypatch):
        client = OpenRouterClient(api_key="test-key")
        messages = [LLMMessage(role="user", content="test")]
        calls = []

        def fake_complete(messages, model, temperature=None, max_tokens=None, response_format=None):
            calls.append(model)
            if model == "primary/model":
                raise LLMError("primary failed")
            return LLMResponse(content="ok", model=model, tokens_used=10)

        monkeypatch.setattr(client, "complete", fake_complete)
        result = client.complete_with_fallback(messages, models=["primary/model", "fallback/model"])

        assert result.model == "fallback/model"
        assert calls == ["primary/model", "fallback/model"]

    def test_fallback_respects_config_chain_and_deduplicates(self, monkeypatch):
        config = LLMConfig(fallback_models=["model/c", "model/b"])
        client = OpenRouterClient(config=config, api_key="test-key")
        messages = [LLMMessage(role="user", content="test")]
        calls = []

        def fake_complete(messages, model, temperature=None, max_tokens=None, response_format=None):
            calls.append(model)
            if model == "model/c":
                return LLMResponse(content="ok", model=model, tokens_used=5)
            raise LLMError(f"{model} failed")

        monkeypatch.setattr(client, "complete", fake_complete)
        result = client.complete_with_fallback(messages, models=["model/a", "model/b"])

        assert result.model == "model/c"
        assert calls == ["model/a", "model/b", "model/c"]

    def test_authentication_error_is_not_swallowed(self, monkeypatch):
        client = OpenRouterClient(api_key="test-key")
        messages = [LLMMessage(role="user", content="test")]

        def fake_complete(messages, model, temperature=None, max_tokens=None, response_format=None):
            raise AuthenticationError("bad key")

        monkeypatch.setattr(client, "complete", fake_complete)
        with pytest.raises(AuthenticationError, match="bad key"):
            client.complete_with_fallback(messages, models=["primary/model", "fallback/model"])


@requires_openrouter
class TestOpenRouterClientLive:
    """Live API tests — only run when OPENROUTER_API_KEY is set."""

    @pytest.fixture
    def client(self):
        c = OpenRouterClient()
        yield c
        c.close()

    @pytest.fixture
    def model_id(self):
        """Use a fast, cheap model for testing."""
        # Read from models.yaml or use a sensible default
        from pathlib import Path
        from src.core.config import load_model_registry
        config_dir = Path(__file__).parent.parent / "config"
        try:
            reg = load_model_registry(config_dir=config_dir)
            return reg.get_model("research")  # Typically a fast model
        except Exception:
            return "google/gemini-2.5-flash-preview"

    def test_simple_completion(self, client, model_id):
        messages = [
            LLMMessage(role="user", content="Reply with exactly: HELLO"),
        ]
        response = client.complete(messages, model=model_id, max_tokens=50)
        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        assert response.tokens_used > 0

    def test_json_completion(self, client, model_id):
        messages = [
            LLMMessage(
                role="user",
                content='Return a JSON object with keys "status" and "count". Status should be "ok" and count should be 42.',
            ),
        ]
        result = client.complete_json(messages, model=model_id)
        assert isinstance(result, dict)
        assert "status" in result
