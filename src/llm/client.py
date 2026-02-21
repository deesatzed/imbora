"""OpenRouter LLM client for The Associate.

Adapted from cbarch's LLMClient pattern (httpx, OpenAI-compatible endpoint).
Adds OpenRouter auth headers, retry logic, and structured JSON parsing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

import httpx

from src.core.config import LLMConfig
from src.core.exceptions import (
    AuthenticationError,
    LLMError,
    ModelNotFoundError,
    RateLimitError,
    ResponseParseError,
)

logger = logging.getLogger("associate.llm")


class LLMMessage:
    """A single message in a conversation."""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMResponse:
    """Parsed response from the LLM."""

    def __init__(
        self,
        content: str,
        model: str,
        tokens_used: int = 0,
        raw: Optional[dict] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        self.content = content
        self.model = model
        self.tokens_used = tokens_used
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.raw = raw or {}


class OpenRouterClient:
    """Async-capable HTTP client for OpenRouter's OpenAI-compatible API.

    The user manages model selection via config/models.yaml.
    This client never hardcodes model IDs.
    """

    def __init__(self, config: Optional[LLMConfig] = None, api_key: Optional[str] = None):
        self.config = config or LLMConfig()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = self.config.base_url.rstrip("/")
        self._client: Optional[httpx.Client] = None
        self._model_failure_counts: dict[str, int] = {}
        self._model_cooldown_until: dict[str, float] = {}

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self.config.timeout_seconds),
            )
        return self._client

    def complete(
        self,
        messages: list[LLMMessage],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        """Send a chat completion request to OpenRouter.

        Args:
            messages: Conversation messages.
            model: OpenRouter model ID (e.g., "anthropic/claude-sonnet-4-20250514").
            temperature: Sampling temperature (default from config).
            max_tokens: Max response tokens (default from config).
            response_format: Optional format constraint (e.g., {"type": "json_object"}).

        Returns:
            LLMResponse with content, model, and token usage.
        """
        if not self.api_key:
            raise AuthenticationError("OPENROUTER_API_KEY not set")

        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature if temperature is not None else self.config.default_temperature,
            "max_tokens": max_tokens or self.config.default_max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/the-associate",
            "X-Title": "The Associate",
        }

        return self._request_with_retry(
            payload,
            headers,
            max_retries=self.config.provider_retries + 1,
            backoff_base_seconds=self.config.provider_backoff_seconds,
        )

    def complete_with_fallback(
        self,
        messages: list[LLMMessage],
        models: list[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        """Try a model chain in order with retry/backoff per model.

        Inspired by ZeroClaw's ReliableProvider strategy:
        - Retry each model locally first.
        - Fail over to fallback models when retries are exhausted.
        """
        chain: list[str] = []
        for model in [*models, *self.config.fallback_models]:
            if model and model not in chain:
                chain.append(model)

        if not chain:
            raise LLMError("No models provided for completion")

        failures: list[str] = []
        cooldown_skips: list[str] = []
        for model in chain:
            remaining = self._cooldown_remaining_seconds(model)
            if remaining > 0:
                cooldown_skips.append(f"{model}: cooling down ({remaining:.1f}s)")
                logger.warning(
                    "Skipping model '%s' due to cooldown (%.1fs remaining)",
                    model,
                    remaining,
                )
                continue

            try:
                response = self.complete(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                self._record_model_success(model)
                return response
            except AuthenticationError:
                raise
            except Exception as e:
                failures.append(f"{model}: {e}")
                self._record_model_failure(model, error=e)
                logger.warning("Model '%s' failed, trying next fallback", model)
                continue

        details = [*cooldown_skips, *failures]
        if not details:
            details = ["No available models (all filtered)"]
        raise LLMError("All models failed.\n" + "\n".join(details))

    def complete_json(
        self,
        messages: list[LLMMessage],
        model: str,
        temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        """Send a completion request expecting JSON output.

        Strips markdown code fences if present (common LLM behavior).
        """
        response = self.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return _parse_json_response(response.content)

    def _request_with_retry(
        self,
        payload: dict,
        headers: dict,
        max_retries: int = 3,
        backoff_base_seconds: float = 2.0,
    ) -> LLMResponse:
        """Execute request with exponential backoff on retryable errors."""
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                resp = self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )

                if resp.status_code == 401:
                    raise AuthenticationError("Invalid API key")
                if resp.status_code == 404:
                    raise ModelNotFoundError(f"Model not found: {payload.get('model')}")
                if resp.status_code == 429:
                    delay = _backoff_delay(attempt, backoff_base_seconds)
                    logger.warning("Rate limited. Waiting %.1fs before retry %d", delay, attempt + 1)
                    time.sleep(delay)
                    continue
                if resp.status_code >= 500:
                    delay = _backoff_delay(attempt, backoff_base_seconds)
                    logger.warning("Server error %d. Waiting %.1fs", resp.status_code, delay)
                    time.sleep(delay)
                    continue

                resp.raise_for_status()
                data = resp.json()

                content = data["choices"][0]["message"]["content"]
                model = data.get("model", payload.get("model", "unknown"))
                usage = data.get("usage", {})
                tokens = usage.get("total_tokens", 0)
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                logger.debug("LLM response: model=%s tokens=%d", model, tokens)
                return LLMResponse(
                    content=content, model=model, tokens_used=tokens, raw=data,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                )

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                delay = _backoff_delay(attempt, backoff_base_seconds)
                logger.warning("Network error: %s. Waiting %.1fs", e, delay)
                time.sleep(delay)
            except (AuthenticationError, ModelNotFoundError):
                raise
            except RateLimitError:
                raise
            except Exception as e:
                last_error = e
                if attempt == max_retries - 1:
                    break
                delay = _backoff_delay(attempt, backoff_base_seconds)
                logger.warning("Unexpected error: %s. Waiting %.1fs", e, delay)
                time.sleep(delay)

        raise LLMError(f"Request failed after {max_retries} attempts: {last_error}")

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
        self._model_failure_counts.clear()
        self._model_cooldown_until.clear()

    def get_model_failover_state(self) -> dict[str, dict[str, float | int]]:
        """Expose model cooldown/failure state for diagnostics."""
        now = time.monotonic()
        state: dict[str, dict[str, float | int]] = {}
        models = set(self._model_failure_counts) | set(self._model_cooldown_until)
        for model in models:
            until = self._model_cooldown_until.get(model, 0.0)
            remaining = max(0.0, until - now)
            state[model] = {
                "failure_count": self._model_failure_counts.get(model, 0),
                "cooldown_remaining_seconds": round(remaining, 3),
            }
        return state

    def _cooldown_remaining_seconds(self, model: str) -> float:
        until = self._model_cooldown_until.get(model)
        if until is None:
            return 0.0
        remaining = until - time.monotonic()
        if remaining <= 0:
            self._model_cooldown_until.pop(model, None)
            return 0.0
        return remaining

    def _record_model_success(self, model: str) -> None:
        self._model_failure_counts.pop(model, None)
        self._model_cooldown_until.pop(model, None)

    def _record_model_failure(self, model: str, error: Exception) -> None:
        count = self._model_failure_counts.get(model, 0) + 1
        self._model_failure_counts[model] = count

        threshold = max(1, self.config.model_failure_threshold)
        if count < threshold:
            return

        cooldown = max(1, self.config.model_cooldown_seconds)
        self._model_cooldown_until[model] = time.monotonic() + cooldown
        self._model_failure_counts[model] = 0
        logger.warning(
            "Model '%s' entered cooldown for %ds after %d consecutive failures (%s)",
            model,
            cooldown,
            threshold,
            type(error).__name__,
        )


def _backoff_delay(attempt: int, base_seconds: float = 2.0) -> float:
    """Exponential backoff: 2s, 4s, 8s, ..."""
    return min(base_seconds * (2 ** attempt), 60)


def _parse_json_response(text: str) -> dict[str, Any]:
    """Parse JSON from LLM response, stripping markdown code fences if present."""
    cleaned = text.strip()

    # Strip markdown code fences
    fence_pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
    match = re.match(fence_pattern, cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ResponseParseError(f"Failed to parse JSON from LLM response: {e}\nRaw: {text[:500]}")
