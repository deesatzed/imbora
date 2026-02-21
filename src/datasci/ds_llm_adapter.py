"""Adapter bridging the DS pipeline's LLM interface to the real OpenRouterClient.

The DS pipeline agents call:
    llm_client.complete_with_fallback(
        model=str, fallback_models=list[str],
        messages=list[dict], max_tokens=int, temperature=float,
    )

But the real OpenRouterClient expects:
    complete_with_fallback(
        messages=list[LLMMessage], models=list[str],
        max_tokens=int, temperature=float,
    )

This adapter translates between the two interfaces.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.llm.client import LLMMessage, LLMResponse, OpenRouterClient

logger = logging.getLogger("associate.datasci.llm_adapter")


class DSLLMAdapter:
    """Adapts the DS pipeline's LLM calling convention to OpenRouterClient."""

    def __init__(self, client: OpenRouterClient):
        self.client = client
        self.call_count = 0

    def complete_with_fallback(
        self,
        model: str = "",
        fallback_models: Optional[list[str]] = None,
        messages: Optional[list[dict[str, str]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> LLMResponse:
        """Translate DS pipeline call convention to OpenRouterClient.

        Args:
            model: Primary model ID.
            fallback_models: Fallback model IDs.
            messages: List of {"role": ..., "content": ...} dicts.
            max_tokens: Max response tokens.
            temperature: Sampling temperature.

        Returns:
            LLMResponse from the real OpenRouterClient.
        """
        self.call_count += 1

        # Build the models chain
        models: list[str] = []
        if model:
            models.append(model)
        if fallback_models:
            for m in fallback_models:
                if m and m not in models:
                    models.append(m)

        if not models:
            raise ValueError("No models provided to DSLLMAdapter")

        # Convert dict messages to LLMMessage objects
        llm_messages: list[LLMMessage] = []
        for msg in (messages or []):
            llm_messages.append(LLMMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
            ))

        logger.debug(
            "DS LLM call #%d: model=%s, fallbacks=%s, %d messages, max_tokens=%d",
            self.call_count, models[0], models[1:], len(llm_messages), max_tokens,
        )

        return self.client.complete_with_fallback(
            messages=llm_messages,
            models=models,
            max_tokens=max_tokens,
            temperature=temperature,
        )
