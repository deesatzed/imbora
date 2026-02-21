"""Model router for The Associate.

Resolves agent roles (builder, sentinel, council, etc.) to OpenRouter model IDs
using the user-managed config/models.yaml file.
"""

from __future__ import annotations

import logging

from src.core.config import ModelRegistry
from src.core.exceptions import ConfigError

logger = logging.getLogger("associate.llm.router")


class ModelRouter:
    """Maps agent roles to LLM model IDs.

    The user manages config/models.yaml. This router reads it and resolves
    role names to OpenRouter model IDs. It also enforces the constraint
    that Council must use a different model family than Builder.
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def get_model(self, role: str) -> str:
        """Resolve a role to its configured model ID.

        Args:
            role: Agent role name (builder, sentinel, council, librarian, research).

        Returns:
            OpenRouter model ID string.

        Raises:
            ConfigError: If role not found in models.yaml.
        """
        model = self.registry.get_model(role)
        logger.debug("Resolved role '%s' -> model '%s'", role, model)
        return model

    def get_model_chain(self, role: str) -> list[str]:
        """Resolve a role to [primary, fallbacks...], de-duplicated."""
        primary = self.get_model(role)
        fallbacks = self.registry.get_fallback_models(role)

        chain: list[str] = []
        for model in [primary, *fallbacks]:
            if model and model not in chain:
                chain.append(model)

        logger.debug("Resolved model chain for role '%s': %s", role, chain)
        return chain

    def get_council_model(self, builder_model: str) -> str:
        """Get the council model, verifying it's a different family than builder.

        The Council's value comes from perspective diversity. Using the same
        model family defeats the purpose.

        Args:
            builder_model: The model ID used by the Builder agent.

        Returns:
            Council model ID.

        Raises:
            ConfigError: If council model is same family as builder.
        """
        council_model = self.get_model("council")

        builder_family = _extract_family(builder_model)
        council_family = _extract_family(council_model)

        if builder_family == council_family:
            logger.warning(
                "Council model '%s' is same family as builder '%s'. "
                "Update config/models.yaml to use a different provider for council.",
                council_model, builder_model,
            )

        return council_model

    def list_roles(self) -> dict[str, str]:
        """Return all configured role -> model mappings."""
        return dict(self.registry.roles)


def _extract_family(model_id: str) -> str:
    """Extract the provider/family from an OpenRouter model ID.

    OpenRouter format: 'provider/model-name'
    Examples: 'anthropic/claude-sonnet-4' -> 'anthropic'
              'google/gemini-2.5-flash' -> 'google'
              'openai/gpt-4o' -> 'openai'
    """
    if "/" in model_id:
        return model_id.split("/")[0].lower()
    return model_id.lower()
