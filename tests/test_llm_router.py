"""Tests for src/llm/router.py — model routing and family extraction."""

import pytest

from src.core.config import ModelRegistry
from src.core.exceptions import ConfigError
from src.llm.router import ModelRouter, _extract_family


class TestExtractFamily:
    def test_anthropic(self):
        assert _extract_family("anthropic/claude-sonnet-4-20250514") == "anthropic"

    def test_google(self):
        assert _extract_family("google/gemini-2.5-flash-preview") == "google"

    def test_openai(self):
        assert _extract_family("openai/gpt-4o") == "openai"

    def test_no_provider(self):
        assert _extract_family("gpt-4o") == "gpt-4o"

    def test_case_insensitive(self):
        assert _extract_family("Anthropic/Claude-Sonnet-4") == "anthropic"


class TestModelRouter:
    @pytest.fixture
    def registry(self):
        return ModelRegistry(roles={
            "builder": "anthropic/claude-sonnet-4-20250514",
            "sentinel": "anthropic/claude-sonnet-4-20250514",
            "council": "google/gemini-2.5-flash-preview",
            "librarian": "anthropic/claude-sonnet-4-20250514",
            "research": "google/gemini-2.5-flash-preview",
        }, fallbacks={
            "builder": [
                "google/gemini-2.5-flash-preview",
                "openai/gpt-4o-mini",
            ]
        })

    @pytest.fixture
    def router(self, registry):
        return ModelRouter(registry)

    def test_get_model_builder(self, router):
        model = router.get_model("builder")
        assert model == "anthropic/claude-sonnet-4-20250514"

    def test_get_model_council(self, router):
        model = router.get_model("council")
        assert model == "google/gemini-2.5-flash-preview"

    def test_get_model_missing_role(self, router):
        with pytest.raises(ConfigError, match="No model configured"):
            router.get_model("nonexistent")

    def test_get_council_model_different_family(self, router):
        builder_model = "anthropic/claude-sonnet-4-20250514"
        council_model = router.get_council_model(builder_model)
        assert council_model == "google/gemini-2.5-flash-preview"
        assert _extract_family(builder_model) != _extract_family(council_model)

    def test_get_council_model_same_family_warns(self, router, caplog):
        """Council should log a warning if same family as builder."""
        same_family_registry = ModelRegistry(roles={
            "builder": "anthropic/claude-sonnet-4",
            "council": "anthropic/claude-haiku-4",
        })
        same_family_router = ModelRouter(same_family_registry)
        import logging
        with caplog.at_level(logging.WARNING, logger="associate.llm.router"):
            council = same_family_router.get_council_model("anthropic/claude-sonnet-4")
        assert council == "anthropic/claude-haiku-4"
        assert "same family" in caplog.text.lower()

    def test_list_roles(self, router, registry):
        roles = router.list_roles()
        assert roles == dict(registry.roles)
        assert "builder" in roles
        assert "council" in roles

    def test_get_model_chain_primary_and_fallbacks(self, router):
        chain = router.get_model_chain("builder")
        assert chain == [
            "anthropic/claude-sonnet-4-20250514",
            "google/gemini-2.5-flash-preview",
            "openai/gpt-4o-mini",
        ]

    def test_get_model_chain_deduplicates(self):
        reg = ModelRegistry(
            roles={"builder": "openai/gpt-4o"},
            fallbacks={"builder": ["openai/gpt-4o", "google/gemini-2.5-flash-preview"]},
        )
        chain = ModelRouter(reg).get_model_chain("builder")
        assert chain == ["openai/gpt-4o", "google/gemini-2.5-flash-preview"]
