"""Tests for src/core/config.py — YAML cascade config loader."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.core.config import (
    AppConfig,
    DatabaseConfig,
    EmbeddingsConfig,
    LLMConfig,
    LoggingConfig,
    ModelRegistry,
    OrchestratorConfig,
    PromptLoader,
    SecurityConfig,
    SOTAppRConfig,
    SearchConfig,
    _deep_merge,
    load_config,
    load_model_registry,
)
from src.core.exceptions import ConfigError


class TestDatabaseConfig:
    def test_defaults(self):
        c = DatabaseConfig()
        assert c.backend == "postgresql"
        assert c.host == "localhost"
        assert c.port == 5433
        assert c.dbname == "the_associate"

    def test_connection_string(self):
        c = DatabaseConfig(user="test", password="pw", host="db.local", port=5433, dbname="mydb")
        assert c.connection_string == "postgresql://test:pw@db.local:5433/mydb"


class TestLLMConfig:
    def test_defaults(self):
        c = LLMConfig()
        assert c.provider == "openrouter"
        assert "openrouter.ai" in c.base_url
        assert c.default_temperature == 0.3
        assert c.default_max_tokens == 4096
        assert c.timeout_seconds == 120
        assert c.provider_retries == 2
        assert c.provider_backoff_seconds == 2.0
        assert c.fallback_models == []
        assert c.model_failure_threshold == 2
        assert c.model_cooldown_seconds == 90


class TestSearchConfig:
    def test_defaults(self):
        c = SearchConfig()
        assert c.provider == "tavily"
        assert c.max_results == 3


class TestEmbeddingsConfig:
    def test_defaults(self):
        c = EmbeddingsConfig()
        assert c.model == "all-MiniLM-L6-v2"
        assert c.dimension == 384


class TestOrchestratorConfig:
    def test_defaults(self):
        c = OrchestratorConfig()
        assert c.max_retries_per_task == 5
        assert c.council_trigger_threshold == 2
        assert c.council_max_invocations == 3
        assert c.checkpoint_strategy == "git_stash"
        assert c.enable_multi_candidate_arbitration is False
        assert c.arbitration_candidate_count == 2
        assert c.arbitration_low_confidence_threshold == 0.55
        assert c.arbitration_conflict_trigger_count == 1
        assert c.enable_prebuild_self_correction is False
        assert c.self_correction_confidence_threshold == 0.60
        assert c.self_correction_conflict_threshold == 1


class TestAppConfig:
    def test_all_defaults(self):
        c = AppConfig()
        assert isinstance(c.database, DatabaseConfig)
        assert isinstance(c.llm, LLMConfig)
        assert isinstance(c.search, SearchConfig)
        assert isinstance(c.embeddings, EmbeddingsConfig)
        assert isinstance(c.orchestrator, OrchestratorConfig)
        assert isinstance(c.logging, LoggingConfig)
        assert isinstance(c.security, SecurityConfig)
        assert isinstance(c.sotappr, SOTAppRConfig)


class TestSecurityConfig:
    def test_defaults(self):
        c = SecurityConfig()
        assert c.autonomy_level == "supervised"
        assert c.workspace_only is True
        assert "git" in c.allowed_commands
        assert "/etc" in c.forbidden_paths
        assert c.max_actions_per_hour == 200
        assert c.sanitize_env is True


class TestSOTAppRConfig:
    def test_defaults(self):
        c = SOTAppRConfig()
        assert c.governance_pack == "balanced"
        assert c.require_human_review_before_done is False
        assert c.max_files_changed_per_task == 25
        assert ".env" in c.protected_paths
        assert c.max_estimated_cost_per_task_usd > 0
        assert c.transfer_arbitration_candidate_limit == 4
        assert c.enable_protocol_effectiveness_gate is True
        assert c.enable_protocol_decay_policy is True


class TestModelRegistry:
    def test_get_model_exists(self):
        reg = ModelRegistry(roles={"builder": "anthropic/claude-sonnet-4-20250514"})
        assert reg.get_model("builder") == "anthropic/claude-sonnet-4-20250514"

    def test_get_model_missing(self):
        reg = ModelRegistry(roles={"builder": "anthropic/claude-sonnet-4-20250514"})
        with pytest.raises(ConfigError, match="No model configured for role 'council'"):
            reg.get_model("council")

    def test_empty_registry(self):
        reg = ModelRegistry()
        with pytest.raises(ConfigError):
            reg.get_model("builder")

    def test_get_fallback_models(self):
        reg = ModelRegistry(
            roles={"builder": "openai/gpt-4o"},
            fallbacks={"builder": ["google/gemini-2.5-flash-preview"]},
        )
        assert reg.get_fallback_models("builder") == ["google/gemini-2.5-flash-preview"]
        assert reg.get_fallback_models("sentinel") == []


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"database": {"host": "localhost", "port": 5432}}
        override = {"database": {"host": "db.prod"}}
        result = _deep_merge(base, override)
        assert result["database"]["host"] == "db.prod"
        assert result["database"]["port"] == 5432

    def test_base_unchanged(self):
        base = {"a": {"x": 1}}
        override = {"a": {"x": 2}}
        _deep_merge(base, override)
        assert base["a"]["x"] == 1  # original not mutated


class TestLoadConfig:
    def test_from_project_config_dir(self):
        config_dir = Path(__file__).parent.parent / "config"
        if not config_dir.exists():
            pytest.skip("config/ directory not found")
        config = load_config(config_dir=config_dir)
        assert isinstance(config, AppConfig)
        assert config.database.backend == "postgresql"

    def test_from_temp_yaml(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            default_yaml = Path(tmpdir) / "default.yaml"
            default_yaml.write_text(yaml.dump({
                "database": {"host": "custom-host", "port": 9999},
                "llm": {"default_temperature": 0.7},
            }))
            config = load_config(config_dir=Path(tmpdir))
            assert config.database.host == "custom-host"
            assert config.database.port == 9999
            assert config.llm.default_temperature == 0.7

    def test_env_overlay(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            default_yaml = Path(tmpdir) / "default.yaml"
            default_yaml.write_text(yaml.dump({"database": {"host": "localhost"}}))
            test_yaml = Path(tmpdir) / "test.yaml"
            test_yaml.write_text(yaml.dump({"database": {"host": "test-db"}}))
            config = load_config(config_dir=Path(tmpdir), env="test")
            assert config.database.host == "test-db"

    def test_production_overlay_from_project_config(self, config_dir, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        production_yaml = config_dir / "production.yaml"
        if not production_yaml.exists():
            pytest.skip("config/production.yaml not found")

        config = load_config(config_dir=config_dir, env="production")
        assert config.sotappr.governance_pack == "strict"
        assert config.sotappr.require_human_review_before_done is True

    def test_database_url_override(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://usr:pw@remotehost:5555/proddb")
        with tempfile.TemporaryDirectory() as tmpdir:
            default_yaml = Path(tmpdir) / "default.yaml"
            default_yaml.write_text(yaml.dump({"database": {"host": "localhost"}}))
            config = load_config(config_dir=Path(tmpdir))
            assert config.database.host == "remotehost"
            assert config.database.port == 5555
            assert config.database.user == "usr"
            assert config.database.dbname == "proddb"

    def test_empty_config_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(config_dir=Path(tmpdir))
            assert isinstance(config, AppConfig)


class TestLoadModelRegistry:
    def test_from_project_models_yaml(self):
        config_dir = Path(__file__).parent.parent / "config"
        if not config_dir.exists():
            pytest.skip("config/ directory not found")
        reg = load_model_registry(config_dir=config_dir)
        assert isinstance(reg, ModelRegistry)
        assert "builder" in reg.roles

    def test_from_temp_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_yaml = Path(tmpdir) / "models.yaml"
            models_yaml.write_text(yaml.dump({
                "roles": {"builder": "openai/gpt-4o", "sentinel": "anthropic/claude-sonnet-4"}
            }))
            reg = load_model_registry(config_dir=Path(tmpdir))
            assert reg.get_model("builder") == "openai/gpt-4o"
            assert reg.get_model("sentinel") == "anthropic/claude-sonnet-4"


class TestPromptLoader:
    def test_load_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "test_prompt.txt"
            prompt_file.write_text("  You are an engineer.  \n")
            loader = PromptLoader(prompts_dir=Path(tmpdir))
            result = loader.load("test_prompt.txt")
            assert result == "You are an engineer."

    def test_load_missing_file_returns_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = PromptLoader(prompts_dir=Path(tmpdir))
            result = loader.load("nonexistent.txt", default="fallback text")
            assert result == "fallback text"

    def test_load_missing_file_empty_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = PromptLoader(prompts_dir=Path(tmpdir))
            result = loader.load("nonexistent.txt")
            assert result == ""

    def test_load_multiline_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "multi.txt"
            prompt_file.write_text("Line 1\nLine 2\nLine 3")
            loader = PromptLoader(prompts_dir=Path(tmpdir))
            result = loader.load("multi.txt")
            assert "Line 1" in result
            assert "Line 3" in result
            assert result.count("\n") == 2

    def test_default_prompts_dir(self):
        loader = PromptLoader()
        assert loader.prompts_dir.name == "prompts"
        assert loader.prompts_dir.parent.name == "config"

    def test_load_from_project_prompts_dir(self):
        """Verify PromptLoader can read from the actual config/prompts/ dir."""
        prompts_dir = Path(__file__).parent.parent / "config" / "prompts"
        if not prompts_dir.exists():
            pytest.skip("config/prompts/ directory not found")
        loader = PromptLoader(prompts_dir=prompts_dir)
        # Should not raise even if directory is empty
        result = loader.load("nonexistent.txt", default="ok")
        assert result == "ok"
