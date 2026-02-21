"""Configuration loader for The Associate.

Loads config from a YAML cascade: config/default.yaml is always loaded,
then environment-specific overrides, then environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional  # noqa: F811 — used across all config classes

import yaml
from pydantic import BaseModel, Field

from src.core.exceptions import ConfigError


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

class DatabaseConfig(BaseModel):
    backend: str = "postgresql"
    host: str = "localhost"
    port: int = 5433
    dbname: str = "the_associate"
    user: str = "associate"
    password: str = "associate"

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"


class LLMConfig(BaseModel):
    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    default_temperature: float = 0.3
    default_max_tokens: int = 4096
    timeout_seconds: int = 120
    provider_retries: int = 2
    provider_backoff_seconds: float = 2.0
    fallback_models: list[str] = Field(default_factory=list)
    model_failure_threshold: int = 2
    model_cooldown_seconds: int = 90


class SearchConfig(BaseModel):
    provider: str = "tavily"
    max_results: int = 3


class EmbeddingsConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384


class OrchestratorConfig(BaseModel):
    max_retries_per_task: int = 5
    council_trigger_threshold: int = 2
    council_max_invocations: int = 3
    checkpoint_strategy: str = "git_stash"
    max_tokens_per_task: int = 100_000
    max_task_age_minutes: int = 30
    enable_multi_candidate_arbitration: bool = False
    arbitration_candidate_count: int = 2
    arbitration_low_confidence_threshold: float = 0.55
    arbitration_conflict_trigger_count: int = 1
    enable_prebuild_self_correction: bool = False
    self_correction_confidence_threshold: float = 0.60
    self_correction_conflict_threshold: int = 1


class SentinelConfig(BaseModel):
    llm_deep_check: bool = False
    drift_threshold: float = 0.40
    quality_score_threshold: float = 0.60  # Item 5: min quality score to pass
    rubrics_dir: Optional[str] = None  # Item 6: task-type-specific rubrics directory


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class SecurityConfig(BaseModel):
    autonomy_level: str = "supervised"
    workspace_only: bool = True
    allowed_commands: list[str] = Field(
        default_factory=lambda: [
            "git",
            "npm",
            "cargo",
            "ls",
            "cat",
            "grep",
            "find",
            "echo",
            "pwd",
            "wc",
            "head",
            "tail",
            "pytest",
            "python",
            "python3",
        ]
    )
    forbidden_paths: list[str] = Field(
        default_factory=lambda: [
            "/etc",
            "/root",
            "/home",
            "/usr",
            "/bin",
            "/sbin",
            "/lib",
            "/opt",
            "/boot",
            "/dev",
            "/proc",
            "/sys",
            "/var",
            "/tmp",
            "~/.ssh",
            "~/.gnupg",
            "~/.aws",
            "~/.config",
        ]
    )
    max_actions_per_hour: int = 200
    sanitize_env: bool = True
    safe_env_vars: list[str] = Field(
        default_factory=lambda: [
            "PATH",
            "HOME",
            "TERM",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "USER",
            "SHELL",
            "TMPDIR",
            "VIRTUAL_ENV",
            "PYTHONPATH",
        ]
    )


class TokenTrackingConfig(BaseModel):
    """Item 1: Token cost tracking configuration."""
    enabled: bool = True
    jsonl_path: str = "artifacts/sotappr/token_costs.jsonl"
    cost_per_1k_input: float = 0.01
    cost_per_1k_output: float = 0.03


class SOTAppRConfig(BaseModel):
    governance_pack: str = "balanced"
    require_human_review_before_done: bool = False
    max_files_changed_per_task: int = 25
    protected_paths: list[str] = Field(
        default_factory=lambda: [
            ".env",
            ".env.production",
            ".git/",
            "secrets/",
            "credentials/",
        ]
    )
    max_estimated_cost_per_task_usd: float = 5.0
    estimated_cost_per_1k_tokens_usd: float = 0.02
    report_archive_dir: str = "artifacts/sotappr"
    observability_jsonl_path: str = "artifacts/sotappr/events.jsonl"
    observability_metrics_path: str = "artifacts/sotappr/metrics.json"
    skills_dir: Optional[str] = None  # Item 13: path to skills/ directory
    enable_wrapup_workflow: bool = False  # Item 9: partial output salvage on timeout
    transfer_arbitration_candidate_limit: int = 4
    transfer_arbitration_top_k: int = 1
    enable_protocol_effectiveness_gate: bool = True
    min_protocol_effectiveness_samples: int = 3
    min_protocol_effectiveness_score: float = 0.50
    enable_protocol_decay_policy: bool = True
    protocol_decay_min_samples: int = 5
    protocol_decay_effectiveness_threshold: float = 0.35
    protocol_decay_drift_risk_threshold: float = 0.75


class DataScienceConfig(BaseModel):
    """Configuration for the SOTA data science pipeline module."""
    enabled: bool = False
    artifacts_dir: str = "artifacts/datasci"
    small_dataset_threshold: int = 10_000
    medium_dataset_threshold: int = 50_000
    cv_folds: int = 5
    llm_fe_generations: int = 5
    llm_fe_population_size: int = 10
    conformal_alpha: float = 0.10
    pareto_objectives: list[str] = Field(
        default_factory=lambda: ["balanced_accuracy", "interpretability", "speed"]
    )
    max_model_training_tokens: int = 50_000
    enable_tabpfn: bool = True
    enable_tabicl: bool = True
    enable_nam: bool = True
    cleanlab_enabled: bool = True
    enable_augmentation: bool = True
    imbalance_threshold: float = 3.0
    max_augmentation_ratio: float = 5.0
    augmentation_strategy: str = "auto"
    enable_imbens: bool = True
    enable_calibration: bool = True
    enable_threshold_optimization: bool = True
    enable_fe_assessment: bool = True


class AppConfig(BaseModel):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    sentinel: SentinelConfig = Field(default_factory=SentinelConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    sotappr: SOTAppRConfig = Field(default_factory=SOTAppRConfig)
    token_tracking: TokenTrackingConfig = Field(default_factory=TokenTrackingConfig)
    datasci: DataScienceConfig = Field(default_factory=DataScienceConfig)


# ---------------------------------------------------------------------------
# Model registry (models.yaml)
# ---------------------------------------------------------------------------

class ModelRegistry(BaseModel):
    """Maps agent roles to OpenRouter model IDs."""
    roles: dict[str, str] = Field(default_factory=dict)
    fallbacks: dict[str, list[str]] = Field(default_factory=dict)
    evaluation_api_key: Optional[str] = None  # Item 12: separate eval credentials
    evaluation_model: Optional[str] = None  # Item 12: override model for sentinel eval

    def get_model(self, role: str) -> str:
        if role not in self.roles:
            raise ConfigError(f"No model configured for role '{role}'. Update config/models.yaml.")
        return self.roles[role]

    def get_fallback_models(self, role: str) -> list[str]:
        return list(self.fallbacks.get(role, []))

    def get_evaluation_model(self) -> Optional[str]:
        """Return the evaluation-specific model override (Item 12)."""
        return self.evaluation_model


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_config(
    config_dir: Optional[Path] = None,
    env: Optional[str] = None,
) -> AppConfig:
    """Load application config from YAML cascade.

    Order: default.yaml -> {env}.yaml -> env vars (DATABASE_URL, etc.)
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "config"

    # Base config
    merged = _load_yaml(config_dir / "default.yaml")

    # Environment overlay
    if env:
        overlay = _load_yaml(config_dir / f"{env}.yaml")
        merged = _deep_merge(merged, overlay)

    # Environment variable overrides
    db_url = os.getenv("DATABASE_URL")
    if db_url and "database" in merged:
        # Parse DATABASE_URL into components if provided
        # Format: postgresql://user:pass@host:port/dbname
        if db_url.startswith("postgresql://"):
            from urllib.parse import urlparse
            parsed = urlparse(db_url)
            merged.setdefault("database", {})
            if parsed.hostname:
                merged["database"]["host"] = parsed.hostname
            if parsed.port:
                merged["database"]["port"] = parsed.port
            if parsed.username:
                merged["database"]["user"] = parsed.username
            if parsed.password:
                merged["database"]["password"] = parsed.password
            if parsed.path and len(parsed.path) > 1:
                merged["database"]["dbname"] = parsed.path[1:]

    return AppConfig(**merged)


def load_model_registry(config_dir: Optional[Path] = None) -> ModelRegistry:
    """Load the model registry from models.yaml."""
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "config"

    data = _load_yaml(config_dir / "models.yaml")
    return ModelRegistry(**data)


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

class PromptLoader:
    """Loads prompt templates from config/prompts/ directory.

    Falls back to hardcoded defaults if the file doesn't exist,
    enabling backward compatibility while allowing prompt iteration
    without code changes.
    """

    def __init__(self, prompts_dir: Optional[Path] = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "config" / "prompts"
        self.prompts_dir = prompts_dir

    def load(self, name: str, default: str = "") -> str:
        """Load a prompt template by filename.

        Args:
            name: Filename within config/prompts/ (e.g. "builder_system.txt").
            default: Fallback text if file doesn't exist.

        Returns:
            Prompt text (stripped of leading/trailing whitespace).
        """
        path = self.prompts_dir / name
        if path.exists():
            return path.read_text().strip()
        return default
