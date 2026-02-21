"""Shared fixtures for The Associate tests.

All tests use REAL dependencies — no mocks, no placeholders.
Tests requiring external services use skip markers when unavailable.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from project root so DATABASE_URL, API keys, etc. are available
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from src.core.config import (
    AppConfig,
    DatabaseConfig,
    EmbeddingsConfig,
    LLMConfig,
    ModelRegistry,
    load_config,
    load_model_registry,
)
from src.core.models import Project, Task, TaskStatus


# ---------------------------------------------------------------------------
# Service availability checks
# ---------------------------------------------------------------------------

def _postgres_available() -> bool:
    """Check if PostgreSQL is reachable."""
    try:
        import psycopg
        config = _get_db_config()
        conn = psycopg.connect(config.connection_string, connect_timeout=5)
        conn.close()
        return True
    except Exception:
        return False


def _openrouter_key_set() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


def _get_db_config() -> DatabaseConfig:
    """Build a DatabaseConfig from environment or defaults."""
    db_url = os.getenv("DATABASE_URL")
    if db_url and db_url.startswith("postgresql://"):
        from urllib.parse import urlparse
        parsed = urlparse(db_url)
        return DatabaseConfig(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            dbname=(parsed.path[1:] if parsed.path and len(parsed.path) > 1 else "the_associate"),
            user=parsed.username or "associate",
            password=parsed.password or "associate",
        )
    return DatabaseConfig()


# Skip markers
requires_postgres = pytest.mark.skipif(
    not _postgres_available(),
    reason="PostgreSQL not available",
)

requires_openrouter = pytest.mark.skipif(
    not _openrouter_key_set(),
    reason="OPENROUTER_API_KEY not set",
)


def _datasci_deps_available() -> bool:
    """Check if heavy DS dependencies (tabpfn, cleanlab, etc.) are importable."""
    try:
        import sklearn  # noqa: F401
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


requires_datasci_deps = pytest.mark.skipif(
    not _datasci_deps_available(),
    reason="datasci dependencies not installed (pip install -e '.[datasci]')",
)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_dir() -> Path:
    return Path(__file__).parent.parent / "config"


@pytest.fixture
def app_config(config_dir: Path) -> AppConfig:
    return load_config(config_dir=config_dir)


@pytest.fixture
def model_registry(config_dir: Path) -> ModelRegistry:
    return load_model_registry(config_dir=config_dir)


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_config() -> DatabaseConfig:
    return _get_db_config()


@pytest.fixture
def db_engine(db_config):
    """Real PostgreSQL engine — creates schema, yields, cleans up."""
    from src.db.engine import DatabaseEngine
    engine = DatabaseEngine(db_config)
    engine.initialize_schema()
    yield engine
    engine.close()


@pytest.fixture
def repository(db_engine):
    from src.db.repository import Repository
    return Repository(db_engine)


@pytest.fixture
def sample_project() -> Project:
    return Project(
        name="test-project",
        repo_path="/tmp/test-repo",
        tech_stack={"language": "python", "framework": "fastapi"},
        project_rules="No wildcard imports",
        banned_dependencies=["flask", "django"],
    )


@pytest.fixture
def sample_task(sample_project: Project) -> Task:
    return Task(
        project_id=sample_project.id,
        title="Implement user auth",
        description="Add JWT-based authentication to the API",
        priority=10,
    )
