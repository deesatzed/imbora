"""Tests for src/core/factory.py — ComponentFactory.

Requires a running PostgreSQL instance. Uses real component initialization.
"""

from pathlib import Path

import pytest

from src.core.factory import ComponentBundle, ComponentFactory

from tests.conftest import requires_postgres


@requires_postgres
class TestComponentFactory:
    @pytest.fixture
    def config_dir(self):
        return Path(__file__).parent.parent / "config"

    def test_create_returns_bundle(self, config_dir):
        bundle = ComponentFactory.create(config_dir=config_dir)
        try:
            assert isinstance(bundle, ComponentBundle)
            assert bundle.config is not None
            assert bundle.model_registry is not None
            assert bundle.db_engine is not None
            assert bundle.repository is not None
            assert bundle.embedding_engine is not None
            assert bundle.llm_client is not None
            assert bundle.model_router is not None
        finally:
            ComponentFactory.close(bundle)

    def test_schema_initialized(self, config_dir):
        bundle = ComponentFactory.create(config_dir=config_dir)
        try:
            tables = bundle.db_engine.fetch_all(
                """SELECT table_name FROM information_schema.tables
                   WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"""
            )
            table_names = {r["table_name"] for r in tables}
            assert "tasks" in table_names
            assert "projects" in table_names
        finally:
            ComponentFactory.close(bundle)

    def test_model_router_resolves_roles(self, config_dir):
        bundle = ComponentFactory.create(config_dir=config_dir)
        try:
            model = bundle.model_router.get_model("builder")
            assert isinstance(model, str) and "/" in model  # Must be provider/model format
        finally:
            ComponentFactory.close(bundle)

    def test_close_shuts_down(self, config_dir):
        bundle = ComponentFactory.create(config_dir=config_dir)
        ComponentFactory.close(bundle)
        # After close, db engine connection should be closed
        assert bundle.db_engine._conn is None or bundle.db_engine._conn.closed
