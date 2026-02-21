"""Tests for src/db/engine.py — PostgreSQL engine.

Requires a running PostgreSQL instance. Skipped if unavailable.
"""

import pytest

from src.core.exceptions import DatabaseError, SchemaInitError
from src.db.engine import DatabaseEngine

from tests.conftest import requires_postgres


@requires_postgres
class TestDatabaseEngine:
    def test_connect_and_query(self, db_engine):
        """Engine connects and can execute a simple query."""
        result = db_engine.fetch_one("SELECT 1 AS num")
        assert result is not None
        assert result["num"] == 1

    def test_schema_initialized(self, db_engine):
        """Schema creates all 6 tables."""
        tables = db_engine.fetch_all(
            """SELECT table_name FROM information_schema.tables
               WHERE table_schema = 'public'
               AND table_type = 'BASE TABLE'
               ORDER BY table_name"""
        )
        table_names = {r["table_name"] for r in tables}
        expected = {"projects", "tasks", "hypothesis_log", "methodologies", "peer_reviews", "context_snapshots"}
        assert expected.issubset(table_names), f"Missing tables: {expected - table_names}"

    def test_execute(self, db_engine):
        """execute() runs DML without returning results."""
        db_engine.execute(
            "INSERT INTO projects (id, name, repo_path) VALUES (gen_random_uuid(), 'test-exec', '/tmp')"
        )
        row = db_engine.fetch_one("SELECT * FROM projects WHERE name = 'test-exec'")
        assert row is not None
        assert row["name"] == "test-exec"
        # Cleanup
        db_engine.execute("DELETE FROM projects WHERE name = 'test-exec'")

    def test_fetch_all(self, db_engine):
        """fetch_all() returns a list of dicts."""
        rows = db_engine.fetch_all("SELECT 1 AS a UNION SELECT 2 AS a ORDER BY a")
        assert len(rows) == 2
        assert rows[0]["a"] == 1
        assert rows[1]["a"] == 2

    def test_fetch_one_no_results(self, db_engine):
        """fetch_one() returns None when no rows match."""
        row = db_engine.fetch_one(
            "SELECT * FROM projects WHERE id = '00000000-0000-0000-0000-000000000000'"
        )
        assert row is None

    def test_transaction_commit(self, db_engine):
        """Transaction commits on success."""
        with db_engine.transaction() as cur:
            cur.execute(
                "INSERT INTO projects (id, name, repo_path) VALUES (gen_random_uuid(), 'txn-test', '/tmp')"
            )
        row = db_engine.fetch_one("SELECT * FROM projects WHERE name = 'txn-test'")
        assert row is not None
        # Cleanup
        db_engine.execute("DELETE FROM projects WHERE name = 'txn-test'")

    def test_transaction_rollback(self, db_engine):
        """Transaction rolls back on exception."""
        try:
            with db_engine.transaction() as cur:
                cur.execute(
                    "INSERT INTO projects (id, name, repo_path) VALUES (gen_random_uuid(), 'rollback-test', '/tmp')"
                )
                raise RuntimeError("Force rollback")
        except RuntimeError:
            pass
        row = db_engine.fetch_one("SELECT * FROM projects WHERE name = 'rollback-test'")
        assert row is None

    def test_pgvector_extension(self, db_engine):
        """pgvector extension is available."""
        row = db_engine.fetch_one("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        assert row is not None, "pgvector extension not installed"

    def test_close_and_reconnect(self, db_engine):
        """Engine reconnects after close."""
        db_engine.close()
        # Accessing conn property should trigger reconnect
        result = db_engine.fetch_one("SELECT 1 AS num")
        assert result["num"] == 1
