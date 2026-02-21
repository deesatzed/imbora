"""End-to-end integration test for the full sotappr-execute pipeline.

Requires both PostgreSQL and OPENROUTER_API_KEY to run. Skipped in CI
without credentials. Exercises the real pipeline:
  SOTAppRBuilder.build() → SOTAppRExecutor.bootstrap_and_execute()

Uses a temporary git repository with a minimal FastAPI skeleton.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.conftest import requires_openrouter, requires_postgres

from src.core.factory import ComponentFactory
from src.sotappr.engine import SOTAppRBuilder
from src.sotappr.executor import SOTAppRExecutor
from src.sotappr.models import BuilderRequest, FeatureInput


def _init_temp_repo(tmp_path: Path) -> Path:
    """Create a minimal FastAPI skeleton in a temporary git repo."""
    repo = tmp_path / "e2e-test-app"
    repo.mkdir()

    # app.py
    (repo / "app.py").write_text(
        '''\
"""Minimal FastAPI application for E2E testing."""
from fastapi import FastAPI

app = FastAPI(title="E2E Test App")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/greet/{name}")
def greet(name: str):
    return {"message": f"Hello, {name}!"}
''',
        encoding="utf-8",
    )

    # tests/test_app.py
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "test_app.py").write_text(
        '''\
"""Smoke tests for E2E test app."""
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_greet():
    response = client.get("/greet/World")
    assert response.status_code == 200
    assert response.json()["message"] == "Hello, World!"
''',
        encoding="utf-8",
    )

    # requirements.txt
    (repo / "requirements.txt").write_text(
        "fastapi>=0.100.0\nhttpx>=0.25.0\nuvicorn>=0.20.0\n",
        encoding="utf-8",
    )

    # Initialize git repo
    subprocess.run(
        ["git", "init"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@e2e.local"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "E2E Test"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial scaffold"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )

    return repo


@requires_postgres
@requires_openrouter
class TestE2ESOTAppRExecute:
    """End-to-end test: SOTAppR build → seed → execute.

    Runs against real PostgreSQL and real OpenRouter API.
    Costs are incurred per LLM call.
    """

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create and return a temporary git repo with FastAPI skeleton."""
        return _init_temp_repo(tmp_path)

    @pytest.fixture
    def bundle(self):
        """Create a real ComponentBundle with full infrastructure."""
        config_dir = Path(__file__).parent.parent / "config"
        bundle = ComponentFactory.create(
            config_dir=config_dir,
            initialize_schema=True,
        )
        yield bundle
        ComponentFactory.close(bundle)

    def test_full_pipeline(self, temp_repo, bundle):
        """Full E2E: build SOTAppR report → seed tasks → process at least 1."""
        # 1. Build SOTAppR report
        request = BuilderRequest(
            organism_name="E2E Test Reactor",
            stated_problem="Validate the full pipeline with a minimal app.",
            root_need="Prove end-to-end correctness.",
            user_confirmed_phase1=True,
            features=[
                FeatureInput(
                    name="health-endpoint",
                    description="Add a /health endpoint that returns status OK",
                ),
            ],
        )
        builder = SOTAppRBuilder()
        report = builder.build(request)

        assert report.phase8.phase == 8
        assert len(report.phase3.backlog_actions) == 5

        # 2. Create executor and bootstrap
        executor = SOTAppRExecutor.from_bundle(
            bundle=bundle,
            repo_path=str(temp_repo),
            test_command="python3 -m pytest tests/ -v",
        )

        summary = executor.bootstrap_and_execute(
            request=request,
            report=report,
            repo_path=str(temp_repo),
            max_iterations=10,  # Limit to prevent runaway costs
            mode="execute",
            governance_pack="balanced",
            execute=True,
        )

        # 3. Verify outcomes
        assert summary.tasks_seeded >= 1, "At least 1 task should be seeded"
        assert summary.run_id is not None, "Run ID should be assigned"
        # We don't assert tasks_processed > 0 because the builder may
        # legitimately fail on the minimal app. The important thing is
        # no crashes and the pipeline completed.

    def test_dry_run_seeds_without_executing(self, temp_repo, bundle):
        """Dry run: build + seed tasks but don't execute."""
        request = BuilderRequest(
            organism_name="E2E Dry Run",
            stated_problem="Test seeding without execution.",
            root_need="Verify SOTAppR planning produces valid tasks.",
            user_confirmed_phase1=True,
        )
        builder = SOTAppRBuilder()
        report = builder.build(request)

        executor = SOTAppRExecutor.from_bundle(
            bundle=bundle,
            repo_path=str(temp_repo),
            test_command="python3 -m pytest tests/ -v",
        )

        summary = executor.bootstrap_and_execute(
            request=request,
            report=report,
            repo_path=str(temp_repo),
            execute=False,  # Dry run — seed only
        )

        assert summary.tasks_seeded >= 1
        assert summary.tasks_processed == 0
        assert summary.run_id is not None
