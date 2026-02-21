"""Tests for src/agents/state_manager.py — State Manager agent.

Tests use real git repos and real PostgreSQL when available.
"""

import uuid

import pytest

from src.agents.state_manager import StateManager
from src.core.models import (
    AgentResult,
    HypothesisEntry,
    HypothesisOutcome,
    PeerReview,
    Project,
    Task,
    TaskContext,
    TaskStatus,
)
from src.tools.shell import run_command

from tests.conftest import requires_postgres


def _init_git_repo(path):
    """Initialize a fresh git repo with an initial commit."""
    run_command("git init", cwd=str(path))
    run_command("git config user.email 'test@test.com'", cwd=str(path))
    run_command("git config user.name 'Test'", cwd=str(path))
    (path / "initial.txt").write_text("initial content")
    run_command("git add -A", cwd=str(path))
    run_command("git commit -m 'initial commit'", cwd=str(path))


@requires_postgres
class TestStateManagerWithDB:
    @pytest.fixture
    def project(self, repository):
        p = Project(
            name="test-project",
            repo_path="/tmp/test",
            tech_stack={"lang": "python"},
        )
        repository.create_project(p)
        return p

    @pytest.fixture
    def task(self, repository, project):
        t = Task(
            project_id=project.id,
            title="Test task",
            description="Test description",
        )
        repository.create_task(t)
        return t

    @pytest.fixture
    def state_manager(self, repository, tmp_path):
        _init_git_repo(tmp_path)
        return StateManager(repository=repository, repo_path=str(tmp_path))

    def test_process_returns_task_context(self, state_manager, task):
        result = state_manager.run(task)
        assert result.status == "success"
        assert "task_context" in result.data

    def test_task_context_has_correct_task(self, state_manager, task):
        result = state_manager.run(task)
        ctx = TaskContext(**result.data["task_context"])
        assert ctx.task.title == "Test task"

    def test_no_forbidden_approaches_initially(self, state_manager, task):
        result = state_manager.run(task)
        ctx = TaskContext(**result.data["task_context"])
        assert ctx.forbidden_approaches == []

    def test_forbidden_approaches_from_hypothesis_log(self, state_manager, task, repository):
        # Log a failed hypothesis
        entry = HypothesisEntry(
            task_id=task.id,
            attempt_number=1,
            approach_summary="Tried using regex parsing",
            outcome=HypothesisOutcome.FAILURE,
            error_signature="SyntaxError",
        )
        repository.log_hypothesis(entry)

        result = state_manager.run(task)
        ctx = TaskContext(**result.data["task_context"])
        assert len(ctx.forbidden_approaches) == 1
        assert "regex parsing" in ctx.forbidden_approaches[0]

    def test_multiple_forbidden_approaches(self, state_manager, task, repository):
        for i in range(3):
            entry = HypothesisEntry(
                task_id=task.id,
                attempt_number=i + 1,
                approach_summary=f"Approach {i + 1}",
                outcome=HypothesisOutcome.FAILURE,
                error_signature=f"Error{i}",
            )
            repository.log_hypothesis(entry)

        result = state_manager.run(task)
        ctx = TaskContext(**result.data["task_context"])
        assert len(ctx.forbidden_approaches) == 3

    def test_council_diagnosis_injected(self, state_manager, task, repository):
        review = PeerReview(
            task_id=task.id,
            model_used="test/model",
            diagnosis="Try a different algorithm",
            recommended_approach="Use dynamic programming",
        )
        repository.save_peer_review(review)

        result = state_manager.run(task)
        ctx = TaskContext(**result.data["task_context"])
        assert ctx.previous_council_diagnosis is not None
        assert "different algorithm" in ctx.previous_council_diagnosis

    def test_checkpoint_created(self, state_manager, task, tmp_path):
        # Add a file to stash
        (tmp_path / "work.txt").write_text("work in progress")

        result = state_manager.run(task)
        ctx = TaskContext(**result.data["task_context"])
        assert ctx.checkpoint_ref  # Should have stash ref

        # File should be gone (stashed)
        assert not (tmp_path / "work.txt").exists()

    def test_rewind_restores_checkpoint(self, state_manager, task, tmp_path):
        # Create a change and checkpoint it
        (tmp_path / "work.txt").write_text("work in progress")
        result = state_manager.run(task)
        ctx = TaskContext(**result.data["task_context"])

        # Rewind should restore
        success = state_manager.rewind_to_checkpoint(task, ctx.checkpoint_ref)
        assert success is True
        assert (tmp_path / "work.txt").exists()


class TestStateManagerNonGitRepo:
    """Test behavior when repo path is not a git repo."""

    @requires_postgres
    def test_skips_checkpoint_for_non_git(self, repository, tmp_path):
        project = Project(name="test", repo_path=str(tmp_path))
        repository.create_project(project)
        task = Task(project_id=project.id, title="Test", description="Test")
        repository.create_task(task)

        sm = StateManager(repository=repository, repo_path=str(tmp_path))
        result = sm.run(task)
        assert result.status == "success"
        ctx = TaskContext(**result.data["task_context"])
        assert ctx.checkpoint_ref == ""
