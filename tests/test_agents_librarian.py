"""Tests for src/agents/librarian.py — Methodology RAG agent."""

import uuid

import pytest

from src.agents.librarian import Librarian
from src.core.models import (
    AgentResult,
    ContextBrief,
    Methodology,
    Project,
    ResearchBrief,
    Task,
    TaskContext,
    TaskStatus,
)
from src.memory.hypothesis_tracker import HypothesisTracker
from src.memory.methodology_store import MethodologyStore
from tests.conftest import requires_postgres


# ---------------------------------------------------------------------------
# Input extraction tests
# ---------------------------------------------------------------------------


class TestLibrarianInputExtraction:
    @pytest.fixture
    def librarian(self):
        """Create Librarian with stubs for testing extraction logic only."""

        class StubStore:
            def find_similar(self, query, limit=3):
                return []

        class StubTracker:
            def get_enriched_forbidden_approaches(self, task_id, project_id):
                return []
            def get_forbidden_approaches(self, task_id):
                return []

        class StubRepo:
            def get_project(self, pid):
                return None

        return Librarian(
            methodology_store=StubStore(),
            hypothesis_tracker=StubTracker(),
            repository=StubRepo(),
        )

    def test_extract_task_context(self, librarian):
        task = Task(project_id=uuid.uuid4(), title="Test", description="Desc")
        tc = TaskContext(task=task, forbidden_approaches=["old approach"])

        tc_out, rb_out = librarian._extract_inputs({"task_context": tc})
        assert tc_out.task.title == "Test"
        assert rb_out is None

    def test_extract_task_context_with_research(self, librarian):
        task = Task(project_id=uuid.uuid4(), title="Test", description="Desc")
        tc = TaskContext(task=task)
        rb = ResearchBrief(live_docs=[{"title": "Doc", "url": "http://x", "snippet": "s"}])

        tc_out, rb_out = librarian._extract_inputs({"task_context": tc, "research_brief": rb})
        assert tc_out.task.title == "Test"
        assert rb_out is not None
        assert len(rb_out.live_docs) == 1

    def test_extract_from_dict(self, librarian):
        task = Task(project_id=uuid.uuid4(), title="DictTask", description="D")
        tc_dict = TaskContext(task=task).model_dump()
        rb_dict = ResearchBrief(live_docs=[]).model_dump()

        tc_out, rb_out = librarian._extract_inputs({"task_context": tc_dict, "research_brief": rb_dict})
        assert tc_out.task.title == "DictTask"
        assert rb_out is not None

    def test_extract_direct_task_context(self, librarian):
        task = Task(project_id=uuid.uuid4(), title="Direct", description="D")
        tc = TaskContext(task=task)

        tc_out, rb_out = librarian._extract_inputs(tc)
        assert tc_out.task.title == "Direct"
        assert rb_out is None

    def test_extract_invalid_raises(self, librarian):
        with pytest.raises(ValueError, match="unexpected input type"):
            librarian._extract_inputs("invalid string")

    def test_extract_missing_task_context_raises(self, librarian):
        with pytest.raises(ValueError):
            librarian._extract_inputs({"task_context": "invalid"})


# ---------------------------------------------------------------------------
# Process tests with stubs
# ---------------------------------------------------------------------------


class TestLibrarianProcess:
    def test_process_returns_context_brief(self):
        class StubStore:
            def find_similar(self, query, limit=3):
                return []

        class StubTracker:
            def get_enriched_forbidden_approaches(self, task_id, project_id):
                return ["Don't use approach X"]
            def get_forbidden_approaches(self, task_id):
                return []

        class StubRepo:
            def get_project(self, pid):
                return Project(
                    id=pid,
                    name="test",
                    repo_path="/tmp",
                    project_rules="No wildcard imports",
                )

        librarian = Librarian(
            methodology_store=StubStore(),
            hypothesis_tracker=StubTracker(),
            repository=StubRepo(),
        )

        task = Task(
            project_id=uuid.uuid4(),
            title="Implement auth",
            description="Add JWT auth",
        )
        tc = TaskContext(task=task)

        result = librarian.run({"task_context": tc})
        assert result.status == "success"
        assert "context_brief" in result.data
        assert result.data["forbidden_count"] == 1
        assert result.data["past_solutions_count"] == 0
        assert result.data["retrieval_confidence"] == 0.0
        assert result.data["retrieval_conflict_count"] == 0

    def test_process_includes_live_docs(self):
        class StubStore:
            def find_similar(self, query, limit=3):
                return []

        class StubTracker:
            def get_enriched_forbidden_approaches(self, task_id, project_id):
                return []
            def get_forbidden_approaches(self, task_id):
                return []

        class StubRepo:
            def get_project(self, pid):
                return None

        librarian = Librarian(
            methodology_store=StubStore(),
            hypothesis_tracker=StubTracker(),
            repository=StubRepo(),
        )

        task = Task(project_id=uuid.uuid4(), title="Test", description="Desc")
        tc = TaskContext(task=task)
        rb = ResearchBrief(live_docs=[{"title": "Doc", "url": "http://x", "snippet": "s"}])

        result = librarian.run({"task_context": tc, "research_brief": rb})
        assert result.data["live_docs_count"] == 1

    def test_process_handles_store_failure(self):
        class FailingStore:
            def find_similar(self, query, limit=3):
                raise Exception("DB down")

        class StubTracker:
            def get_enriched_forbidden_approaches(self, task_id, project_id):
                return []
            def get_forbidden_approaches(self, task_id):
                return []

        class StubRepo:
            def get_project(self, pid):
                return None

        librarian = Librarian(
            methodology_store=FailingStore(),
            hypothesis_tracker=StubTracker(),
            repository=StubRepo(),
        )

        task = Task(project_id=uuid.uuid4(), title="Test", description="D")
        tc = TaskContext(task=task)

        result = librarian.run({"task_context": tc})
        assert result.status == "success"
        assert result.data["past_solutions_count"] == 0

    def test_process_uses_retrieval_signals_when_available(self):
        methodology = Methodology(problem_description="JWT auth", solution_code="code")
        hybrid_result = type("HybridResult", (), {"methodology": methodology})()

        class StubStore:
            def find_similar_with_signals(self, query, limit=3):
                del query, limit
                return [hybrid_result], {
                    "retrieval_confidence": 0.82,
                    "conflict_count": 1,
                    "conflicts": ["JWT auth conflict"],
                }

        class StubTracker:
            def get_enriched_forbidden_approaches(self, task_id, project_id):
                return []
            def get_forbidden_approaches(self, task_id):
                return []

        class StubRepo:
            def get_project(self, pid):
                return None

        librarian = Librarian(
            methodology_store=StubStore(),
            hypothesis_tracker=StubTracker(),
            repository=StubRepo(),
        )

        task = Task(project_id=uuid.uuid4(), title="Auth", description="JWT")
        tc = TaskContext(task=task)
        result = librarian.run({"task_context": tc})

        assert result.status == "success"
        assert result.data["past_solutions_count"] == 1
        assert result.data["retrieval_confidence"] == 0.82
        assert result.data["retrieval_conflict_count"] == 1

    def test_process_handles_tracker_failure(self):
        class StubStore:
            def find_similar(self, query, limit=3):
                return []

        class FailingTracker:
            def get_enriched_forbidden_approaches(self, task_id, project_id):
                raise Exception("DB down")
            def get_forbidden_approaches(self, task_id):
                raise Exception("Also down")

        class StubRepo:
            def get_project(self, pid):
                return None

        librarian = Librarian(
            methodology_store=StubStore(),
            hypothesis_tracker=FailingTracker(),
            repository=StubRepo(),
        )

        task = Task(project_id=uuid.uuid4(), title="Test", description="D")
        tc = TaskContext(task=task)

        result = librarian.run({"task_context": tc})
        assert result.status == "success"
        assert result.data["forbidden_count"] == 0
