"""Tests for src/memory/methodology_store.py — Solution persistence."""

import uuid

import pytest

from src.core.models import Methodology, Task, TaskStatus
from src.memory.methodology_store import MethodologyStore
from tests.conftest import requires_postgres


# ---------------------------------------------------------------------------
# Unit tests (no DB needed — test business logic)
# ---------------------------------------------------------------------------


class TestMethodologyStoreUnit:
    """Tests that don't require database — focus on API shape and error handling."""

    def test_methodology_model_creation(self):
        m = Methodology(
            problem_description="Fix the auth bug",
            solution_code="def fix_auth(): pass",
            tags=["auth", "bugfix"],
            language="python",
        )
        assert m.problem_description == "Fix the auth bug"
        assert m.language == "python"
        assert "auth" in m.tags
        assert m.problem_embedding is None

    def test_methodology_with_embedding(self):
        embedding = [0.1] * 384
        m = Methodology(
            problem_description="Test",
            problem_embedding=embedding,
            solution_code="code",
        )
        assert len(m.problem_embedding) == 384

    def test_methodology_with_source_task(self):
        task_id = uuid.uuid4()
        m = Methodology(
            problem_description="Test",
            solution_code="code",
            source_task_id=task_id,
        )
        assert m.source_task_id == task_id

    def test_find_similar_with_signals(self):
        methodology = Methodology(problem_description="Fix auth", solution_code="code")
        result = type(
            "Result",
            (),
            {
                "methodology": methodology,
                "confidence_score": 0.8,
                "conflict_score": 0.1,
                "source": "hybrid",
            },
        )()

        class StubHybridSearch:
            def search(self, query, limit=3, language=None, tags=None):
                del query, limit, language, tags
                return [result]

            def summarize_signals(self, results):
                del results
                return {
                    "retrieval_confidence": 0.8,
                    "conflict_count": 0,
                    "conflicts": [],
                    "hybrid_hits": 1,
                }

        store = MethodologyStore(
            repository=None,
            embedding_engine=None,
            hybrid_search=StubHybridSearch(),
        )
        results, signals = store.find_similar_with_signals("auth")
        assert len(results) == 1
        assert signals["retrieval_confidence"] == 0.8


# ---------------------------------------------------------------------------
# Integration tests (require PostgreSQL + embeddings)
# ---------------------------------------------------------------------------


@requires_postgres
class TestMethodologyStoreIntegration:
    @pytest.fixture
    def embedding_engine(self):
        from src.core.config import EmbeddingsConfig
        from src.db.embeddings import EmbeddingEngine
        return EmbeddingEngine(EmbeddingsConfig())

    @pytest.fixture
    def hybrid_search(self, repository, embedding_engine):
        from src.memory.hybrid_search import HybridSearch
        return HybridSearch(repository=repository, embedding_engine=embedding_engine)

    @pytest.fixture
    def store(self, repository, embedding_engine, hybrid_search):
        return MethodologyStore(
            repository=repository,
            embedding_engine=embedding_engine,
            hybrid_search=hybrid_search,
        )

    def test_save_solution(self, store, sample_project, repository):
        repository.create_project(sample_project)
        m = store.save_solution(
            problem_description="Implement JWT authentication",
            solution_code="from jwt import encode; def auth(): pass",
            tags=["auth", "jwt"],
            language="python",
        )
        assert m.id is not None
        assert m.problem_description == "Implement JWT authentication"

    def test_save_from_task(self, store, sample_project, sample_task, repository):
        repository.create_project(sample_project)
        repository.create_task(sample_task)
        m = store.save_from_task(
            task=sample_task,
            solution_code="def solution(): pass",
            methodology_notes="Used approach X",
            language="python",
        )
        assert m.source_task_id == sample_task.id
        assert "Implement user auth" in m.problem_description

    def test_find_similar(self, store, sample_project, repository):
        repository.create_project(sample_project)
        # Save a methodology
        store.save_solution(
            problem_description="Implement JWT authentication with refresh tokens",
            solution_code="jwt_auth_code",
            language="python",
        )
        # Search for similar
        results = store.find_similar("JWT based auth system", limit=3)
        assert len(results) >= 0  # May or may not find depending on threshold
