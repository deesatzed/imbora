"""Tests for src/memory/hybrid_search.py — Two-Key hybrid search."""

import uuid

import pytest

from src.core.config import EmbeddingsConfig
from src.core.models import Methodology
from src.memory.hybrid_search import HybridSearch, HybridSearchResult


# ---------------------------------------------------------------------------
# HybridSearchResult unit tests
# ---------------------------------------------------------------------------


class TestHybridSearchResult:
    def test_basic_creation(self):
        m = Methodology(
            problem_description="Test problem",
            solution_code="print('hello')",
        )
        r = HybridSearchResult(methodology=m, vector_score=0.8, text_score=0.6, combined_score=0.72)
        assert r.vector_score == 0.8
        assert r.text_score == 0.6
        assert r.combined_score == 0.72
        assert r.source == "hybrid"

    def test_repr(self):
        m = Methodology(problem_description="Test", solution_code="code")
        r = HybridSearchResult(methodology=m, combined_score=0.5)
        assert "combined=0.500" in repr(r)

    def test_source_default(self):
        m = Methodology(problem_description="Test", solution_code="code")
        r = HybridSearchResult(methodology=m)
        assert r.source == "hybrid"


# ---------------------------------------------------------------------------
# HybridSearch merge logic tests (in-memory, no DB/embedding needed)
# ---------------------------------------------------------------------------


class TestHybridSearchMerge:
    """Test the merge and scoring logic without external dependencies."""

    def _make_search(self):
        """Create a HybridSearch with null-safe stubs for testing merge logic."""
        # We'll test _merge_results and _apply_filters directly
        return HybridSearch.__new__(HybridSearch)

    def _make_result(self, mid=None, vec=0.0, txt=0.0, source="vector", **kwargs):
        m = Methodology(
            id=mid or uuid.uuid4(),
            problem_description="test",
            solution_code="code",
            **kwargs,
        )
        return HybridSearchResult(
            methodology=m,
            vector_score=vec,
            text_score=txt,
            source=source,
        )

    def test_merge_deduplicates_by_id(self):
        hs = self._make_search()
        hs.vector_weight = 0.6
        hs.text_weight = 0.4

        mid = uuid.uuid4()
        vec_results = [self._make_result(mid=mid, vec=0.9)]
        txt_results = [self._make_result(mid=mid, txt=0.8, source="text")]

        merged = hs._merge_results(vec_results, txt_results)
        assert len(merged) == 1
        assert merged[0].source == "hybrid"
        assert merged[0].vector_score == 0.9
        assert merged[0].text_score == 0.8

    def test_merge_combines_scores(self):
        hs = self._make_search()
        hs.vector_weight = 0.6
        hs.text_weight = 0.4

        mid = uuid.uuid4()
        vec_results = [self._make_result(mid=mid, vec=0.8)]
        txt_results = [self._make_result(mid=mid, txt=0.6, source="text")]

        merged = hs._merge_results(vec_results, txt_results)
        expected = 0.6 * 0.8 + 0.4 * 0.6  # 0.48 + 0.24 = 0.72
        assert abs(merged[0].combined_score - expected) < 0.001
        assert 0.0 <= merged[0].confidence_score <= 1.0
        assert 0.0 <= merged[0].conflict_score <= 1.0

    def test_merge_unique_entries_preserved(self):
        hs = self._make_search()
        hs.vector_weight = 0.6
        hs.text_weight = 0.4

        vec_results = [self._make_result(vec=0.9)]
        txt_results = [self._make_result(txt=0.7, source="text")]

        merged = hs._merge_results(vec_results, txt_results)
        assert len(merged) == 2

    def test_merge_empty_inputs(self):
        hs = self._make_search()
        hs.vector_weight = 0.6
        hs.text_weight = 0.4

        merged = hs._merge_results([], [])
        assert merged == []

    def test_merge_vector_only(self):
        hs = self._make_search()
        hs.vector_weight = 0.6
        hs.text_weight = 0.4

        vec_results = [self._make_result(vec=0.9)]
        merged = hs._merge_results(vec_results, [])
        assert len(merged) == 1
        assert merged[0].combined_score == 0.6 * 0.9

    def test_merge_text_only(self):
        hs = self._make_search()
        hs.vector_weight = 0.6
        hs.text_weight = 0.4

        txt_results = [self._make_result(txt=0.7, source="text")]
        merged = hs._merge_results([], txt_results)
        assert len(merged) == 1
        assert merged[0].combined_score == 0.4 * 0.7

    def test_apply_filters_by_language(self):
        hs = self._make_search()

        m1 = Methodology(problem_description="t", solution_code="c", language="python")
        m2 = Methodology(problem_description="t", solution_code="c", language="rust")
        results = [
            HybridSearchResult(methodology=m1, combined_score=0.9),
            HybridSearchResult(methodology=m2, combined_score=0.8),
        ]

        filtered = hs._apply_filters(results, language="python")
        assert len(filtered) == 1
        assert filtered[0].methodology.language == "python"

    def test_apply_filters_by_tags(self):
        hs = self._make_search()

        m1 = Methodology(problem_description="t", solution_code="c", tags=["api", "auth"])
        m2 = Methodology(problem_description="t", solution_code="c", tags=["database"])
        results = [
            HybridSearchResult(methodology=m1, combined_score=0.9),
            HybridSearchResult(methodology=m2, combined_score=0.8),
        ]

        filtered = hs._apply_filters(results, tags=["auth"])
        assert len(filtered) == 1
        assert "auth" in filtered[0].methodology.tags

    def test_apply_filters_case_insensitive(self):
        hs = self._make_search()

        m1 = Methodology(problem_description="t", solution_code="c", language="Python")
        results = [HybridSearchResult(methodology=m1, combined_score=0.9)]

        filtered = hs._apply_filters(results, language="python")
        assert len(filtered) == 1

    def test_apply_filters_no_filters(self):
        hs = self._make_search()

        results = [
            HybridSearchResult(
                methodology=Methodology(problem_description="t", solution_code="c"),
                combined_score=0.5,
            )
        ]
        filtered = hs._apply_filters(results)
        assert len(filtered) == 1

    def test_summarize_signals_reports_conflicts(self):
        hs = self._make_search()

        r1 = self._make_result(vec=0.95, txt=0.9, source="hybrid")
        r2 = self._make_result(vec=0.95, txt=0.1, source="hybrid")
        r1.confidence_score, r1.conflict_score = hs._derive_memory_signals(r1)
        r2.confidence_score, r2.conflict_score = hs._derive_memory_signals(r2)

        summary = hs.summarize_signals([r1, r2])
        assert summary["retrieval_confidence"] > 0
        assert summary["conflict_count"] == 1
        assert len(summary["conflicts"]) == 1
