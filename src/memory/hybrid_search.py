"""Two-Key hybrid search for methodology retrieval.

Combines pgvector semantic similarity with PostgreSQL full-text search (tsvector),
then merges and deduplicates results. Adapted from HMLR's hybrid search pattern
and GrokFlow's EnhancedGUKS merge strategy (P13).

The two search backends:
1. Vector search: Repository.find_similar_methodologies() — pgvector cosine distance
2. Text search: Repository.search_methodologies_text() — tsvector + ts_rank

This module orchestrates both, normalizes scores, and returns top-K merged results.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from src.core.models import Methodology
from src.db.embeddings import EmbeddingEngine
from src.db.repository import Repository

logger = logging.getLogger("associate.memory.hybrid_search")


class HybridSearchResult:
    """A single hybrid search result with combined score."""

    def __init__(
        self,
        methodology: Methodology,
        vector_score: float = 0.0,
        text_score: float = 0.0,
        combined_score: float = 0.0,
        confidence_score: float = 0.0,
        conflict_score: float = 0.0,
        source: str = "hybrid",
    ):
        self.methodology = methodology
        self.vector_score = vector_score
        self.text_score = text_score
        self.combined_score = combined_score
        self.confidence_score = confidence_score
        self.conflict_score = conflict_score
        self.source = source

    def __repr__(self) -> str:
        return (
            f"HybridSearchResult(id={self.methodology.id}, "
            f"combined={self.combined_score:.3f}, "
            f"vec={self.vector_score:.3f}, txt={self.text_score:.3f}, "
            f"conf={self.confidence_score:.3f}, conflict={self.conflict_score:.3f})"
        )


class HybridSearch:
    """Two-Key search combining vector similarity and full-text search.

    Injected dependencies:
        repository: Database access for both search backends.
        embedding_engine: Encodes query text to vectors for similarity search.

    Configuration:
        vector_weight: Weight for vector similarity score (0.0-1.0).
        text_weight: Weight for text search score (0.0-1.0).
        min_score: Minimum combined score to include in results.
    """

    def __init__(
        self,
        repository: Repository,
        embedding_engine: EmbeddingEngine,
        vector_weight: float = 0.6,
        text_weight: float = 0.4,
        min_score: float = 0.1,
        max_conflict_score: float = 0.85,
    ):
        self.repository = repository
        self.embedding_engine = embedding_engine
        self.vector_weight = vector_weight
        self.text_weight = text_weight
        self.min_score = min_score
        self.max_conflict_score = max_conflict_score

    def search(
        self,
        query: str,
        limit: int = 5,
        language: Optional[str] = None,
        tags: Optional[list[str]] = None,
        file_paths: Optional[list[str]] = None,
        scope: Optional[str] = None,
    ) -> list[HybridSearchResult]:
        """Execute hybrid search combining vector and text results.

        Args:
            query: Natural language search query.
            limit: Maximum results to return.
            language: Optional filter by programming language.
            tags: Optional filter by tags.
            file_paths: Optional filter by files_affected overlap (Item 4).
            scope: Optional scope filter — "project", "global", or None for both.

        Returns:
            List of HybridSearchResult sorted by combined score descending.
        """
        # Fetch more candidates than needed for better merge quality
        fetch_limit = limit * 3

        # 1. Vector search
        vector_results = self._vector_search(query, fetch_limit)

        # 2. Text search
        text_results = self._text_search(query, fetch_limit)

        # 3. Merge and deduplicate
        merged = self._merge_results(vector_results, text_results)

        # 4. Apply scope filter (Item 2)
        if scope:
            merged = [r for r in merged if r.methodology.scope == scope]

        # 5. Apply context filters
        if language or tags or file_paths:
            merged = self._apply_filters(merged, language, tags, file_paths)

        # 6. Sort by combined score and limit
        merged.sort(key=lambda r: r.combined_score, reverse=True)

        # 7. Apply minimum score threshold
        merged = [r for r in merged if r.combined_score >= self.min_score]
        merged = [r for r in merged if r.conflict_score <= self.max_conflict_score]

        return merged[:limit]

    def _vector_search(
        self, query: str, limit: int
    ) -> list[HybridSearchResult]:
        """Execute vector similarity search."""
        try:
            embedding = self.embedding_engine.encode(query)
            raw_results = self.repository.find_similar_methodologies(embedding, limit=limit)

            results = []
            for methodology, similarity in raw_results:
                results.append(
                    HybridSearchResult(
                        methodology=methodology,
                        vector_score=max(0.0, similarity),  # Clamp to non-negative
                        source="vector",
                    )
                )
            logger.debug("Vector search returned %d results for: %s", len(results), query[:50])
            return results

        except Exception as e:
            logger.warning("Vector search failed (falling back to text-only): %s", e)
            return []

    def _text_search(self, query: str, limit: int) -> list[HybridSearchResult]:
        """Execute full-text search."""
        try:
            raw_results = self.repository.search_methodologies_text(query, limit=limit)

            if not raw_results:
                return []

            # Normalize text scores to 0.0-1.0 range
            # Since we don't get ts_rank scores back from repository, assign
            # descending scores based on position (first result = highest rank)
            results = []
            for i, methodology in enumerate(raw_results):
                # Linear decay: first result gets 1.0, last gets something > 0
                text_score = 1.0 - (i / max(len(raw_results), 1))
                results.append(
                    HybridSearchResult(
                        methodology=methodology,
                        text_score=text_score,
                        source="text",
                    )
                )
            logger.debug("Text search returned %d results for: %s", len(results), query[:50])
            return results

        except Exception as e:
            logger.warning("Text search failed (falling back to vector-only): %s", e)
            return []

    def _merge_results(
        self,
        vector_results: list[HybridSearchResult],
        text_results: list[HybridSearchResult],
    ) -> list[HybridSearchResult]:
        """Merge and deduplicate results from both search backends.

        When a methodology appears in both result sets, its scores are combined
        using the configured weights. Unique results keep their single-source score.
        """
        # Index by methodology ID for deduplication
        merged: dict[uuid.UUID, HybridSearchResult] = {}

        # Add vector results
        for r in vector_results:
            mid = r.methodology.id
            merged[mid] = HybridSearchResult(
                methodology=r.methodology,
                vector_score=r.vector_score,
                text_score=0.0,
                source="vector",
            )

        # Merge text results
        for r in text_results:
            mid = r.methodology.id
            if mid in merged:
                # Methodology found in both — merge scores
                existing = merged[mid]
                existing.text_score = r.text_score
                existing.source = "hybrid"
            else:
                merged[mid] = HybridSearchResult(
                    methodology=r.methodology,
                    vector_score=0.0,
                    text_score=r.text_score,
                    source="text",
                )

        # Calculate combined scores
        for result in merged.values():
            result.combined_score = (
                self.vector_weight * result.vector_score
                + self.text_weight * result.text_score
            )
            result.confidence_score, result.conflict_score = self._derive_memory_signals(result)

        return list(merged.values())

    def summarize_signals(self, results: list[HybridSearchResult]) -> dict[str, float | int | list[str]]:
        """Aggregate confidence/conflict signals across retrieval results."""
        if not results:
            return {
                "retrieval_confidence": 0.0,
                "conflict_count": 0,
                "conflicts": [],
                "hybrid_hits": 0,
            }

        confidence = sum(r.confidence_score for r in results) / len(results)
        conflicts: list[str] = []
        hybrid_hits = 0
        for r in results:
            if r.source == "hybrid":
                hybrid_hits += 1
            if r.conflict_score >= 0.60:
                conflicts.append(
                    f"{r.methodology.problem_description[:100]} (conflict={r.conflict_score:.2f})"
                )

        return {
            "retrieval_confidence": round(confidence, 3),
            "conflict_count": len(conflicts),
            "conflicts": conflicts[:3],
            "hybrid_hits": hybrid_hits,
        }

    def _apply_filters(
        self,
        results: list[HybridSearchResult],
        language: Optional[str] = None,
        tags: Optional[list[str]] = None,
        file_paths: Optional[list[str]] = None,
    ) -> list[HybridSearchResult]:
        """Apply context filters to search results.

        Args:
            results: Search results to filter.
            language: Optional filter by programming language.
            tags: Optional filter by tags.
            file_paths: Optional filter by files_affected overlap (Item 4).
        """
        filtered = results

        if language:
            filtered = [
                r for r in filtered
                if r.methodology.language and r.methodology.language.lower() == language.lower()
            ]

        if tags:
            tag_set = {t.lower() for t in tags}
            filtered = [
                r for r in filtered
                if tag_set & {t.lower() for t in r.methodology.tags}
            ]

        if file_paths:
            path_set = {p.lower() for p in file_paths}
            filtered = [
                r for r in filtered
                if path_set & {f.lower() for f in (r.methodology.files_affected or [])}
            ]

        return filtered

    @staticmethod
    def _derive_memory_signals(result: HybridSearchResult) -> tuple[float, float]:
        """Infer retrieval confidence and conflict from score agreement."""
        if result.source == "hybrid":
            agreement = 1.0 - min(1.0, abs(result.vector_score - result.text_score))
            confidence = 0.50 + 0.50 * agreement
            conflict = 1.0 - agreement
            return confidence, conflict

        primary = max(result.vector_score, result.text_score)
        confidence = 0.30 + 0.70 * max(0.0, min(1.0, primary))
        return confidence, 0.0
