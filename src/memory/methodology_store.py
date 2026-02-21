"""Methodology store for solution persistence and retrieval.

Wraps Repository and EmbeddingEngine to provide a higher-level API
for saving solutions (with automatic embedding) and finding similar
past solutions via hybrid search.

This is the persistence layer for The Associate's accumulated knowledge —
every successfully completed task saves its methodology here so future
tasks can benefit from past experience.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from src.core.models import Methodology, Task
from src.db.embeddings import EmbeddingEngine
from src.db.repository import Repository
from src.memory.hybrid_search import HybridSearch, HybridSearchResult

logger = logging.getLogger("associate.memory.methodology_store")


class MethodologyStore:
    """High-level API for methodology persistence and retrieval.

    Injected dependencies:
        repository: Database access for CRUD operations.
        embedding_engine: Encodes problem descriptions to vectors.
        hybrid_search: Two-Key search for finding similar methodologies.
    """

    def __init__(
        self,
        repository: Repository,
        embedding_engine: EmbeddingEngine,
        hybrid_search: HybridSearch,
    ):
        self.repository = repository
        self.embedding_engine = embedding_engine
        self.hybrid_search = hybrid_search

    # Quality filter thresholds (Item 5)
    MIN_SOLUTION_LENGTH = 50
    MIN_ATTEMPTS_FOR_TRIVIAL = 1

    def save_solution(
        self,
        problem_description: str,
        solution_code: str,
        source_task_id: Optional[uuid.UUID] = None,
        methodology_notes: Optional[str] = None,
        tags: Optional[list[str]] = None,
        language: Optional[str] = None,
        scope: str = "project",
        methodology_type: Optional[str] = None,
        files_affected: Optional[list[str]] = None,
    ) -> Methodology:
        """Save a solution with automatic embedding.

        Args:
            problem_description: Natural language description of the problem solved.
            solution_code: The code that solved the problem.
            source_task_id: ID of the task that produced this solution.
            methodology_notes: Optional notes about the approach.
            tags: Optional tags for filtering.
            language: Programming language of the solution.
            scope: "project" or "global" (Item 2).
            methodology_type: BUG_FIX/PATTERN/DECISION/GOTCHA (Item 3).
            files_affected: List of file paths this solution touched (Item 4).

        Returns:
            The saved Methodology model.
        """
        # Generate embedding for the problem description
        try:
            embedding = self.embedding_engine.encode(problem_description)
            logger.debug(
                "Generated embedding (%d dims) for: %s",
                len(embedding), problem_description[:50],
            )
        except Exception as e:
            logger.warning("Embedding generation failed — saving without vector: %s", e)
            embedding = None

        methodology = Methodology(
            problem_description=problem_description,
            problem_embedding=embedding,
            solution_code=solution_code,
            methodology_notes=methodology_notes,
            source_task_id=source_task_id,
            tags=tags or [],
            language=language,
            scope=scope,
            methodology_type=methodology_type,
            files_affected=files_affected or [],
        )

        saved = self.repository.save_methodology(methodology)
        logger.info(
            "Saved methodology %s (scope=%s, type=%s) for: %s",
            saved.id, scope, methodology_type, problem_description[:80],
        )
        return saved

    def save_from_task(
        self,
        task: Task,
        solution_code: str,
        methodology_notes: Optional[str] = None,
        tags: Optional[list[str]] = None,
        language: Optional[str] = None,
        files_affected: Optional[list[str]] = None,
        methodology_type: Optional[str] = None,
        scope: str = "project",
    ) -> Optional[Methodology]:
        """Save a methodology derived from a completed task.

        Applies quality filter (Item 5): only saves if the task was non-trivial
        (attempt_count >= MIN_ATTEMPTS_FOR_TRIVIAL) or the solution code exceeds
        MIN_SOLUTION_LENGTH. Returns None if filtered out.

        Args:
            task: The completed task.
            solution_code: The code that completed the task.
            methodology_notes: Optional notes about the approach.
            tags: Optional tags for filtering.
            language: Programming language of the solution.
            files_affected: File paths changed by this solution (Item 4).
            methodology_type: BUG_FIX/PATTERN/DECISION/GOTCHA (Item 3).
            scope: "project" or "global" (Item 2).

        Returns:
            The saved Methodology model, or None if quality filter rejected it.
        """
        # Quality filter (Item 5)
        if not self._passes_quality_filter(task, solution_code):
            logger.info(
                "Methodology for task '%s' filtered out (trivial, %d chars, %d attempts)",
                task.title, len(solution_code), task.attempt_count,
            )
            return None

        # Infer methodology_type from attempt history if not provided (Item 3)
        if methodology_type is None:
            methodology_type = self._infer_methodology_type(task)

        problem = f"{task.title}: {task.description}"
        return self.save_solution(
            problem_description=problem,
            solution_code=solution_code,
            source_task_id=task.id,
            methodology_notes=methodology_notes,
            tags=tags,
            language=language,
            scope=scope,
            methodology_type=methodology_type,
            files_affected=files_affected,
        )

    def _passes_quality_filter(self, task: Task, solution_code: str) -> bool:
        """Check if a methodology passes the quality gate (Item 5).

        Saves are allowed when:
        - Task had at least MIN_ATTEMPTS_FOR_TRIVIAL attempts (non-trivial), OR
        - Solution code exceeds MIN_SOLUTION_LENGTH chars
        """
        if task.attempt_count >= self.MIN_ATTEMPTS_FOR_TRIVIAL:
            return True
        if len(solution_code) > self.MIN_SOLUTION_LENGTH:
            return True
        return False

    def _infer_methodology_type(self, task: Task) -> str:
        """Infer methodology type from task characteristics (Item 3)."""
        title_lower = task.title.lower()
        desc_lower = task.description.lower()
        combined = f"{title_lower} {desc_lower}"

        if any(kw in combined for kw in ("fix", "bug", "error", "crash", "broken", "issue")):
            return "BUG_FIX"
        if any(kw in combined for kw in ("decide", "choose", "select", "adr", "architecture")):
            return "DECISION"
        if any(kw in combined for kw in ("gotcha", "caveat", "warning", "pitfall", "trap")):
            return "GOTCHA"
        return "PATTERN"

    def find_similar(
        self,
        query: str,
        limit: int = 3,
        language: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[HybridSearchResult]:
        """Find similar past methodologies using hybrid search.

        Args:
            query: Problem description or task description to search for.
            limit: Maximum number of results.
            language: Optional filter by programming language.
            tags: Optional filter by tags.

        Returns:
            List of HybridSearchResult sorted by relevance.
        """
        results = self.hybrid_search.search(
            query=query,
            limit=limit,
            language=language,
            tags=tags,
        )
        logger.info(
            "Found %d similar methodologies for: %s",
            len(results), query[:80],
        )
        return results

    def find_similar_with_signals(
        self,
        query: str,
        limit: int = 3,
        language: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> tuple[list[HybridSearchResult], dict[str, Any]]:
        """Find similar methodologies and aggregate retrieval confidence/conflicts."""
        results = self.find_similar(
            query=query,
            limit=limit,
            language=language,
            tags=tags,
        )
        signals = self.hybrid_search.summarize_signals(results)
        return results, signals

    def get_by_task(self, task_id: uuid.UUID) -> list[Methodology]:
        """Get all methodologies saved from a specific task.

        Args:
            task_id: The source task ID.

        Returns:
            List of Methodology models linked to this task.
        """
        # Search text for the task ID reference
        # This uses the repository's text search on methodology_notes
        # which would contain task references
        all_results = self.repository.search_methodologies_text(str(task_id), limit=20)
        return [m for m in all_results if m.source_task_id == task_id]
