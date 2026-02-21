"""Hypothesis tracker with cross-task failure pattern analysis.

Tracks attempted approaches per task and analyzes failure patterns
across an entire project. Adapted from GrokFlow's GUKSAnalytics
error pattern detection (P10).

Key capabilities:
- Record attempts with error signatures
- Detect duplicate errors within a task
- Detect recurring bugs across tasks (cross-task pattern mining)
- Calculate urgency for failure patterns
- Provide forbidden approach lists enriched with project-wide patterns
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from typing import Any, Optional

from src.core.models import HypothesisEntry, HypothesisOutcome
from src.db.repository import Repository

logger = logging.getLogger("associate.memory.hypothesis_tracker")


# Error category keywords adapted from GrokFlow's _categorize_patterns
# (translated from TypeScript to Python domain)
ERROR_CATEGORIES: dict[str, list[str]] = {
    "type_error": ["typeerror", "type error", "unexpected type", "not callable", "has no attribute"],
    "import_error": ["importerror", "modulenotfounderror", "no module named", "cannot import"],
    "attribute_error": ["attributeerror", "has no attribute", "no such attribute"],
    "value_error": ["valueerror", "invalid literal", "invalid value"],
    "key_error": ["keyerror", "key not found", "missing key"],
    "index_error": ["indexerror", "index out of range", "list index"],
    "connection_error": ["connectionerror", "connection refused", "timeout", "could not connect"],
    "database_error": ["operationalerror", "integrityerror", "programmingerror", "duplicate key"],
    "permission_error": ["permissionerror", "permission denied", "access denied", "unauthorized"],
    "file_error": ["filenotfounderror", "no such file", "is a directory", "not a file"],
    "async_error": ["asyncio", "coroutine", "event loop", "awaitable"],
    "api_error": ["api error", "http error", "status code", "rate limit"],
    "validation_error": ["validationerror", "validation failed", "schema", "pydantic"],
    "syntax_error": ["syntaxerror", "invalid syntax", "unexpected token"],
    "test_failure": ["assertionerror", "assert", "expected", "actual"],
}


class FailurePattern:
    """A recurring failure pattern detected across tasks."""

    def __init__(
        self,
        error_signature: str,
        count: int,
        task_ids: set[uuid.UUID],
        category: str,
        urgency: str,
        example_approaches: list[str],
        successful_resolution: Optional[str] = None,
    ):
        self.error_signature = error_signature
        self.count = count
        self.task_ids = task_ids
        self.category = category
        self.urgency = urgency
        self.example_approaches = example_approaches
        self.successful_resolution = successful_resolution

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_signature": self.error_signature,
            "count": self.count,
            "task_count": len(self.task_ids),
            "category": self.category,
            "urgency": self.urgency,
            "example_approaches": self.example_approaches[:3],
            "has_resolution": self.successful_resolution is not None,
        }


class HypothesisTracker:
    """Tracks hypotheses and detects cross-task failure patterns.

    Injected dependencies:
        repository: Database access for hypothesis_log queries.
    """

    def __init__(self, repository: Repository):
        self.repository = repository

    def record_attempt(
        self,
        task_id: uuid.UUID,
        attempt_number: int,
        approach_summary: str,
        outcome: HypothesisOutcome,
        error_signature: Optional[str] = None,
        error_full: Optional[str] = None,
        files_changed: Optional[list[str]] = None,
        duration_seconds: Optional[float] = None,
        model_used: Optional[str] = None,
    ) -> HypothesisEntry:
        """Record an attempt in the hypothesis log.

        Args:
            task_id: The task being worked on.
            attempt_number: Which attempt this is.
            approach_summary: What was tried.
            outcome: SUCCESS or FAILURE.
            error_signature: Normalized error for deduplication.
            error_full: Full error text.
            files_changed: Files that were modified.
            duration_seconds: How long the attempt took.
            model_used: Which LLM model was used.

        Returns:
            The saved HypothesisEntry.
        """
        entry = HypothesisEntry(
            task_id=task_id,
            attempt_number=attempt_number,
            approach_summary=approach_summary,
            outcome=outcome,
            error_signature=error_signature,
            error_full=error_full,
            files_changed=files_changed or [],
            duration_seconds=duration_seconds,
            model_used=model_used,
        )
        return self.repository.log_hypothesis(entry)

    def get_forbidden_approaches(self, task_id: uuid.UUID) -> list[str]:
        """Get forbidden approaches for a task (local failures only).

        Args:
            task_id: The task to query.

        Returns:
            List of approach summaries that failed.
        """
        failed = self.repository.get_failed_approaches(task_id)
        return [
            f"Attempt #{e.attempt_number}: {e.approach_summary}"
            + (f" (error: {e.error_signature})" if e.error_signature else "")
            for e in failed
        ]

    def has_duplicate_error(self, task_id: uuid.UUID, error_signature: str) -> bool:
        """Check if this exact error has been seen before for this task.

        Args:
            task_id: The task to check.
            error_signature: Normalized error signature.

        Returns:
            True if this error already exists in the hypothesis log.
        """
        return self.repository.has_duplicate_error(task_id, error_signature)

    def get_enriched_forbidden_approaches(
        self, task_id: uuid.UUID, project_id: uuid.UUID
    ) -> list[str]:
        """Get forbidden approaches enriched with project-wide failure patterns.

        Combines task-local failures with recurring project-wide patterns.
        This is the cross-task pattern mining capability (Dirac P4 + GrokFlow P10).

        Args:
            task_id: The current task.
            project_id: The project for cross-task analysis.

        Returns:
            Enriched list of forbidden approaches including project patterns.
        """
        # Task-local forbidden approaches
        local = self.get_forbidden_approaches(task_id)

        # Project-wide recurring patterns
        patterns = self.get_common_failure_patterns(project_id, min_count=2)

        # Add project patterns as additional context
        project_warnings = []
        for pattern in patterns:
            if pattern.urgency in ("critical", "high"):
                warning = (
                    f"[PROJECT PATTERN - {pattern.urgency.upper()}] "
                    f"Error '{pattern.error_signature[:100]}' has occurred "
                    f"{pattern.count} times across {len(pattern.task_ids)} tasks"
                )
                if pattern.successful_resolution:
                    warning += f". Successful resolution: {pattern.successful_resolution}"
                project_warnings.append(warning)

        return local + project_warnings

    def get_common_failure_patterns(
        self,
        project_id: uuid.UUID,
        min_count: int = 2,
    ) -> list[FailurePattern]:
        """Detect recurring failure patterns across all tasks in a project.

        Adapted from GrokFlow's detect_recurring_bugs(). Groups failures
        by normalized error signature, counts occurrences, calculates urgency,
        and checks for successful resolutions.

        Args:
            project_id: The project to analyze.
            min_count: Minimum occurrence count to report.

        Returns:
            List of FailurePattern sorted by urgency then count.
        """
        # Get all tasks for this project
        from src.core.models import TaskStatus

        all_statuses = [
            TaskStatus.PENDING, TaskStatus.CODING, TaskStatus.REVIEWING,
            TaskStatus.STUCK, TaskStatus.DONE, TaskStatus.RESEARCHING,
        ]

        all_tasks = []
        for status in all_statuses:
            all_tasks.extend(self.repository.get_tasks_by_status(project_id, status))

        if not all_tasks:
            return []

        # Collect all failures across all tasks
        error_groups: dict[str, list[tuple[uuid.UUID, HypothesisEntry]]] = defaultdict(list)
        success_map: dict[str, list[str]] = defaultdict(list)

        for task in all_tasks:
            failed = self.repository.get_failed_approaches(task.id)
            for entry in failed:
                if entry.error_signature:
                    error_groups[entry.error_signature].append((task.id, entry))

            # Also check for successful resolutions of the same errors
            # (entries where outcome is SUCCESS for a task that previously had this error)
            hypothesis_count = self.repository.get_hypothesis_count(task.id)
            if hypothesis_count > 0 and task.status == TaskStatus.DONE:
                for entry in failed:
                    if entry.error_signature:
                        # This task had this error but eventually succeeded
                        success_map[entry.error_signature].append(
                            f"Task '{task.title}' resolved this error"
                        )

        # Build failure patterns
        patterns = []
        for error_sig, occurrences in error_groups.items():
            if len(occurrences) < min_count:
                continue

            task_ids = {task_id for task_id, _ in occurrences}
            approaches = [entry.approach_summary for _, entry in occurrences]
            category = _categorize_error(error_sig)
            urgency = _calculate_urgency(len(occurrences), len(task_ids))
            resolution = success_map.get(error_sig, [None])[0]

            patterns.append(
                FailurePattern(
                    error_signature=error_sig,
                    count=len(occurrences),
                    task_ids=task_ids,
                    category=category,
                    urgency=urgency,
                    example_approaches=approaches,
                    successful_resolution=resolution,
                )
            )

        # Sort by urgency (critical first) then by count
        urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        patterns.sort(key=lambda p: (urgency_order.get(p.urgency, 4), -p.count))

        logger.info(
            "Found %d recurring failure patterns across %d tasks in project %s",
            len(patterns), len(all_tasks), project_id,
        )
        return patterns


def _categorize_error(error_signature: str) -> str:
    """Categorize an error by its signature.

    Adapted from GrokFlow's _categorize_patterns with Python-domain keywords.

    Args:
        error_signature: Normalized error string.

    Returns:
        Category name string.
    """
    sig_lower = error_signature.lower()

    for category, keywords in ERROR_CATEGORIES.items():
        for keyword in keywords:
            if keyword in sig_lower:
                return category

    return "unknown"


def _calculate_urgency(count: int, num_tasks: int) -> str:
    """Calculate urgency based on frequency and spread.

    Adapted from GrokFlow's _calculate_urgency.

    Args:
        count: Total number of occurrences.
        num_tasks: Number of distinct tasks affected.

    Returns:
        Urgency level: "critical", "high", "medium", or "low".
    """
    if count >= 5 or num_tasks >= 3:
        return "critical"
    if count >= 3 or num_tasks >= 2:
        return "high"
    if count >= 2:
        return "medium"
    return "low"


def normalize_error_for_dedup(error_text: str) -> str:
    """Enhanced error normalization for cross-task deduplication.

    Extends the base normalize_error_signature with additional
    normalizations adapted from GrokFlow's _normalize_error:
    removes quoted strings, variable names, and specific values.

    Args:
        error_text: Raw error text.

    Returns:
        Normalized error signature.
    """
    from src.llm.response_parser import normalize_error_signature

    sig = error_text.strip()

    # Remove UUIDs BEFORE base normalization (base replaces numbers first)
    sig = re.sub(
        r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',
        '<UUID>', sig,
    )

    # Remove quoted strings BEFORE base normalization
    sig = re.sub(r"'[^']*'", "'<STR>'", sig)
    sig = re.sub(r'"[^"]*"', '"<STR>"', sig)

    # Apply base normalization (timestamps, paths, line numbers, addresses, whitespace)
    sig = normalize_error_signature(sig)

    # Remove remaining standalone numbers
    sig = re.sub(r'\b\d+\b', '<NUM>', sig)

    # Collapse repeated normalized tokens
    sig = re.sub(r'(<\w+>)\s*\1+', r'\1', sig)

    return sig
