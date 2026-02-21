"""Tests for src/memory/hypothesis_tracker.py — Cross-task failure patterns."""

import uuid

import pytest

from src.core.models import HypothesisEntry, HypothesisOutcome, Task, TaskStatus
from src.memory.hypothesis_tracker import (
    ERROR_CATEGORIES,
    FailurePattern,
    HypothesisTracker,
    _calculate_urgency,
    _categorize_error,
    normalize_error_for_dedup,
)
from tests.conftest import requires_postgres


# ---------------------------------------------------------------------------
# Error categorization tests
# ---------------------------------------------------------------------------


class TestCategorizeError:
    def test_type_error(self):
        assert _categorize_error("TypeError: int is not callable") == "type_error"

    def test_import_error(self):
        assert _categorize_error("ModuleNotFoundError: No module named 'foo'") == "import_error"

    def test_connection_error(self):
        assert _categorize_error("ConnectionError: Connection refused on port 5432") == "connection_error"

    def test_database_error(self):
        assert _categorize_error("IntegrityError: duplicate key value violates") == "database_error"

    def test_validation_error(self):
        assert _categorize_error("ValidationError: field required") == "validation_error"

    def test_file_error(self):
        assert _categorize_error("FileNotFoundError: No such file or directory") == "file_error"

    def test_test_failure(self):
        assert _categorize_error("AssertionError: Expected 5 but got 3") == "test_failure"

    def test_api_error(self):
        assert _categorize_error("HTTP Error 429: rate limit exceeded") == "api_error"

    def test_unknown_error(self):
        assert _categorize_error("Something went sideways in the pipeline") == "unknown"

    def test_case_insensitive(self):
        assert _categorize_error("TYPEERROR: test") == "type_error"

    def test_all_categories_have_keywords(self):
        for category, keywords in ERROR_CATEGORIES.items():
            assert len(keywords) > 0, f"Category '{category}' has no keywords"


# ---------------------------------------------------------------------------
# Urgency calculation tests
# ---------------------------------------------------------------------------


class TestCalculateUrgency:
    def test_critical_high_count(self):
        assert _calculate_urgency(count=5, num_tasks=1) == "critical"

    def test_critical_many_tasks(self):
        assert _calculate_urgency(count=2, num_tasks=3) == "critical"

    def test_high(self):
        assert _calculate_urgency(count=3, num_tasks=1) == "high"

    def test_high_two_tasks(self):
        assert _calculate_urgency(count=2, num_tasks=2) == "high"

    def test_medium(self):
        assert _calculate_urgency(count=2, num_tasks=1) == "medium"

    def test_low(self):
        assert _calculate_urgency(count=1, num_tasks=1) == "low"


# ---------------------------------------------------------------------------
# Error normalization tests
# ---------------------------------------------------------------------------


class TestNormalizeErrorForDedup:
    def test_removes_quoted_strings(self):
        sig = normalize_error_for_dedup("KeyError: 'some_key_name'")
        assert "some_key_name" not in sig
        assert "<STR>" in sig

    def test_removes_uuids(self):
        sig = normalize_error_for_dedup("Task 12345678-1234-1234-1234-123456789abc failed")
        assert "12345678-1234" not in sig
        assert "<UUID>" in sig

    def test_removes_numbers(self):
        sig = normalize_error_for_dedup("Expected 42 but got 99")
        assert "42" not in sig
        assert "99" not in sig

    def test_collapses_repeated_tokens(self):
        sig = normalize_error_for_dedup("line 123 line 456")
        # Should have normalized numbers and collapsed
        assert sig.count("<NUM>") <= 2  # Could be collapsed

    def test_preserves_error_type(self):
        sig = normalize_error_for_dedup("TypeError: unsupported operand")
        assert "TypeError" in sig or "typeerror" in sig.lower()


# ---------------------------------------------------------------------------
# FailurePattern model tests
# ---------------------------------------------------------------------------


class TestFailurePattern:
    def test_to_dict(self):
        fp = FailurePattern(
            error_signature="TypeError: <test>",
            count=3,
            task_ids={uuid.uuid4(), uuid.uuid4()},
            category="type_error",
            urgency="high",
            example_approaches=["approach 1", "approach 2", "approach 3", "approach 4"],
        )
        d = fp.to_dict()
        assert d["count"] == 3
        assert d["task_count"] == 2
        assert d["category"] == "type_error"
        assert d["urgency"] == "high"
        assert len(d["example_approaches"]) == 3  # Truncated to 3
        assert d["has_resolution"] is False

    def test_with_resolution(self):
        fp = FailurePattern(
            error_signature="test",
            count=1,
            task_ids={uuid.uuid4()},
            category="unknown",
            urgency="low",
            example_approaches=[],
            successful_resolution="Fixed by adding null check",
        )
        assert fp.to_dict()["has_resolution"] is True


# ---------------------------------------------------------------------------
# HypothesisTracker integration tests (require PostgreSQL)
# ---------------------------------------------------------------------------


@requires_postgres
class TestHypothesisTrackerIntegration:
    @pytest.fixture
    def tracker(self, repository):
        return HypothesisTracker(repository=repository)

    def test_record_attempt(self, tracker, sample_project, sample_task, repository):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        entry = tracker.record_attempt(
            task_id=sample_task.id,
            attempt_number=1,
            approach_summary="Tried approach A",
            outcome=HypothesisOutcome.FAILURE,
            error_signature="TypeError: test",
            error_full="Full traceback here",
        )
        assert entry.id is not None
        assert entry.task_id == sample_task.id

    def test_get_forbidden_approaches(self, tracker, sample_project, sample_task, repository):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        # Record two failures
        tracker.record_attempt(
            task_id=sample_task.id,
            attempt_number=1,
            approach_summary="Used raw SQL",
            outcome=HypothesisOutcome.FAILURE,
            error_signature="IntegrityError: duplicate key",
        )
        tracker.record_attempt(
            task_id=sample_task.id,
            attempt_number=2,
            approach_summary="Used ORM upsert",
            outcome=HypothesisOutcome.FAILURE,
            error_signature="AttributeError: no attribute 'upsert'",
        )

        forbidden = tracker.get_forbidden_approaches(sample_task.id)
        assert len(forbidden) == 2
        assert "Used raw SQL" in forbidden[0]
        assert "Used ORM upsert" in forbidden[1]

    def test_has_duplicate_error(self, tracker, sample_project, sample_task, repository):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        tracker.record_attempt(
            task_id=sample_task.id,
            attempt_number=1,
            approach_summary="Test",
            outcome=HypothesisOutcome.FAILURE,
            error_signature="TypeError: test",
        )

        assert tracker.has_duplicate_error(sample_task.id, "TypeError: test") is True
        assert tracker.has_duplicate_error(sample_task.id, "ValueError: other") is False

    def test_get_common_failure_patterns(self, tracker, sample_project, repository):
        repository.create_project(sample_project)

        # Create two tasks
        task1 = Task(
            project_id=sample_project.id,
            title="Task 1",
            description="First task",
        )
        task2 = Task(
            project_id=sample_project.id,
            title="Task 2",
            description="Second task",
        )
        repository.create_task(task1)
        repository.create_task(task2)

        # Same error across both tasks
        tracker.record_attempt(
            task_id=task1.id, attempt_number=1,
            approach_summary="Approach A",
            outcome=HypothesisOutcome.FAILURE,
            error_signature="TypeError: int not callable",
        )
        tracker.record_attempt(
            task_id=task2.id, attempt_number=1,
            approach_summary="Approach B",
            outcome=HypothesisOutcome.FAILURE,
            error_signature="TypeError: int not callable",
        )

        patterns = tracker.get_common_failure_patterns(sample_project.id, min_count=2)
        assert len(patterns) >= 1
        assert patterns[0].error_signature == "TypeError: int not callable"
        assert patterns[0].count >= 2
        assert len(patterns[0].task_ids) >= 2

    def test_get_enriched_forbidden_approaches(self, tracker, sample_project, sample_task, repository):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        # Record a local failure
        tracker.record_attempt(
            task_id=sample_task.id,
            attempt_number=1,
            approach_summary="Local failure",
            outcome=HypothesisOutcome.FAILURE,
            error_signature="local error",
        )

        enriched = tracker.get_enriched_forbidden_approaches(
            task_id=sample_task.id,
            project_id=sample_project.id,
        )
        # Should contain at least the local failure
        assert len(enriched) >= 1
        assert "Local failure" in enriched[0]
