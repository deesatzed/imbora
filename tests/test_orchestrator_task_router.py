"""Tests for src/orchestrator/task_router.py — task state machine.

Tests use real Repository with PostgreSQL when available, otherwise
tests the transition logic directly.
"""

import pytest

from src.core.exceptions import AgentError
from src.core.models import TaskStatus
from src.orchestrator.task_router import VALID_TRANSITIONS, TaskRouter
from tests.conftest import requires_postgres


class TestValidTransitions:
    """Test the state transition graph directly."""

    def test_pending_can_go_to_coding(self):
        assert TaskStatus.CODING in VALID_TRANSITIONS[TaskStatus.PENDING]

    def test_pending_can_go_to_researching(self):
        assert TaskStatus.RESEARCHING in VALID_TRANSITIONS[TaskStatus.PENDING]

    def test_coding_can_go_to_reviewing(self):
        assert TaskStatus.REVIEWING in VALID_TRANSITIONS[TaskStatus.CODING]

    def test_coding_can_go_to_researching(self):
        assert TaskStatus.RESEARCHING in VALID_TRANSITIONS[TaskStatus.CODING]

    def test_coding_can_retry(self):
        assert TaskStatus.CODING in VALID_TRANSITIONS[TaskStatus.CODING]

    def test_coding_can_go_stuck(self):
        assert TaskStatus.STUCK in VALID_TRANSITIONS[TaskStatus.CODING]

    def test_reviewing_can_go_done(self):
        assert TaskStatus.DONE in VALID_TRANSITIONS[TaskStatus.REVIEWING]

    def test_reviewing_can_go_back_to_coding(self):
        assert TaskStatus.CODING in VALID_TRANSITIONS[TaskStatus.REVIEWING]

    def test_stuck_is_terminal(self):
        assert len(VALID_TRANSITIONS[TaskStatus.STUCK]) == 0

    def test_done_is_terminal(self):
        assert len(VALID_TRANSITIONS[TaskStatus.DONE]) == 0

    def test_pending_cannot_go_to_done(self):
        assert TaskStatus.DONE not in VALID_TRANSITIONS[TaskStatus.PENDING]


@requires_postgres
class TestTaskRouterWithDB:
    """Integration tests with real PostgreSQL."""

    @pytest.fixture
    def router(self, repository):
        return TaskRouter(repository)

    @pytest.fixture
    def db_task(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)
        return sample_task

    def test_transition_pending_to_coding(self, router, db_task):
        result = router.transition(db_task, TaskStatus.CODING, reason="starting build")
        assert result.status == TaskStatus.CODING

    def test_transition_invalid_raises(self, router, db_task):
        with pytest.raises(AgentError, match="Invalid transition"):
            router.transition(db_task, TaskStatus.DONE)

    def test_mark_stuck(self, router, db_task):
        router.transition(db_task, TaskStatus.CODING)
        result = router.mark_stuck(db_task, "test failure")
        assert result.status == TaskStatus.STUCK

    def test_mark_done_flow(self, router, db_task):
        router.transition(db_task, TaskStatus.CODING)
        router.transition(db_task, TaskStatus.REVIEWING)
        result = router.mark_done(db_task)
        assert result.status == TaskStatus.DONE

    def test_increment_attempt(self, router, db_task):
        assert db_task.attempt_count == 0
        router.increment_attempt(db_task)
        assert db_task.attempt_count == 1
        router.increment_attempt(db_task)
        assert db_task.attempt_count == 2

    def test_increment_council(self, router, db_task):
        assert db_task.council_count == 0
        router.increment_council(db_task)
        assert db_task.council_count == 1

    def test_should_trigger_council(self, router, db_task):
        assert not router.should_trigger_council(db_task, threshold=2)
        db_task.attempt_count = 2
        assert router.should_trigger_council(db_task, threshold=2)

    def test_should_not_trigger_council_after_first(self, router, db_task):
        db_task.attempt_count = 3
        db_task.council_count = 1
        assert not router.should_trigger_council(db_task, threshold=2)

    def test_should_mark_stuck(self, router, db_task):
        assert not router.should_mark_stuck(db_task, max_council=3)
        db_task.council_count = 3
        assert router.should_mark_stuck(db_task, max_council=3)

    def test_transition_history_summary(self, router, db_task):
        summary = router.get_transition_history_summary(db_task)
        assert "Implement user auth" in summary
        assert "PENDING" in summary
