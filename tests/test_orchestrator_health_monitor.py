"""Tests for src/orchestrator/health_monitor.py — Dirac Pattern 1."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from src.core.config import OrchestratorConfig
from src.core.models import Task, TaskStatus
from src.orchestrator.health_monitor import HealthMonitor

from tests.conftest import requires_postgres


class TestHealthMonitorCircuitBreaker:
    """Test circuit breaker logic without DB dependency."""

    @pytest.fixture
    def config(self):
        return OrchestratorConfig()

    def test_circuit_breaker_closed_initially(self, config):
        # Use a minimal repository stand-in for non-DB tests
        monitor = HealthMonitor(repository=None, config=config)
        assert monitor.is_circuit_breaker_open is False

    def test_circuit_breaker_opens_after_3_failures(self, config):
        monitor = HealthMonitor(repository=None, config=config)
        monitor.record_llm_failure()
        monitor.record_llm_failure()
        assert monitor.is_circuit_breaker_open is False
        monitor.record_llm_failure()
        assert monitor.is_circuit_breaker_open is True

    def test_circuit_breaker_resets_on_success(self, config):
        monitor = HealthMonitor(repository=None, config=config)
        monitor.record_llm_failure()
        monitor.record_llm_failure()
        monitor.record_llm_failure()
        assert monitor.is_circuit_breaker_open is True

        monitor.record_llm_success()
        assert monitor.is_circuit_breaker_open is False

    def test_circuit_breaker_check_returns_failed(self, config):
        monitor = HealthMonitor(repository=None, config=config)
        for _ in range(3):
            monitor.record_llm_failure()

        check = monitor._check_circuit_breaker()
        assert check.passed is False
        assert "OPEN" in check.message


class TestHealthMonitorTokenBudget:
    @pytest.fixture
    def config(self):
        return OrchestratorConfig()

    def test_under_budget(self, config):
        monitor = HealthMonitor(repository=None, config=config, max_tokens_per_task=10000)
        check = monitor.check_token_budget(5000)
        assert check.passed is True

    def test_over_budget(self, config):
        monitor = HealthMonitor(repository=None, config=config, max_tokens_per_task=10000)
        check = monitor.check_token_budget(15000)
        assert check.passed is False
        assert "exceeded" in check.message

    def test_exact_budget(self, config):
        monitor = HealthMonitor(repository=None, config=config, max_tokens_per_task=10000)
        check = monitor.check_token_budget(10000)
        assert check.passed is True  # Equal is OK, only > triggers


@requires_postgres
class TestHealthMonitorWithDB:
    @pytest.fixture
    def config(self):
        return OrchestratorConfig()

    @pytest.fixture(autouse=True)
    def clean_in_progress_tasks(self, db_engine):
        """Reset any in-progress tasks left by other tests so health checks start clean."""
        db_engine.execute(
            "UPDATE tasks SET status = 'DONE' WHERE status IN ('CODING', 'REVIEWING', 'RESEARCHING')"
        )
        yield

    @pytest.fixture
    def monitor(self, repository, config):
        return HealthMonitor(
            repository=repository,
            config=config,
            max_task_age_minutes=30,
        )

    def test_no_stuck_tasks(self, monitor):
        check = monitor._check_stuck_tasks()
        assert check.passed is True

    def test_run_checks_all_pass(self, monitor):
        checks = monitor.run_checks()
        assert all(c.passed for c in checks)


class TestHealthMonitorCooldownReset:
    """Additional circuit breaker edge cases."""

    @pytest.fixture
    def config(self):
        return OrchestratorConfig()

    def test_circuit_breaker_resets_after_cooldown(self, config):
        """Set breaker, advance time past cooldown, check reset via run_checks."""
        monitor = HealthMonitor(repository=None, config=config)
        for _ in range(3):
            monitor.record_llm_failure()
        assert monitor.is_circuit_breaker_open is True

        # Manually set the cooldown to the past
        monitor._circuit_breaker_until = datetime.now(UTC) - timedelta(seconds=1)

        # _check_circuit_breaker should detect expiry and reset
        check = monitor._check_circuit_breaker()
        assert check.passed is True
        assert "reset" in check.message.lower()
        assert monitor._circuit_breaker_open is False
        assert monitor._consecutive_llm_failures == 0

    def test_stuck_tasks_handles_naive_datetime(self, config):
        """Task with naive updated_at → treated as UTC, no crash."""
        # Create a task with a naive (no tzinfo) datetime far in the past
        old_task = Task(
            project_id=uuid.uuid4(),
            title="Old stuck task",
            description="This task has a naive datetime",
            status=TaskStatus.CODING,
        )
        # Manually set a naive datetime (no timezone info)
        old_task.updated_at = datetime(2020, 1, 1, 0, 0, 0)  # naive, no tzinfo

        class FakeRepoForStuck:
            def get_in_progress_tasks(self):
                return [old_task]

        monitor = HealthMonitor(
            repository=FakeRepoForStuck(),
            config=config,
            max_task_age_minutes=30,
        )
        check = monitor._check_stuck_tasks()
        # The naive datetime should be treated as UTC and detected as stuck
        assert check.passed is False
        assert "1 stuck" in check.message

    def test_is_circuit_breaker_open_auto_resets(self, config):
        """Property is_circuit_breaker_open checks expiry and auto-resets."""
        monitor = HealthMonitor(repository=None, config=config)
        for _ in range(3):
            monitor.record_llm_failure()
        assert monitor.is_circuit_breaker_open is True

        # Set cooldown to past
        monitor._circuit_breaker_until = datetime.now(UTC) - timedelta(seconds=10)

        # Property should auto-reset
        assert monitor.is_circuit_breaker_open is False
        assert monitor._circuit_breaker_open is False
        assert monitor._consecutive_llm_failures == 0
        assert monitor._circuit_breaker_until is None
