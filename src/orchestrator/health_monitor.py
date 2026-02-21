"""Health monitor for The Associate — Dirac Pattern 1.

Runs at the start of each orchestrator loop iteration to detect and
remediate anomalies before processing the next task.

Checks:
1. Stuck tasks — tasks in CODING/REVIEWING too long
2. Repeated identical errors — same error_signature across tasks
3. Token budget overrun — task exceeded max token budget
4. LLM circuit breaker — consecutive LLM failures
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Optional

from src.core.config import OrchestratorConfig
from src.core.models import Task, TaskStatus
from src.db.repository import Repository

logger = logging.getLogger("associate.orchestrator.health_monitor")


class HealthCheck:
    """Result of a single health check."""

    def __init__(
        self,
        check_name: str,
        passed: bool,
        message: str = "",
        remediation: str | None = None,
    ):
        self.check_name = check_name
        self.passed = passed
        self.message = message
        self.remediation = remediation


class HealthMonitor:
    """Anomaly detection and remediation at the start of each loop iteration.

    Injected dependencies:
        repository: Database access for task queries.
        config: Orchestrator configuration for thresholds.
    """

    def __init__(
        self,
        repository: Repository,
        config: OrchestratorConfig,
        max_task_age_minutes: int = 30,
        max_tokens_per_task: int = 100_000,
    ):
        self.repository = repository
        self.config = config
        self.max_task_age_minutes = max_task_age_minutes
        self.max_tokens_per_task = max_tokens_per_task
        self._consecutive_llm_failures = 0
        self._circuit_breaker_open = False
        self._circuit_breaker_until: Optional[datetime] = None

    def run_checks(self) -> list[HealthCheck]:
        """Run all health checks. Returns list of check results."""
        checks: list[HealthCheck] = []

        checks.append(self._check_stuck_tasks())
        checks.append(self._check_circuit_breaker())

        failed = [c for c in checks if not c.passed]
        if failed:
            logger.warning(
                "Health checks: %d/%d failed: %s",
                len(failed), len(checks),
                ", ".join(c.check_name for c in failed),
            )
        else:
            logger.debug("Health checks: all %d passed", len(checks))

        return checks

    def _check_stuck_tasks(self) -> HealthCheck:
        """Find tasks stuck in processing states too long."""
        in_progress = self.repository.get_in_progress_tasks()

        stuck_tasks: list[Task] = []
        cutoff = datetime.now(UTC) - timedelta(minutes=self.max_task_age_minutes)

        for task in in_progress:
            # Check if task.updated_at is timezone-aware
            updated = task.updated_at
            if updated.tzinfo is None:
                # Assume UTC for naive datetimes from DB
                updated = updated.replace(tzinfo=UTC)
            if updated < cutoff:
                stuck_tasks.append(task)

        if not stuck_tasks:
            return HealthCheck("stuck_tasks", passed=True, message="No stuck tasks")

        task_summaries = [
            f"'{t.title}' (status={t.status.value}, updated={t.updated_at.isoformat()})"
            for t in stuck_tasks
        ]
        return HealthCheck(
            "stuck_tasks",
            passed=False,
            message=f"{len(stuck_tasks)} stuck task(s): {', '.join(task_summaries)}",
            remediation="Consider rewinding and retrying or marking as STUCK",
        )

    def _check_circuit_breaker(self) -> HealthCheck:
        """Check if the LLM circuit breaker is open."""
        if not self._circuit_breaker_open:
            return HealthCheck("circuit_breaker", passed=True, message="Circuit breaker closed")

        if self._circuit_breaker_until and datetime.now(UTC) > self._circuit_breaker_until:
            # Reset the circuit breaker
            self._circuit_breaker_open = False
            self._consecutive_llm_failures = 0
            self._circuit_breaker_until = None
            logger.info("Circuit breaker reset after cooldown")
            return HealthCheck(
                "circuit_breaker", passed=True,
                message="Circuit breaker reset after cooldown",
            )

        remaining = ""
        if self._circuit_breaker_until:
            delta = self._circuit_breaker_until - datetime.now(UTC)
            remaining = f" ({delta.seconds}s remaining)"
        return HealthCheck(
            "circuit_breaker",
            passed=False,
            message=f"Circuit breaker OPEN — {self._consecutive_llm_failures} consecutive LLM failures{remaining}",
            remediation="Wait for cooldown or check LLM provider status",
        )

    def record_llm_success(self) -> None:
        """Record a successful LLM call — resets the failure counter."""
        self._consecutive_llm_failures = 0
        if self._circuit_breaker_open:
            self._circuit_breaker_open = False
            self._circuit_breaker_until = None
            logger.info("Circuit breaker closed after successful LLM call")

    def record_llm_failure(self) -> None:
        """Record a failed LLM call. Opens circuit breaker after 3 consecutive failures."""
        self._consecutive_llm_failures += 1
        logger.warning("Consecutive LLM failures: %d", self._consecutive_llm_failures)

        if self._consecutive_llm_failures >= 3 and not self._circuit_breaker_open:
            self._circuit_breaker_open = True
            # Backoff: 30s * failure_count (capped at 5 minutes)
            cooldown_seconds = min(30 * self._consecutive_llm_failures, 300)
            self._circuit_breaker_until = datetime.now(UTC) + timedelta(seconds=cooldown_seconds)
            logger.warning(
                "Circuit breaker OPENED — cooling down for %ds",
                cooldown_seconds,
            )

    def check_token_budget(self, task_tokens: int) -> HealthCheck:
        """Check if a task has exceeded its token budget.

        Args:
            task_tokens: Total tokens consumed by the task so far.

        Returns:
            HealthCheck result.
        """
        if task_tokens > self.max_tokens_per_task:
            return HealthCheck(
                "token_budget",
                passed=False,
                message=f"Token budget exceeded: {task_tokens} > {self.max_tokens_per_task}",
                remediation="Mark task as STUCK with reason 'token_budget_exceeded'",
            )
        return HealthCheck(
            "token_budget",
            passed=True,
            message=f"Token budget OK: {task_tokens}/{self.max_tokens_per_task}",
        )

    @property
    def is_circuit_breaker_open(self) -> bool:
        """Check if the circuit breaker is currently open."""
        if self._circuit_breaker_open and self._circuit_breaker_until:
            if datetime.now(UTC) > self._circuit_breaker_until:
                self._circuit_breaker_open = False
                self._consecutive_llm_failures = 0
                self._circuit_breaker_until = None
                return False
        return self._circuit_breaker_open
