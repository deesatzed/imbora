"""Pipeline metrics collector for The Associate — Dirac Pattern 3.

Records per-agent execution data during pipeline runs:
  {task_id, agent_name, started_at, completed_at, duration_seconds, tokens_used, status, error}

Enables: CLI status with per-stage timing, bottleneck identification, cost tracking.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Optional

logger = logging.getLogger("associate.orchestrator.metrics")


@dataclass
class AgentMetric:
    """Single agent execution record within a pipeline run."""
    task_id: uuid.UUID
    agent_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    tokens_used: int = 0
    status: str = "pending"
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.status == "success"


@dataclass
class PipelineRun:
    """Aggregated metrics for a full pipeline execution on one task."""
    task_id: uuid.UUID
    run_number: int
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: Optional[datetime] = None
    agent_metrics: list[AgentMetric] = field(default_factory=list)
    total_tokens: int = 0
    outcome: str = "in_progress"  # "success", "failure", "stuck"

    @property
    def total_duration(self) -> float:
        return sum(m.duration_seconds for m in self.agent_metrics)

    @property
    def bottleneck_agent(self) -> Optional[str]:
        if not self.agent_metrics:
            return None
        slowest = max(self.agent_metrics, key=lambda m: m.duration_seconds)
        return slowest.agent_name


class PipelineMetrics:
    """Collects and aggregates pipeline execution metrics.

    Used by pipeline.py to wrap each agent.run() call with timing
    and token tracking.
    """

    def __init__(self):
        self._runs: list[PipelineRun] = []
        self._current_run: Optional[PipelineRun] = None

    def start_run(self, task_id: uuid.UUID, run_number: int) -> PipelineRun:
        """Begin tracking a new pipeline run."""
        run = PipelineRun(
            task_id=task_id,
            run_number=run_number,
            started_at=datetime.now(UTC),
        )
        self._current_run = run
        self._runs.append(run)
        logger.info("Pipeline run #%d started for task %s", run_number, task_id)
        return run

    def start_agent(self, task_id: uuid.UUID, agent_name: str) -> AgentMetric:
        """Begin tracking an agent execution within the current run."""
        metric = AgentMetric(
            task_id=task_id,
            agent_name=agent_name,
            started_at=datetime.now(UTC),
        )
        if self._current_run:
            self._current_run.agent_metrics.append(metric)
        return metric

    def complete_agent(
        self,
        metric: AgentMetric,
        status: str,
        tokens_used: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Record completion of an agent execution."""
        metric.completed_at = datetime.now(UTC)
        metric.status = status
        metric.tokens_used = tokens_used
        metric.error = error
        metric.duration_seconds = (metric.completed_at - metric.started_at).total_seconds()

        if self._current_run:
            self._current_run.total_tokens += tokens_used

        logger.info(
            "Agent '%s': status=%s, duration=%.2fs, tokens=%d",
            metric.agent_name, status, metric.duration_seconds, tokens_used,
        )

    def complete_run(self, outcome: str) -> Optional[PipelineRun]:
        """Record completion of the current pipeline run."""
        if self._current_run is None:
            return None

        run = self._current_run
        run.completed_at = datetime.now(UTC)
        run.outcome = outcome
        self._current_run = None

        logger.info(
            "Pipeline run #%d complete: outcome=%s, duration=%.2fs, tokens=%d, bottleneck=%s",
            run.run_number,
            outcome,
            run.total_duration,
            run.total_tokens,
            run.bottleneck_agent or "none",
        )
        return run

    def get_runs(self, task_id: Optional[uuid.UUID] = None) -> list[PipelineRun]:
        """Get all recorded runs, optionally filtered by task."""
        if task_id:
            return [r for r in self._runs if r.task_id == task_id]
        return list(self._runs)

    def get_latest_run(self) -> Optional[PipelineRun]:
        """Get the most recent run."""
        return self._runs[-1] if self._runs else None

    def get_summary(self) -> dict:
        """Get an aggregate summary of all recorded runs."""
        if not self._runs:
            return {"total_runs": 0}

        total_tokens = sum(r.total_tokens for r in self._runs)
        total_duration = sum(r.total_duration for r in self._runs)
        outcomes = {}
        for r in self._runs:
            outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1

        return {
            "total_runs": len(self._runs),
            "total_tokens": total_tokens,
            "total_duration_seconds": round(total_duration, 2),
            "outcomes": outcomes,
        }
