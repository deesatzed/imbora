"""Tests for src/orchestrator/metrics.py — Dirac Pattern 3 pipeline metrics."""

import uuid
from datetime import UTC, datetime

import pytest

from src.orchestrator.metrics import AgentMetric, PipelineMetrics, PipelineRun


class TestAgentMetric:
    def test_succeeded_true(self):
        metric = AgentMetric(
            task_id=uuid.uuid4(),
            agent_name="TestAgent",
            started_at=datetime.now(UTC),
            status="success",
        )
        assert metric.succeeded is True

    def test_succeeded_false(self):
        metric = AgentMetric(
            task_id=uuid.uuid4(),
            agent_name="TestAgent",
            started_at=datetime.now(UTC),
            status="failure",
        )
        assert metric.succeeded is False


class TestPipelineRun:
    def test_total_duration(self):
        run = PipelineRun(task_id=uuid.uuid4(), run_number=1)
        task_id = run.task_id
        run.agent_metrics = [
            AgentMetric(task_id=task_id, agent_name="A", started_at=datetime.now(UTC), duration_seconds=1.5),
            AgentMetric(task_id=task_id, agent_name="B", started_at=datetime.now(UTC), duration_seconds=2.5),
        ]
        assert run.total_duration == 4.0

    def test_bottleneck_agent(self):
        run = PipelineRun(task_id=uuid.uuid4(), run_number=1)
        task_id = run.task_id
        run.agent_metrics = [
            AgentMetric(task_id=task_id, agent_name="Fast", started_at=datetime.now(UTC), duration_seconds=0.5),
            AgentMetric(task_id=task_id, agent_name="Slow", started_at=datetime.now(UTC), duration_seconds=5.0),
            AgentMetric(task_id=task_id, agent_name="Medium", started_at=datetime.now(UTC), duration_seconds=2.0),
        ]
        assert run.bottleneck_agent == "Slow"

    def test_bottleneck_none_when_empty(self):
        run = PipelineRun(task_id=uuid.uuid4(), run_number=1)
        assert run.bottleneck_agent is None


class TestPipelineMetrics:
    @pytest.fixture
    def metrics(self):
        return PipelineMetrics()

    @pytest.fixture
    def task_id(self):
        return uuid.uuid4()

    def test_start_run(self, metrics, task_id):
        run = metrics.start_run(task_id, run_number=1)
        assert run.task_id == task_id
        assert run.run_number == 1
        assert run.outcome == "in_progress"

    def test_start_and_complete_agent(self, metrics, task_id):
        metrics.start_run(task_id, 1)
        metric = metrics.start_agent(task_id, "Builder")
        assert metric.agent_name == "Builder"
        assert metric.status == "pending"

        metrics.complete_agent(metric, status="success", tokens_used=100)
        assert metric.status == "success"
        assert metric.tokens_used == 100
        assert metric.completed_at is not None
        assert metric.duration_seconds >= 0

    def test_complete_run(self, metrics, task_id):
        metrics.start_run(task_id, 1)
        run = metrics.complete_run("success")
        assert run.outcome == "success"
        assert run.completed_at is not None

    def test_complete_run_none_when_no_run(self, metrics):
        assert metrics.complete_run("success") is None

    def test_get_runs_all(self, metrics):
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()
        metrics.start_run(id1, 1)
        metrics.complete_run("success")
        metrics.start_run(id2, 1)
        metrics.complete_run("failure")

        all_runs = metrics.get_runs()
        assert len(all_runs) == 2

    def test_get_runs_filtered(self, metrics):
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()
        metrics.start_run(id1, 1)
        metrics.complete_run("success")
        metrics.start_run(id2, 1)
        metrics.complete_run("failure")

        filtered = metrics.get_runs(task_id=id1)
        assert len(filtered) == 1
        assert filtered[0].task_id == id1

    def test_get_latest_run(self, metrics, task_id):
        metrics.start_run(task_id, 1)
        metrics.complete_run("success")
        metrics.start_run(task_id, 2)
        metrics.complete_run("failure")

        latest = metrics.get_latest_run()
        assert latest.run_number == 2

    def test_get_latest_run_none(self, metrics):
        assert metrics.get_latest_run() is None

    def test_token_accumulation(self, metrics, task_id):
        metrics.start_run(task_id, 1)
        m1 = metrics.start_agent(task_id, "A")
        metrics.complete_agent(m1, "success", tokens_used=50)
        m2 = metrics.start_agent(task_id, "B")
        metrics.complete_agent(m2, "success", tokens_used=75)
        run = metrics.complete_run("success")

        assert run.total_tokens == 125

    def test_get_summary(self, metrics):
        id1 = uuid.uuid4()
        metrics.start_run(id1, 1)
        m = metrics.start_agent(id1, "Agent")
        metrics.complete_agent(m, "success", tokens_used=100)
        metrics.complete_run("success")

        summary = metrics.get_summary()
        assert summary["total_runs"] == 1
        assert summary["total_tokens"] == 100
        assert summary["outcomes"]["success"] == 1

    def test_get_summary_empty(self, metrics):
        summary = metrics.get_summary()
        assert summary == {"total_runs": 0}
