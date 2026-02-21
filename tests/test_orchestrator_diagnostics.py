"""Tests for src/orchestrator/diagnostics.py — Dirac Pattern 6 diagnostic snapshots."""

import json
import uuid

import pytest

from src.core.models import AgentResult
from src.orchestrator.diagnostics import (
    DiagnosticEntry,
    DiagnosticsCollector,
    RunDiagnostics,
    _summarize_data,
)


class TestRunDiagnostics:
    def test_total_duration(self):
        from datetime import UTC, datetime
        diag = RunDiagnostics(task_id=uuid.uuid4(), run_number=1)
        diag.entries = [
            DiagnosticEntry(
                timestamp=datetime.now(UTC), task_id=diag.task_id,
                agent_name="A", status="success", duration_seconds=1.5,
            ),
            DiagnosticEntry(
                timestamp=datetime.now(UTC), task_id=diag.task_id,
                agent_name="B", status="success", duration_seconds=2.5,
            ),
        ]
        assert diag.total_duration == 4.0

    def test_total_tokens(self):
        from datetime import UTC, datetime
        diag = RunDiagnostics(task_id=uuid.uuid4(), run_number=1)
        diag.entries = [
            DiagnosticEntry(
                timestamp=datetime.now(UTC), task_id=diag.task_id,
                agent_name="A", status="success", duration_seconds=1.0, tokens_used=50,
            ),
            DiagnosticEntry(
                timestamp=datetime.now(UTC), task_id=diag.task_id,
                agent_name="B", status="success", duration_seconds=1.0, tokens_used=75,
            ),
        ]
        assert diag.total_tokens == 125

    def test_to_dict(self):
        diag = RunDiagnostics(task_id=uuid.uuid4(), run_number=1)
        d = diag.to_dict()
        assert "task_id" in d
        assert d["run_number"] == 1
        assert d["outcome"] == "in_progress"
        assert isinstance(d["entries"], list)
        assert isinstance(d["events"], list)


class TestDiagnosticsCollector:
    @pytest.fixture
    def collector(self):
        return DiagnosticsCollector(max_history=10)

    @pytest.fixture
    def task_id(self):
        return uuid.uuid4()

    def test_start_run(self, collector, task_id):
        diag = collector.start_run(task_id, 1, run_id="run-1", trace_id="trace-1")
        assert diag.task_id == task_id
        assert diag.run_number == 1
        assert diag.run_id == "run-1"
        assert diag.trace_id == "trace-1"

    def test_record_agent(self, collector, task_id):
        collector.start_run(task_id, 1, run_id="run-a", trace_id="trace-a")
        result = AgentResult(agent_name="Builder", status="success", duration_seconds=1.5)
        entry = collector.record_agent(
            task_id, "Builder", result, tokens_used=100, input_summary="test input",
        )
        assert entry.agent_name == "Builder"
        assert entry.tokens_used == 100
        assert entry.status == "success"
        assert entry.run_id == "run-a"
        assert entry.trace_id == "trace-a"

        current = collector.get_current()
        assert len(current.entries) == 1

    def test_complete_run(self, collector, task_id):
        collector.start_run(task_id, 1)
        completed = collector.complete_run("success")
        assert completed.outcome == "success"
        assert collector.get_current() is None

    def test_complete_run_none(self, collector):
        assert collector.complete_run("success") is None

    def test_dump_to_file(self, collector, task_id, tmp_path):
        collector.start_run(task_id, 1)
        result = AgentResult(agent_name="Test", status="success", duration_seconds=0.5)
        collector.record_agent(task_id, "Test", result)
        collector.complete_run("success")

        path = collector.dump_to_file(tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total_runs"] == 1
        assert len(data["runs"]) == 1

    def test_dump_to_specific_file(self, collector, task_id, tmp_path):
        collector.start_run(task_id, 1)
        collector.complete_run("success")

        specific_file = tmp_path / "my_diag.json"
        path = collector.dump_to_file(specific_file)
        assert path == specific_file
        assert path.exists()

    def test_dump_filtered_by_task(self, collector, tmp_path):
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()

        collector.start_run(id1, 1)
        collector.complete_run("success")
        collector.start_run(id2, 1)
        collector.complete_run("failure")

        path = collector.dump_to_file(tmp_path, task_id=id1)
        data = json.loads(path.read_text())
        assert data["total_runs"] == 1

    def test_max_history(self):
        collector = DiagnosticsCollector(max_history=3)
        for i in range(5):
            collector.start_run(uuid.uuid4(), i + 1)
            collector.complete_run("success")

        history = collector.get_history(limit=10)
        assert len(history) == 3

    def test_get_history_limit(self, collector):
        for i in range(5):
            collector.start_run(uuid.uuid4(), i + 1)
            collector.complete_run("success")

        history = collector.get_history(limit=2)
        assert len(history) == 2

    def test_error_recording(self, collector, task_id):
        collector.start_run(task_id, 1)
        result = AgentResult(
            agent_name="Builder", status="failure",
            error="Tests failed", duration_seconds=2.0,
        )
        entry = collector.record_agent(task_id, "Builder", result)
        assert entry.error == "Tests failed"

        completed = collector.complete_run("failure", error_summary="Builder failed")
        assert completed.error_summary == "Builder failed"

    def test_record_event_appends_to_current_run(self, collector, task_id):
        collector.start_run(task_id, 1, run_id="run-b", trace_id="trace-b")
        collector.record_event(
            "retry_decision",
            {"retry_cause": "tests_failed"},
            task_id=task_id,
        )
        current = collector.get_current()
        assert current is not None
        assert len(current.events) == 1
        assert current.events[0]["event_type"] == "retry_decision"
        assert current.events[0]["run_id"] == "run-b"


class TestSummarizeData:
    def test_empty_data(self):
        assert _summarize_data({}) == ""

    def test_with_keys(self):
        result = _summarize_data({"a": 1, "b": 2})
        assert "a" in result
        assert "b" in result

    def test_truncation(self):
        long_keys = {f"very_long_key_{i}": i for i in range(100)}
        result = _summarize_data(long_keys, max_len=50)
        assert len(result) <= 53  # 50 + "..."
