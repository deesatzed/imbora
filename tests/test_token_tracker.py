"""Tests for src/llm/token_tracker.py — Token-level cost tracking."""

import json
import os
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

from src.core.models import TokenCostRecord
from src.llm.token_tracker import TokenTracker


def test_record_returns_token_cost_record():
    tracker = TokenTracker()
    result = tracker.record(model="openai/gpt-4o", input_tokens=100, output_tokens=200)
    assert isinstance(result, TokenCostRecord)
    assert result.model_used == "openai/gpt-4o"
    assert result.input_tokens == 100
    assert result.output_tokens == 200


def test_record_computes_total_tokens_when_zero():
    tracker = TokenTracker()
    result = tracker.record(model="test-model", input_tokens=50, output_tokens=75)
    assert result.total_tokens == 125


def test_record_preserves_explicit_total_tokens():
    tracker = TokenTracker()
    result = tracker.record(
        model="test-model", input_tokens=50, output_tokens=75, total_tokens=999
    )
    assert result.total_tokens == 999


def test_record_estimates_cost_when_not_provided():
    tracker = TokenTracker(cost_per_1k_input=0.01, cost_per_1k_output=0.03)
    result = tracker.record(model="test-model", input_tokens=1000, output_tokens=1000)
    # 1000/1000 * 0.01 + 1000/1000 * 0.03 = 0.04
    assert abs(result.cost_usd - 0.04) < 1e-9


def test_record_uses_explicit_cost_when_provided():
    tracker = TokenTracker(cost_per_1k_input=0.01, cost_per_1k_output=0.03)
    result = tracker.record(
        model="test-model", input_tokens=1000, output_tokens=1000, cost_usd=0.99
    )
    assert result.cost_usd == 0.99


def test_set_context_attaches_task_and_role():
    tracker = TokenTracker()
    task_id = uuid.uuid4()
    run_id = uuid.uuid4()
    tracker.set_context(task_id=task_id, run_id=run_id, agent_role="builder")
    result = tracker.record(model="test-model", input_tokens=10, output_tokens=20)
    assert result.task_id == task_id
    assert result.run_id == run_id
    assert result.agent_role == "builder"


def test_set_context_can_clear():
    tracker = TokenTracker()
    task_id = uuid.uuid4()
    tracker.set_context(task_id=task_id, agent_role="sentinel")
    tracker.set_context()  # clear
    result = tracker.record(model="test-model", input_tokens=10, output_tokens=20)
    assert result.task_id is None
    assert result.run_id is None
    assert result.agent_role == ""


def test_get_session_totals_accumulates():
    tracker = TokenTracker()
    tracker.record(model="m1", input_tokens=100, output_tokens=200, cost_usd=0.01)
    tracker.record(model="m2", input_tokens=50, output_tokens=100, cost_usd=0.005)

    totals = tracker.get_session_totals()
    assert totals["total_input_tokens"] == 150
    assert totals["total_output_tokens"] == 300
    assert totals["total_tokens"] == 450
    assert abs(totals["total_cost_usd"] - 0.015) < 1e-9
    assert totals["call_count"] == 2


def test_get_session_totals_is_copy():
    tracker = TokenTracker()
    tracker.record(model="m1", input_tokens=10, output_tokens=20, cost_usd=0.001)
    totals = tracker.get_session_totals()
    totals["call_count"] = 999  # mutate the copy
    assert tracker.get_session_totals()["call_count"] == 1  # original unchanged


def test_get_task_totals_tracks_per_task():
    tracker = TokenTracker()
    task_a = uuid.uuid4()
    task_b = uuid.uuid4()

    tracker.set_context(task_id=task_a, agent_role="builder")
    tracker.record(model="m1", input_tokens=100, output_tokens=200, cost_usd=0.01)

    tracker.set_context(task_id=task_b, agent_role="sentinel")
    tracker.record(model="m2", input_tokens=50, output_tokens=100, cost_usd=0.005)

    tracker.set_context(task_id=task_a, agent_role="council")
    tracker.record(model="m3", input_tokens=25, output_tokens=50, cost_usd=0.002)

    totals_a = tracker.get_task_totals(task_a)
    assert totals_a["total_input_tokens"] == 125
    assert totals_a["total_output_tokens"] == 250
    assert totals_a["call_count"] == 2

    totals_b = tracker.get_task_totals(task_b)
    assert totals_b["total_input_tokens"] == 50
    assert totals_b["call_count"] == 1


def test_get_task_totals_returns_zeros_for_unknown_task():
    tracker = TokenTracker()
    totals = tracker.get_task_totals(uuid.uuid4())
    assert totals["total_input_tokens"] == 0
    assert totals["total_output_tokens"] == 0
    assert totals["total_tokens"] == 0
    assert totals["total_cost_usd"] == 0.0
    assert totals["call_count"] == 0


def test_jsonl_writing_to_tempfile():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = str(Path(tmpdir) / "costs.jsonl")
        tracker = TokenTracker(jsonl_path=jsonl_path)
        task_id = uuid.uuid4()
        tracker.set_context(task_id=task_id, agent_role="builder")
        tracker.record(model="openai/gpt-4o", input_tokens=100, output_tokens=200, cost_usd=0.01)
        tracker.record(model="anthropic/claude-sonnet", input_tokens=50, output_tokens=75, cost_usd=0.005)

        lines = Path(jsonl_path).read_text().strip().split("\n")
        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        assert entry1["model_used"] == "openai/gpt-4o"
        assert entry1["input_tokens"] == 100
        assert entry1["output_tokens"] == 200
        assert entry1["cost_usd"] == 0.01
        assert entry1["agent_role"] == "builder"
        assert entry1["task_id"] == str(task_id)
        assert "created_at" in entry1

        entry2 = json.loads(lines[1])
        assert entry2["model_used"] == "anthropic/claude-sonnet"


def test_jsonl_not_written_when_path_is_none():
    tracker = TokenTracker(jsonl_path=None)
    tracker.record(model="test", input_tokens=10, output_tokens=20)
    # No error, no file — just confirm it completes cleanly
    assert tracker.get_session_totals()["call_count"] == 1


def test_jsonl_creates_parent_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = str(Path(tmpdir) / "nested" / "deep" / "costs.jsonl")
        tracker = TokenTracker(jsonl_path=jsonl_path)
        tracker.record(model="test", input_tokens=10, output_tokens=20, cost_usd=0.001)
        assert Path(jsonl_path).exists()
        lines = Path(jsonl_path).read_text().strip().split("\n")
        assert len(lines) == 1


def test_jsonl_write_failure_handled_gracefully():
    """Trigger the JSONL exception handler (lines 183-184) by pointing
    jsonl_path at a directory, causing open() to raise IsADirectoryError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use the directory itself as the path — open(dir, "a") will fail
        tracker = TokenTracker(jsonl_path=tmpdir)
        # Should not raise — exception is caught and logged
        result = tracker.record(model="test", input_tokens=10, output_tokens=20, cost_usd=0.001)
        assert isinstance(result, TokenCostRecord)
        assert tracker.get_session_totals()["call_count"] == 1


def test_persist_to_db_with_save_token_cost_method():
    saved = []

    class FakeRepo:
        def save_token_cost(self, record):
            saved.append(record)

    repo = FakeRepo()
    tracker = TokenTracker(repository=repo)
    tracker.record(model="test", input_tokens=10, output_tokens=20, cost_usd=0.001)
    assert len(saved) == 1
    assert isinstance(saved[0], TokenCostRecord)


def test_persist_to_db_skipped_when_no_repository():
    tracker = TokenTracker(repository=None)
    # Should not raise
    tracker.record(model="test", input_tokens=10, output_tokens=20)
    assert tracker.get_session_totals()["call_count"] == 1


def test_persist_to_db_skipped_when_repo_lacks_method():
    repo = SimpleNamespace()  # no save_token_cost attribute
    tracker = TokenTracker(repository=repo)
    # Should not raise
    tracker.record(model="test", input_tokens=10, output_tokens=20)
    assert tracker.get_session_totals()["call_count"] == 1


def test_persist_to_db_handles_exception_gracefully():
    class FailingRepo:
        def save_token_cost(self, record):
            raise RuntimeError("DB connection lost")

    tracker = TokenTracker(repository=FailingRepo())
    # Should not raise — exception is caught and logged
    result = tracker.record(model="test", input_tokens=10, output_tokens=20)
    assert isinstance(result, TokenCostRecord)
    assert tracker.get_session_totals()["call_count"] == 1


def test_record_with_no_context_leaves_task_id_none():
    tracker = TokenTracker()
    result = tracker.record(model="test", input_tokens=10, output_tokens=20)
    assert result.task_id is None
    assert result.run_id is None
    assert result.agent_role == ""


def test_session_totals_start_at_zero():
    tracker = TokenTracker()
    totals = tracker.get_session_totals()
    assert totals["total_input_tokens"] == 0
    assert totals["total_output_tokens"] == 0
    assert totals["total_tokens"] == 0
    assert totals["total_cost_usd"] == 0.0
    assert totals["call_count"] == 0


def test_cost_estimation_with_custom_rates():
    tracker = TokenTracker(cost_per_1k_input=0.005, cost_per_1k_output=0.015)
    result = tracker.record(model="test", input_tokens=2000, output_tokens=3000)
    # 2000/1000 * 0.005 + 3000/1000 * 0.015 = 0.01 + 0.045 = 0.055
    assert abs(result.cost_usd - 0.055) < 1e-9
