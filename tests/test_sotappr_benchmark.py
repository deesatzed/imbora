"""Tests for SOTAppR benchmark snapshot and regression gate helpers."""

from __future__ import annotations

from src.sotappr.benchmark import (
    BenchmarkTolerances,
    build_run_metrics,
    compare_to_baseline,
    make_frozen_snapshot,
    summarize_benchmark_runs,
)


def test_build_run_metrics_counts_retry_and_trace_events() -> None:
    run = {
        "id": "run-1",
        "status": "completed",
        "tasks_seeded": 4,
        "tasks_processed": 4,
        "elapsed_hours": 1.5,
        "estimated_cost_usd": 0.2,
    }
    artifacts = [
        {"artifact_type": "retry_telemetry"},
        {"artifact_type": "retry_telemetry"},
        {"artifact_type": "trace_event"},
        {"artifact_type": "arbitration_decision"},
    ]
    metrics = build_run_metrics(run_row=run, artifacts=artifacts)
    assert metrics["quality_score"] == 1.0
    assert metrics["retry_events"] == 2
    assert metrics["trace_events"] == 1
    assert metrics["arbitration_events"] == 1


def test_compare_to_baseline_detects_quality_regression() -> None:
    baseline = {
        "aggregate": {
            "quality_success_rate": 1.0,
            "avg_elapsed_hours": 1.0,
            "avg_estimated_cost_usd": 0.2,
            "retry_events_per_task": 0.1,
            "rollback_rate": 0.1,
        }
    }
    current = {
        "quality_success_rate": 0.7,
        "avg_elapsed_hours": 1.0,
        "avg_estimated_cost_usd": 0.2,
        "retry_events_per_task": 0.1,
        "rollback_rate": 0.1,
    }
    result = compare_to_baseline(
        baseline=baseline,
        current_aggregate=current,
        tolerances=BenchmarkTolerances(quality_drop_abs=0.05),
    )
    assert result["passed"] is False
    quality_check = [row for row in result["checks"] if row["metric"] == "quality_success_rate"][0]
    assert quality_check["passed"] is False


def test_make_frozen_snapshot_contains_aggregate_and_runs() -> None:
    runs = [
        {
            "run_id": "run-a",
            "completed": True,
            "quality_score": 1.0,
            "done_ratio": 1.0,
            "elapsed_hours": 1.0,
            "estimated_cost_usd": 0.1,
            "tasks_processed": 3,
            "retry_events": 0,
        },
        {
            "run_id": "run-b",
            "completed": False,
            "quality_score": 0.4,
            "done_ratio": 0.8,
            "elapsed_hours": 1.2,
            "estimated_cost_usd": 0.2,
            "tasks_processed": 2,
            "retry_events": 1,
        },
    ]
    snapshot = make_frozen_snapshot(
        run_metrics=runs,
        project_id="proj-1",
        tolerances=BenchmarkTolerances(),
    )
    assert snapshot["project_id"] == "proj-1"
    assert len(snapshot["run_ids"]) == 2
    assert snapshot["aggregate"]["total_runs"] == 2
    assert "tolerances" in snapshot


def test_summarize_benchmark_runs_handles_empty() -> None:
    summary = summarize_benchmark_runs([])
    assert summary["total_runs"] == 0
    assert summary["quality_success_rate"] == 0.0
