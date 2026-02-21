"""Benchmark snapshot + regression gate helpers for SOTAppR runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class BenchmarkTolerances:
    """Allowed benchmark drift relative to a frozen baseline."""

    quality_drop_abs: float = 0.05
    latency_increase_pct: float = 0.20
    cost_increase_pct: float = 0.20
    retry_increase_pct: float = 0.25
    rollback_increase_pct: float = 0.15

    def to_dict(self) -> dict[str, float]:
        return {
            "quality_drop_abs": float(self.quality_drop_abs),
            "latency_increase_pct": float(self.latency_increase_pct),
            "cost_increase_pct": float(self.cost_increase_pct),
            "retry_increase_pct": float(self.retry_increase_pct),
            "rollback_increase_pct": float(self.rollback_increase_pct),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "BenchmarkTolerances":
        raw = payload or {}
        return cls(
            quality_drop_abs=float(raw.get("quality_drop_abs", 0.05)),
            latency_increase_pct=float(raw.get("latency_increase_pct", 0.20)),
            cost_increase_pct=float(raw.get("cost_increase_pct", 0.20)),
            retry_increase_pct=float(raw.get("retry_increase_pct", 0.25)),
            rollback_increase_pct=float(raw.get("rollback_increase_pct", 0.15)),
        )


def build_run_metrics(
    *,
    run_row: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build normalized benchmark metrics for one SOTAppR run."""
    status = str(run_row.get("status") or "unknown")
    tasks_seeded = int(run_row.get("tasks_seeded") or 0)
    tasks_processed = int(run_row.get("tasks_processed") or 0)
    elapsed_hours = float(run_row.get("elapsed_hours") or 0.0)
    estimated_cost_usd = float(run_row.get("estimated_cost_usd") or 0.0)
    done_ratio = (
        float(tasks_processed) / float(tasks_seeded)
        if tasks_seeded > 0
        else (1.0 if tasks_processed > 0 else 0.0)
    )

    retry_events = 0
    arbitration_events = 0
    trace_events = 0
    for row in artifacts:
        artifact_type = str(row.get("artifact_type") or "")
        if artifact_type == "retry_telemetry":
            retry_events += 1
        elif artifact_type == "arbitration_decision":
            arbitration_events += 1
        elif artifact_type == "trace_event":
            trace_events += 1

    completed = status == "completed"
    quality_score = 0.5 * (1.0 if completed else 0.0) + 0.5 * min(max(done_ratio, 0.0), 1.0)

    return {
        "run_id": str(run_row.get("id")),
        "status": status,
        "tasks_seeded": tasks_seeded,
        "tasks_processed": tasks_processed,
        "done_ratio": round(done_ratio, 6),
        "completed": completed,
        "quality_score": round(quality_score, 6),
        "elapsed_hours": round(elapsed_hours, 6),
        "estimated_cost_usd": round(estimated_cost_usd, 6),
        "retry_events": retry_events,
        "arbitration_events": arbitration_events,
        "trace_events": trace_events,
    }


def summarize_benchmark_runs(run_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate run-level benchmark metrics into a regression baseline/current summary."""
    total_runs = len(run_metrics)
    if total_runs == 0:
        return {
            "total_runs": 0,
            "quality_success_rate": 0.0,
            "avg_quality_score": 0.0,
            "avg_done_ratio": 0.0,
            "avg_elapsed_hours": 0.0,
            "avg_estimated_cost_usd": 0.0,
            "retry_events_per_task": 0.0,
            "rollback_rate": 0.0,
            "total_tasks_processed": 0,
            "total_retry_events": 0,
        }

    total_tasks_processed = sum(int(row.get("tasks_processed") or 0) for row in run_metrics)
    total_retry_events = sum(int(row.get("retry_events") or 0) for row in run_metrics)

    quality_success_rate = sum(1 for row in run_metrics if bool(row.get("completed"))) / total_runs
    avg_quality_score = sum(float(row.get("quality_score") or 0.0) for row in run_metrics) / total_runs
    avg_done_ratio = sum(float(row.get("done_ratio") or 0.0) for row in run_metrics) / total_runs
    avg_elapsed_hours = sum(float(row.get("elapsed_hours") or 0.0) for row in run_metrics) / total_runs
    avg_cost = sum(float(row.get("estimated_cost_usd") or 0.0) for row in run_metrics) / total_runs
    rollback_rate = sum(1 for row in run_metrics if int(row.get("retry_events") or 0) > 0) / total_runs
    retry_events_per_task = total_retry_events / max(1, total_tasks_processed)

    return {
        "total_runs": total_runs,
        "quality_success_rate": round(quality_success_rate, 6),
        "avg_quality_score": round(avg_quality_score, 6),
        "avg_done_ratio": round(avg_done_ratio, 6),
        "avg_elapsed_hours": round(avg_elapsed_hours, 6),
        "avg_estimated_cost_usd": round(avg_cost, 6),
        "retry_events_per_task": round(retry_events_per_task, 6),
        "rollback_rate": round(rollback_rate, 6),
        "total_tasks_processed": int(total_tasks_processed),
        "total_retry_events": int(total_retry_events),
    }


def make_frozen_snapshot(
    *,
    run_metrics: list[dict[str, Any]],
    project_id: str | None,
    tolerances: BenchmarkTolerances,
) -> dict[str, Any]:
    """Build a portable frozen benchmark snapshot payload."""
    return {
        "schema_version": "1",
        "created_at": datetime.now(UTC).isoformat(),
        "project_id": project_id,
        "run_ids": [str(row.get("run_id")) for row in run_metrics],
        "aggregate": summarize_benchmark_runs(run_metrics),
        "tolerances": tolerances.to_dict(),
        "runs": run_metrics,
    }


def compare_to_baseline(
    *,
    baseline: dict[str, Any],
    current_aggregate: dict[str, Any],
    tolerances: BenchmarkTolerances,
) -> dict[str, Any]:
    """Compare current benchmark summary against a frozen baseline and emit pass/fail evidence."""
    base = baseline.get("aggregate") or {}
    checks: list[dict[str, Any]] = []

    checks.append(
        _compare_lower_bound(
            metric="quality_success_rate",
            baseline=float(base.get("quality_success_rate") or 0.0),
            current=float(current_aggregate.get("quality_success_rate") or 0.0),
            allowed_drop=tolerances.quality_drop_abs,
        )
    )
    checks.append(
        _compare_pct_increase(
            metric="avg_elapsed_hours",
            baseline=float(base.get("avg_elapsed_hours") or 0.0),
            current=float(current_aggregate.get("avg_elapsed_hours") or 0.0),
            allowed_pct=tolerances.latency_increase_pct,
        )
    )
    checks.append(
        _compare_pct_increase(
            metric="avg_estimated_cost_usd",
            baseline=float(base.get("avg_estimated_cost_usd") or 0.0),
            current=float(current_aggregate.get("avg_estimated_cost_usd") or 0.0),
            allowed_pct=tolerances.cost_increase_pct,
        )
    )
    checks.append(
        _compare_pct_increase(
            metric="retry_events_per_task",
            baseline=float(base.get("retry_events_per_task") or 0.0),
            current=float(current_aggregate.get("retry_events_per_task") or 0.0),
            allowed_pct=tolerances.retry_increase_pct,
        )
    )
    checks.append(
        _compare_pct_increase(
            metric="rollback_rate",
            baseline=float(base.get("rollback_rate") or 0.0),
            current=float(current_aggregate.get("rollback_rate") or 0.0),
            allowed_pct=tolerances.rollback_increase_pct,
        )
    )

    passed = all(bool(row.get("passed")) for row in checks)
    return {
        "passed": passed,
        "checks": checks,
        "baseline_aggregate": base,
        "current_aggregate": current_aggregate,
    }


def collect_run_metrics_from_repository(
    *,
    repository: Any,
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect normalized run metrics + artifact telemetry from repository rows."""
    collected: list[dict[str, Any]] = []
    for row in runs:
        run_id_raw = row.get("id")
        if run_id_raw is None:
            continue
        artifacts: list[dict[str, Any]] = []
        if hasattr(repository, "list_sotappr_artifacts"):
            try:
                artifacts = repository.list_sotappr_artifacts(
                    run_id=UUID(str(run_id_raw)),
                    limit=500,
                )
            except Exception:
                artifacts = []
        collected.append(build_run_metrics(run_row=row, artifacts=artifacts))
    return collected


def _compare_lower_bound(metric: str, baseline: float, current: float, allowed_drop: float) -> dict[str, Any]:
    floor = baseline - allowed_drop
    passed = current >= floor
    return {
        "metric": metric,
        "baseline": round(baseline, 6),
        "current": round(current, 6),
        "allowed_drop_abs": round(allowed_drop, 6),
        "threshold": round(floor, 6),
        "passed": passed,
    }


def _compare_pct_increase(metric: str, baseline: float, current: float, allowed_pct: float) -> dict[str, Any]:
    if baseline <= 0:
        # With no baseline signal, only fail if current is also meaningfully non-zero.
        threshold = 0.0
        passed = current <= 0.0
    else:
        threshold = baseline * (1.0 + allowed_pct)
        passed = current <= threshold
    return {
        "metric": metric,
        "baseline": round(baseline, 6),
        "current": round(current, 6),
        "allowed_increase_pct": round(allowed_pct, 6),
        "threshold": round(threshold, 6),
        "passed": passed,
    }
