"""Benchmark runner — executes DS pipeline across benchmark datasets.

Runs each dataset through the full DSPipelineOrchestrator, collects
metrics, and compares results against published SOTA baselines from
the benchmark catalog.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.core.models import Project
from src.datasci.benchmark.catalog import (
    BENCHMARK_CATALOG,
    DatasetEntry,
)
from src.datasci.benchmark.loader import BenchmarkLoader

logger = logging.getLogger("associate.datasci.benchmark.runner")


@dataclass
class DatasetResult:
    """Result of running the pipeline on one dataset."""

    dataset_name: str
    tier: str
    status: str  # "success", "failed", "skipped"
    error: str | None = None
    pipeline_metrics: dict = field(default_factory=dict)
    best_auc: float | None = None
    best_f1: float | None = None
    sota_auc: float | None = None
    sota_f1: float | None = None
    auc_gap: float | None = None
    f1_gap: float | None = None
    phases_completed: int = 0
    elapsed_seconds: float = 0.0


class BenchmarkRunner:
    """Runs the DS pipeline across multiple benchmark datasets.

    For each dataset:
    1. Load via BenchmarkLoader
    2. Create a project in the repository
    3. Execute the pipeline
    4. Extract metrics from pipeline state
    5. Compare against SOTA
    """

    def __init__(
        self,
        pipeline: Any,
        loader: BenchmarkLoader,
        repository: Any,
    ):
        """Initialize runner with pipeline, loader, and repository.

        Args:
            pipeline: DSPipelineOrchestrator instance
                (has .run(input_data) -> DSPipelineState).
            loader: BenchmarkLoader instance
                (has .load(entry) -> Path).
            repository: Repository instance
                (has .create_project(project) -> Project).
        """
        self.pipeline = pipeline
        self.loader = loader
        self.repository = repository

    def run_all(
        self,
        catalog: dict[str, DatasetEntry] | None = None,
        tiers: list[str] | None = None,
        max_datasets: int | None = None,
    ) -> list[DatasetResult]:
        """Run pipeline across all (or filtered) datasets.

        Args:
            catalog: Optional catalog override.
                If None, uses BENCHMARK_CATALOG.
            tiers: Optional tier filter
                (e.g. ["mild", "moderate"]).
            max_datasets: Optional limit on number of
                datasets to run.

        Returns:
            List of DatasetResult, one per dataset executed.
        """
        active_catalog = catalog if catalog is not None else (
            dict(BENCHMARK_CATALOG)
        )

        # Filter by tiers if specified
        if tiers is not None:
            active_catalog = {
                key: entry
                for key, entry in active_catalog.items()
                if entry.imbalance_tier in tiers
            }

        # Sort by key for deterministic ordering
        sorted_entries = sorted(
            active_catalog.values(),
            key=lambda e: e.name,
        )

        # Limit by max_datasets if specified
        if max_datasets is not None and max_datasets > 0:
            sorted_entries = sorted_entries[:max_datasets]

        logger.info(
            "Starting benchmark run: %d datasets "
            "(tiers=%s, max=%s)",
            len(sorted_entries),
            tiers,
            max_datasets,
        )

        results: list[DatasetResult] = []
        for i, entry in enumerate(sorted_entries, start=1):
            logger.info(
                "--- Dataset %d/%d: %s (tier=%s) ---",
                i,
                len(sorted_entries),
                entry.name,
                entry.imbalance_tier,
            )
            result = self.run_single(entry)
            results.append(result)

            logger.info(
                "Dataset %s: status=%s, best_auc=%s, "
                "best_f1=%s, elapsed=%.1fs",
                entry.name,
                result.status,
                result.best_auc,
                result.best_f1,
                result.elapsed_seconds,
            )

        # Summary log
        success_count = sum(
            1 for r in results if r.status == "success"
        )
        failed_count = sum(
            1 for r in results if r.status == "failed"
        )
        skipped_count = sum(
            1 for r in results if r.status == "skipped"
        )
        logger.info(
            "Benchmark run complete: %d success, %d failed, "
            "%d skipped out of %d total",
            success_count,
            failed_count,
            skipped_count,
            len(results),
        )

        return results

    def run_single(
        self,
        entry: DatasetEntry,
    ) -> DatasetResult:
        """Run pipeline on a single dataset entry.

        Args:
            entry: The DatasetEntry to run.

        Returns:
            DatasetResult with status, metrics, and SOTA gaps.
        """
        start_time = time.time()

        # Step 1: Load dataset via BenchmarkLoader
        try:
            csv_path = self.loader.load(entry)
        except Exception as exc:
            elapsed = time.time() - start_time
            logger.warning(
                "Skipping %s: load failed: %s",
                entry.name,
                exc,
            )
            return DatasetResult(
                dataset_name=entry.name,
                tier=entry.imbalance_tier,
                status="skipped",
                error=str(exc),
                elapsed_seconds=round(elapsed, 2),
                sota_auc=entry.sota_auc,
                sota_f1=entry.sota_f1,
            )

        # Step 2: Create a project in the repository
        try:
            project = Project(
                name=f"benchmark-{entry.name}",
                repo_path=str(csv_path.parent),
                tech_stack={
                    "type": "datasci",
                    "problem_type": entry.problem_type,
                },
                project_rules="",
            )
            project = self.repository.create_project(project)
        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error(
                "Failed to create project for %s: %s",
                entry.name,
                exc,
            )
            return DatasetResult(
                dataset_name=entry.name,
                tier=entry.imbalance_tier,
                status="failed",
                error=f"Project creation failed: {exc}",
                elapsed_seconds=round(elapsed, 2),
                sota_auc=entry.sota_auc,
                sota_f1=entry.sota_f1,
            )

        # Step 3: Build input data and run the pipeline
        input_data: dict[str, Any] = {
            "dataset_path": str(csv_path),
            "target_column": entry.target_column,
            "project_id": str(project.id),
            "problem_type": entry.problem_type,
        }

        try:
            state = self.pipeline.run(input_data)
        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error(
                "Pipeline failed for %s: %s",
                entry.name,
                exc,
            )
            return DatasetResult(
                dataset_name=entry.name,
                tier=entry.imbalance_tier,
                status="failed",
                error=f"Pipeline execution failed: {exc}",
                elapsed_seconds=round(elapsed, 2),
                sota_auc=entry.sota_auc,
                sota_f1=entry.sota_f1,
            )

        # Step 4: Extract metrics from pipeline state
        elapsed = time.time() - start_time
        metrics = self._extract_metrics(state)
        phases = self._count_phases(state)

        best_auc = metrics.get("best_auc")
        best_f1 = metrics.get("best_f1")

        # Step 5: Compute gaps vs SOTA
        auc_gap: float | None = None
        if best_auc is not None and entry.sota_auc is not None:
            auc_gap = round(best_auc - entry.sota_auc, 6)

        f1_gap: float | None = None
        if best_f1 is not None and entry.sota_f1 is not None:
            f1_gap = round(best_f1 - entry.sota_f1, 6)

        return DatasetResult(
            dataset_name=entry.name,
            tier=entry.imbalance_tier,
            status="success",
            pipeline_metrics=metrics,
            best_auc=best_auc,
            best_f1=best_f1,
            sota_auc=entry.sota_auc,
            sota_f1=entry.sota_f1,
            auc_gap=auc_gap,
            f1_gap=f1_gap,
            phases_completed=phases,
            elapsed_seconds=round(elapsed, 2),
        )

    def _extract_metrics(self, state: Any) -> dict:
        """Extract key metrics from pipeline state.

        Checks evaluation_report, ensemble_report, and
        training_report for AUC and F1 scores. Uses .get()
        defensively since exact keys may vary.

        Args:
            state: DSPipelineState instance.

        Returns:
            Dict of extracted metric values.
        """
        metrics: dict[str, Any] = {}

        # Try evaluation_report first
        if state.evaluation_report is not None:
            er = state.evaluation_report
            if isinstance(er, dict):
                # Try multiple key variants
                metrics["eval_auc"] = (
                    er.get("auc") or er.get("roc_auc")
                )
                metrics["eval_f1"] = (
                    er.get("f1_macro") or er.get("f1_score")
                )
                # Nested metrics dict
                inner = er.get("metrics", {})
                if inner:
                    metrics["eval_auc"] = inner.get(
                        "auc",
                        inner.get("roc_auc", metrics.get("eval_auc")),
                    )
                    metrics["eval_f1"] = inner.get(
                        "f1_macro",
                        inner.get("f1_score", metrics.get("eval_f1")),
                    )
            else:
                # Pydantic model — use attribute access
                ps = getattr(er, "predictive_scores", {})
                if isinstance(ps, dict):
                    metrics["eval_auc"] = (
                        ps.get("auc") or ps.get("roc_auc")
                    )
                    metrics["eval_f1"] = (
                        ps.get("f1_macro") or ps.get("f1_score")
                    )

        # Try ensemble_report
        if state.ensemble_report is not None:
            ens = state.ensemble_report
            if isinstance(ens, dict):
                metrics["ensemble_auc"] = (
                    ens.get("best_auc")
                    or ens.get("roc_auc")
                )
                metrics["ensemble_f1"] = (
                    ens.get("best_f1")
                    or ens.get("f1_score")
                )
            else:
                # Pydantic model — check pareto_front
                pareto = getattr(ens, "pareto_front", [])
                if pareto and isinstance(pareto, list):
                    for point in pareto:
                        if isinstance(point, dict):
                            if "roc_auc" in point:
                                prev = metrics.get(
                                    "ensemble_auc",
                                )
                                val = point["roc_auc"]
                                if prev is None or (
                                    val is not None
                                    and val > prev
                                ):
                                    metrics["ensemble_auc"] = (
                                        val
                                    )
                            if "f1_score" in point:
                                prev = metrics.get(
                                    "ensemble_f1",
                                )
                                val = point["f1_score"]
                                if prev is None or (
                                    val is not None
                                    and val > prev
                                ):
                                    metrics["ensemble_f1"] = (
                                        val
                                    )

        # Try training_report
        if state.training_report is not None:
            tr = state.training_report
            if isinstance(tr, dict):
                candidates = tr.get("candidates", [])
            else:
                candidates = getattr(tr, "candidates", [])
            for cand in candidates:
                if isinstance(cand, dict):
                    score = cand.get("mean_score")
                else:
                    score = getattr(cand, "mean_score", None)
                if score is not None:
                    key = "training_best_score"
                    prev = metrics.get(key)
                    if prev is None or score > prev:
                        metrics[key] = score

        # Best AUC/F1 = max across all collected values
        auc_vals = [
            v for k, v in metrics.items()
            if "auc" in k and v is not None
        ]
        f1_vals = [
            v for k, v in metrics.items()
            if "f1" in k and v is not None
        ]
        metrics["best_auc"] = (
            max(auc_vals) if auc_vals else None
        )
        metrics["best_f1"] = (
            max(f1_vals) if f1_vals else None
        )

        return metrics

    def _count_phases(self, state: Any) -> int:
        """Count non-None report fields on pipeline state.

        Args:
            state: DSPipelineState instance.

        Returns:
            Number of phases that produced a report.
        """
        report_attrs = [
            "audit_report",
            "eda_report",
            "fe_assessment",
            "feature_report",
            "augmentation_report",
            "training_report",
            "ensemble_report",
            "evaluation_report",
            "deployment_package",
        ]
        return sum(
            1 for attr in report_attrs
            if getattr(state, attr, None) is not None
        )
