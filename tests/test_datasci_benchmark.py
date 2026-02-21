"""Tests for the benchmark system — catalog, loader, runner, and report.

Uses real-shaped in-memory fakes for the pipeline, loader, and repository.
No mocks. All fakes implement the same interfaces as the real components.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.core.models import Project
from src.datasci.benchmark.catalog import (
    BENCHMARK_CATALOG,
    DatasetEntry,
    get_by_source,
    get_by_tier,
    get_catalog,
)
from src.datasci.benchmark.loader import BenchmarkLoader
from src.datasci.benchmark.report import BenchmarkReport, BenchmarkReporter
from src.datasci.benchmark.runner import BenchmarkRunner, DatasetResult
from src.datasci.models import (
    DataAuditReport,
    DSPipelineState,
    EDAReport,
    EnsembleReport,
    EvaluationReport,
    FeatureEngineeringReport,
    ModelTrainingReport,
)

# ---------------------------------------------------------------------------
# Real-shaped fakes
# ---------------------------------------------------------------------------


class FakePipeline:
    """In-memory pipeline returning deterministic DSPipelineState.

    Implements the same run(input_data) -> DSPipelineState interface
    as DSPipelineOrchestrator. Returns configurable results.
    """

    def __init__(
        self,
        should_fail: bool = False,
        eval_auc: float = 0.85,
        eval_f1: float = 0.80,
    ):
        self.should_fail = should_fail
        self.eval_auc = eval_auc
        self.eval_f1 = eval_f1
        self.run_calls: list[dict] = []

    def run(self, input_data: dict[str, Any]) -> DSPipelineState:
        self.run_calls.append(input_data)

        if self.should_fail:
            raise RuntimeError("Pipeline execution failed (fake)")

        project_id = uuid.UUID(input_data["project_id"])
        state = DSPipelineState(
            dataset_path=input_data["dataset_path"],
            target_column=input_data["target_column"],
            problem_type=input_data.get("problem_type", "classification"),
            project_id=project_id,
            audit_report=DataAuditReport(
                dataset_path=input_data["dataset_path"],
                row_count=100,
                column_count=10,
            ),
            eda_report=EDAReport(
                llm_synthesis="Fake EDA",
            ),
            feature_report=FeatureEngineeringReport(
                original_feature_count=9,
            ),
            training_report=ModelTrainingReport(
                candidates=[
                    {
                        "model_name": "xgb",
                        "model_type": "xgboost",
                        "mean_score": 0.83,
                        "std_score": 0.02,
                    },
                ],
                best_candidate="xgb",
            ),
            ensemble_report=EnsembleReport(
                pareto_front=[
                    {
                        "roc_auc": self.eval_auc,
                        "f1_score": self.eval_f1,
                    },
                ],
            ),
            evaluation_report=EvaluationReport(
                predictive_scores={
                    "roc_auc": self.eval_auc,
                    "f1_score": self.eval_f1,
                    "accuracy": 0.87,
                },
            ),
        )
        return state


class FakeLoader:
    """In-memory loader that writes small CSV files without network.

    Implements the same load(entry) -> Path interface as BenchmarkLoader.
    """

    def __init__(
        self,
        cache_dir: Path,
        should_fail_names: set[str] | None = None,
    ):
        self.cache_dir = cache_dir
        self.should_fail_names = should_fail_names or set()
        self.load_calls: list[str] = []

    def load(self, entry: DatasetEntry) -> Path:
        self.load_calls.append(entry.name)

        if entry.name in self.should_fail_names:
            raise RuntimeError(f"Cannot load {entry.name} (fake)")

        csv_path = self.cache_dir / f"{entry.name}.csv"
        if not csv_path.exists():
            df = pd.DataFrame({
                "feature_0": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_1": [0.1, 0.2, 0.3, 0.4, 0.5],
                entry.target_column: [0, 1, 0, 1, 0],
            })
            df.to_csv(csv_path, index=False)
        return csv_path

    def is_cached(self, entry: DatasetEntry) -> bool:
        return (self.cache_dir / f"{entry.name}.csv").exists()

    def clear_cache(self) -> None:
        pass

    def cache_status(self) -> dict[str, bool]:
        return {}


class FakeRepository:
    """In-memory repository for benchmark tests."""

    def __init__(self, should_fail: bool = False):
        self.projects: dict[uuid.UUID, Project] = {}
        self.should_fail = should_fail

    def create_project(self, project: Project) -> Project:
        if self.should_fail:
            raise RuntimeError("Repository unavailable (fake)")
        self.projects[project.id] = project
        return project

    def create_ds_experiment(self, **kwargs) -> uuid.UUID:
        return uuid.uuid4()

    def update_ds_experiment(self, **kwargs) -> None:
        pass


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_test_catalog() -> dict[str, DatasetEntry]:
    """Small test catalog with one entry per tier."""
    return {
        "test_mild": DatasetEntry(
            name="Test Mild",
            source="sklearn",
            source_id="breast_cancer",
            target_column="target",
            imbalance_tier="mild",
            approximate_ir=1.7,
            sota_auc=0.99,
            sota_f1=0.97,
            n_rows=569,
            n_features=30,
        ),
        "test_moderate": DatasetEntry(
            name="Test Moderate",
            source="openml",
            source_id=38,
            target_column="class",
            imbalance_tier="moderate",
            approximate_ir=7.4,
            sota_auc=0.95,
            n_rows=3772,
            n_features=29,
        ),
        "test_severe": DatasetEntry(
            name="Test Severe",
            source="imblearn",
            source_id="abalone_19",
            target_column="target",
            imbalance_tier="severe",
            approximate_ir=16.4,
            sota_f1=0.12,
            n_rows=4177,
            n_features=8,
        ),
        "test_extreme": DatasetEntry(
            name="Test Extreme",
            source="openml",
            source_id=1597,
            target_column="Class",
            imbalance_tier="extreme",
            approximate_ir=577.0,
            sota_auc=0.98,
            n_rows=284807,
            n_features=30,
        ),
    }


# ===================================================================
# CATALOG TESTS
# ===================================================================


class TestCatalog:
    """Tests for the benchmark catalog."""

    def test_catalog_has_entries(self):
        catalog = get_catalog()
        assert len(catalog) >= 20
        assert len(catalog) == len(BENCHMARK_CATALOG)

    def test_all_tiers_represented(self):
        tiers = {e.imbalance_tier for e in BENCHMARK_CATALOG.values()}
        assert "mild" in tiers
        assert "moderate" in tiers
        assert "severe" in tiers
        assert "extreme" in tiers

    def test_get_by_tier_mild(self):
        mild = get_by_tier("mild")
        assert len(mild) >= 3
        assert all(e.imbalance_tier == "mild" for e in mild)

    def test_get_by_tier_nonexistent(self):
        result = get_by_tier("nonexistent")
        assert result == []

    def test_get_by_source_sklearn(self):
        sklearn_entries = get_by_source("sklearn")
        assert len(sklearn_entries) >= 1
        assert all(e.source == "sklearn" for e in sklearn_entries)

    def test_get_by_source_openml(self):
        openml_entries = get_by_source("openml")
        assert len(openml_entries) >= 10
        assert all(e.source == "openml" for e in openml_entries)

    def test_get_by_source_imblearn(self):
        imblearn_entries = get_by_source("imblearn")
        assert len(imblearn_entries) >= 3
        assert all(e.source == "imblearn" for e in imblearn_entries)

    def test_all_entries_have_required_fields(self):
        for key, entry in BENCHMARK_CATALOG.items():
            assert entry.name, f"{key}: name is empty"
            assert entry.source in (
                "sklearn", "openml", "imblearn"
            ), f"{key}: invalid source={entry.source}"
            assert entry.source_id, f"{key}: source_id is empty"
            assert entry.target_column, f"{key}: target is empty"
            assert entry.problem_type == "classification", (
                f"{key}: unexpected problem_type"
            )
            assert entry.approximate_ir >= 1.0, (
                f"{key}: IR should be >= 1.0"
            )
            assert entry.n_rows > 0, f"{key}: n_rows should be > 0"
            assert entry.n_features > 0, (
                f"{key}: n_features should be > 0"
            )

    def test_no_duplicate_names(self):
        names = [e.name for e in BENCHMARK_CATALOG.values()]
        assert len(names) == len(set(names))

    def test_dataset_entry_is_frozen(self):
        entry = list(BENCHMARK_CATALOG.values())[0]
        with pytest.raises(AttributeError):
            entry.name = "modified"  # type: ignore[misc]


# ===================================================================
# LOADER TESTS
# ===================================================================


class TestLoader:
    """Tests for BenchmarkLoader."""

    def test_loader_creates_cache_dir(self, tmp_path):
        cache_dir = tmp_path / "test_cache"
        BenchmarkLoader(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_load_sklearn_breast_cancer(self, tmp_path):
        loader = BenchmarkLoader(cache_dir=tmp_path)
        entry = BENCHMARK_CATALOG["breast_cancer_wisconsin"]
        csv_path = loader.load(entry)
        assert csv_path.exists()
        assert csv_path.suffix == ".csv"
        df = pd.read_csv(csv_path)
        assert len(df) == 569
        assert entry.target_column in df.columns

    def test_cache_hit(self, tmp_path):
        loader = BenchmarkLoader(cache_dir=tmp_path)
        entry = BENCHMARK_CATALOG["breast_cancer_wisconsin"]
        path1 = loader.load(entry)
        path2 = loader.load(entry)
        assert path1 == path2

    def test_is_cached_false(self, tmp_path):
        loader = BenchmarkLoader(cache_dir=tmp_path)
        entry = BENCHMARK_CATALOG["breast_cancer_wisconsin"]
        assert not loader.is_cached(entry)

    def test_is_cached_true(self, tmp_path):
        loader = BenchmarkLoader(cache_dir=tmp_path)
        entry = BENCHMARK_CATALOG["breast_cancer_wisconsin"]
        loader.load(entry)
        assert loader.is_cached(entry)

    def test_cache_status(self, tmp_path):
        loader = BenchmarkLoader(cache_dir=tmp_path)
        entry = BENCHMARK_CATALOG["breast_cancer_wisconsin"]
        loader.load(entry)
        status = loader.cache_status()
        assert entry.name in status

    def test_clear_cache(self, tmp_path):
        loader = BenchmarkLoader(cache_dir=tmp_path)
        entry = BENCHMARK_CATALOG["breast_cancer_wisconsin"]
        loader.load(entry)
        assert loader.is_cached(entry)
        loader.clear_cache()
        assert not loader.is_cached(entry)

    def test_unknown_source_raises_valueerror(self, tmp_path):
        loader = BenchmarkLoader(cache_dir=tmp_path)
        entry = DatasetEntry(
            name="bad_source",
            source="unknown",
            source_id="x",
            target_column="y",
            n_rows=1,
            n_features=1,
        )
        with pytest.raises(ValueError, match="Unknown source"):
            loader.load(entry)

    def test_unknown_sklearn_raises_runtimeerror(self, tmp_path):
        loader = BenchmarkLoader(cache_dir=tmp_path)
        entry = DatasetEntry(
            name="bad_sklearn",
            source="sklearn",
            source_id="nonexistent_dataset",
            target_column="y",
            n_rows=1,
            n_features=1,
        )
        with pytest.raises(RuntimeError, match="No sklearn loader"):
            loader.load(entry)


# ===================================================================
# RUNNER TESTS
# ===================================================================


class TestRunner:
    """Tests for BenchmarkRunner."""

    def test_run_single_success(self, tmp_path):
        fake_loader = FakeLoader(cache_dir=tmp_path)
        fake_pipeline = FakePipeline(eval_auc=0.90, eval_f1=0.85)
        fake_repo = FakeRepository()
        runner = BenchmarkRunner(
            pipeline=fake_pipeline,
            loader=fake_loader,
            repository=fake_repo,
        )

        catalog = _make_test_catalog()
        entry = catalog["test_mild"]
        result = runner.run_single(entry)

        assert result.status == "success"
        assert result.dataset_name == "Test Mild"
        assert result.tier == "mild"
        assert result.best_auc == 0.90
        assert result.best_f1 == 0.85
        assert result.sota_auc == 0.99
        assert result.auc_gap is not None
        assert result.auc_gap < 0  # below SOTA
        assert result.phases_completed >= 5
        assert result.elapsed_seconds >= 0

    def test_run_single_load_failure_skips(self, tmp_path):
        fake_loader = FakeLoader(
            cache_dir=tmp_path,
            should_fail_names={"Test Mild"},
        )
        fake_pipeline = FakePipeline()
        fake_repo = FakeRepository()
        runner = BenchmarkRunner(
            pipeline=fake_pipeline,
            loader=fake_loader,
            repository=fake_repo,
        )

        catalog = _make_test_catalog()
        entry = catalog["test_mild"]
        result = runner.run_single(entry)

        assert result.status == "skipped"
        assert result.error is not None
        assert "Cannot load" in result.error

    def test_run_single_pipeline_failure(self, tmp_path):
        fake_loader = FakeLoader(cache_dir=tmp_path)
        fake_pipeline = FakePipeline(should_fail=True)
        fake_repo = FakeRepository()
        runner = BenchmarkRunner(
            pipeline=fake_pipeline,
            loader=fake_loader,
            repository=fake_repo,
        )

        catalog = _make_test_catalog()
        entry = catalog["test_mild"]
        result = runner.run_single(entry)

        assert result.status == "failed"
        assert result.error is not None
        assert "Pipeline execution failed" in result.error

    def test_run_single_repo_failure(self, tmp_path):
        fake_loader = FakeLoader(cache_dir=tmp_path)
        fake_pipeline = FakePipeline()
        fake_repo = FakeRepository(should_fail=True)
        runner = BenchmarkRunner(
            pipeline=fake_pipeline,
            loader=fake_loader,
            repository=fake_repo,
        )

        catalog = _make_test_catalog()
        entry = catalog["test_mild"]
        result = runner.run_single(entry)

        assert result.status == "failed"
        assert "Project creation failed" in result.error

    def test_run_all_with_catalog(self, tmp_path):
        fake_loader = FakeLoader(cache_dir=tmp_path)
        fake_pipeline = FakePipeline()
        fake_repo = FakeRepository()
        runner = BenchmarkRunner(
            pipeline=fake_pipeline,
            loader=fake_loader,
            repository=fake_repo,
        )

        catalog = _make_test_catalog()
        results = runner.run_all(catalog=catalog)

        assert len(results) == 4
        assert all(r.status == "success" for r in results)

    def test_run_all_with_tier_filter(self, tmp_path):
        fake_loader = FakeLoader(cache_dir=tmp_path)
        fake_pipeline = FakePipeline()
        fake_repo = FakeRepository()
        runner = BenchmarkRunner(
            pipeline=fake_pipeline,
            loader=fake_loader,
            repository=fake_repo,
        )

        catalog = _make_test_catalog()
        results = runner.run_all(
            catalog=catalog, tiers=["mild"],
        )

        assert len(results) == 1
        assert results[0].tier == "mild"

    def test_run_all_with_max_datasets(self, tmp_path):
        fake_loader = FakeLoader(cache_dir=tmp_path)
        fake_pipeline = FakePipeline()
        fake_repo = FakeRepository()
        runner = BenchmarkRunner(
            pipeline=fake_pipeline,
            loader=fake_loader,
            repository=fake_repo,
        )

        catalog = _make_test_catalog()
        results = runner.run_all(catalog=catalog, max_datasets=2)

        assert len(results) == 2

    def test_run_all_mixed_success_failure(self, tmp_path):
        fake_loader = FakeLoader(
            cache_dir=tmp_path,
            should_fail_names={"Test Severe"},
        )
        fake_pipeline = FakePipeline()
        fake_repo = FakeRepository()
        runner = BenchmarkRunner(
            pipeline=fake_pipeline,
            loader=fake_loader,
            repository=fake_repo,
        )

        catalog = _make_test_catalog()
        results = runner.run_all(catalog=catalog)

        assert len(results) == 4
        statuses = {r.dataset_name: r.status for r in results}
        assert statuses["Test Severe"] == "skipped"
        success_count = sum(
            1 for r in results if r.status == "success"
        )
        assert success_count == 3

    def test_run_all_creates_projects(self, tmp_path):
        fake_loader = FakeLoader(cache_dir=tmp_path)
        fake_pipeline = FakePipeline()
        fake_repo = FakeRepository()
        runner = BenchmarkRunner(
            pipeline=fake_pipeline,
            loader=fake_loader,
            repository=fake_repo,
        )

        catalog = _make_test_catalog()
        runner.run_all(catalog=catalog)

        assert len(fake_repo.projects) == 4

    def test_gap_computation_positive(self, tmp_path):
        """When pipeline beats SOTA, gap should be positive."""
        fake_loader = FakeLoader(cache_dir=tmp_path)
        fake_pipeline = FakePipeline(eval_auc=1.0, eval_f1=1.0)
        fake_repo = FakeRepository()
        runner = BenchmarkRunner(
            pipeline=fake_pipeline,
            loader=fake_loader,
            repository=fake_repo,
        )

        entry = DatasetEntry(
            name="Test Beat SOTA",
            source="sklearn",
            source_id="breast_cancer",
            target_column="target",
            imbalance_tier="mild",
            approximate_ir=1.7,
            sota_auc=0.90,
            sota_f1=0.85,
            n_rows=569,
            n_features=30,
        )
        result = runner.run_single(entry)

        assert result.auc_gap is not None
        assert result.auc_gap > 0
        assert result.f1_gap is not None
        assert result.f1_gap > 0

    def test_gap_none_when_no_sota(self, tmp_path):
        """When SOTA is None, gap should be None."""
        fake_loader = FakeLoader(cache_dir=tmp_path)
        fake_pipeline = FakePipeline()
        fake_repo = FakeRepository()
        runner = BenchmarkRunner(
            pipeline=fake_pipeline,
            loader=fake_loader,
            repository=fake_repo,
        )

        entry = DatasetEntry(
            name="Test No SOTA",
            source="sklearn",
            source_id="breast_cancer",
            target_column="target",
            imbalance_tier="mild",
            approximate_ir=1.7,
            n_rows=569,
            n_features=30,
        )
        result = runner.run_single(entry)

        assert result.auc_gap is None


# ===================================================================
# REPORT TESTS
# ===================================================================


class TestReport:
    """Tests for BenchmarkReporter."""

    def _make_results(self) -> list[DatasetResult]:
        return [
            DatasetResult(
                dataset_name="DS Mild",
                tier="mild",
                status="success",
                best_auc=0.92,
                best_f1=0.88,
                sota_auc=0.99,
                sota_f1=0.97,
                auc_gap=-0.07,
                f1_gap=-0.09,
                phases_completed=7,
                elapsed_seconds=12.5,
            ),
            DatasetResult(
                dataset_name="DS Moderate",
                tier="moderate",
                status="success",
                best_auc=0.88,
                best_f1=0.80,
                sota_auc=0.95,
                auc_gap=-0.07,
                phases_completed=7,
                elapsed_seconds=25.3,
            ),
            DatasetResult(
                dataset_name="DS Severe",
                tier="severe",
                status="failed",
                error="Pipeline crashed",
                elapsed_seconds=2.1,
            ),
            DatasetResult(
                dataset_name="DS Extreme",
                tier="extreme",
                status="skipped",
                error="Download failed",
                elapsed_seconds=0.5,
            ),
        ]

    def test_compile_counts(self):
        reporter = BenchmarkReporter()
        report = reporter.compile(self._make_results())

        assert report.total_datasets == 4
        assert report.successful == 2
        assert report.failed == 1
        assert report.skipped == 1

    def test_compile_tier_grouping(self):
        reporter = BenchmarkReporter()
        report = reporter.compile(self._make_results())

        assert "mild" in report.by_tier
        assert "moderate" in report.by_tier
        assert "severe" in report.by_tier
        assert "extreme" in report.by_tier
        assert len(report.by_tier["mild"]) == 1

    def test_compile_avg_gaps(self):
        reporter = BenchmarkReporter()
        report = reporter.compile(self._make_results())

        assert report.avg_auc_gap is not None
        # Both successful results have auc_gap = -0.07
        assert abs(report.avg_auc_gap - (-0.07)) < 0.001

    def test_compile_best_worst(self):
        reporter = BenchmarkReporter()
        results = [
            DatasetResult(
                dataset_name="Good",
                tier="mild",
                status="success",
                auc_gap=0.05,
            ),
            DatasetResult(
                dataset_name="Bad",
                tier="severe",
                status="success",
                auc_gap=-0.15,
            ),
        ]
        report = reporter.compile(results)

        assert report.best_relative_performance == "Good"
        assert report.worst_relative_performance == "Bad"

    def test_to_json(self):
        reporter = BenchmarkReporter()
        report = reporter.compile(self._make_results())
        json_str = reporter.to_json(report)

        data = json.loads(json_str)
        assert data["total_datasets"] == 4
        assert data["successful"] == 2
        assert isinstance(data["results"], list)

    def test_to_table_has_header(self):
        reporter = BenchmarkReporter()
        report = reporter.compile(self._make_results())
        table = reporter.to_table(report)

        assert "Dataset" in table
        assert "Tier" in table
        assert "Status" in table
        assert "AUC" in table
        assert "Summary" in table

    def test_to_table_contains_dataset_names(self):
        reporter = BenchmarkReporter()
        report = reporter.compile(self._make_results())
        table = reporter.to_table(report)

        assert "DS Mild" in table
        assert "DS Moderate" in table

    def test_to_table_summary_line(self):
        reporter = BenchmarkReporter()
        report = reporter.compile(self._make_results())
        table = reporter.to_table(report)

        assert "2/4 datasets successful" in table
        assert "1 failed" in table
        assert "1 skipped" in table

    def test_to_table_gap_formatting(self):
        reporter = BenchmarkReporter()
        results = [
            DatasetResult(
                dataset_name="Above",
                tier="mild",
                status="success",
                best_auc=1.0,
                sota_auc=0.95,
                auc_gap=0.05,
            ),
        ]
        report = reporter.compile(results)
        table = reporter.to_table(report)

        assert "+0.050" in table


# ===================================================================
# EDGE CASE TESTS
# ===================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_results_report(self):
        reporter = BenchmarkReporter()
        report = reporter.compile([])

        assert report.total_datasets == 0
        assert report.successful == 0
        assert report.avg_auc_gap is None
        assert report.best_relative_performance == ""

    def test_all_failed_report(self):
        reporter = BenchmarkReporter()
        results = [
            DatasetResult(
                dataset_name=f"fail_{i}",
                tier="mild",
                status="failed",
                error="boom",
            )
            for i in range(3)
        ]
        report = reporter.compile(results)

        assert report.successful == 0
        assert report.failed == 3
        assert report.avg_auc_gap is None

    def test_run_all_empty_catalog(self, tmp_path):
        fake_loader = FakeLoader(cache_dir=tmp_path)
        fake_pipeline = FakePipeline()
        fake_repo = FakeRepository()
        runner = BenchmarkRunner(
            pipeline=fake_pipeline,
            loader=fake_loader,
            repository=fake_repo,
        )

        results = runner.run_all(catalog={})
        assert results == []

    def test_dataset_result_defaults(self):
        result = DatasetResult(
            dataset_name="test",
            tier="mild",
            status="success",
        )
        assert result.error is None
        assert result.pipeline_metrics == {}
        assert result.best_auc is None
        assert result.elapsed_seconds == 0.0

    def test_benchmark_report_defaults(self):
        report = BenchmarkReport(results=[])
        assert report.total_datasets == 0
        assert report.by_tier == {}
        assert report.avg_auc_gap is None

    def test_to_table_empty_report(self):
        reporter = BenchmarkReporter()
        report = reporter.compile([])
        table = reporter.to_table(report)

        assert "Summary" in table
        assert "0/0 datasets successful" in table

    def test_to_json_empty_report(self):
        reporter = BenchmarkReporter()
        report = reporter.compile([])
        json_str = reporter.to_json(report)
        data = json.loads(json_str)
        assert data["total_datasets"] == 0
        assert data["results"] == []

    def test_format_time_short(self):
        reporter = BenchmarkReporter()
        assert reporter._fmt_time(45.3) == "45.3s"

    def test_format_time_long(self):
        reporter = BenchmarkReporter()
        assert reporter._fmt_time(125.0) == "2m 5s"

    def test_format_metric_none(self):
        reporter = BenchmarkReporter()
        assert reporter._fmt_metric(None) == "N/A"

    def test_format_metric_value(self):
        reporter = BenchmarkReporter()
        assert reporter._fmt_metric(0.95123) == "0.951"

    def test_format_gap_positive(self):
        reporter = BenchmarkReporter()
        assert reporter._fmt_gap(0.05) == "+0.050"

    def test_format_gap_negative(self):
        reporter = BenchmarkReporter()
        assert reporter._fmt_gap(-0.03) == "-0.030"

    def test_format_gap_none(self):
        reporter = BenchmarkReporter()
        assert reporter._fmt_gap(None) == "N/A"
