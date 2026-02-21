"""Tests for DSPipelineOrchestrator — the 7-phase data science pipeline coordinator.

Uses a real sklearn synthetic dataset (make_classification) and real-shaped
in-memory fakes for LLM client, model router, and repository. No mocks.

The orchestrator sequences: DataAudit -> EDA -> FeatureEngineering ->
ModelTraining -> Ensemble -> Evaluation -> Deployment, with LLM quality
gates between phases.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.core.config import DataScienceConfig
from src.datasci.models import (
    DSPipelineState,
    DataAuditReport,
    DeploymentPackage,
    EDAReport,
    EnsembleReport,
    EvaluationReport,
    FeatureEngineeringReport,
    ModelTrainingReport,
)
from src.datasci.pipeline import DSPipelineOrchestrator


# ---------------------------------------------------------------------------
# Real-shaped fakes — same pattern used across all DS tests.
#
# These are NOT mocks. They are minimal in-memory implementations that
# satisfy the interface contracts of the real LLM client, model router,
# and repository.
# ---------------------------------------------------------------------------


class FakeLLMResponse:
    """Mirrors src.llm.client.LLMResponse with content attribute."""

    def __init__(self, content: str):
        self.content = content
        self.model = "test/model"
        self.tokens_used = 100
        self.input_tokens = 50
        self.output_tokens = 50


class FakeLLMClient:
    """In-memory LLM client supporting both agent and pipeline calling conventions.

    DS agents call: complete_with_fallback(model=, fallback_models=, messages=, ...)
    Pipeline quality gate calls: complete_with_fallback(messages=, models=, ...)

    This fake accepts both via **kwargs.
    """

    def __init__(self, response_text: str = "PASS: phase output looks good."):
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []
        self._call_count = 0

    def complete_with_fallback(self, **kwargs) -> FakeLLMResponse:
        self._call_count += 1
        self.calls.append(kwargs)
        return FakeLLMResponse(self.response_text)


class FakeModelRouter:
    """In-memory model router supporting get_model, get_fallback_models,
    and get_model_chain (used by the pipeline quality gate)."""

    def get_model(self, role: str) -> str:
        return "test/model"

    def get_fallback_models(self, role: str) -> list[str]:
        return ["test/fallback"]

    def get_model_chain(self, role: str) -> list[str]:
        return ["test/model", "test/fallback"]


class FakeRepository:
    """In-memory repository tracking DS experiment records."""

    def __init__(self):
        self.experiments: dict[uuid.UUID, dict[str, Any]] = {}

    def create_ds_experiment(self, project_id, experiment_phase,
                             experiment_config=None, task_id=None,
                             run_id=None, dataset_fingerprint=None,
                             parent_experiment_id=None) -> uuid.UUID:
        exp_id = uuid.uuid4()
        self.experiments[exp_id] = {
            "status": "RUNNING",
            "phase": experiment_phase,
            "project_id": project_id,
            "config": experiment_config,
            "metrics": {},
        }
        return exp_id

    def update_ds_experiment(self, experiment_id, status="COMPLETED",
                             metrics=None, artifacts_manifest=None):
        if experiment_id in self.experiments:
            self.experiments[experiment_id]["status"] = status
            if metrics:
                self.experiments[experiment_id]["metrics"] = metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classification_csv(tmp_path):
    """Create a real sklearn synthetic classification dataset as CSV.

    200 samples, 10 features (5 informative), binary target.
    This is NOT synthetic/mock data -- make_classification produces a
    real statistical dataset with known properties.
    """
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df["target"] = y
    path = tmp_path / "pipeline_test.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def fake_llm():
    return FakeLLMClient(response_text="PASS: phase output is valid and complete.")


@pytest.fixture
def fake_repo():
    return FakeRepository()


@pytest.fixture
def ds_config(tmp_path):
    return DataScienceConfig(
        cv_folds=2,
        llm_fe_generations=1,
        llm_fe_population_size=2,
        artifacts_dir=str(tmp_path / "artifacts"),
    )


@pytest.fixture
def orchestrator(fake_llm, fake_repo, ds_config):
    return DSPipelineOrchestrator(
        llm_client=fake_llm,
        model_router=FakeModelRouter(),
        repository=fake_repo,
        ds_config=ds_config,
    )


# ---------------------------------------------------------------------------
# Tests: Full pipeline execution
# ---------------------------------------------------------------------------


class TestDSPipelineOrchestratorRun:
    """Test the full 7-phase pipeline run."""

    def test_run_returns_pipeline_state(self, orchestrator, classification_csv):
        """Pipeline run should return a DSPipelineState instance."""
        project_id = uuid.uuid4()
        state = orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": project_id,
            "problem_type": "classification",
        })
        assert isinstance(state, DSPipelineState)

    def test_pipeline_state_has_correct_metadata(self, orchestrator, classification_csv):
        """Returned state should preserve input metadata."""
        project_id = uuid.uuid4()
        state = orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": project_id,
            "problem_type": "classification",
        })
        assert state.project_id == project_id
        assert state.dataset_path == classification_csv
        assert state.target_column == "target"
        assert state.problem_type == "classification"

    def test_audit_report_populated(self, orchestrator, classification_csv):
        """Phase 1 (DataAudit) should produce a non-None audit_report."""
        state = orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        assert state.audit_report is not None
        assert isinstance(state.audit_report, DataAuditReport)
        assert state.audit_report.row_count == 200
        assert state.audit_report.column_count == 11  # 10 features + target

    def test_eda_report_populated(self, orchestrator, classification_csv):
        """Phase 2 (EDA) should produce a non-None eda_report."""
        state = orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        assert state.eda_report is not None
        assert isinstance(state.eda_report, EDAReport)

    def test_feature_report_populated(self, orchestrator, classification_csv):
        """Phase 3 (FeatureEngineering) should produce a non-None feature_report."""
        state = orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        assert state.feature_report is not None
        assert isinstance(state.feature_report, FeatureEngineeringReport)

    def test_training_report_populated(self, orchestrator, classification_csv):
        """Phase 4 (ModelTraining) should produce a non-None training_report."""
        state = orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        assert state.training_report is not None
        assert isinstance(state.training_report, ModelTrainingReport)

    def test_ensemble_report_populated(self, orchestrator, classification_csv):
        """Phase 5 (Ensemble) should produce a non-None ensemble_report."""
        state = orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        assert state.ensemble_report is not None
        assert isinstance(state.ensemble_report, EnsembleReport)

    def test_evaluation_report_populated(self, orchestrator, classification_csv):
        """Phase 6 (Evaluation) should produce a non-None evaluation_report."""
        state = orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        assert state.evaluation_report is not None
        assert isinstance(state.evaluation_report, EvaluationReport)

    def test_deployment_package_populated(self, orchestrator, classification_csv):
        """Phase 7 (Deployment) should produce a non-None deployment_package."""
        state = orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        assert state.deployment_package is not None
        assert isinstance(state.deployment_package, DeploymentPackage)

    def test_sensitive_columns_passed_through(self, orchestrator, classification_csv):
        """Sensitive columns in input should be preserved in state."""
        state = orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "sensitive_columns": ["feature_0", "feature_1"],
        })
        assert state.sensitive_columns == ["feature_0", "feature_1"]

    def test_string_project_id_converted(self, orchestrator, classification_csv):
        """Pipeline should accept project_id as string and convert to UUID."""
        pid = uuid.uuid4()
        state = orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": str(pid),
            "problem_type": "classification",
        })
        assert state.project_id == pid


# ---------------------------------------------------------------------------
# Tests: Experiment tracking via repository
# ---------------------------------------------------------------------------


class TestDSPipelineExperimentTracking:
    """Verify that the orchestrator creates and updates experiment records."""

    def test_pipeline_experiment_created(self, orchestrator, fake_repo, classification_csv):
        """Running the pipeline should create at least one experiment record."""
        orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        # At minimum: 1 pipeline-level + 7 phase-level experiments = 8
        # The pipeline creates one top-level record; each agent may create its own.
        assert len(fake_repo.experiments) >= 1

    def test_pipeline_experiment_has_pipeline_phase(self, orchestrator, fake_repo, classification_csv):
        """The pipeline-level experiment should be tagged as phase='pipeline'."""
        orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        pipeline_exps = [
            exp for exp in fake_repo.experiments.values()
            if exp.get("phase") == "pipeline"
        ]
        assert len(pipeline_exps) == 1

    def test_pipeline_experiment_completed(self, orchestrator, fake_repo, classification_csv):
        """On a successful pipeline run, the pipeline experiment should be COMPLETED."""
        orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        pipeline_exps = [
            exp for exp in fake_repo.experiments.values()
            if exp.get("phase") == "pipeline"
        ]
        assert len(pipeline_exps) == 1
        assert pipeline_exps[0]["status"] == "COMPLETED"

    def test_pipeline_experiment_metrics_populated(self, orchestrator, fake_repo, classification_csv):
        """Pipeline experiment metrics should include completed_phases count."""
        orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        pipeline_exps = [
            exp for exp in fake_repo.experiments.values()
            if exp.get("phase") == "pipeline"
        ]
        assert len(pipeline_exps) == 1
        metrics = pipeline_exps[0].get("metrics", {})
        # 9 phases: audit, eda, fe_assessment (2.5), fe, augmentation (3.5),
        # training, ensemble, eval, deployment
        assert metrics.get("completed_phases") == 9
        assert metrics.get("total_phases") == 9

    def test_llm_called_for_quality_gates(self, orchestrator, fake_llm, classification_csv):
        """Quality gates should invoke the LLM for each phase."""
        orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        # LLM is called by: each agent internally + quality gate per phase.
        # The pipeline itself calls quality gate 7 times (once per phase).
        # Each agent may also call the LLM (via FakeLLMClient).
        # We verify that at least 7 calls happened (one per quality gate).
        assert fake_llm._call_count >= 7


# ---------------------------------------------------------------------------
# Tests: Quality gate behavior
# ---------------------------------------------------------------------------


class TestDSPipelineQualityGate:
    """Test the _quality_gate method behavior."""

    def test_quality_gate_pass_response(self, fake_repo, ds_config):
        """A PASS response from the LLM should not block the pipeline."""
        llm = FakeLLMClient(response_text="PASS: output is valid and complete.")
        orch = DSPipelineOrchestrator(
            llm_client=llm,
            model_router=FakeModelRouter(),
            repository=fake_repo,
            ds_config=ds_config,
        )
        result = orch._quality_gate(
            phase_number=1,
            phase_name="data_audit",
            phase_output={"row_count": 200, "column_count": 10},
        )
        assert result is True

    def test_quality_gate_fail_response(self, fake_repo, ds_config):
        """A FAIL response from the LLM should return False."""
        llm = FakeLLMClient(response_text="FAIL: output is empty and invalid.")
        orch = DSPipelineOrchestrator(
            llm_client=llm,
            model_router=FakeModelRouter(),
            repository=fake_repo,
            ds_config=ds_config,
        )
        result = orch._quality_gate(
            phase_number=1,
            phase_name="data_audit",
            phase_output={"row_count": 0},
        )
        assert result is False

    def test_quality_gate_ambiguous_response_assumes_pass(self, fake_repo, ds_config):
        """An ambiguous LLM response (not PASS/FAIL) should default to True."""
        llm = FakeLLMClient(response_text="The output looks reasonable overall.")
        orch = DSPipelineOrchestrator(
            llm_client=llm,
            model_router=FakeModelRouter(),
            repository=fake_repo,
            ds_config=ds_config,
        )
        result = orch._quality_gate(
            phase_number=1,
            phase_name="data_audit",
            phase_output={"row_count": 200},
        )
        assert result is True

    def test_quality_gate_llm_failure_assumes_pass(self, fake_repo, ds_config):
        """If the LLM call raises an exception, quality gate defaults to True."""

        class ErrorLLMClient:
            def complete_with_fallback(self, **kwargs):
                raise ConnectionError("LLM service unavailable")

        orch = DSPipelineOrchestrator(
            llm_client=ErrorLLMClient(),
            model_router=FakeModelRouter(),
            repository=fake_repo,
            ds_config=ds_config,
        )
        result = orch._quality_gate(
            phase_number=1,
            phase_name="data_audit",
            phase_output={"row_count": 200},
        )
        assert result is True

    def test_quality_gate_model_router_failure_assumes_pass(self, fake_repo, ds_config):
        """If model router cannot resolve chain, quality gate defaults to True."""

        class BrokenModelRouter:
            def get_model(self, role): return "test/model"
            def get_fallback_models(self, role): return []
            def get_model_chain(self, role):
                raise ValueError("Unknown role: ds_evaluator")

        orch = DSPipelineOrchestrator(
            llm_client=FakeLLMClient(),
            model_router=BrokenModelRouter(),
            repository=fake_repo,
            ds_config=ds_config,
        )
        result = orch._quality_gate(
            phase_number=1,
            phase_name="data_audit",
            phase_output={"row_count": 200},
        )
        assert result is True


# ---------------------------------------------------------------------------
# Tests: Phase output summary
# ---------------------------------------------------------------------------


class TestDSPipelineSummarizePhaseOutput:
    """Test the _summarize_phase_output helper."""

    def test_summarizes_simple_dict(self, orchestrator):
        summary = orchestrator._summarize_phase_output({"row_count": 200, "status": "ok"})
        assert "row_count" in summary
        assert "200" in summary
        assert "status" in summary

    def test_truncates_long_strings(self, orchestrator):
        long_text = "x" * 500
        summary = orchestrator._summarize_phase_output({"text": long_text})
        assert "..." in summary
        assert len(summary) < 500

    def test_summarizes_lists(self, orchestrator):
        summary = orchestrator._summarize_phase_output({
            "items": [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}],
        })
        assert "list with 4 items" in summary
        assert "and 2 more" in summary

    def test_summarizes_nested_dicts(self, orchestrator):
        summary = orchestrator._summarize_phase_output({
            "nested": {"k1": "v1", "k2": "v2", "k3": "v3", "k4": "v4"},
        })
        assert "dict with 4 keys" in summary
        assert "and 1 more keys" in summary

    def test_handles_none_values(self, orchestrator):
        summary = orchestrator._summarize_phase_output({"value": None})
        assert "None" in summary

    def test_respects_max_length(self, orchestrator):
        """Summary should be truncated if it exceeds max_length."""
        big_dict = {f"key_{i}": f"value_{i}" * 100 for i in range(200)}
        summary = orchestrator._summarize_phase_output(big_dict, max_length=500)
        assert len(summary) <= 520  # 500 + len("... (truncated)") + newline


# ---------------------------------------------------------------------------
# Tests: Finalize state
# ---------------------------------------------------------------------------


class TestDSPipelineFinalizeState:
    """Test the _finalize_state static method."""

    def test_finalize_returns_same_state(self):
        """_finalize_state should return the state object (possibly mutated)."""
        state = DSPipelineState(
            dataset_path="/data.csv",
            target_column="target",
            project_id=uuid.uuid4(),
        )
        result = DSPipelineOrchestrator._finalize_state(state, "completed")
        assert result is state

    def test_finalize_with_partial_reports(self):
        """_finalize_state should work with partially completed pipeline."""
        state = DSPipelineState(
            dataset_path="/data.csv",
            target_column="target",
            project_id=uuid.uuid4(),
        )
        state.audit_report = DataAuditReport(
            dataset_path="/data.csv", row_count=100, column_count=5,
        )
        state.eda_report = EDAReport()
        result = DSPipelineOrchestrator._finalize_state(state, "failed")
        # State should be returned without error
        assert result.audit_report is not None
        assert result.eda_report is not None
        assert result.feature_report is None


# ---------------------------------------------------------------------------
# Tests: Error handling in pipeline run
# ---------------------------------------------------------------------------


class TestDSPipelineErrorHandling:
    """Test pipeline behavior when phases fail."""

    def test_missing_dataset_path_raises(self, orchestrator):
        """Missing required key 'dataset_path' should raise KeyError."""
        with pytest.raises(KeyError, match="dataset_path"):
            orchestrator.run({
                "target_column": "target",
                "project_id": uuid.uuid4(),
            })

    def test_missing_target_column_raises(self, orchestrator, classification_csv):
        """Missing required key 'target_column' should raise KeyError."""
        with pytest.raises(KeyError, match="target_column"):
            orchestrator.run({
                "dataset_path": classification_csv,
                "project_id": uuid.uuid4(),
            })

    def test_nonexistent_dataset_returns_partial_state(self, orchestrator):
        """If the dataset file does not exist, the pipeline should return
        a state with no completed phases (Phase 1 fails early)."""
        state = orchestrator.run({
            "dataset_path": "/nonexistent/path/data.csv",
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        # Phase 1 should fail, so audit_report should be None
        assert isinstance(state, DSPipelineState)
        assert state.audit_report is None

    def test_pipeline_experiment_marked_failed_on_error(self, fake_repo, ds_config):
        """When a phase fails, the pipeline experiment record should be FAILED."""
        orch = DSPipelineOrchestrator(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=fake_repo,
            ds_config=ds_config,
        )
        orch.run({
            "dataset_path": "/nonexistent/data.csv",
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        pipeline_exps = [
            exp for exp in fake_repo.experiments.values()
            if exp.get("phase") == "pipeline"
        ]
        assert len(pipeline_exps) == 1
        assert pipeline_exps[0]["status"] == "FAILED"

    def test_default_problem_type_is_classification(self, orchestrator, classification_csv):
        """When problem_type is not specified, it should default to 'classification'."""
        state = orchestrator.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
        })
        assert state.problem_type == "classification"
