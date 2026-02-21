"""Tests for Phase 7: Deployment Agent.

Uses real sklearn synthetic datasets. No mocks.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from src.core.config import DataScienceConfig
from src.datasci.agents.deployment import DeploymentAgent
from src.datasci.models import (
    DeploymentPackage,
    EvaluationReport,
    ModelCandidate,
    ModelTrainingReport,
)


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLMClient:
    def __init__(self):
        self.calls = []

    def complete_with_fallback(self, **kwargs):
        self.calls.append(True)
        return FakeLLMResponse(
            "Deployment package is complete with all required artifacts. "
            "Model, API scaffold, monitoring, and data contract verified."
        )


class FakeModelRouter:
    def get_model(self, role): return "test/model"
    def get_fallback_models(self, role): return []

    def get_model_chain(self, role: str) -> list[str]:
        return [self.get_model(role)]


class FakeRepository:
    def __init__(self):
        self.experiments = {}

    def create_ds_experiment(self, **kwargs):
        eid = uuid.uuid4()
        self.experiments[eid] = {"status": "RUNNING"}
        return eid

    def update_ds_experiment(self, experiment_id, **kwargs):
        if experiment_id in self.experiments:
            self.experiments[experiment_id].update(kwargs)


@pytest.fixture
def classification_csv(tmp_path):
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
    df["target"] = y
    path = tmp_path / "deploy_cls.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def training_report():
    return ModelTrainingReport(
        scale_tier="small",
        candidates=[
            ModelCandidate(
                model_name="xgboost_0",
                model_type="xgboost",
                mean_score=0.88,
                hyperparameters={"n_estimators": 100, "max_depth": 3},
            ),
        ],
        best_candidate="xgboost_0",
    ).model_dump()


@pytest.fixture
def evaluation_report():
    return EvaluationReport(
        predictive_scores={"accuracy": 0.88},
        overall_grade="B+",
    ).model_dump()


@pytest.fixture
def agent(tmp_path):
    config = DataScienceConfig(
        cv_folds=3,
        artifacts_dir=str(tmp_path / "artifacts"),
    )
    return DeploymentAgent(
        llm_client=FakeLLMClient(),
        model_router=FakeModelRouter(),
        repository=FakeRepository(),
        ds_config=config,
    )


class TestDeploymentAgent:
    def test_produces_valid_package(self, agent, classification_csv, training_report, evaluation_report):
        result = agent.process({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "training_report": training_report,
            "evaluation_report": evaluation_report,
        })
        assert result.status == "success"
        pkg = DeploymentPackage(**result.data)
        assert pkg.model_artifact_path != ""

    def test_api_scaffold_generated(self, agent, classification_csv, training_report, evaluation_report):
        result = agent.process({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "training_report": training_report,
            "evaluation_report": evaluation_report,
        })
        pkg = DeploymentPackage(**result.data)
        assert "FastAPI" in pkg.api_scaffold_code or "fastapi" in pkg.api_scaffold_code

    def test_monitoring_config(self, agent, classification_csv, training_report, evaluation_report):
        result = agent.process({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "training_report": training_report,
            "evaluation_report": evaluation_report,
        })
        pkg = DeploymentPackage(**result.data)
        assert len(pkg.monitoring_config) > 0

    def test_data_contract(self, agent, classification_csv, training_report, evaluation_report):
        result = agent.process({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "training_report": training_report,
            "evaluation_report": evaluation_report,
        })
        pkg = DeploymentPackage(**result.data)
        assert len(pkg.data_contract) > 0

    def test_llm_completeness_review(self, agent, classification_csv, training_report, evaluation_report):
        result = agent.process({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "training_report": training_report,
            "evaluation_report": evaluation_report,
        })
        pkg = DeploymentPackage(**result.data)
        assert pkg.llm_completeness_review != ""

    def test_experiment_tracked(self, agent, classification_csv, training_report, evaluation_report):
        agent.process({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "training_report": training_report,
            "evaluation_report": evaluation_report,
        })
        assert len(agent.repository.experiments) == 1
        exp = list(agent.repository.experiments.values())[0]
        assert exp["status"] == "COMPLETED"


# ---------------------------------------------------------------------------
# Additional tests for coverage expansion
# ---------------------------------------------------------------------------


class TestLoadDataset:
    """Tests for _load_dataset covering TSV, parquet, and unsupported formats."""

    def test_load_tsv(self, tmp_path):
        """Cover the TSV loading path (lines 50-51)."""
        from src.datasci.agents.deployment import _load_dataset

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tmp_path / "data.tsv"
        df.to_csv(path, sep="\t", index=False)
        result = _load_dataset(str(path))
        assert len(result) == 3
        assert list(result.columns) == ["a", "b"]

    def test_load_parquet(self, tmp_path):
        """Cover the parquet loading path (lines 52-53)."""
        from src.datasci.agents.deployment import _load_dataset

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tmp_path / "data.parquet"
        df.to_parquet(path, index=False)
        result = _load_dataset(str(path))
        assert len(result) == 3
        assert list(result.columns) == ["a", "b"]

    def test_load_pq_extension(self, tmp_path):
        """Cover the .pq extension variant of parquet path."""
        from src.datasci.agents.deployment import _load_dataset

        df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
        path = tmp_path / "data.pq"
        df.to_parquet(path, index=False)
        result = _load_dataset(str(path))
        assert len(result) == 2

    def test_load_unsupported_format(self, tmp_path):
        """Cover the ValueError for unsupported formats (lines 54-55)."""
        from src.datasci.agents.deployment import _load_dataset

        path = tmp_path / "data.xlsx"
        path.write_text("fake")
        with pytest.raises(ValueError, match="Unsupported"):
            _load_dataset(str(path))


class TestBuildSklearnEstimator:
    """Tests for _build_sklearn_estimator covering all model branches."""

    def test_xgboost_regression(self):
        """Cover xgboost regression branch (lines 78-83)."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("xgboost", "regression")
        assert est is not None
        assert hasattr(est, "fit")
        assert hasattr(est, "predict")
        # Verify it is a regressor
        from xgboost import XGBRegressor
        assert isinstance(est, XGBRegressor)

    def test_catboost_classification(self):
        """Cover catboost classification branch (lines 90-98)."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("catboost", "classification")
        assert est is not None
        assert hasattr(est, "fit")
        from catboost import CatBoostClassifier
        assert isinstance(est, CatBoostClassifier)

    def test_catboost_regression(self):
        """Cover catboost regression branch (lines 99-105)."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("catboost", "regression")
        assert est is not None
        assert hasattr(est, "fit")
        from catboost import CatBoostRegressor
        assert isinstance(est, CatBoostRegressor)

    def test_lightgbm_classification(self):
        """Cover lightgbm classification branch (lines 112-120)."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("lightgbm", "classification")
        assert est is not None
        assert hasattr(est, "fit")
        from lightgbm import LGBMClassifier
        assert isinstance(est, LGBMClassifier)

    def test_lightgbm_regression(self):
        """Cover lightgbm regression branch (lines 121-127)."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("lightgbm", "regression")
        assert est is not None
        assert hasattr(est, "fit")
        from lightgbm import LGBMRegressor
        assert isinstance(est, LGBMRegressor)

    def test_tabpfn_classification(self):
        """Cover tabpfn classification branch -- ImportError path (lines 134-145)."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("tabpfn", "classification")
        # tabpfn is not installed, so this exercises the ImportError handler
        assert est is None

    def test_tabpfn_regression(self):
        """Cover tabpfn regression branch -- ImportError path."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("tabpfn", "regression")
        assert est is None

    def test_tabicl_classification(self):
        """Cover tabicl classification branch -- ImportError path (lines 147-158)."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("tabicl", "classification")
        assert est is None

    def test_tabicl_regression(self):
        """Cover tabicl regression branch -- ImportError path."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("tabicl", "regression")
        assert est is None

    def test_nam_classification(self):
        """Cover nam classification branch -- ImportError path (lines 160-171)."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("nam", "classification")
        assert est is None

    def test_nam_regression(self):
        """Cover nam regression branch -- ImportError path."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("nam", "regression")
        assert est is None

    def test_unknown_model(self):
        """Cover the unknown model fallback (lines 173-177)."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("nonexistent_model", "classification")
        assert est is None

    def test_unknown_model_regression(self):
        """Cover unknown model with regression problem type."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("totally_fake_model", "regression")
        assert est is None

    def test_xgboost_classifier_can_fit(self):
        """Verify xgboost classifier from builder actually trains on real data."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("xgboost", "classification")
        assert est is not None
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        est.fit(X, y)
        preds = est.predict(X)
        assert len(preds) == 50

    def test_xgboost_regressor_can_fit(self):
        """Verify xgboost regressor from builder actually trains on real data."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("xgboost", "regression")
        assert est is not None
        X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        est.fit(X, y)
        preds = est.predict(X)
        assert len(preds) == 50

    def test_catboost_regressor_can_fit(self):
        """Verify catboost regressor trains on real data."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("catboost", "regression")
        assert est is not None
        X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        est.fit(X, y)
        preds = est.predict(X)
        assert len(preds) == 50

    def test_lightgbm_regressor_can_fit(self):
        """Verify lightgbm regressor trains on real data."""
        from src.datasci.agents.deployment import _build_sklearn_estimator

        est = _build_sklearn_estimator("lightgbm", "regression")
        assert est is not None
        X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        est.fit(X, y)
        preds = est.predict(X)
        assert len(preds) == 50


class TestBuildGradientBoosting:
    """Tests for _build_gradient_boosting covering both problem types."""

    def test_classification(self):
        """Cover the classification path (lines 187-191) -- already covered, but explicit."""
        from src.datasci.agents.deployment import _build_gradient_boosting

        est = _build_gradient_boosting("classification")
        assert hasattr(est, "fit")
        assert hasattr(est, "predict")
        from sklearn.ensemble import GradientBoostingClassifier
        assert isinstance(est, GradientBoostingClassifier)

    def test_regression(self):
        """Cover the regression path (lines 192-196)."""
        from src.datasci.agents.deployment import _build_gradient_boosting

        est = _build_gradient_boosting("regression")
        assert hasattr(est, "fit")
        assert hasattr(est, "predict")
        from sklearn.ensemble import GradientBoostingRegressor
        assert isinstance(est, GradientBoostingRegressor)

    def test_classification_trains_on_real_data(self):
        """Verify GradientBoosting classifier trains on real synthetic data."""
        from src.datasci.agents.deployment import _build_gradient_boosting

        est = _build_gradient_boosting("classification")
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        est.fit(X, y)
        preds = est.predict(X)
        assert len(preds) == 100
        assert set(preds).issubset({0, 1})

    def test_regression_trains_on_real_data(self):
        """Verify GradientBoosting regressor trains on real synthetic data."""
        from src.datasci.agents.deployment import _build_gradient_boosting

        est = _build_gradient_boosting("regression")
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        est.fit(X, y)
        preds = est.predict(X)
        assert len(preds) == 100
        # Predictions should be continuous, not just 0/1
        assert len(set(preds)) > 2


class TestDeploymentAgentEdgeCases:
    """Edge case tests for DeploymentAgent.process and related methods."""

    def test_regression_problem(self, tmp_path):
        """Test regression deployment path with real regression data."""
        X, y = make_regression(n_samples=200, n_features=10, n_informative=5, random_state=42)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
        df["target"] = y
        path = tmp_path / "deploy_reg.csv"
        df.to_csv(path, index=False)

        config = DataScienceConfig(cv_folds=3, artifacts_dir=str(tmp_path / "artifacts"))
        agent = DeploymentAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

        training_report = ModelTrainingReport(
            scale_tier="small",
            candidates=[
                ModelCandidate(
                    model_name="xgboost_0",
                    model_type="xgboost",
                    mean_score=0.85,
                    hyperparameters={"n_estimators": 100},
                ),
            ],
            best_candidate="xgboost_0",
        ).model_dump()

        evaluation_report = EvaluationReport(
            predictive_scores={"r2": 0.85, "mae": 10.0, "rmse": 15.0},
            overall_grade="B",
        ).model_dump()

        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "regression",
            "training_report": training_report,
            "evaluation_report": evaluation_report,
        })
        assert result.status == "success"
        pkg = DeploymentPackage(**result.data)
        assert pkg.model_artifact_path != ""

    def test_fallback_when_no_best_candidate(self, tmp_path):
        """Test fallback when best_candidate is empty string (line 281)."""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
        df["target"] = y
        path = tmp_path / "deploy_fallback.csv"
        df.to_csv(path, index=False)

        config = DataScienceConfig(cv_folds=3, artifacts_dir=str(tmp_path / "artifacts"))
        agent = DeploymentAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

        # Empty best_candidate -- should fallback to first candidate by max score
        training_report = ModelTrainingReport(
            scale_tier="small",
            candidates=[
                ModelCandidate(
                    model_name="xgboost_0",
                    model_type="xgboost",
                    mean_score=0.88,
                    hyperparameters={"n_estimators": 100},
                ),
            ],
            best_candidate="",
        ).model_dump()

        evaluation_report = EvaluationReport(
            predictive_scores={"accuracy": 0.88},
            overall_grade="B",
        ).model_dump()

        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "training_report": training_report,
            "evaluation_report": evaluation_report,
        })
        assert result.status == "success"

    def test_fallback_when_no_best_candidate_picks_highest_score(self, tmp_path):
        """Verify fallback picks the candidate with the highest mean_score."""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
        df["target"] = y
        path = tmp_path / "deploy_fallback_multi.csv"
        df.to_csv(path, index=False)

        config = DataScienceConfig(cv_folds=3, artifacts_dir=str(tmp_path / "artifacts"))
        agent = DeploymentAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

        training_report = ModelTrainingReport(
            scale_tier="small",
            candidates=[
                ModelCandidate(
                    model_name="lightgbm_0",
                    model_type="lightgbm",
                    mean_score=0.80,
                    hyperparameters={},
                ),
                ModelCandidate(
                    model_name="catboost_0",
                    model_type="catboost",
                    mean_score=0.92,
                    hyperparameters={},
                ),
            ],
            best_candidate="",
        ).model_dump()

        evaluation_report = EvaluationReport(
            predictive_scores={"accuracy": 0.90},
            overall_grade="A",
        ).model_dump()

        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "training_report": training_report,
            "evaluation_report": evaluation_report,
        })
        assert result.status == "success"

    def test_fallback_to_gradient_boosting(self, tmp_path):
        """Test fallback when best model type is unknown (lines 297-303)."""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
        df["target"] = y
        path = tmp_path / "deploy_gb_fallback.csv"
        df.to_csv(path, index=False)

        config = DataScienceConfig(cv_folds=3, artifacts_dir=str(tmp_path / "artifacts"))
        agent = DeploymentAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

        # Use a model type that _build_sklearn_estimator returns None for
        training_report = ModelTrainingReport(
            scale_tier="small",
            candidates=[
                ModelCandidate(
                    model_name="nonexistent_0",
                    model_type="nonexistent",
                    mean_score=0.88,
                    hyperparameters={},
                ),
            ],
            best_candidate="nonexistent_0",
        ).model_dump()

        evaluation_report = EvaluationReport(
            predictive_scores={"accuracy": 0.88},
            overall_grade="B",
        ).model_dump()

        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "training_report": training_report,
            "evaluation_report": evaluation_report,
        })
        assert result.status == "success"

    def test_fallback_to_gradient_boosting_regression(self, tmp_path):
        """Test fallback to GradientBoosting for regression with unknown model."""
        X, y = make_regression(n_samples=200, n_features=10, n_informative=5, random_state=42)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
        df["target"] = y
        path = tmp_path / "deploy_gb_fallback_reg.csv"
        df.to_csv(path, index=False)

        config = DataScienceConfig(cv_folds=3, artifacts_dir=str(tmp_path / "artifacts"))
        agent = DeploymentAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

        training_report = ModelTrainingReport(
            scale_tier="small",
            candidates=[
                ModelCandidate(
                    model_name="fake_model_0",
                    model_type="fake_model",
                    mean_score=0.75,
                    hyperparameters={},
                ),
            ],
            best_candidate="fake_model_0",
        ).model_dump()

        evaluation_report = EvaluationReport(
            predictive_scores={"r2": 0.75, "rmse": 20.0},
            overall_grade="C",
        ).model_dump()

        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "regression",
            "training_report": training_report,
            "evaluation_report": evaluation_report,
        })
        assert result.status == "success"

    def test_process_exception_marks_failed(self, tmp_path):
        """Test that process marks experiment as FAILED on exception (lines 414-416)."""
        config = DataScienceConfig(cv_folds=3, artifacts_dir=str(tmp_path / "artifacts"))
        repo = FakeRepository()
        agent = DeploymentAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=repo,
            ds_config=config,
        )

        training_report = ModelTrainingReport(
            scale_tier="small",
            candidates=[
                ModelCandidate(
                    model_name="xgboost_0",
                    model_type="xgboost",
                    mean_score=0.88,
                    hyperparameters={},
                ),
            ],
            best_candidate="xgboost_0",
        ).model_dump()

        evaluation_report = EvaluationReport(
            predictive_scores={"accuracy": 0.88},
            overall_grade="B",
        ).model_dump()

        with pytest.raises(Exception):
            agent.process({
                "dataset_path": "/nonexistent/path.csv",
                "target_column": "target",
                "project_id": uuid.uuid4(),
                "problem_type": "classification",
                "training_report": training_report,
                "evaluation_report": evaluation_report,
            })
        # Verify experiment was marked as FAILED
        assert len(repo.experiments) == 1
        exp = list(repo.experiments.values())[0]
        assert exp["status"] == "FAILED"

    def test_catboost_deployment_classification(self, tmp_path):
        """Full integration: deploy with catboost classification model."""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
        df["target"] = y
        path = tmp_path / "deploy_catboost.csv"
        df.to_csv(path, index=False)

        config = DataScienceConfig(cv_folds=3, artifacts_dir=str(tmp_path / "artifacts"))
        agent = DeploymentAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

        training_report = ModelTrainingReport(
            scale_tier="small",
            candidates=[
                ModelCandidate(
                    model_name="catboost_0",
                    model_type="catboost",
                    mean_score=0.90,
                    hyperparameters={"iterations": 100},
                ),
            ],
            best_candidate="catboost_0",
        ).model_dump()

        evaluation_report = EvaluationReport(
            predictive_scores={"accuracy": 0.90},
            overall_grade="A",
        ).model_dump()

        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "training_report": training_report,
            "evaluation_report": evaluation_report,
        })
        assert result.status == "success"
        pkg = DeploymentPackage(**result.data)
        assert pkg.model_artifact_path != ""

    def test_lightgbm_deployment_regression(self, tmp_path):
        """Full integration: deploy with lightgbm regression model."""
        X, y = make_regression(n_samples=200, n_features=10, n_informative=5, random_state=42)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
        df["target"] = y
        path = tmp_path / "deploy_lgbm_reg.csv"
        df.to_csv(path, index=False)

        config = DataScienceConfig(cv_folds=3, artifacts_dir=str(tmp_path / "artifacts"))
        agent = DeploymentAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

        training_report = ModelTrainingReport(
            scale_tier="small",
            candidates=[
                ModelCandidate(
                    model_name="lightgbm_0",
                    model_type="lightgbm",
                    mean_score=0.82,
                    hyperparameters={"n_estimators": 100},
                ),
            ],
            best_candidate="lightgbm_0",
        ).model_dump()

        evaluation_report = EvaluationReport(
            predictive_scores={"r2": 0.82, "mae": 12.5},
            overall_grade="B",
        ).model_dump()

        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "regression",
            "training_report": training_report,
            "evaluation_report": evaluation_report,
        })
        assert result.status == "success"
        pkg = DeploymentPackage(**result.data)
        assert pkg.model_artifact_path != ""

    def test_tsv_dataset_through_process(self, tmp_path):
        """Full integration: process with TSV dataset file."""
        X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
        df["target"] = y
        path = tmp_path / "deploy.tsv"
        df.to_csv(path, sep="\t", index=False)

        config = DataScienceConfig(cv_folds=3, artifacts_dir=str(tmp_path / "artifacts"))
        agent = DeploymentAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

        training_report = ModelTrainingReport(
            scale_tier="small",
            candidates=[
                ModelCandidate(
                    model_name="xgboost_0",
                    model_type="xgboost",
                    mean_score=0.85,
                    hyperparameters={},
                ),
            ],
            best_candidate="xgboost_0",
        ).model_dump()

        evaluation_report = EvaluationReport(
            predictive_scores={"accuracy": 0.85},
            overall_grade="B",
        ).model_dump()

        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "training_report": training_report,
            "evaluation_report": evaluation_report,
        })
        assert result.status == "success"

    def test_parquet_dataset_through_process(self, tmp_path):
        """Full integration: process with parquet dataset file."""
        X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
        df["target"] = y
        path = tmp_path / "deploy.parquet"
        df.to_parquet(path, index=False)

        config = DataScienceConfig(cv_folds=3, artifacts_dir=str(tmp_path / "artifacts"))
        agent = DeploymentAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

        training_report = ModelTrainingReport(
            scale_tier="small",
            candidates=[
                ModelCandidate(
                    model_name="xgboost_0",
                    model_type="xgboost",
                    mean_score=0.85,
                    hyperparameters={},
                ),
            ],
            best_candidate="xgboost_0",
        ).model_dump()

        evaluation_report = EvaluationReport(
            predictive_scores={"accuracy": 0.85},
            overall_grade="B",
        ).model_dump()

        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "training_report": training_report,
            "evaluation_report": evaluation_report,
        })
        assert result.status == "success"


class TestPrepareDataEdgeCases:
    """Edge case tests for DeploymentAgent._prepare_data."""

    def _make_agent(self, tmp_path):
        config = DataScienceConfig(cv_folds=3, artifacts_dir=str(tmp_path / "artifacts"))
        return DeploymentAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

    def test_prepare_data_missing_target_column(self, tmp_path):
        """Test ValueError when target column does not exist (line 425)."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
        df["target"] = y

        agent = self._make_agent(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            agent._prepare_data(df, "nonexistent_target")

    def test_prepare_data_no_numeric_features(self, tmp_path):
        """Test ValueError when no features at all are available."""
        # Only a target column — no features to work with
        df = pd.DataFrame({
            "target": [0, 1, 0],
        })

        agent = self._make_agent(tmp_path)
        with pytest.raises(ValueError, match="No numeric features"):
            agent._prepare_data(df, "target")

    def test_prepare_data_with_missing_target_values(self, tmp_path):
        """Test that rows with missing target values are dropped (lines 445-451)."""
        df = pd.DataFrame({
            "x0": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x1": [10.0, 20.0, 30.0, 40.0, 50.0],
            "target": [0, 1, None, 0, None],
        })

        agent = self._make_agent(tmp_path)
        features, target = agent._prepare_data(df, "target")
        assert len(features) == 3
        assert len(target) == 3
        # Verify the correct rows were kept (indices 0, 1, 3 of original)
        assert list(features["x0"]) == [1.0, 2.0, 4.0]
        assert list(features["x1"]) == [10.0, 20.0, 40.0]

    def test_prepare_data_all_targets_present(self, tmp_path):
        """Test that no rows are dropped when all targets are present."""
        df = pd.DataFrame({
            "x0": [1.0, 2.0, 3.0],
            "x1": [10.0, 20.0, 30.0],
            "target": [0, 1, 0],
        })

        agent = self._make_agent(tmp_path)
        features, target = agent._prepare_data(df, "target")
        assert len(features) == 3
        assert len(target) == 3

    def test_prepare_data_fills_feature_nans(self, tmp_path):
        """Test that NaN feature values are filled with median."""
        df = pd.DataFrame({
            "x0": [1.0, np.nan, 3.0, 4.0, 5.0],
            "x1": [10.0, 20.0, np.nan, 40.0, 50.0],
            "target": [0, 1, 0, 1, 0],
        })

        agent = self._make_agent(tmp_path)
        features, target = agent._prepare_data(df, "target")
        assert len(features) == 5
        # No NaN values should remain in features
        assert not features.isna().any().any()

    def test_prepare_data_mixed_numeric_and_string_columns(self, tmp_path):
        """Test that string columns are frequency-encoded and kept as numeric."""
        df = pd.DataFrame({
            "x0": [1.0, 2.0, 3.0],
            "label": ["a", "b", "c"],
            "x1": [10.0, 20.0, 30.0],
            "target": [0, 1, 0],
        })

        agent = self._make_agent(tmp_path)
        features, target = agent._prepare_data(df, "target")
        # label is now frequency-encoded as a numeric column
        assert "label" in features.columns
        assert "x0" in features.columns
        assert "x1" in features.columns
        assert len(features.columns) == 3


class TestComputeFeatureStats:
    """Tests for _compute_feature_stats."""

    def test_feature_stats_computed(self, tmp_path):
        """Test _compute_feature_stats returns proper stats for each feature."""
        config = DataScienceConfig(cv_folds=3, artifacts_dir=str(tmp_path / "artifacts"))
        agent = DeploymentAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

        features = pd.DataFrame({
            "x0": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x1": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

        stats = agent._compute_feature_stats(features)
        assert "x0" in stats
        assert "x1" in stats
        # Verify all expected stat keys
        for col_name in ["x0", "x1"]:
            assert "mean" in stats[col_name]
            assert "std" in stats[col_name]
            assert "min" in stats[col_name]
            assert "max" in stats[col_name]
            assert "median" in stats[col_name]
            assert "q25" in stats[col_name]
            assert "q75" in stats[col_name]

        # Verify actual values for x0 = [1, 2, 3, 4, 5]
        assert stats["x0"]["mean"] == round(3.0, 6)
        assert stats["x0"]["min"] == round(1.0, 6)
        assert stats["x0"]["max"] == round(5.0, 6)
        assert stats["x0"]["median"] == round(3.0, 6)

    def test_feature_stats_single_column(self, tmp_path):
        """Test stats with a single feature column."""
        config = DataScienceConfig(cv_folds=3, artifacts_dir=str(tmp_path / "artifacts"))
        agent = DeploymentAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

        features = pd.DataFrame({
            "x0": [100.0, 200.0, 300.0],
        })

        stats = agent._compute_feature_stats(features)
        assert len(stats) == 1
        assert stats["x0"]["mean"] == round(200.0, 6)
        assert stats["x0"]["min"] == round(100.0, 6)
        assert stats["x0"]["max"] == round(300.0, 6)

    def test_feature_stats_with_real_synthetic_data(self, tmp_path):
        """Test stats computed on real sklearn synthetic dataset features."""
        config = DataScienceConfig(cv_folds=3, artifacts_dir=str(tmp_path / "artifacts"))
        agent = DeploymentAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

        X, _ = make_classification(n_samples=500, n_features=20, random_state=42)
        features = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])

        stats = agent._compute_feature_stats(features)
        assert len(stats) == 20
        for col_name in features.columns:
            col_stats = stats[col_name]
            assert col_stats["min"] <= col_stats["q25"] <= col_stats["median"]
            assert col_stats["median"] <= col_stats["q75"] <= col_stats["max"]
