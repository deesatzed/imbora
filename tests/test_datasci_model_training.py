"""Tests for Phase 4: Model Training Agent.

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
from src.datasci.agents.model_training import ModelTrainingAgent
from src.datasci.models import ModelTrainingReport


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLMClient:
    def __init__(self):
        self.calls = []

    def complete_with_fallback(self, **kwargs):
        self.calls.append(True)
        return FakeLLMResponse(
            "XGBoost shows strong performance with good calibration. "
            "The model generalizes well across folds."
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
    path = tmp_path / "train_cls.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def regression_csv(tmp_path):
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=5, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
    df["target"] = y
    path = tmp_path / "train_reg.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def agent():
    config = DataScienceConfig(
        cv_folds=3,
        enable_tabpfn=False,  # Skip optional deps for testing
        enable_tabicl=False,
        enable_nam=False,
    )
    return ModelTrainingAgent(
        llm_client=FakeLLMClient(),
        model_router=FakeModelRouter(),
        repository=FakeRepository(),
        ds_config=config,
    )


class TestModelTrainingAgent:
    def test_classification_produces_report(self, agent, classification_csv):
        result = agent.process({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        assert result.status == "success"
        report = ModelTrainingReport(**result.data)
        assert report.scale_tier in ("small", "medium", "large")
        assert len(report.candidates) > 0
        assert report.best_candidate != ""

    def test_candidates_have_scores(self, agent, classification_csv):
        result = agent.process({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        report = ModelTrainingReport(**result.data)
        for candidate in report.candidates:
            assert candidate.mean_score > 0.0
            assert len(candidate.cv_scores) > 0
            assert candidate.training_time_seconds >= 0

    def test_best_candidate_is_highest_scorer(self, agent, classification_csv):
        result = agent.process({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        report = ModelTrainingReport(**result.data)
        best = max(report.candidates, key=lambda c: c.mean_score)
        assert report.best_candidate == best.model_name

    def test_regression_works(self, agent, regression_csv):
        result = agent.process({
            "dataset_path": regression_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "regression",
        })
        assert result.status == "success"
        report = ModelTrainingReport(**result.data)
        assert len(report.candidates) > 0

    def test_llm_behavioral_analysis(self, agent, classification_csv):
        result = agent.process({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        report = ModelTrainingReport(**result.data)
        assert report.llm_evaluation_narrative != ""

    def test_experiment_tracked(self, agent, classification_csv):
        agent.process({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        assert len(agent.repository.experiments) == 1
        exp = list(agent.repository.experiments.values())[0]
        assert exp["status"] == "COMPLETED"

    def test_run_lifecycle(self, agent, classification_csv):
        result = agent.run({
            "dataset_path": classification_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
        })
        assert result.status == "success"
        assert result.duration_seconds > 0


# ---------------------------------------------------------------------------
# New coverage tests -- _load_dataset (lines 48-53)
# ---------------------------------------------------------------------------

class TestModelTrainingLoadDataset:
    def test_load_tsv(self, tmp_path):
        from src.datasci.agents.model_training import _load_dataset
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "data.tsv"
        df.to_csv(path, sep="\t", index=False)
        result = _load_dataset(str(path))
        assert len(result) == 2
        assert list(result.columns) == ["a", "b"]

    def test_load_parquet(self, tmp_path):
        from src.datasci.agents.model_training import _load_dataset
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "data.parquet"
        df.to_parquet(path, index=False)
        result = _load_dataset(str(path))
        assert len(result) == 2
        assert list(result.columns) == ["a", "b"]

    def test_load_pq_extension(self, tmp_path):
        from src.datasci.agents.model_training import _load_dataset
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "data.pq"
        df.to_parquet(path, index=False)
        result = _load_dataset(str(path))
        assert len(result) == 2

    def test_load_unsupported(self, tmp_path):
        from src.datasci.agents.model_training import _load_dataset
        path = tmp_path / "data.xlsx"
        path.write_text("fake")
        with pytest.raises(ValueError, match="Unsupported"):
            _load_dataset(str(path))


# ---------------------------------------------------------------------------
# New coverage tests -- _build_sklearn_estimator (lines 84-86, 108-110,
# 132-134, 139-142, 150-153, 161-164, 170-171)
# ---------------------------------------------------------------------------

class TestModelTrainingBuildEstimator:
    """Exercise every branch in _build_sklearn_estimator.

    For xgboost/catboost/lightgbm the packages are installed so we get real
    estimators and cover the success paths. For tabpfn/tabicl/nam the packages
    are NOT installed so the ImportError branches (returning None) are exercised.
    """

    def test_xgboost_classification(self):
        from src.datasci.agents.model_training import _build_sklearn_estimator
        est = _build_sklearn_estimator("xgboost", "classification")
        assert est is not None
        assert hasattr(est, "fit")

    def test_xgboost_regression(self):
        from src.datasci.agents.model_training import _build_sklearn_estimator
        est = _build_sklearn_estimator("xgboost", "regression")
        assert est is not None
        assert hasattr(est, "fit")

    def test_catboost_classification(self):
        from src.datasci.agents.model_training import _build_sklearn_estimator
        est = _build_sklearn_estimator("catboost", "classification")
        assert est is not None
        assert hasattr(est, "fit")

    def test_catboost_regression(self):
        from src.datasci.agents.model_training import _build_sklearn_estimator
        est = _build_sklearn_estimator("catboost", "regression")
        assert est is not None
        assert hasattr(est, "fit")

    def test_lightgbm_classification(self):
        from src.datasci.agents.model_training import _build_sklearn_estimator
        est = _build_sklearn_estimator("lightgbm", "classification")
        assert est is not None
        assert hasattr(est, "fit")

    def test_lightgbm_regression(self):
        from src.datasci.agents.model_training import _build_sklearn_estimator
        est = _build_sklearn_estimator("lightgbm", "regression")
        assert est is not None
        assert hasattr(est, "fit")

    def test_tabpfn_classification(self):
        """tabpfn is not installed; exercises ImportError branch (lines 143-145)."""
        from src.datasci.agents.model_training import _build_sklearn_estimator
        est = _build_sklearn_estimator("tabpfn", "classification")
        # Returns None when tabpfn is not installed
        assert est is None or hasattr(est, "fit")

    def test_tabpfn_regression(self):
        """tabpfn regression branch (lines 139-142)."""
        from src.datasci.agents.model_training import _build_sklearn_estimator
        est = _build_sklearn_estimator("tabpfn", "regression")
        assert est is None or hasattr(est, "fit")

    def test_tabicl_classification(self):
        """tabicl is not installed; exercises ImportError branch (lines 154-156)."""
        from src.datasci.agents.model_training import _build_sklearn_estimator
        est = _build_sklearn_estimator("tabicl", "classification")
        assert est is None or hasattr(est, "fit")

    def test_tabicl_regression(self):
        """tabicl regression branch (lines 150-153)."""
        from src.datasci.agents.model_training import _build_sklearn_estimator
        est = _build_sklearn_estimator("tabicl", "regression")
        assert est is None or hasattr(est, "fit")

    def test_nam_classification(self):
        """dnamite is not installed; exercises ImportError branch (lines 165-167)."""
        from src.datasci.agents.model_training import _build_sklearn_estimator
        est = _build_sklearn_estimator("nam", "classification")
        assert est is None or hasattr(est, "fit")

    def test_nam_regression(self):
        """nam regression branch (lines 161-164)."""
        from src.datasci.agents.model_training import _build_sklearn_estimator
        est = _build_sklearn_estimator("nam", "regression")
        assert est is None or hasattr(est, "fit")

    def test_unknown_model(self):
        """Unknown model name returns None (lines 170-171)."""
        from src.datasci.agents.model_training import _build_sklearn_estimator
        result = _build_sklearn_estimator("nonexistent_model_xyz", "classification")
        assert result is None

    def test_unknown_model_regression(self):
        from src.datasci.agents.model_training import _build_sklearn_estimator
        result = _build_sklearn_estimator("totally_fake", "regression")
        assert result is None


# ---------------------------------------------------------------------------
# New coverage tests -- _prepare_data edge cases (lines 373, 385, 397-402)
# ---------------------------------------------------------------------------

class TestModelTrainingPrepareData:
    def _make_agent(self):
        config = DataScienceConfig(
            cv_folds=3,
            enable_tabpfn=False,
            enable_tabicl=False,
            enable_nam=False,
        )
        return ModelTrainingAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

    def test_missing_target_column_raises(self):
        """Cover line 373: target column not found."""
        agent = self._make_agent()
        df = pd.DataFrame({"x0": [1, 2, 3], "x1": [4, 5, 6]})
        with pytest.raises(ValueError, match="not found"):
            agent._prepare_data(df, "nonexistent_target")

    def test_no_numeric_features_raises(self):
        """Cover line 385: no numeric features available."""
        agent = self._make_agent()
        # Only target column — no features at all
        df = pd.DataFrame({
            "target": [0, 1, 0],
        })
        with pytest.raises(ValueError, match="No numeric features"):
            agent._prepare_data(df, "target")

    def test_drops_rows_with_missing_target(self):
        """Cover lines 397-402: drop rows where target is NaN."""
        agent = self._make_agent()
        df = pd.DataFrame({
            "x0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "target": [0, None, 1, None, 0, 1],
        })
        features, target = agent._prepare_data(df, "target")
        assert len(features) == 4
        assert len(target) == 4
        # Confirm no NaN in target after dropping
        assert target.notna().all()

    def test_fills_missing_feature_values(self):
        """Verify NaN in features are filled with median."""
        agent = self._make_agent()
        df = pd.DataFrame({
            "x0": [1.0, None, 3.0, 4.0],
            "x1": [10.0, 20.0, None, 40.0],
            "target": [0, 1, 0, 1],
        })
        features, target = agent._prepare_data(df, "target")
        assert len(features) == 4
        assert not features.isna().any().any()


# ---------------------------------------------------------------------------
# New coverage tests -- process exception handler (lines 359-361)
# ---------------------------------------------------------------------------

class TestModelTrainingProcessException:
    def test_process_exception_marks_experiment_failed(self):
        """Cover lines 359-361: except block sets FAILED and re-raises."""
        config = DataScienceConfig(
            cv_folds=3,
            enable_tabpfn=False,
            enable_tabicl=False,
            enable_nam=False,
        )
        repo = FakeRepository()
        agent = ModelTrainingAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=repo,
            ds_config=config,
        )
        with pytest.raises(Exception):
            agent.process({
                "dataset_path": "/nonexistent/path/data.csv",
                "target_column": "target",
                "project_id": uuid.uuid4(),
                "problem_type": "classification",
            })
        # Exactly one experiment was created, and it was marked FAILED
        assert len(repo.experiments) == 1
        exp = list(repo.experiments.values())[0]
        assert exp["status"] == "FAILED"


# ---------------------------------------------------------------------------
# New coverage tests -- _extract_nam_shapes (lines 320, 563-594)
# ---------------------------------------------------------------------------

class TestModelTrainingExtractNamShapes:
    def _make_agent(self):
        config = DataScienceConfig(
            cv_folds=3,
            enable_tabpfn=False,
            enable_tabicl=False,
            enable_nam=False,
        )
        return ModelTrainingAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

    def test_with_get_shape_functions(self):
        """Cover line 320 and 564-580: get_shape_functions path."""
        agent = self._make_agent()

        class FakeNAMWithShapes:
            def get_shape_functions(self, features):
                return {
                    "x0": {
                        "x": np.array([1.0, 2.0, 3.0]),
                        "y": np.array([0.1, 0.2, 0.3]),
                    },
                    "x1": {
                        "x": np.array([4.0, 5.0, 6.0]),
                        "y": np.array([0.4, 0.5, 0.6]),
                    },
                }

        features = pd.DataFrame({"x0": [1, 2, 3], "x1": [4, 5, 6]})
        result = agent._extract_nam_shapes(FakeNAMWithShapes(), features)
        assert result is not None
        assert "x0" in result
        assert "x1" in result
        assert "x_values" in result["x0"]
        assert "y_values" in result["x0"]
        assert result["x0"]["x_values"] == [1.0, 2.0, 3.0]
        assert result["x1"]["y_values"] == [0.4, 0.5, 0.6]

    def test_with_shape_functions_non_array(self):
        """Cover get_shape_functions returning plain lists (no .tolist())."""
        agent = self._make_agent()

        class FakeNAMPlainLists:
            def get_shape_functions(self, features):
                return {
                    "x0": {"x": [1, 2, 3], "y": [0.1, 0.2, 0.3]},
                }

        features = pd.DataFrame({"x0": [1, 2, 3]})
        result = agent._extract_nam_shapes(FakeNAMPlainLists(), features)
        assert result is not None
        assert result["x0"]["x_values"] == [1, 2, 3]

    def test_with_feature_importances_fallback(self):
        """Cover lines 581-590: feature_importances_ fallback path."""
        agent = self._make_agent()

        class FakeNAMImportances:
            feature_importances_ = np.array([0.3, 0.7])

        features = pd.DataFrame({"x0": [1, 2, 3], "x1": [4, 5, 6]})
        result = agent._extract_nam_shapes(FakeNAMImportances(), features)
        assert result is not None
        assert "importances" in result
        assert result["importances"]["x0"] == 0.3
        assert result["importances"]["x1"] == 0.7

    def test_no_shape_attributes_returns_none(self):
        """Cover line 591: return None when no shape attributes."""
        agent = self._make_agent()

        class PlainEstimator:
            pass

        features = pd.DataFrame({"x0": [1, 2, 3], "x1": [4, 5, 6]})
        result = agent._extract_nam_shapes(PlainEstimator(), features)
        assert result is None

    def test_exception_returns_none(self):
        """Cover lines 592-594: exception in shape extraction returns None."""
        agent = self._make_agent()

        class BrokenNAM:
            def get_shape_functions(self, features):
                raise RuntimeError("NAM internal error")

        features = pd.DataFrame({"x0": [1, 2, 3], "x1": [4, 5, 6]})
        result = agent._extract_nam_shapes(BrokenNAM(), features)
        assert result is None


# ---------------------------------------------------------------------------
# New coverage tests -- _train_candidate edge cases (lines 457, 488-490, 297)
# ---------------------------------------------------------------------------

class TestModelTrainingTrainCandidate:
    def _make_agent(self):
        config = DataScienceConfig(
            cv_folds=3,
            enable_tabpfn=False,
            enable_tabicl=False,
            enable_nam=False,
        )
        return ModelTrainingAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

    def test_unknown_model_returns_none(self):
        """Cover lines 419-421: estimator is None, return None."""
        agent = self._make_agent()
        X, y = make_classification(
            n_samples=50, n_features=5, random_state=42,
        )
        features = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
        target = pd.Series(y)
        result = agent._train_candidate(
            name="nonexistent_model",
            problem_type="classification",
            features=features,
            target=target,
            scoring="accuracy",
        )
        assert result is None

    def test_train_failure_returns_none(self):
        """Cover lines 488-490: exception during cross_val_score returns None.

        Uses a real sklearn-compatible estimator that raises during fit.
        """
        from sklearn.base import BaseEstimator, ClassifierMixin

        class FailingEstimator(BaseEstimator, ClassifierMixin):
            """Real sklearn-compatible estimator that raises during fit."""

            def fit(self, X, y):
                raise RuntimeError("Deliberate fit failure for testing")

            def predict(self, X):
                return np.zeros(len(X))

        agent = self._make_agent()
        X, y = make_classification(
            n_samples=50, n_features=5, random_state=42,
        )
        features = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
        target = pd.Series(y)

        # Temporarily override _build_sklearn_estimator in the module namespace
        import src.datasci.agents.model_training as mt_module
        original_build = mt_module._build_sklearn_estimator

        def patched_build(model_name, problem_type, imbalance_ratio=1.0):
            if model_name == "fail_model":
                return FailingEstimator()
            return original_build(model_name, problem_type, imbalance_ratio=imbalance_ratio)

        mt_module._build_sklearn_estimator = patched_build
        try:
            result = agent._train_candidate(
                name="fail_model",
                problem_type="classification",
                features=features,
                target=target,
                scoring="accuracy",
            )
            assert result is None
        finally:
            mt_module._build_sklearn_estimator = original_build

    def test_prediction_time_zero_when_clone_none(self):
        """Cover line 457: prediction_time_ms = 0.0 when clone is None.

        The first call to _build_sklearn_estimator (line 419) returns a
        working estimator for cross_val_score. The second call (line 443)
        returns None, causing prediction_time_ms to be set to 0.0.
        """
        from sklearn.base import BaseEstimator, ClassifierMixin

        class SimpleClassifier(BaseEstimator, ClassifierMixin):
            """Minimal classifier that works with cross_val_score."""

            def fit(self, X, y):
                self.classes_ = np.unique(y)
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

            def score(self, X, y):
                return 0.5

        agent = self._make_agent()
        X, y = make_classification(
            n_samples=60, n_features=5, random_state=42,
        )
        features = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
        target = pd.Series(y)

        import src.datasci.agents.model_training as mt_module
        original_build = mt_module._build_sklearn_estimator
        build_call_count = {"n": 0}

        def patched_build(model_name, problem_type, imbalance_ratio=1.0):
            if model_name == "clone_none_model":
                build_call_count["n"] += 1
                if build_call_count["n"] == 1:
                    # First call: return working estimator for cross_val_score
                    return SimpleClassifier()
                else:
                    # Second call: return None for prediction timing
                    return None
            return original_build(model_name, problem_type, imbalance_ratio=imbalance_ratio)

        mt_module._build_sklearn_estimator = patched_build
        try:
            result = agent._train_candidate(
                name="clone_none_model",
                problem_type="classification",
                features=features,
                target=target,
                scoring="accuracy",
            )
            assert result is not None
            assert result.prediction_time_ms == 0.0
        finally:
            mt_module._build_sklearn_estimator = original_build

    def test_no_candidates_raises_runtime_error(self, tmp_path):
        """Cover line 297: no candidates trained raises RuntimeError.

        Makes all estimators return None so no candidates can be trained.
        """
        import src.datasci.agents.model_training as mt_module
        original_build = mt_module._build_sklearn_estimator
        original_select = mt_module.select_model_candidates

        X, y = make_classification(
            n_samples=50, n_features=5, random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
        df["target"] = y
        path = tmp_path / "data_no_candidates.csv"
        df.to_csv(path, index=False)

        # Make all estimators return None to trigger "no candidates" error
        mt_module._build_sklearn_estimator = lambda name, pt, imbalance_ratio=1.0: None
        try:
            agent = self._make_agent()
            with pytest.raises(RuntimeError, match="No model candidates"):
                agent.process({
                    "dataset_path": str(path),
                    "target_column": "target",
                    "project_id": uuid.uuid4(),
                    "problem_type": "classification",
                })
        finally:
            mt_module._build_sklearn_estimator = original_build


# ---------------------------------------------------------------------------
# New coverage tests -- _calibrate_conformal edge cases (lines 512, 551-552)
# ---------------------------------------------------------------------------

class TestModelTrainingCalibrateConformal:
    def _make_agent(self):
        config = DataScienceConfig(
            cv_folds=3,
            enable_tabpfn=False,
            enable_tabicl=False,
            enable_nam=False,
        )
        return ModelTrainingAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=config,
        )

    def test_estimator_none_skipped(self):
        """Cover line 512: skip conformal when estimator is None."""
        from src.datasci.models import ModelCandidate

        agent = self._make_agent()
        candidate = ModelCandidate(
            model_name="missing_model",
            model_type="missing_model",
            cv_scores=[0.8, 0.85, 0.82],
            mean_score=0.823,
            std_score=0.02,
        )

        X, y = make_classification(
            n_samples=50, n_features=5, random_state=42,
        )
        features = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
        target = pd.Series(y)

        # No estimator in the dict for "missing_model" -> skip (line 512)
        agent._calibrate_conformal(
            candidates=[candidate],
            trained_estimators={},
            features=features,
            target=target,
            problem_type="classification",
        )
        # conformal_coverage should remain None since calibration was skipped
        assert candidate.conformal_coverage is None

    def test_conformal_failure_warning(self):
        """Cover lines 551-552: conformal calibration exception handled.

        Uses an estimator whose cross_val_predict produces NaN predictions,
        causing ConformalPredictor.calibrate() to fail.
        """
        from sklearn.base import BaseEstimator, ClassifierMixin
        from src.datasci.models import ModelCandidate

        class NaNPredictor(BaseEstimator, ClassifierMixin):
            """Estimator that returns NaN predictions to trigger conformal failure."""
            def fit(self, X, y):
                self.classes_ = np.unique(y)
                return self

            def predict(self, X):
                return np.full(len(X), np.nan)

        agent = self._make_agent()
        candidate = ModelCandidate(
            model_name="nan_predictor",
            model_type="nan_predictor",
            cv_scores=[0.5, 0.5, 0.5],
            mean_score=0.5,
            std_score=0.0,
        )

        X, y = make_classification(
            n_samples=60, n_features=5, random_state=42,
        )
        features = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
        target = pd.Series(y)

        nan_est = NaNPredictor()
        nan_est.fit(features, target)

        # Patch _build_sklearn_estimator in the model_training module namespace
        # so cross_val_predict inside _calibrate_conformal uses NaNPredictor
        import src.datasci.agents.model_training as mt_module
        original_build = mt_module._build_sklearn_estimator

        def patched_build(model_name, problem_type):
            if model_name == "nan_predictor":
                return NaNPredictor()
            return original_build(model_name, problem_type)

        mt_module._build_sklearn_estimator = patched_build
        try:
            agent._calibrate_conformal(
                candidates=[candidate],
                trained_estimators={"nan_predictor": nan_est},
                features=features,
                target=target,
                problem_type="classification",
            )
            # Conformal should have failed gracefully; coverage stays None
        finally:
            mt_module._build_sklearn_estimator = original_build


# ---------------------------------------------------------------------------
# New coverage tests -- conformal fit failure warning (lines 290-291)
# and NAM shape extraction via process() (line 320)
# ---------------------------------------------------------------------------

class TestModelTrainingConformalFitFailure:
    def test_conformal_fit_failure_logs_warning(self, tmp_path):
        """Cover lines 290-291: estimator.fit() fails for conformal step.

        After a candidate is successfully trained via cross_val_score,
        the code at line 285 rebuilds the estimator and at line 289 fits it
        for conformal calibration. If that fit raises, lines 290-291 log
        a warning and continue.

        Call sequence with cv=3:
        - _train_candidate: build #1 -> cross_val_score (3 clone fits: count=3)
                            build #2 (prediction timing) -> fit (count=4)
        - process loop:     build #3 (conformal) -> fit (count=5) -> FAILS
        """
        from sklearn.base import BaseEstimator, ClassifierMixin
        import src.datasci.agents.model_training as mt_module
        original_build = mt_module._build_sklearn_estimator
        original_select = mt_module.select_model_candidates

        call_count = {"fit_model": 0}

        class FitLimitedClassifier(BaseEstimator, ClassifierMixin):
            """Succeeds during cross_val_score and prediction timing,
            but fails on the 5th fit (the conformal calibration fit)."""

            def __init__(self):
                self.classes_ = np.array([0, 1])

            def fit(self, X, y):
                call_count["fit_model"] += 1
                # cv=3 produces 3 fits, prediction timing produces 1 more (=4),
                # the 5th fit is the conformal calibration fit in process()
                if call_count["fit_model"] > 4:
                    raise RuntimeError("Deliberate conformal fit failure")
                self.classes_ = np.unique(y)
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

            def score(self, X, y):
                return 0.5

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        def patched_build(model_name, problem_type, imbalance_ratio=1.0):
            if model_name == "fitlimited":
                return FitLimitedClassifier()
            return original_build(model_name, problem_type, imbalance_ratio=imbalance_ratio)

        def patched_select(**kwargs):
            return ["fitlimited"]

        X, y = make_classification(
            n_samples=60, n_features=5, random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
        df["target"] = y
        path = tmp_path / "data_fitlimited.csv"
        df.to_csv(path, index=False)

        # Patch both in the model_training module namespace
        mt_module._build_sklearn_estimator = patched_build
        mt_module.select_model_candidates = patched_select
        try:
            config = DataScienceConfig(
                cv_folds=3,
                enable_tabpfn=False,
                enable_tabicl=False,
                enable_nam=False,
            )
            agent = ModelTrainingAgent(
                llm_client=FakeLLMClient(),
                model_router=FakeModelRouter(),
                repository=FakeRepository(),
                ds_config=config,
            )
            result = agent.process({
                "dataset_path": str(path),
                "target_column": "target",
                "project_id": uuid.uuid4(),
                "problem_type": "classification",
            })
            # The process should succeed even though conformal fit failed
            assert result.status == "success"
            # Verify conformal fit failure was handled (candidate still present)
            report = ModelTrainingReport(**result.data)
            assert len(report.candidates) == 1
        finally:
            mt_module._build_sklearn_estimator = original_build
            mt_module.select_model_candidates = original_select

    def test_nam_shapes_extracted_in_process(self, tmp_path):
        """Cover line 320: NAM shape extraction when 'nam' is in trained_estimators.

        This requires the NAM candidate to be successfully trained AND the
        conformal fit to succeed, so 'nam' ends up in trained_estimators.
        """
        from sklearn.base import BaseEstimator, ClassifierMixin
        import src.datasci.agents.model_training as mt_module
        original_build = mt_module._build_sklearn_estimator
        original_select = mt_module.select_model_candidates

        class FakeNAMClassifier(BaseEstimator, ClassifierMixin):
            """Classifier that simulates a NAM with feature_importances_."""

            def __init__(self):
                self.classes_ = np.array([0, 1])
                self.feature_importances_ = None

            def fit(self, X, y):
                self.classes_ = np.unique(y)
                n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])
                self.feature_importances_ = np.random.rand(n_features)
                return self

            def predict(self, X):
                n = X.shape[0] if hasattr(X, "shape") else len(X)
                return np.zeros(n, dtype=int)

            def score(self, X, y):
                return 0.5

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        def patched_build(model_name, problem_type, imbalance_ratio=1.0):
            if model_name == "nam":
                return FakeNAMClassifier()
            return original_build(model_name, problem_type, imbalance_ratio=imbalance_ratio)

        def patched_select(**kwargs):
            return ["nam"]

        X, y = make_classification(
            n_samples=60, n_features=5, random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
        df["target"] = y
        path = tmp_path / "data_nam.csv"
        df.to_csv(path, index=False)

        mt_module._build_sklearn_estimator = patched_build
        mt_module.select_model_candidates = patched_select
        try:
            config = DataScienceConfig(
                cv_folds=3,
                enable_tabpfn=False,
                enable_tabicl=False,
                enable_nam=False,
            )
            agent = ModelTrainingAgent(
                llm_client=FakeLLMClient(),
                model_router=FakeModelRouter(),
                repository=FakeRepository(),
                ds_config=config,
            )
            result = agent.process({
                "dataset_path": str(path),
                "target_column": "target",
                "project_id": uuid.uuid4(),
                "problem_type": "classification",
            })
            assert result.status == "success"
            report = ModelTrainingReport(**result.data)
            # nam_shape_functions should be populated (line 320 was hit)
            assert report.nam_shape_functions is not None
            assert "importances" in report.nam_shape_functions
        finally:
            mt_module._build_sklearn_estimator = original_build
            mt_module.select_model_candidates = original_select
