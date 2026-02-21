"""Tests for Phase 3.5: Data Augmentation Agent and strategy selection.

Uses real sklearn synthetic datasets with imbalanced classes. No mocks.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.core.config import DataScienceConfig
from src.datasci.agents.data_augmentation import (
    DataAugmentationAgent,
    select_augmentation_strategy,
)
from src.datasci.models import AugmentationReport


# ---------------------------------------------------------------------------
# Real-shaped fakes (matching real interface signatures)
# ---------------------------------------------------------------------------

class FakeLLMResponse:
    def __init__(self, content: str = "LLM response"):
        self.content = content


class FakeLLMClient:
    def __init__(self):
        self.calls = []

    def complete_with_fallback(self, **kwargs):
        self.calls.append(True)
        return FakeLLMResponse(
            "Augmentation quality is acceptable. The adversarial validation "
            "score indicates reasonable synthetic data quality."
        )


class FakeModelRouter:
    def get_model(self, role):
        return "test/model"

    def get_fallback_models(self, role):
        return []

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ds_config():
    return DataScienceConfig(
        enable_augmentation=True,
        augmentation_strategy="auto",
        imbalance_threshold=3.0,
        max_augmentation_ratio=5.0,
    )


@pytest.fixture
def agent(ds_config):
    return DataAugmentationAgent(
        llm_client=FakeLLMClient(),
        model_router=FakeModelRouter(),
        repository=FakeRepository(),
        ds_config=ds_config,
    )


@pytest.fixture
def imbalanced_csv(tmp_path):
    """Create a CSV with imbalanced binary classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        weights=[0.9, 0.1],
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
    df["target"] = y
    path = tmp_path / "imbalanced.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def balanced_csv(tmp_path):
    """Create a CSV with balanced classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
    df["target"] = y
    path = tmp_path / "balanced.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def regression_csv(tmp_path):
    """Create a CSV with regression data."""
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=5, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
    df["target"] = y
    path = tmp_path / "regression.csv"
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Tests: select_augmentation_strategy()
# ---------------------------------------------------------------------------

class TestSelectAugmentationStrategy:
    """Test the standalone strategy selection function."""

    def test_augmentation_disabled_returns_none(self):
        config = DataScienceConfig(enable_augmentation=False)
        result = select_augmentation_strategy(
            imbalance_ratio=5.0,
            has_text_columns=False,
            has_categorical_columns=False,
            config=config,
        )
        assert result == "none"

    def test_explicit_strategy_overrides_auto(self):
        config = DataScienceConfig(
            enable_augmentation=True,
            augmentation_strategy="smote",
        )
        result = select_augmentation_strategy(
            imbalance_ratio=5.0,
            has_text_columns=False,
            has_categorical_columns=False,
            config=config,
        )
        assert result == "smote"

    def test_below_threshold_returns_none(self):
        config = DataScienceConfig(
            enable_augmentation=True,
            augmentation_strategy="auto",
            imbalance_threshold=3.0,
        )
        result = select_augmentation_strategy(
            imbalance_ratio=2.0,
            has_text_columns=False,
            has_categorical_columns=False,
            config=config,
        )
        assert result == "none"

    def test_high_imbalance_with_text_returns_llm_synth(self):
        config = DataScienceConfig(
            enable_augmentation=True,
            augmentation_strategy="auto",
            imbalance_threshold=3.0,
        )
        result = select_augmentation_strategy(
            imbalance_ratio=10.0,
            has_text_columns=True,
            has_categorical_columns=False,
            config=config,
        )
        assert result == "llm_synth"

    def test_high_imbalance_no_text_returns_adasyn(self):
        config = DataScienceConfig(
            enable_augmentation=True,
            augmentation_strategy="auto",
            imbalance_threshold=3.0,
        )
        result = select_augmentation_strategy(
            imbalance_ratio=10.0,
            has_text_columns=False,
            has_categorical_columns=False,
            config=config,
        )
        assert result == "adasyn"

    def test_moderate_imbalance_with_categorical_returns_smotenc(self):
        config = DataScienceConfig(
            enable_augmentation=True,
            augmentation_strategy="auto",
            imbalance_threshold=3.0,
        )
        result = select_augmentation_strategy(
            imbalance_ratio=5.0,
            has_text_columns=False,
            has_categorical_columns=True,
            config=config,
        )
        assert result == "smotenc"

    def test_moderate_imbalance_numeric_only_returns_smote(self):
        config = DataScienceConfig(
            enable_augmentation=True,
            augmentation_strategy="auto",
            imbalance_threshold=3.0,
        )
        result = select_augmentation_strategy(
            imbalance_ratio=5.0,
            has_text_columns=False,
            has_categorical_columns=False,
            config=config,
        )
        assert result == "smote"

    def test_at_threshold_returns_smote(self):
        """Exactly at threshold should trigger augmentation."""
        config = DataScienceConfig(
            enable_augmentation=True,
            augmentation_strategy="auto",
            imbalance_threshold=3.0,
        )
        result = select_augmentation_strategy(
            imbalance_ratio=3.0,
            has_text_columns=False,
            has_categorical_columns=False,
            config=config,
        )
        assert result == "smote"


# ---------------------------------------------------------------------------
# Tests: DataAugmentationAgent.process()
# ---------------------------------------------------------------------------

class TestDataAugmentationAgent:
    """Test the full agent pipeline."""

    def test_regression_passthrough(self, agent, regression_csv):
        """Regression datasets should pass through without augmentation."""
        result = agent.process({
            "dataset_path": regression_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "regression",
        })
        assert result.status == "success"
        report = AugmentationReport(**result.data)
        assert report.augmented is False
        assert report.strategy_used == "none"
        assert report.samples_generated == 0

    def test_not_imbalanced_passthrough(self, agent, balanced_csv):
        """Balanced datasets should pass through without augmentation."""
        result = agent.process({
            "dataset_path": balanced_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "eda_report": {
                "is_imbalanced": False,
                "imbalance_ratio": 1.0,
            },
        })
        assert result.status == "success"
        report = AugmentationReport(**result.data)
        assert report.augmented is False
        assert report.strategy_used == "none"

    def test_imbalanced_none_when_strategy_none(self, agent, imbalanced_csv):
        """When strategy evaluates to 'none', pass through."""
        # Set imbalanced but augmentation disabled
        agent.ds_config = DataScienceConfig(
            enable_augmentation=False,
        )
        result = agent.process({
            "dataset_path": imbalanced_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "eda_report": {
                "is_imbalanced": True,
                "imbalance_ratio": 5.0,
            },
            "audit_report": {
                "column_profiles": [
                    {"name": f"x{i}", "dtype": "numeric"} for i in range(10)
                ],
            },
        })
        assert result.status == "success"
        report = AugmentationReport(**result.data)
        assert report.augmented is False

    def test_experiment_tracking(self, agent, regression_csv):
        """Verify experiment is tracked in the repository."""
        agent.process({
            "dataset_path": regression_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "regression",
        })
        assert len(agent.repository.experiments) == 1
        exp = list(agent.repository.experiments.values())[0]
        assert exp["status"] == "COMPLETED"

    def test_process_exception_marks_failed(self, agent):
        """Exception during processing should mark experiment FAILED."""
        with pytest.raises(Exception):
            agent.process({
                "dataset_path": "/nonexistent/path/data.csv",
                "target_column": "target",
                "project_id": uuid.uuid4(),
                "problem_type": "classification",
                "eda_report": {
                    "is_imbalanced": True,
                    "imbalance_ratio": 5.0,
                },
                "audit_report": {
                    "column_profiles": [],
                },
            })
        assert len(agent.repository.experiments) == 1
        exp = list(agent.repository.experiments.values())[0]
        assert exp["status"] == "FAILED"

    def test_null_imbalance_ratio_passthrough(self, agent, balanced_csv):
        """None imbalance_ratio should pass through."""
        result = agent.process({
            "dataset_path": balanced_csv,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "eda_report": {
                "is_imbalanced": True,
                "imbalance_ratio": None,
            },
        })
        assert result.status == "success"
        report = AugmentationReport(**result.data)
        assert report.augmented is False


# ---------------------------------------------------------------------------
# Tests: AugmentationReport serialization
# ---------------------------------------------------------------------------

class TestAugmentationReportSerialization:
    def test_default_report(self):
        report = AugmentationReport()
        assert report.augmented is False
        assert report.strategy_used == "none"
        assert report.samples_generated == 0

    def test_round_trip(self):
        report = AugmentationReport(
            augmented=True,
            strategy_used="smote",
            original_class_counts={"0": 180, "1": 20},
            augmented_class_counts={"0": 180, "1": 100},
            samples_generated=80,
            augmented_dataset_path="/tmp/augmented.parquet",
            adversarial_validation_score=0.55,
            quality_score=0.45,
            llm_quality_review="Augmentation quality is acceptable.",
        )
        dumped = report.model_dump()
        restored = AugmentationReport(**dumped)
        assert restored.augmented is True
        assert restored.strategy_used == "smote"
        assert restored.samples_generated == 80
        assert restored.original_class_counts == {"0": 180, "1": 20}
        assert restored.augmented_class_counts == {"0": 180, "1": 100}
        assert restored.adversarial_validation_score == 0.55
        assert restored.quality_score == 0.45
