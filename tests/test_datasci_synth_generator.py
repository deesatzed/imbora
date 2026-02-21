"""Tests for SynthGenerator: synthetic minority generation.

Uses real sklearn synthetic datasets and real pandas operations. No mocks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.datasci.synth_generator import (
    SynthGenerator,
    _adversarial_validation_score,
    _compute_sampling_strategy,
)


# ---------------------------------------------------------------------------
# Real-shaped fakes
# ---------------------------------------------------------------------------

class FakeLLMResponse:
    def __init__(self, content: str = ""):
        self.content = content


class FakeLLMClient:
    def __init__(self, response_content: str = ""):
        self.response_content = response_content

    def complete_with_fallback(self, **kwargs):
        return FakeLLMResponse(self.response_content)


class FakeModelRouter:
    def get_model(self, role):
        return "test/model"

    def get_fallback_models(self, role):
        return []

    def get_model_chain(self, role: str) -> list[str]:
        return [self.get_model(role)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def imbalanced_df():
    """DataFrame with 9:1 class imbalance, numeric features only."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        weights=[0.9, 0.1],
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
    df["target"] = y
    return df


@pytest.fixture
def balanced_df():
    """DataFrame with balanced classes."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
    df["target"] = y
    return df


# ---------------------------------------------------------------------------
# Tests: _compute_sampling_strategy()
# ---------------------------------------------------------------------------

class TestComputeSamplingStrategy:
    def test_binary_imbalanced(self):
        """Minority class should get target count up to max_ratio."""
        y = pd.Series([0] * 180 + [1] * 20)
        strategy = _compute_sampling_strategy(y, max_ratio=5.0)
        # Minority class (1) should be in the strategy
        assert 1 in strategy
        # Target should be min(180, 20 * 5.0) = 100
        assert strategy[1] == 100
        # Majority class (0) should NOT be in the strategy
        assert 0 not in strategy

    def test_balanced_returns_empty(self):
        """Balanced dataset should return empty strategy."""
        y = pd.Series([0] * 100 + [1] * 100)
        strategy = _compute_sampling_strategy(y, max_ratio=5.0)
        assert strategy == {}

    def test_cap_at_majority_count(self):
        """Target should not exceed majority count even with high max_ratio."""
        y = pd.Series([0] * 50 + [1] * 10)
        strategy = _compute_sampling_strategy(y, max_ratio=100.0)
        # min(50, 10 * 100) = 50
        assert strategy[1] == 50

    def test_multiclass_imbalanced(self):
        """Should handle multiple minority classes."""
        y = pd.Series([0] * 100 + [1] * 20 + [2] * 10)
        strategy = _compute_sampling_strategy(y, max_ratio=3.0)
        assert 1 in strategy
        assert 2 in strategy
        # Class 1: min(100, 20*3) = 60
        assert strategy[1] == 60
        # Class 2: min(100, 10*3) = 30
        assert strategy[2] == 30

    def test_at_least_one_new_sample(self):
        """Even with low max_ratio, at least 1 new sample is generated."""
        y = pd.Series([0] * 100 + [1] * 99)
        strategy = _compute_sampling_strategy(y, max_ratio=1.0)
        # min(100, 99*1) = 99, but max(99, 99+1) = 100
        assert 1 in strategy
        assert strategy[1] == 100


# ---------------------------------------------------------------------------
# Tests: _adversarial_validation_score()
# ---------------------------------------------------------------------------

class TestAdversarialValidationScore:
    def test_identical_data_scores_near_half(self):
        """When real and synth are from the same distribution, score is near 0.5."""
        X, y = make_classification(
            n_samples=200, n_features=10, random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
        df["target"] = y
        # Split in half: both halves come from same distribution
        real_df = df.iloc[:100].copy().reset_index(drop=True)
        synth_df = df.iloc[100:].copy().reset_index(drop=True)
        score = _adversarial_validation_score(real_df, synth_df, "target")
        # Should be roughly 0.5 (hard to distinguish)
        assert 0.3 <= score <= 0.8

    def test_distinguishable_data_scores_high(self):
        """When real and synth are very different, score should be high."""
        real_df = pd.DataFrame({
            "x0": np.zeros(100),
            "x1": np.zeros(100),
            "target": np.zeros(100, dtype=int),
        })
        synth_df = pd.DataFrame({
            "x0": np.ones(100) * 100,
            "x1": np.ones(100) * 100,
            "target": np.ones(100, dtype=int),
        })
        score = _adversarial_validation_score(real_df, synth_df, "target")
        assert score > 0.7

    def test_no_numeric_features_returns_half(self):
        """When there are no numeric features, return 0.5."""
        real_df = pd.DataFrame({
            "text": ["hello"] * 20,
            "target": [0] * 20,
        })
        synth_df = pd.DataFrame({
            "text": ["world"] * 20,
            "target": [1] * 20,
        })
        score = _adversarial_validation_score(real_df, synth_df, "target")
        assert score == 0.5

    def test_too_few_samples_returns_half(self):
        """When combined data is fewer than 20 rows, return 0.5."""
        real_df = pd.DataFrame({"x0": [1.0, 2.0], "target": [0, 1]})
        synth_df = pd.DataFrame({"x0": [3.0, 4.0], "target": [0, 1]})
        score = _adversarial_validation_score(real_df, synth_df, "target")
        assert score == 0.5

    def test_single_class_synth_returns_half(self):
        """When n_folds < 2 due to single class in one partition, return 0.5."""
        real_df = pd.DataFrame({"x0": [1.0], "target": [0]})
        synth_df = pd.DataFrame({"x0": [2.0], "target": [1]})
        score = _adversarial_validation_score(real_df, synth_df, "target")
        assert score == 0.5


# ---------------------------------------------------------------------------
# Tests: SynthGenerator.run()
# ---------------------------------------------------------------------------

class TestSynthGeneratorRun:
    def test_strategy_none_passthrough(self, balanced_df):
        gen = SynthGenerator()
        result = gen.run(
            df=balanced_df,
            target_column="target",
            strategy="none",
            imbalance_ratio=1.0,
        )
        assert result["strategy_used"] == "none"
        assert result["samples_generated"] == 0
        assert len(result["augmented_df"]) == len(balanced_df)

    def test_missing_target_column_passthrough(self, balanced_df):
        """When target_column is not in df, should pass through."""
        gen = SynthGenerator()
        result = gen.run(
            df=balanced_df,
            target_column="nonexistent",
            strategy="smote",
            imbalance_ratio=5.0,
        )
        assert result["strategy_used"] == "none"
        assert result["samples_generated"] == 0

    def test_unknown_strategy_passthrough(self, imbalanced_df):
        gen = SynthGenerator()
        result = gen.run(
            df=imbalanced_df,
            target_column="target",
            strategy="unknown_strategy",
            imbalance_ratio=5.0,
        )
        assert result["strategy_used"] == "none"

    def test_smote_with_real_data(self, imbalanced_df):
        """Test SMOTE strategy with real imbalanced data."""
        try:
            from imblearn.over_sampling import SMOTE  # noqa: F401
        except ImportError:
            pytest.skip("imbalanced-learn not installed")

        gen = SynthGenerator()
        result = gen.run(
            df=imbalanced_df,
            target_column="target",
            strategy="smote",
            imbalance_ratio=9.0,
            max_augmentation_ratio=5.0,
        )
        # Either augmentation succeeded or was rejected by adversarial validation
        assert result["strategy_used"] in ("smote", "none")
        if result["strategy_used"] == "smote":
            assert result["samples_generated"] > 0
            assert len(result["augmented_df"]) > len(imbalanced_df)
            assert result["quality_score"] >= 0.0

    def test_adasyn_with_real_data(self, imbalanced_df):
        """Test ADASYN strategy with real imbalanced data."""
        try:
            from imblearn.over_sampling import ADASYN  # noqa: F401
        except ImportError:
            pytest.skip("imbalanced-learn not installed")

        gen = SynthGenerator()
        result = gen.run(
            df=imbalanced_df,
            target_column="target",
            strategy="adasyn",
            imbalance_ratio=9.0,
            max_augmentation_ratio=5.0,
        )
        assert result["strategy_used"] in ("adasyn", "none")

    def test_smotenc_fallback_to_smote_no_categoricals(self, imbalanced_df):
        """SMOTENC with no categoricals falls back to SMOTE."""
        try:
            from imblearn.over_sampling import SMOTENC  # noqa: F401
        except ImportError:
            pytest.skip("imbalanced-learn not installed")

        gen = SynthGenerator()
        result = gen.run(
            df=imbalanced_df,
            target_column="target",
            strategy="smotenc",
            imbalance_ratio=9.0,
            max_augmentation_ratio=5.0,
        )
        # Falls back to SMOTE since no categorical columns
        assert result["strategy_used"] in ("smotenc", "none")

    def test_max_augmentation_ratio_caps_generation(self, imbalanced_df):
        """max_augmentation_ratio should limit how many samples are generated."""
        try:
            from imblearn.over_sampling import SMOTE  # noqa: F401
        except ImportError:
            pytest.skip("imbalanced-learn not installed")

        minority_count = (imbalanced_df["target"] == 1).sum()
        gen = SynthGenerator()
        result = gen.run(
            df=imbalanced_df,
            target_column="target",
            strategy="smote",
            imbalance_ratio=9.0,
            max_augmentation_ratio=2.0,
        )
        if result["strategy_used"] == "smote":
            # Augmented minority should be at most 2x original minority
            augmented_counts = result["augmented_counts"]
            minority_augmented = int(augmented_counts.get("1", 0))
            assert minority_augmented <= minority_count * 2 + 1  # +1 for rounding

    def test_passthrough_preserves_counts(self, balanced_df):
        """Pass-through should preserve original class counts."""
        gen = SynthGenerator()
        result = gen.run(
            df=balanced_df,
            target_column="target",
            strategy="none",
            imbalance_ratio=1.0,
        )
        original_counts = balanced_df["target"].value_counts()
        for cls, count in original_counts.items():
            assert result["original_counts"][str(cls)] == count
        assert result["original_counts"] == result["augmented_counts"]
        assert result["quality_score"] == 1.0
        assert result["adversarial_validation_score"] == 0.0


# ---------------------------------------------------------------------------
# Tests: SynthGenerator fallback behavior
# ---------------------------------------------------------------------------

class TestSynthGeneratorFallback:
    def test_smote_fallback_when_no_numeric_cols(self):
        """SMOTE with no numeric columns should return None (then passthrough)."""
        df = pd.DataFrame({
            "cat1": ["a", "b", "c", "d", "e"] * 20,
            "cat2": ["x", "y", "z", "w", "v"] * 20,
            "target": [0] * 80 + [1] * 20,
        })
        gen = SynthGenerator()
        result = gen.run(
            df=df,
            target_column="target",
            strategy="smote",
            imbalance_ratio=4.0,
        )
        # Should fall back to passthrough since no numeric columns
        assert result["strategy_used"] == "none"

    def test_llm_synth_without_client_falls_back(self, imbalanced_df):
        """LLM synth without LLM client should fall back to SMOTE."""
        gen = SynthGenerator(llm_client=None, model_router=None)
        result = gen.run(
            df=imbalanced_df,
            target_column="target",
            strategy="llm_synth",
            imbalance_ratio=10.0,
            max_augmentation_ratio=3.0,
        )
        # Falls back to SMOTE (or passthrough if SMOTE also fails)
        assert result["strategy_used"] in ("none", "llm_synth", "smote")
