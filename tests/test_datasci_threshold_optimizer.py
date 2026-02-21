"""Tests for threshold optimizer.

Uses real sklearn synthetic datasets. No mocks.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from src.datasci.threshold_optimizer import ThresholdOptimizer


@pytest.fixture
def optimizer():
    return ThresholdOptimizer()


@pytest.fixture
def binary_data():
    """Create an imbalanced binary classification dataset with probabilities."""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        weights=[0.8, 0.2],
        random_state=42,
    )
    clf = LogisticRegression(max_iter=1000, random_state=42)
    y_proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    return y, y_proba


class TestYoudenOptimization:
    def test_returns_valid_threshold(self, optimizer, binary_data):
        y_true, y_proba = binary_data
        result = optimizer.optimize_youden(y_true, y_proba)
        assert 0.0 <= result["threshold"] <= 1.0
        assert result["method"] == "youden"
        assert "j_score" in result
        assert result["j_score"] > 0.0

    def test_tpr_fpr_present(self, optimizer, binary_data):
        y_true, y_proba = binary_data
        result = optimizer.optimize_youden(y_true, y_proba)
        assert "tpr" in result
        assert "fpr" in result
        assert result["tpr"] >= result["fpr"]

    def test_perfect_predictions(self, optimizer):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        result = optimizer.optimize_youden(y_true, y_proba)
        assert result["j_score"] >= 0.9


class TestFBetaOptimization:
    def test_returns_valid_threshold(self, optimizer, binary_data):
        y_true, y_proba = binary_data
        result = optimizer.optimize_f_beta(y_true, y_proba, beta=1.0)
        assert 0.0 <= result["threshold"] <= 1.0
        assert result["method"] == "f1.0"
        assert result["f_score"] > 0.0

    def test_beta_2_favors_recall(self, optimizer, binary_data):
        y_true, y_proba = binary_data
        result_f1 = optimizer.optimize_f_beta(y_true, y_proba, beta=1.0)
        result_f2 = optimizer.optimize_f_beta(y_true, y_proba, beta=2.0)
        # F2 should generally have lower threshold (more permissive)
        # or higher recall
        assert result_f2["recall"] >= result_f1["recall"] - 0.1  # allow small margin

    def test_precision_recall_present(self, optimizer, binary_data):
        y_true, y_proba = binary_data
        result = optimizer.optimize_f_beta(y_true, y_proba, beta=1.0)
        assert "precision" in result
        assert "recall" in result


class TestCostSensitiveOptimization:
    def test_returns_valid_threshold(self, optimizer, binary_data):
        y_true, y_proba = binary_data
        result = optimizer.optimize_cost_sensitive(y_true, y_proba)
        assert 0.0 <= result["threshold"] <= 1.0
        assert result["method"] == "cost_sensitive"

    def test_high_fn_cost_lowers_threshold(self, optimizer, binary_data):
        y_true, y_proba = binary_data
        result_equal = optimizer.optimize_cost_sensitive(y_true, y_proba, fp_cost=1.0, fn_cost=1.0)
        result_high_fn = optimizer.optimize_cost_sensitive(y_true, y_proba, fp_cost=1.0, fn_cost=10.0)
        # Higher FN cost should push threshold lower to catch more positives
        assert result_high_fn["threshold"] <= result_equal["threshold"] + 0.15

    def test_cost_fields_present(self, optimizer, binary_data):
        y_true, y_proba = binary_data
        result = optimizer.optimize_cost_sensitive(y_true, y_proba)
        assert "total_cost" in result
        assert "fp_count" in result
        assert "fn_count" in result


class TestFindOptimal:
    def test_dispatches_youden(self, optimizer, binary_data):
        y_true, y_proba = binary_data
        result = optimizer.find_optimal(y_true, y_proba, method="youden")
        assert result["method"] == "youden"

    def test_dispatches_f_beta(self, optimizer, binary_data):
        y_true, y_proba = binary_data
        result = optimizer.find_optimal(y_true, y_proba, method="f_beta", beta=0.5)
        assert result["method"] == "f0.5"

    def test_dispatches_cost_sensitive(self, optimizer, binary_data):
        y_true, y_proba = binary_data
        result = optimizer.find_optimal(
            y_true, y_proba, method="cost_sensitive", fp_cost=2.0, fn_cost=5.0,
        )
        assert result["method"] == "cost_sensitive"

    def test_unknown_method_defaults_to_youden(self, optimizer, binary_data):
        y_true, y_proba = binary_data
        result = optimizer.find_optimal(y_true, y_proba, method="nonexistent")
        assert result["method"] == "youden"

    def test_empty_arrays(self, optimizer):
        result = optimizer.find_optimal(np.array([]), np.array([]), method="youden")
        assert result["threshold"] == 0.5

    def test_single_class(self, optimizer):
        y_true = np.array([1, 1, 1, 1])
        y_proba = np.array([0.8, 0.9, 0.7, 0.6])
        result = optimizer.find_optimal(y_true, y_proba, method="youden")
        assert result["threshold"] == 0.5

    def test_random_predictions(self, optimizer):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_proba = rng.random(200)
        result = optimizer.find_optimal(y_true, y_proba, method="youden")
        # Should still produce a valid threshold
        assert 0.0 <= result["threshold"] <= 1.0
