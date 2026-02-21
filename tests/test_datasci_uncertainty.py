"""Tests for uncertainty quantification and routing."""

from __future__ import annotations

import numpy as np
import pytest

from src.datasci.uncertainty import (
    ConformalPredictor,
    compute_ensemble_uncertainty,
    compute_prediction_disagreement,
    route_by_uncertainty,
)


class TestPredictionDisagreement:
    def test_perfect_agreement_classification(self):
        preds = {
            "model_a": np.array([0, 1, 0, 1, 0]),
            "model_b": np.array([0, 1, 0, 1, 0]),
            "model_c": np.array([0, 1, 0, 1, 0]),
        }
        disagreement = compute_prediction_disagreement(preds)
        assert np.allclose(disagreement, 0.0)

    def test_total_disagreement_classification(self):
        preds = {
            "model_a": np.array([0, 0, 0]),
            "model_b": np.array([1, 1, 1]),
            "model_c": np.array([2, 2, 2]),
        }
        disagreement = compute_prediction_disagreement(preds)
        # Each model disagrees with the others — 2/3 disagree with majority
        assert np.allclose(disagreement, 2 / 3)

    def test_partial_disagreement_classification(self):
        preds = {
            "model_a": np.array([0, 1, 0]),
            "model_b": np.array([0, 1, 1]),
            "model_c": np.array([0, 0, 1]),
        }
        disagreement = compute_prediction_disagreement(preds)
        # Sample 0: all agree → 0
        assert disagreement[0] == pytest.approx(0.0)
        # Sample 1: 2 agree (1), 1 disagrees (0) → 1/3
        assert disagreement[1] == pytest.approx(1 / 3)
        # Sample 2: 2 agree (1), 1 disagrees (0) → 1/3
        assert disagreement[2] == pytest.approx(1 / 3)

    def test_single_model(self):
        preds = {"model_a": np.array([0, 1, 0, 1])}
        disagreement = compute_prediction_disagreement(preds)
        assert np.allclose(disagreement, 0.0)

    def test_regression_predictions(self):
        preds = {
            "model_a": np.array([10.0, 20.0, 30.0]),
            "model_b": np.array([10.0, 20.0, 30.0]),
        }
        disagreement = compute_prediction_disagreement(preds)
        # Perfect agreement → zero disagreement
        assert np.allclose(disagreement, 0.0)

    def test_regression_high_disagreement(self):
        preds = {
            "model_a": np.array([10.0, 20.0, 30.0]),
            "model_b": np.array([50.0, 80.0, 90.0]),
        }
        disagreement = compute_prediction_disagreement(preds)
        # High CV → high disagreement
        assert np.all(disagreement > 0.0)

    def test_output_shape(self):
        preds = {
            "m1": np.array([1, 2, 3, 4, 5]),
            "m2": np.array([1, 2, 3, 4, 5]),
        }
        disagreement = compute_prediction_disagreement(preds)
        assert disagreement.shape == (5,)


class TestRouteByUncertainty:
    def test_all_confident(self):
        scores = np.array([0.0, 0.1, 0.05, 0.15])
        result = route_by_uncertainty(scores, threshold=0.3)
        assert np.all(result["confident"])
        assert not np.any(result["uncertain"])

    def test_all_uncertain(self):
        scores = np.array([0.5, 0.8, 0.9, 0.4])
        result = route_by_uncertainty(scores, threshold=0.3)
        assert np.all(result["uncertain"])
        assert not np.any(result["confident"])

    def test_mixed_routing(self):
        scores = np.array([0.1, 0.5, 0.2, 0.8, 0.05])
        result = route_by_uncertainty(scores, threshold=0.3)
        assert result["confident"][0] is np.True_
        assert result["uncertain"][1] is np.True_
        assert result["confident"][2] is np.True_
        assert result["uncertain"][3] is np.True_
        assert result["confident"][4] is np.True_

    def test_threshold_boundary(self):
        scores = np.array([0.3])
        result = route_by_uncertainty(scores, threshold=0.3)
        assert result["uncertain"][0] is np.True_  # >= threshold

    def test_returns_scores(self):
        scores = np.array([0.1, 0.5])
        result = route_by_uncertainty(scores)
        assert np.array_equal(result["uncertainty_scores"], scores)


class TestEnsembleUncertainty:
    def test_confident_predictions(self):
        probs = {
            "m1": np.array([[0.99, 0.01], [0.01, 0.99]]),
            "m2": np.array([[0.98, 0.02], [0.02, 0.98]]),
        }
        entropy = compute_ensemble_uncertainty(probs)
        assert entropy.shape == (2,)
        # Very confident → low entropy
        assert np.all(entropy < 0.5)

    def test_uncertain_predictions(self):
        probs = {
            "m1": np.array([[0.5, 0.5], [0.5, 0.5]]),
            "m2": np.array([[0.5, 0.5], [0.5, 0.5]]),
        }
        entropy = compute_ensemble_uncertainty(probs)
        # Maximum uncertainty for binary → entropy = 1.0
        assert np.allclose(entropy, 1.0, atol=0.01)

    def test_empty_input(self):
        result = compute_ensemble_uncertainty({})
        assert len(result) == 0

    def test_single_model(self):
        probs = {"m1": np.array([[0.8, 0.2], [0.3, 0.7]])}
        entropy = compute_ensemble_uncertainty(probs)
        assert entropy.shape == (2,)

    def test_binary_1d(self):
        probs = {"m1": np.array([0.9, 0.5, 0.1])}
        entropy = compute_ensemble_uncertainty(probs)
        assert entropy.shape == (3,)
        # 0.5 should have highest entropy
        assert entropy[1] > entropy[0]
        assert entropy[1] > entropy[2]


class TestConformalPredictor:
    def test_calibrate_and_predict(self):
        cp = ConformalPredictor(alpha=0.10)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3, 5.9, 7.2, 8.0, 9.1, 10.2])
        cp.calibrate(y_true, y_pred)

        result = cp.predict(np.array([5.0, 10.0]))
        assert "lower" in result
        assert "upper" in result
        assert "width" in result
        assert result["lower"].shape == (2,)
        assert result["upper"].shape == (2,)
        assert np.all(result["upper"] > result["lower"])

    def test_predict_without_calibration_raises(self):
        cp = ConformalPredictor()
        with pytest.raises(RuntimeError, match="calibrated"):
            cp.predict(np.array([1.0, 2.0]))

    def test_coverage_target(self):
        np.random.seed(42)
        n = 1000
        y_true = np.random.normal(0, 1, n)
        noise = np.random.normal(0, 0.5, n)
        y_pred = y_true + noise

        # Split into calibration and test
        cal_true, test_true = y_true[:500], y_true[500:]
        cal_pred, test_pred = y_pred[:500], y_pred[500:]

        cp = ConformalPredictor(alpha=0.10)
        cp.calibrate(cal_true, cal_pred)

        coverage = cp.evaluate_coverage(test_true, test_pred)
        # Conformal prediction guarantees coverage >= 1 - alpha
        assert coverage["empirical_coverage"] >= 0.85  # Allow some slack for finite samples
        assert coverage["target_coverage"] == 0.90
        assert coverage["avg_width"] > 0

    def test_tighter_alpha(self):
        np.random.seed(42)
        n = 500
        y_true = np.random.normal(0, 1, n)
        y_pred = y_true + np.random.normal(0, 0.3, n)

        cp_loose = ConformalPredictor(alpha=0.20)
        cp_tight = ConformalPredictor(alpha=0.05)

        cp_loose.calibrate(y_true, y_pred)
        cp_tight.calibrate(y_true, y_pred)

        result_loose = cp_loose.predict(np.array([0.0]))
        result_tight = cp_tight.predict(np.array([0.0]))

        # Tighter alpha → wider intervals
        assert result_tight["width"][0] > result_loose["width"][0]

    def test_evaluate_coverage_keys(self):
        cp = ConformalPredictor(alpha=0.10)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        cp.calibrate(y_true, y_pred)

        coverage = cp.evaluate_coverage(y_true, y_pred)
        assert "empirical_coverage" in coverage
        assert "target_coverage" in coverage
        assert "avg_width" in coverage
