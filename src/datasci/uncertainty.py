"""Uncertainty quantification and routing.

Provides conformal prediction wrappers (via mapie when available),
prediction disagreement scoring, and uncertainty-based model routing.
Pure computation — no LLM calls.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("associate.datasci.uncertainty")


def compute_prediction_disagreement(
    predictions: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute per-sample disagreement across model predictions.

    For classification: fraction of models that disagree with the majority vote.
    For regression: coefficient of variation across predictions.

    Args:
        predictions: Dict mapping model_name → array of predictions (n_samples,).

    Returns:
        Array of disagreement scores (n_samples,), in [0, 1].
    """
    if len(predictions) < 2:
        # Single model — no disagreement
        first = next(iter(predictions.values()))
        return np.zeros(len(first))

    pred_matrix = np.column_stack(list(predictions.values()))
    n_models = pred_matrix.shape[1]

    # Detect if classification (integer/string) or regression (float)
    is_classification = _is_classification_predictions(pred_matrix[:, 0])

    if is_classification:
        # For each sample, find majority vote and compute fraction disagreeing
        disagreement = np.zeros(pred_matrix.shape[0])
        for i in range(pred_matrix.shape[0]):
            row = pred_matrix[i, :]
            unique, counts = np.unique(row, return_counts=True)
            majority_count = counts.max()
            disagreement[i] = 1.0 - (majority_count / n_models)
        return disagreement
    else:
        # Coefficient of variation per sample
        means = np.mean(pred_matrix, axis=1)
        stds = np.std(pred_matrix, axis=1)
        # Avoid division by zero
        safe_means = np.where(np.abs(means) < 1e-10, 1e-10, means)
        cv = stds / np.abs(safe_means)
        # Normalize to [0, 1] using sigmoid-like transform
        return 1.0 - 1.0 / (1.0 + cv)


def route_by_uncertainty(
    disagreement_scores: np.ndarray,
    threshold: float = 0.3,
) -> dict[str, np.ndarray]:
    """Route samples by uncertainty level.

    Args:
        disagreement_scores: Per-sample disagreement scores.
        threshold: Disagreement threshold for high-uncertainty routing.

    Returns:
        Dict with 'confident' and 'uncertain' boolean masks.
    """
    uncertain_mask = disagreement_scores >= threshold
    return {
        "confident": ~uncertain_mask,
        "uncertain": uncertain_mask,
        "uncertainty_scores": disagreement_scores,
    }


def compute_ensemble_uncertainty(
    probabilities: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute ensemble uncertainty from predicted probabilities.

    Uses predictive entropy across model probability estimates.

    Args:
        probabilities: Dict mapping model_name → array of probabilities
                       (n_samples, n_classes).

    Returns:
        Array of uncertainty scores (n_samples,), higher = more uncertain.
    """
    if len(probabilities) < 1:
        return np.array([])

    prob_list = list(probabilities.values())
    # Average probabilities across models
    avg_probs = np.mean(prob_list, axis=0)

    # Predictive entropy: -sum(p * log(p))
    # Clip to avoid log(0)
    avg_probs = np.clip(avg_probs, 1e-10, 1.0)

    if avg_probs.ndim == 1:
        # Binary: treat as [p, 1-p]
        entropy = -(avg_probs * np.log2(avg_probs) + (1 - avg_probs) * np.log2(1 - avg_probs))
    else:
        entropy = -np.sum(avg_probs * np.log2(avg_probs), axis=1)

    return entropy


def _is_classification_predictions(sample: np.ndarray) -> bool:
    """Heuristic: check if predictions look like classification (integer values)."""
    if sample.dtype in (np.int32, np.int64, np.int16, np.int8):
        return True
    if np.issubdtype(sample.dtype, np.floating):
        # If all values are close to integers, likely classification
        return bool(np.all(np.abs(sample - np.round(sample)) < 1e-6))
    return True  # Default to classification for string-like


class ConformalPredictor:
    """Wrapper for conformal prediction intervals.

    Uses mapie when available, falls back to quantile-based intervals.
    Provides distribution-free coverage guarantees.
    """

    def __init__(self, alpha: float = 0.10):
        """Initialize with desired miscoverage rate.

        Args:
            alpha: Target miscoverage rate. Coverage = 1 - alpha.
                   E.g., alpha=0.10 targets 90% coverage.
        """
        self.alpha = alpha
        self._calibration_scores: np.ndarray | None = None
        self._quantile: float | None = None

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """Calibrate conformal predictor on calibration set.

        Computes nonconformity scores and the quantile threshold.

        Args:
            y_true: True labels/values for calibration set.
            y_pred: Predicted values for calibration set.
        """
        self._calibration_scores = np.abs(y_true - y_pred)
        n = len(self._calibration_scores)
        # Compute the (1-alpha)(1 + 1/n) quantile
        level = min(1.0, (1 - self.alpha) * (1 + 1 / n))
        self._quantile = float(np.quantile(self._calibration_scores, level))
        logger.info(
            "Conformal predictor calibrated: alpha=%.3f, n=%d, quantile=%.4f",
            self.alpha, n, self._quantile,
        )

    def predict(
        self,
        y_pred: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Generate prediction intervals.

        Args:
            y_pred: Point predictions for test set.

        Returns:
            Dict with 'lower', 'upper', 'width' arrays.

        Raises:
            RuntimeError: If not calibrated.
        """
        if self._quantile is None:
            raise RuntimeError("ConformalPredictor must be calibrated before predict()")

        lower = y_pred - self._quantile
        upper = y_pred + self._quantile
        width = upper - lower

        return {
            "lower": lower,
            "upper": upper,
            "width": width,
            "quantile": self._quantile,
        }

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate empirical coverage on a test set.

        Args:
            y_true: True values.
            y_pred: Point predictions.

        Returns:
            Dict with 'empirical_coverage', 'target_coverage', 'avg_width'.
        """
        intervals = self.predict(y_pred)
        covered = (y_true >= intervals["lower"]) & (y_true <= intervals["upper"])
        return {
            "empirical_coverage": float(covered.mean()),
            "target_coverage": 1.0 - self.alpha,
            "avg_width": float(intervals["width"].mean()),
        }
