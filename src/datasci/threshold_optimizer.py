"""Decision threshold optimization for imbalanced classification.

Provides multiple strategies to find the optimal classification threshold
instead of using the default 0.5:
- Youden's J statistic (maximize TPR - FPR)
- F-beta optimization (maximize F-beta score)
- Cost-sensitive optimization (minimize weighted misclassification cost)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("associate.datasci.threshold_optimizer")


class ThresholdOptimizer:
    """Optimize classification decision threshold on validation data."""

    def optimize_youden(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> dict[str, Any]:
        """Youden's J statistic: argmax(TPR - FPR).

        Best general-purpose threshold for balanced sensitivity/specificity.

        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities for the positive class.

        Returns:
            Dict with threshold, method, j_score, tpr, fpr.
        """
        from sklearn.metrics import roc_curve

        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        fpr, tpr, thresholds = roc_curve(y_true, y_proba)

        if len(thresholds) == 0:
            return {
                "threshold": 0.5,
                "method": "youden",
                "j_score": 0.0,
                "tpr": 0.0,
                "fpr": 0.0,
            }

        j_scores = tpr - fpr
        optimal_idx = int(np.argmax(j_scores))

        return {
            "threshold": float(thresholds[optimal_idx]),
            "method": "youden",
            "j_score": float(j_scores[optimal_idx]),
            "tpr": float(tpr[optimal_idx]),
            "fpr": float(fpr[optimal_idx]),
        }

    def optimize_f_beta(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        beta: float = 1.0,
    ) -> dict[str, Any]:
        """Optimal threshold for F-beta score.

        beta < 1 favors precision, beta > 1 favors recall.

        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities for the positive class.
            beta: Beta parameter for F-beta score.

        Returns:
            Dict with threshold, method, f_score, precision, recall.
        """
        from sklearn.metrics import precision_recall_curve

        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        if len(thresholds) == 0:
            return {
                "threshold": 0.5,
                "method": f"f{beta}",
                "f_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        # F-beta: (1 + beta^2) * P * R / (beta^2 * P + R)
        # precision_recall_curve returns one extra element for precision/recall
        f_scores = (
            (1 + beta**2)
            * precision[:-1]
            * recall[:-1]
            / (beta**2 * precision[:-1] + recall[:-1] + 1e-10)
        )

        optimal_idx = int(np.argmax(f_scores))

        return {
            "threshold": float(thresholds[optimal_idx]),
            "method": f"f{beta}",
            "f_score": float(f_scores[optimal_idx]),
            "precision": float(precision[optimal_idx]),
            "recall": float(recall[optimal_idx]),
        }

    def optimize_cost_sensitive(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        fp_cost: float = 1.0,
        fn_cost: float = 1.0,
    ) -> dict[str, Any]:
        """Cost-sensitive threshold optimization.

        Minimizes total misclassification cost: fp_cost * FP + fn_cost * FN.

        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities for the positive class.
            fp_cost: Cost of a false positive.
            fn_cost: Cost of a false negative.

        Returns:
            Dict with threshold, method, total_cost, fp_count, fn_count.
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        # Evaluate cost across candidate thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        best_threshold = 0.5
        best_cost = float("inf")
        best_fp = 0
        best_fn = 0

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            cost = fp_cost * fp + fn_cost * fn

            if cost < best_cost:
                best_cost = cost
                best_threshold = float(t)
                best_fp = fp
                best_fn = fn

        return {
            "threshold": best_threshold,
            "method": "cost_sensitive",
            "total_cost": best_cost,
            "fp_count": best_fp,
            "fn_count": best_fn,
            "fp_cost": fp_cost,
            "fn_cost": fn_cost,
        }

    def find_optimal(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        method: str = "youden",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Dispatch to the requested optimization method.

        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities for the positive class.
            method: One of 'youden', 'f_beta', 'cost_sensitive'.
            **kwargs: Additional arguments passed to the method.

        Returns:
            Dict with threshold, method, and method-specific metrics.
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        # Validate inputs
        if len(y_true) == 0 or len(y_proba) == 0:
            logger.warning("Empty arrays passed to threshold optimizer")
            return {"threshold": 0.5, "method": method}

        if len(np.unique(y_true)) < 2:
            logger.warning("Single class in y_true; threshold optimization not meaningful")
            return {"threshold": 0.5, "method": method}

        if method == "youden":
            return self.optimize_youden(y_true, y_proba)
        elif method == "f_beta":
            beta = kwargs.get("beta", 1.0)
            return self.optimize_f_beta(y_true, y_proba, beta=beta)
        elif method == "cost_sensitive":
            fp_cost = kwargs.get("fp_cost", 1.0)
            fn_cost = kwargs.get("fn_cost", 1.0)
            return self.optimize_cost_sensitive(
                y_true, y_proba, fp_cost=fp_cost, fn_cost=fn_cost,
            )
        else:
            logger.warning("Unknown threshold method '%s'; defaulting to youden", method)
            return self.optimize_youden(y_true, y_proba)
