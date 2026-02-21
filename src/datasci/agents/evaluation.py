"""Phase 6: Evaluation Agent.

Multi-dimensional model evaluation across predictive performance,
robustness, fairness, and interpretability. Uses real sklearn metrics,
numpy noise injection, pandas groupby for fairness analysis, and
LLM for narrative evaluation. Produces an EvaluationReport.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import DataScienceConfig
from src.core.models import AgentResult
from src.datasci.agents.base_ds_agent import BaseDataScienceAgent
from src.datasci.models import (
    EnsembleReport,
    EvaluationReport,
    ModelTrainingReport,
)

logger = logging.getLogger("associate.datasci.evaluation")

EVALUATION_SYSTEM_PROMPT = (
    "You are an expert ML evaluator producing a comprehensive model assessment.\n"
    "Given multi-dimensional evaluation results (predictive scores, robustness,\n"
    "fairness, interpretability, feature importance, and baseline comparison),\n"
    "produce a narrative evaluation covering:\n"
    "1. Predictive quality: Are scores competitive? How far above baseline?\n"
    "2. Robustness: How much do scores degrade under noise? Is the model fragile?\n"
    "3. Fairness: Any significant performance gaps across sensitive groups?\n"
    "4. Interpretability: Is feature importance concentrated or distributed?\n"
    "5. Overall assessment: Grade justification, deployment readiness, risks.\n"
    "6. Recommendations: What should be done before production deployment?\n\n"
    "Reference specific numbers from the evaluation. Be concise but thorough."
)

# Grade thresholds: weighted score -> letter grade
GRADE_THRESHOLDS = [
    (0.90, "A"),
    (0.80, "B"),
    (0.65, "C"),
    (0.50, "D"),
    (0.00, "F"),
]

# Dimension weights for overall grade
DIMENSION_WEIGHTS = {
    "predictive": 0.40,
    "robustness": 0.25,
    "fairness": 0.15,
    "interpretability": 0.20,
}


def _load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from supported formats."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    elif suffix == ".tsv":
        return pd.read_csv(p, sep="\t")
    elif suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")


def _build_sklearn_estimator(
    model_name: str, problem_type: str, imbalance_ratio: float = 1.0,
) -> Any | None:
    """Build an sklearn-compatible estimator for the given model type.

    Uses try/except ImportError for all optional dependencies.
    Returns None if the required package is not installed.

    Args:
        model_name: Name of the model to build.
        problem_type: 'classification' or 'regression'.
        imbalance_ratio: Ratio of majority to minority class.
            When > 1.5 and problem_type is classification, class weight
            parameters are added to supported estimators.
    """
    apply_class_weights = (
        imbalance_ratio > 1.5 and problem_type == "classification"
    )

    if model_name == "xgboost":
        try:
            if problem_type == "classification":
                from xgboost import XGBClassifier
                params: dict[str, Any] = dict(
                    n_estimators=100, max_depth=6,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42, verbosity=0,
                )
                if apply_class_weights:
                    params["scale_pos_weight"] = imbalance_ratio
                return XGBClassifier(**params)
            else:
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators=100, max_depth=6,
                    learning_rate=0.1,
                    random_state=42, verbosity=0,
                )
        except ImportError:
            logger.info(
                "xgboost not installed; skipping XGBoost in evaluation",
            )
            return None

    elif model_name == "catboost":
        try:
            if problem_type == "classification":
                from catboost import CatBoostClassifier
                params = dict(
                    iterations=100, depth=6,
                    learning_rate=0.1,
                    random_seed=42, silent=True,
                )
                if apply_class_weights:
                    params["auto_class_weights"] = "Balanced"
                return CatBoostClassifier(**params)
            else:
                from catboost import CatBoostRegressor
                return CatBoostRegressor(
                    iterations=100, depth=6,
                    learning_rate=0.1,
                    random_seed=42, silent=True,
                )
        except ImportError:
            logger.info(
                "catboost not installed; skipping CatBoost in evaluation",
            )
            return None

    elif model_name == "lightgbm":
        try:
            if problem_type == "classification":
                from lightgbm import LGBMClassifier
                params = dict(
                    n_estimators=100, max_depth=6,
                    learning_rate=0.1,
                    random_state=42, verbose=-1,
                )
                if apply_class_weights:
                    params["is_unbalance"] = True
                return LGBMClassifier(**params)
            else:
                from lightgbm import LGBMRegressor
                return LGBMRegressor(
                    n_estimators=100, max_depth=6,
                    learning_rate=0.1,
                    random_state=42, verbose=-1,
                )
        except ImportError:
            logger.info(
                "lightgbm not installed; skipping LightGBM in evaluation",
            )
            return None

    elif model_name == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            if problem_type == "classification":
                return TabPFNClassifier(device="cpu")
            else:
                return TabPFNRegressor(device="cpu")
        except ImportError:
            logger.info(
                "tabpfn not installed; skipping TabPFN in evaluation",
            )
            return None

    elif model_name == "tabicl":
        try:
            from tabicl import TabICLClassifier, TabICLRegressor
            if problem_type == "classification":
                return TabICLClassifier()
            else:
                return TabICLRegressor()
        except ImportError:
            logger.info(
                "tabicl not installed; skipping TabICL in evaluation",
            )
            return None

    elif model_name == "nam":
        try:
            from dnamite import NAMClassifier, NAMRegressor
            if problem_type == "classification":
                return NAMClassifier(random_state=42)
            else:
                return NAMRegressor(random_state=42)
        except ImportError:
            logger.info(
                "dnamite (NAM) not installed; skipping NAM in evaluation",
            )
            return None

    elif model_name == "self_paced_ensemble":
        try:
            from imbens.ensemble import SelfPacedEnsembleClassifier
            if problem_type == "classification":
                return SelfPacedEnsembleClassifier(
                    n_estimators=50, random_state=42,
                )
            return None
        except ImportError:
            logger.info("imbens not installed; skipping SelfPacedEnsemble")
            return None

    elif model_name == "easy_ensemble":
        try:
            from imbens.ensemble import EasyEnsembleClassifier
            if problem_type == "classification":
                return EasyEnsembleClassifier(
                    n_estimators=50, random_state=42,
                )
            return None
        except ImportError:
            logger.info("imbens not installed; skipping EasyEnsemble")
            return None

    else:
        logger.warning(
            "Unknown model type for evaluation: %s", model_name,
        )
        return None


class EvaluationAgent(BaseDataScienceAgent):
    """Agent for Phase 6: Multi-Dimensional Evaluation.

    Evaluates the best model across multiple dimensions:
    - Predictive performance (accuracy, F1, AUC / R2, MAE, RMSE)
    - Robustness (performance under Gaussian noise)
    - Fairness (per-group accuracy for sensitive columns)
    - Interpretability (feature importance concentration)
    - Baseline comparison (vs. majority class / mean predictor)
    - LLM evaluation narrative
    - Overall letter grade
    """

    def __init__(
        self,
        llm_client: Any,
        model_router: Any,
        repository: Any,
        ds_config: DataScienceConfig,
    ):
        super().__init__(
            name="EvaluationAgent",
            role="ds_evaluator",
            llm_client=llm_client,
            model_router=model_router,
            repository=repository,
            ds_config=ds_config,
        )

    def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Run the multi-dimensional evaluation pipeline.

        Args:
            input_data: Dict with keys:
                - dataset_path: str
                - target_column: str
                - project_id: UUID
                - problem_type: str ('classification' or 'regression')
                - training_report: dict (ModelTrainingReport)
                - ensemble_report: dict (EnsembleReport)
                - sensitive_columns: list[str] (optional)
                - eda_report: dict (optional, may contain imbalance_ratio)
                - task_id: UUID (optional)
                - run_id: UUID (optional)

        Returns:
            AgentResult with EvaluationReport in data.
        """
        dataset_path = input_data["dataset_path"]
        target_column = input_data["target_column"]
        project_id = input_data["project_id"]
        problem_type = input_data.get("problem_type", "classification")
        training_data = input_data.get("training_report", {})
        ensemble_data = input_data.get("ensemble_report", {})
        sensitive_columns = input_data.get("sensitive_columns", [])

        # Extract imbalance_ratio from EDA report
        imbalance_ratio: float = input_data.get(
            "eda_report", {},
        ).get("imbalance_ratio", 1.0)

        # Reconstruct reports from dicts
        training_report = ModelTrainingReport(**training_data)
        _ensemble_report = EnsembleReport(**ensemble_data)

        experiment_id = self._save_experiment(
            project_id=project_id,
            phase="evaluation",
            config={
                "target_column": target_column,
                "problem_type": problem_type,
                "best_candidate": training_report.best_candidate,
                "sensitive_columns": sensitive_columns,
                "n_candidates": len(training_report.candidates),
                "imbalance_ratio": imbalance_ratio,
            },
            task_id=input_data.get("task_id"),
            run_id=input_data.get("run_id"),
        )

        try:
            # Step 1: Load dataset and prepare data
            df = _load_dataset(dataset_path)
            features, target = self._prepare_data(df, target_column)
            logger.info(
                "Loaded dataset for evaluation: %d samples, %d features",
                features.shape[0], features.shape[1],
            )

            # Step 2: Train best model on full data for evaluation
            best_name = training_report.best_candidate
            if not best_name and training_report.candidates:
                best_name = max(
                    training_report.candidates,
                    key=lambda c: c.mean_score,
                ).model_name

            # Compare training best vs ensemble meta-learner
            ensemble_data = input_data.get("ensemble_report", {})
            if isinstance(ensemble_data, dict):
                arch = ensemble_data.get("stacking_architecture", {})
                ensemble_meta = arch.get("meta_cv_score")
                training_best_score = 0.0
                for c in training_report.candidates:
                    if c.model_name == best_name:
                        training_best_score = c.mean_score
                        break
                if (ensemble_meta is not None
                        and ensemble_meta > training_best_score):
                    logger.info(
                        "Ensemble meta-learner (%.4f) outperformed "
                        "training best %s (%.4f)",
                        ensemble_meta, best_name, training_best_score,
                    )

            # Look up model_type from the best candidate
            best_model_type = best_name  # fallback
            for c in training_report.candidates:
                if c.model_name == best_name:
                    best_model_type = c.model_type
                    break

            estimator = _build_sklearn_estimator(
                best_model_type, problem_type,
                imbalance_ratio=imbalance_ratio,
            )
            if estimator is None:
                # Fallback: try GradientBoosting from sklearn
                estimator = self._build_gradient_boosting(problem_type)

            from sklearn.model_selection import cross_val_predict, StratifiedKFold
            if problem_type == "classification":
                cv = StratifiedKFold(
                    n_splits=min(5, len(target)), shuffle=True, random_state=42,
                )
            else:
                cv = 5

            # Get cross-validated predictions and probabilities (honest, no leakage)
            cv_proba = None
            try:
                predictions = cross_val_predict(estimator, features, target, cv=cv)
                if problem_type == "classification" and hasattr(estimator, "predict_proba"):
                    try:
                        cv_proba = cross_val_predict(
                            estimator, features, target, cv=cv, method="predict_proba",
                        )
                    except Exception:
                        logger.debug("cross_val_predict with predict_proba failed; skipping")
                # Fit on full data for downstream uses (robustness, fairness, etc.)
                estimator.fit(features, target)
            except Exception:
                # Fallback to train-on-train if CV fails (e.g., too few samples)
                estimator.fit(features, target)
                predictions = estimator.predict(features)

            # Threshold optimization: use CV probabilities to find optimal threshold
            optimal_threshold = None
            if cv_proba is not None and problem_type == "classification":
                try:
                    from src.datasci.threshold_optimizer import ThresholdOptimizer
                    optimizer = ThresholdOptimizer()
                    if cv_proba.ndim == 2 and cv_proba.shape[1] == 2:
                        threshold_result = optimizer.find_optimal(
                            target.values, cv_proba[:, 1], method="youden",
                        )
                        optimal_threshold = threshold_result.get("threshold", 0.5)
                        # Recompute predictions at optimal threshold
                        predictions = (cv_proba[:, 1] >= optimal_threshold).astype(int)
                        logger.info(
                            "Threshold optimization: optimal=%.3f (method=%s, j=%.3f)",
                            optimal_threshold,
                            threshold_result.get("method", "youden"),
                            threshold_result.get("j_score", 0.0),
                        )
                except Exception as e:
                    logger.warning("Threshold optimization failed: %s", e)

            logger.info(
                "Trained %s for evaluation on %d samples",
                best_name or "GradientBoosting", features.shape[0],
            )

            # Step 3: Predictive scores
            predictive_scores = self._compute_predictive_scores(
                target=target,
                predictions=predictions,
                estimator=estimator,
                features=features,
                problem_type=problem_type,
                cv_proba=cv_proba,
            )
            if optimal_threshold is not None:
                predictive_scores["optimal_threshold"] = round(optimal_threshold, 6)
            logger.info("Predictive scores: %s", predictive_scores)

            # Step 4: Robustness scores
            robustness_scores = self._compute_robustness_scores(
                estimator=estimator,
                features=features,
                target=target,
                problem_type=problem_type,
                predictive_scores=predictive_scores,
            )
            logger.info("Robustness scores: %s", robustness_scores)

            # Step 5: Fairness scores
            fairness_scores = None
            if sensitive_columns:
                fairness_scores = self._compute_fairness_scores(
                    df=df,
                    target_column=target_column,
                    features=features,
                    estimator=estimator,
                    sensitive_columns=sensitive_columns,
                    problem_type=problem_type,
                )
                logger.info("Fairness scores: %s", fairness_scores)

            # Step 6: Interpretability scores
            interpretability_scores = self._compute_interpretability_scores(
                estimator=estimator,
                features=features,
                target=target,
                problem_type=problem_type,
            )
            logger.info(
                "Interpretability scores: %s", interpretability_scores,
            )

            # Step 7: Feature importance
            feature_importance = self._compute_feature_importance(
                features=features,
                target=target,
                problem_type=problem_type,
            )
            logger.info(
                "Computed feature importance for %d features",
                len(feature_importance),
            )

            # Step 7.5: SHAP analysis
            shap_summary = self._compute_shap_summary(
                estimator=estimator,
                features=features,
            )
            if shap_summary is not None:
                logger.info(
                    "SHAP analysis completed: %d features analyzed",
                    len(shap_summary.get("mean_abs_shap", {})),
                )
            else:
                logger.debug("SHAP analysis skipped or unavailable")

            # Step 8: Baseline comparison
            baseline_comparison = self._compute_baseline_comparison(
                target=target,
                predictions=predictions,
                predictive_scores=predictive_scores,
                problem_type=problem_type,
            )
            logger.info("Baseline comparison: %s", baseline_comparison)

            # Step 9: LLM evaluation narrative
            llm_narrative = self._evaluation_narrative(
                predictive_scores=predictive_scores,
                robustness_scores=robustness_scores,
                fairness_scores=fairness_scores,
                interpretability_scores=interpretability_scores,
                baseline_comparison=baseline_comparison,
                feature_importance=feature_importance,
                problem_type=problem_type,
                best_candidate=best_name,
                n_samples=features.shape[0],
                n_features=features.shape[1],
            )

            # Step 10: Overall grade
            overall_grade = self._compute_overall_grade(
                predictive_scores=predictive_scores,
                robustness_scores=robustness_scores,
                fairness_scores=fairness_scores,
                interpretability_scores=interpretability_scores,
                baseline_comparison=baseline_comparison,
            )
            logger.info("Overall grade: %s", overall_grade)

            # Step 11: Recommendations
            recommendations = self._generate_recommendations(
                predictive_scores=predictive_scores,
                robustness_scores=robustness_scores,
                fairness_scores=fairness_scores,
                interpretability_scores=interpretability_scores,
                baseline_comparison=baseline_comparison,
                overall_grade=overall_grade,
            )

            report = EvaluationReport(
                predictive_scores=predictive_scores,
                robustness_scores=robustness_scores,
                fairness_scores=fairness_scores,
                interpretability_scores=interpretability_scores,
                feature_importance=feature_importance,
                shap_summary=shap_summary,
                baseline_comparison=baseline_comparison,
                llm_evaluation_narrative=llm_narrative,
                overall_grade=overall_grade,
                recommendations=recommendations,
            )

            self._complete_experiment(
                experiment_id,
                metrics={
                    "overall_grade": overall_grade,
                    "predictive_scores": predictive_scores,
                    "robustness_mean_drop": robustness_scores.get(
                        "mean_score_drop", 0.0,
                    ),
                    "fairness_computed": fairness_scores is not None,
                    "shap_available": shap_summary is not None,
                },
            )

            return AgentResult(
                agent_name=self.name,
                status="success",
                data=report.model_dump(),
            )

        except Exception:
            self._complete_experiment(experiment_id, status="FAILED")
            raise

    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix and target vector."""
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in dataset"
            )

        target = df[target_column].copy()
        feature_df = df.drop(columns=[target_column]).copy()

        # Encode categorical/object columns before selecting numerics
        cat_cols = feature_df.select_dtypes(
            include=["object", "category", "bool"],
        ).columns.tolist()
        for col in cat_cols:
            freq = feature_df[col].value_counts(normalize=True)
            feature_df[col] = feature_df[col].map(freq).astype(float)

        features = feature_df.select_dtypes(include=["number"]).copy()

        if features.shape[1] == 0:
            raise ValueError(
                "No numeric features available for evaluation"
            )

        features = features.fillna(features.median())

        valid_mask = target.notna()
        if not valid_mask.all():
            n_dropped = (~valid_mask).sum()
            logger.warning(
                "Dropping %d rows with missing target values",
                n_dropped,
            )
            features = features.loc[valid_mask].reset_index(drop=True)
            target = target.loc[valid_mask].reset_index(drop=True)

        return features, target

    def _build_gradient_boosting(self, problem_type: str) -> Any:
        """Build a GradientBoosting estimator as fallback."""
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            GradientBoostingRegressor,
        )

        if problem_type == "classification":
            return GradientBoostingClassifier(
                n_estimators=100, max_depth=4,
                learning_rate=0.1, random_state=42,
            )
        else:
            return GradientBoostingRegressor(
                n_estimators=100, max_depth=4,
                learning_rate=0.1, random_state=42,
            )

    def _compute_predictive_scores(
        self,
        target: pd.Series,
        predictions: np.ndarray,
        estimator: Any,
        features: pd.DataFrame,
        problem_type: str,
        cv_proba: Any = None,
    ) -> dict[str, float]:
        """Compute predictive performance metrics.

        Classification: accuracy, f1 (macro), AUC (if predict_proba).
        Regression: R2, MAE, RMSE.

        Args:
            cv_proba: Cross-validated probability predictions (honest, no leakage).
                      Used for AUC/PR-AUC/Brier if available; falls back to
                      estimator.predict_proba (train-fitted) otherwise.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            roc_auc_score,
        )

        y_true = np.asarray(target)
        y_pred = np.asarray(predictions)

        if problem_type == "classification":
            scores: dict[str, float] = {
                "accuracy": round(
                    float(accuracy_score(y_true, y_pred)), 6,
                ),
                "f1_macro": round(
                    float(f1_score(
                        y_true, y_pred, average="macro",
                        zero_division=0.0,
                    )), 6,
                ),
            }

            # Imbalance-aware metrics
            from sklearn.metrics import (
                balanced_accuracy_score,
                matthews_corrcoef,
            )
            scores["balanced_accuracy"] = round(
                float(balanced_accuracy_score(y_true, y_pred)), 6,
            )
            scores["mcc"] = round(
                float(matthews_corrcoef(y_true, y_pred)), 6,
            )

            # AUC requires probability estimates — prefer cross-validated
            # probabilities (cv_proba) for honest evaluation, fall back to
            # train-fitted estimator.predict_proba if CV probabilities unavailable.
            y_proba = None
            if cv_proba is not None:
                y_proba = np.asarray(cv_proba)
            elif hasattr(estimator, "predict_proba"):
                try:
                    y_proba = estimator.predict_proba(features)
                except Exception as e:
                    logger.warning("predict_proba failed: %s", e)

            if y_proba is not None:
                try:
                    n_classes = y_proba.shape[1] if y_proba.ndim == 2 else 1
                    if n_classes == 2:
                        auc = roc_auc_score(y_true, y_proba[:, 1])
                    elif n_classes > 2:
                        auc = roc_auc_score(
                            y_true, y_proba,
                            multi_class="ovr", average="macro",
                        )
                    else:
                        auc = roc_auc_score(y_true, y_proba)
                    scores["auc"] = round(float(auc), 6)
                    # PR-AUC, Brier, and log_loss
                    from sklearn.metrics import (
                        average_precision_score,
                        brier_score_loss,
                        log_loss,
                    )
                    if n_classes == 2:
                        try:
                            scores["pr_auc"] = round(
                                float(average_precision_score(y_true, y_proba[:, 1])), 6,
                            )
                            scores["brier_score"] = round(
                                float(brier_score_loss(y_true, y_proba[:, 1])), 6,
                            )
                        except Exception as e:
                            logger.warning("PR-AUC/Brier computation failed: %s", e)
                    try:
                        scores["log_loss"] = round(
                            float(log_loss(y_true, y_proba)), 6,
                        )
                    except Exception as e:
                        logger.warning("Log loss computation failed: %s", e)
                except Exception as e:
                    logger.warning(
                        "AUC computation failed: %s", e,
                    )
        else:
            mse = mean_squared_error(y_true, y_pred)
            scores = {
                "r2": round(float(r2_score(y_true, y_pred)), 6),
                "mae": round(
                    float(mean_absolute_error(y_true, y_pred)), 6,
                ),
                "rmse": round(float(np.sqrt(mse)), 6),
            }

        return scores

    def _compute_robustness_scores(
        self,
        estimator: Any,
        features: pd.DataFrame,
        target: pd.Series,
        problem_type: str,
        predictive_scores: dict[str, float],
    ) -> dict[str, float]:
        """Evaluate model robustness under Gaussian noise perturbation.

        Adds Gaussian noise with std = 0.1 * feature_std to each
        feature and measures the resulting score drop.
        """
        from sklearn.metrics import accuracy_score, r2_score

        rng = np.random.default_rng(seed=42)
        feature_stds = features.std().values
        # Guard against zero std
        feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)

        noise = rng.normal(
            loc=0.0,
            scale=0.1 * feature_stds,
            size=features.shape,
        )
        noisy_features = features.values + noise

        y_true = np.asarray(target)

        try:
            noisy_predictions = estimator.predict(noisy_features)

            if problem_type == "classification":
                clean_score = predictive_scores.get("accuracy", 0.0)
                noisy_score = float(
                    accuracy_score(y_true, noisy_predictions),
                )
                primary_metric = "accuracy"
            else:
                clean_score = predictive_scores.get("r2", 0.0)
                noisy_score = float(
                    r2_score(y_true, noisy_predictions),
                )
                primary_metric = "r2"

            score_drop = clean_score - noisy_score
            relative_drop = (
                score_drop / clean_score
                if clean_score != 0
                else 0.0
            )

            return {
                "clean_score": round(clean_score, 6),
                "noisy_score": round(noisy_score, 6),
                "mean_score_drop": round(score_drop, 6),
                "relative_drop_pct": round(relative_drop * 100, 4),
                "noise_std_factor": 0.1,
                "primary_metric": primary_metric,
            }

        except Exception as e:
            logger.warning("Robustness evaluation failed: %s", e)
            return {
                "clean_score": 0.0,
                "noisy_score": 0.0,
                "mean_score_drop": 0.0,
                "relative_drop_pct": 0.0,
                "noise_std_factor": 0.1,
                "primary_metric": "unknown",
                "error": str(e),
            }

    def _compute_fairness_scores(
        self,
        df: pd.DataFrame,
        target_column: str,
        features: pd.DataFrame,
        estimator: Any,
        sensitive_columns: list[str],
        problem_type: str,
    ) -> dict[str, float]:
        """Compute per-group accuracy for each sensitive column.

        Uses pandas groupby on the original dataframe to compute
        per-group performance. Returns a flat dict mapping
        'column_group' to accuracy/score.
        """
        from sklearn.metrics import accuracy_score, r2_score

        # Align indices: use only valid rows matching features
        valid_df = (
            df.loc[features.index].copy()
            if len(features) < len(df)
            else df.copy()
        )

        predictions = estimator.predict(features)
        y_true = np.asarray(valid_df[target_column])

        fairness: dict[str, float] = {}

        for col in sensitive_columns:
            if col not in valid_df.columns:
                logger.warning(
                    "Sensitive column '%s' not found in dataset; "
                    "skipping fairness check",
                    col,
                )
                continue

            groups = valid_df[col]
            unique_groups = groups.unique()

            group_scores: list[float] = []
            for group_val in unique_groups:
                mask = groups == group_val
                if mask.sum() < 5:
                    # Skip very small groups
                    continue

                group_y = y_true[mask]
                group_pred = predictions[mask]

                try:
                    if problem_type == "classification":
                        score = float(
                            accuracy_score(group_y, group_pred),
                        )
                    else:
                        score = float(
                            r2_score(group_y, group_pred),
                        )
                except Exception:
                    score = 0.0

                key = f"{col}_{group_val}"
                fairness[key] = round(score, 6)
                group_scores.append(score)

            # Compute fairness gap for this sensitive column
            if len(group_scores) >= 2:
                fairness[f"{col}_gap"] = round(
                    max(group_scores) - min(group_scores), 6,
                )
                fairness[f"{col}_min_group_score"] = round(
                    min(group_scores), 6,
                )
                fairness[f"{col}_max_group_score"] = round(
                    max(group_scores), 6,
                )

        return fairness

    def _compute_interpretability_scores(
        self,
        estimator: Any,
        features: pd.DataFrame,
        target: pd.Series,
        problem_type: str,
    ) -> dict[str, float]:
        """Compute interpretability scores from feature importance.

        Uses Gini coefficient of feature importances as a measure of
        concentration: 0 = all features equally important (high
        interpretability), 1 = single feature dominates.
        """
        importances = self._get_feature_importances(
            estimator=estimator,
            features=features,
            target=target,
            problem_type=problem_type,
        )

        if importances is None or len(importances) == 0:
            return {
                "gini_concentration": 0.0,
                "interpretability_score": 0.5,
                "n_features_for_80pct": 0,
            }

        # Gini coefficient of importances
        sorted_imp = np.sort(np.abs(importances))
        n = len(sorted_imp)
        if n == 0 or sorted_imp.sum() == 0:
            gini = 0.0
        else:
            index = np.arange(1, n + 1)
            gini = float(
                (2 * np.sum(index * sorted_imp))
                / (n * np.sum(sorted_imp))
                - (n + 1) / n
            )

        # Count features needed for 80% cumulative importance
        cum_imp = np.cumsum(sorted_imp[::-1]) / sorted_imp.sum()
        n_80 = int(np.searchsorted(cum_imp, 0.80) + 1)

        # Interpretability score: higher Gini = more concentrated
        # = more interpretable (fewer features matter)
        interpretability = min(1.0, gini)

        return {
            "gini_concentration": round(gini, 6),
            "interpretability_score": round(interpretability, 6),
            "n_features_for_80pct": n_80,
            "total_features": n,
        }

    def _get_feature_importances(
        self,
        estimator: Any,
        features: pd.DataFrame,
        target: pd.Series,
        problem_type: str,
    ) -> np.ndarray | None:
        """Extract feature importances from the estimator.

        Tries feature_importances_ attribute first, then falls back
        to a fresh GradientBoosting model.
        """
        if hasattr(estimator, "feature_importances_"):
            return np.asarray(estimator.feature_importances_)

        if hasattr(estimator, "coef_"):
            return np.abs(np.asarray(estimator.coef_).flatten())

        # Fallback: train GradientBoosting to get importances
        try:
            gb = self._build_gradient_boosting(problem_type)
            gb.fit(features, target)
            return np.asarray(gb.feature_importances_)
        except Exception as e:
            logger.warning(
                "Could not extract feature importances: %s", e,
            )
            return None

    def _compute_feature_importance(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        problem_type: str,
    ) -> list[dict[str, float]]:
        """Compute feature importance using GradientBoosting.

        Returns a list of {feature_name: importance_score} dicts
        sorted by importance descending.
        """
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            GradientBoostingRegressor,
        )

        if problem_type == "classification":
            gb = GradientBoostingClassifier(
                n_estimators=100, max_depth=4,
                learning_rate=0.1, random_state=42,
            )
        else:
            gb = GradientBoostingRegressor(
                n_estimators=100, max_depth=4,
                learning_rate=0.1, random_state=42,
            )

        try:
            gb.fit(features, target)
            importances = gb.feature_importances_

            result = []
            for col, imp in zip(features.columns, importances):
                result.append({col: round(float(imp), 6)})

            # Sort by importance descending
            result.sort(
                key=lambda d: list(d.values())[0], reverse=True,
            )
            return result

        except Exception as e:
            logger.warning(
                "Feature importance computation failed: %s", e,
            )
            return []

    def _compute_shap_summary(
        self,
        estimator: Any,
        features: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """Compute SHAP feature importance summary.

        Uses the shap library to compute mean absolute SHAP values
        and identify top contributing features. Returns None if shap
        is not installed or computation fails.

        Args:
            estimator: Fitted sklearn-compatible estimator.
            features: Feature DataFrame used for explanation.

        Returns:
            Dict with 'mean_abs_shap' and 'top_5_features', or None.
        """
        try:
            import shap

            # Subsample for performance: max 1000 rows
            sample_size = min(1000, len(features))
            sample_features = features.head(sample_size)

            explainer = shap.Explainer(estimator, sample_features)
            shap_values = explainer(sample_features)

            mean_abs_shap = {}
            top_features_list = []

            for i, col in enumerate(features.columns):
                mean_val = float(np.abs(shap_values[:, i].values).mean())
                mean_abs_shap[col] = mean_val
                top_features_list.append((col, mean_val))

            # Sort by mean absolute SHAP descending, take top 5
            top_features_list.sort(key=lambda x: x[1], reverse=True)
            top_5 = top_features_list[:5]

            shap_summary = {
                "mean_abs_shap": mean_abs_shap,
                "top_5_features": top_5,
            }

            return shap_summary

        except ImportError:
            logger.debug("shap package not installed; skipping SHAP analysis")
            return None
        except Exception as e:
            logger.debug("SHAP analysis failed: %s", e)
            return None

    def _compute_baseline_comparison(
        self,
        target: pd.Series,
        predictions: np.ndarray,
        predictive_scores: dict[str, float],
        problem_type: str,
    ) -> dict[str, float]:
        """Compare model performance against naive baselines.

        Classification: majority class predictor.
        Regression: mean value predictor.
        """
        from sklearn.metrics import (
            accuracy_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        y_true = np.asarray(target)

        if problem_type == "classification":
            # Majority class baseline
            unique, counts = np.unique(y_true, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            baseline_preds = np.full_like(y_true, majority_class)

            baseline_acc = float(
                accuracy_score(y_true, baseline_preds),
            )
            model_acc = predictive_scores.get("accuracy", 0.0)

            return {
                "baseline_type": 0.0,  # sentinel for majority class
                "baseline_accuracy": round(baseline_acc, 6),
                "model_accuracy": round(model_acc, 6),
                "lift_over_baseline": round(
                    model_acc - baseline_acc, 6,
                ),
                "relative_lift_pct": round(
                    (
                        (model_acc - baseline_acc)
                        / baseline_acc
                        * 100
                    )
                    if baseline_acc > 0
                    else 0.0,
                    4,
                ),
            }
        else:
            # Mean predictor baseline
            mean_val = float(np.mean(y_true))
            baseline_preds = np.full_like(
                y_true, mean_val, dtype=float,
            )

            baseline_r2 = float(r2_score(y_true, baseline_preds))
            baseline_mae = float(
                mean_absolute_error(y_true, baseline_preds),
            )
            baseline_rmse = float(
                np.sqrt(mean_squared_error(y_true, baseline_preds)),
            )

            model_r2 = predictive_scores.get("r2", 0.0)
            model_mae = predictive_scores.get("mae", 0.0)
            model_rmse = predictive_scores.get("rmse", 0.0)

            return {
                "baseline_type": 1.0,  # sentinel for mean predictor
                "baseline_r2": round(baseline_r2, 6),
                "baseline_mae": round(baseline_mae, 6),
                "baseline_rmse": round(baseline_rmse, 6),
                "model_r2": round(model_r2, 6),
                "model_mae": round(model_mae, 6),
                "model_rmse": round(model_rmse, 6),
                "r2_lift": round(model_r2 - baseline_r2, 6),
                "mae_improvement": round(
                    baseline_mae - model_mae, 6,
                ),
                "rmse_improvement": round(
                    baseline_rmse - model_rmse, 6,
                ),
            }

    def _compute_overall_grade(
        self,
        predictive_scores: dict[str, float],
        robustness_scores: dict[str, float],
        fairness_scores: dict[str, float] | None,
        interpretability_scores: dict[str, float],
        baseline_comparison: dict[str, float],
    ) -> str:
        """Compute overall letter grade from weighted dimension scores.

        Dimensions: predictive (40%), robustness (25%), fairness (15%),
        interpretability (20%). Fairness weight is redistributed to
        predictive if no sensitive columns were evaluated.
        """
        # Normalize predictive score to 0-1
        # For classification: use balanced_accuracy (fallback accuracy); for regression: use R2
        pred_score = max(
            predictive_scores.get("balanced_accuracy",
                                  predictive_scores.get("accuracy", 0.0)),
            predictive_scores.get("r2", 0.0),
        )

        # Robustness: penalize for large relative drops
        relative_drop = robustness_scores.get(
            "relative_drop_pct", 0.0,
        )
        # Map: 0% drop = 1.0, 10% drop = 0.5, 20%+ drop = 0.0
        robustness_norm = max(0.0, 1.0 - relative_drop / 20.0)

        # Fairness: use 1 - max_gap (lower gap = better)
        if fairness_scores:
            gaps = [
                v for k, v in fairness_scores.items()
                if k.endswith("_gap")
            ]
            if gaps:
                max_gap = max(gaps)
                fairness_norm = max(0.0, 1.0 - max_gap)
            else:
                fairness_norm = 1.0
        else:
            fairness_norm = None

        # Interpretability: use interpretability_score directly
        interp_norm = interpretability_scores.get(
            "interpretability_score", 0.5,
        )

        # Weight and combine
        if fairness_norm is not None:
            weighted_score = (
                DIMENSION_WEIGHTS["predictive"] * pred_score
                + DIMENSION_WEIGHTS["robustness"] * robustness_norm
                + DIMENSION_WEIGHTS["fairness"] * fairness_norm
                + DIMENSION_WEIGHTS["interpretability"] * interp_norm
            )
        else:
            # Redistribute fairness weight to predictive
            adjusted_pred_weight = (
                DIMENSION_WEIGHTS["predictive"]
                + DIMENSION_WEIGHTS["fairness"]
            )
            total_weight = (
                adjusted_pred_weight
                + DIMENSION_WEIGHTS["robustness"]
                + DIMENSION_WEIGHTS["interpretability"]
            )
            weighted_score = (
                adjusted_pred_weight * pred_score
                + DIMENSION_WEIGHTS["robustness"] * robustness_norm
                + DIMENSION_WEIGHTS["interpretability"] * interp_norm
            ) / total_weight

        # Map to letter grade
        for threshold, grade in GRADE_THRESHOLDS:
            if weighted_score >= threshold:
                return grade

        return "F"

    def _generate_recommendations(
        self,
        predictive_scores: dict[str, float],
        robustness_scores: dict[str, float],
        fairness_scores: dict[str, float] | None,
        interpretability_scores: dict[str, float],
        baseline_comparison: dict[str, float],
        overall_grade: str,
    ) -> list[str]:
        """Generate actionable recommendations based on evaluation."""
        recs: list[str] = []

        # Predictive performance
        lift = baseline_comparison.get(
            "lift_over_baseline",
            baseline_comparison.get("r2_lift", 0.0),
        )
        if lift < 0.05:
            recs.append(
                "Model shows minimal lift over baseline. "
                "Consider feature engineering or more complex models."
            )

        # Robustness
        relative_drop = robustness_scores.get(
            "relative_drop_pct", 0.0,
        )
        if relative_drop > 10.0:
            recs.append(
                f"Model shows {relative_drop:.1f}% performance drop "
                f"under noise. Consider regularization, ensemble "
                f"methods, or data augmentation for robustness."
            )

        # Fairness
        if fairness_scores:
            gaps = [
                (k, v) for k, v in fairness_scores.items()
                if k.endswith("_gap")
            ]
            for gap_name, gap_val in gaps:
                if gap_val > 0.10:
                    col = gap_name.replace("_gap", "")
                    recs.append(
                        f"Fairness gap of {gap_val:.2f} detected "
                        f"for sensitive column '{col}'. Consider "
                        f"resampling, fairness constraints, or "
                        f"post-processing calibration."
                    )

        # Interpretability
        n_80 = interpretability_scores.get("n_features_for_80pct", 0)
        total = interpretability_scores.get("total_features", 0)
        if total > 0 and n_80 > total * 0.8:
            recs.append(
                "Feature importance is highly distributed. "
                "Consider feature selection to improve "
                "interpretability and reduce dimensionality."
            )

        # Grade-specific
        if overall_grade in ("D", "F"):
            recs.append(
                "Overall grade is below acceptable threshold. "
                "Revisit data quality, feature engineering, and "
                "model selection before deployment."
            )

        if not recs:
            recs.append(
                "Model evaluation is satisfactory across all "
                "dimensions. Proceed to deployment packaging."
            )

        return recs

    def _evaluation_narrative(
        self,
        predictive_scores: dict[str, float],
        robustness_scores: dict[str, float],
        fairness_scores: dict[str, float] | None,
        interpretability_scores: dict[str, float],
        baseline_comparison: dict[str, float],
        feature_importance: list[dict[str, float]],
        problem_type: str,
        best_candidate: str,
        n_samples: int,
        n_features: int,
    ) -> str:
        """Use LLM to produce evaluation narrative."""
        # Build feature importance summary (top 10)
        top_features = feature_importance[:10]
        fi_lines = []
        for feat_dict in top_features:
            for name, imp in feat_dict.items():
                fi_lines.append(f"  - {name}: {imp:.4f}")

        fairness_section = (
            "Not evaluated (no sensitive columns provided)."
        )
        if fairness_scores:
            fairness_lines = []
            for k, v in fairness_scores.items():
                fairness_lines.append(f"  - {k}: {v:.4f}")
            fairness_section = "\n".join(fairness_lines)

        prompt = (
            "Model Evaluation Results\n"
            "========================\n"
            f"Model: {best_candidate}\n"
            f"Problem type: {problem_type}\n"
            f"Dataset: {n_samples} samples, "
            f"{n_features} features\n\n"
            "Predictive Scores:\n"
            f"{self._format_dict(predictive_scores)}\n\n"
            "Robustness Scores:\n"
            f"{self._format_dict(robustness_scores)}\n\n"
            "Fairness Scores:\n"
            f"{fairness_section}\n\n"
            "Interpretability Scores:\n"
            f"{self._format_dict(interpretability_scores)}\n\n"
            "Top Feature Importances:\n"
            + (
                chr(10).join(fi_lines)
                if fi_lines
                else "  N/A"
            )
            + "\n\n"
            "Baseline Comparison:\n"
            f"{self._format_dict(baseline_comparison)}\n\n"
            "Produce a comprehensive evaluation narrative."
        )

        return self._call_ds_llm(
            prompt=prompt,
            role="ds_evaluator",
            system_prompt=EVALUATION_SYSTEM_PROMPT,
        )

    @staticmethod
    def _format_dict(d: dict[str, Any]) -> str:
        """Format a dict as indented key-value lines."""
        lines = []
        for k, v in d.items():
            if isinstance(v, float):
                lines.append(f"  - {k}: {v:.6f}")
            else:
                lines.append(f"  - {k}: {v}")
        return "\n".join(lines) if lines else "  N/A"
