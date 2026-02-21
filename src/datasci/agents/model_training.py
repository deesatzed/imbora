"""Phase 4: Model Training Agent.

Trains scale-appropriate model candidates using real sklearn cross-validation,
computes per-model performance metrics, runs conformal prediction calibration,
probability calibration (Phase E), and uses LLM for behavioral analysis.
Supports augmented datasets (Phase H) with leakage prevention: augmented data
is used only for fitting, while CV scoring uses the original dataset.
Produces a ModelTrainingReport.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import StratifiedKFold

from src.core.config import DataScienceConfig
from src.core.models import AgentResult
from src.datasci.agents.base_ds_agent import BaseDataScienceAgent
from src.datasci.models import ModelCandidate, ModelTrainingReport
from src.datasci.scale_router import determine_scale_tier, select_model_candidates
from src.datasci.uncertainty import ConformalPredictor

logger = logging.getLogger("associate.datasci.model_training")

MODEL_TRAINING_SYSTEM_PROMPT = (
    "You are an expert ML engineer analyzing model training results.\n"
    "Given a set of model candidates with their cross-validation scores, training times, and\n"
    "uncertainty calibration metrics, provide a behavioral analysis covering:\n"
    "1. Performance comparison: Which models excel and why? Are differences significant?\n"
    "2. Overfitting risk: Compare mean vs std of CV scores. Flag high-variance models.\n"
    "3. Speed-accuracy tradeoff: Is the fastest model competitive with the best?\n"
    "4. Uncertainty quality: Are conformal coverage rates near target? Best calibrated?\n"
    "5. Recommendation: Which 2-3 models should proceed to ensemble, and why?\n\n"
    "Be specific — reference model names, numeric scores, and concrete observations. "
    "Be concise."
)


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
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                    verbosity=0,
                )
                if apply_class_weights:
                    params["scale_pos_weight"] = imbalance_ratio
                return XGBClassifier(**params)
            else:
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0,
                )
        except ImportError:
            logger.info("xgboost not installed; skipping XGBoost candidate")
            return None

    elif model_name == "catboost":
        try:
            if problem_type == "classification":
                from catboost import CatBoostClassifier
                params = dict(
                    iterations=100,
                    depth=6,
                    learning_rate=0.1,
                    random_seed=42,
                    silent=True,
                )
                if apply_class_weights:
                    params["auto_class_weights"] = "Balanced"
                return CatBoostClassifier(**params)
            else:
                from catboost import CatBoostRegressor
                return CatBoostRegressor(
                    iterations=100,
                    depth=6,
                    learning_rate=0.1,
                    random_seed=42,
                    silent=True,
                )
        except ImportError:
            logger.info("catboost not installed; skipping CatBoost candidate")
            return None

    elif model_name == "lightgbm":
        try:
            if problem_type == "classification":
                from lightgbm import LGBMClassifier
                params = dict(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1,
                )
                if apply_class_weights:
                    params["is_unbalance"] = True
                return LGBMClassifier(**params)
            else:
                from lightgbm import LGBMRegressor
                return LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1,
                )
        except ImportError:
            logger.info("lightgbm not installed; skipping LightGBM candidate")
            return None

    elif model_name == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            if problem_type == "classification":
                return TabPFNClassifier(device="cpu")
            else:
                return TabPFNRegressor(device="cpu")
        except ImportError:
            logger.info("tabpfn not installed; skipping TabPFN candidate")
            return None

    elif model_name == "tabicl":
        try:
            from tabicl import TabICLClassifier, TabICLRegressor
            if problem_type == "classification":
                return TabICLClassifier()
            else:
                return TabICLRegressor()
        except ImportError:
            logger.info("tabicl not installed; skipping TabICL candidate")
            return None

    elif model_name == "nam":
        try:
            from dnamite import NAMClassifier, NAMRegressor
            if problem_type == "classification":
                return NAMClassifier(random_state=42)
            else:
                return NAMRegressor(random_state=42)
        except ImportError:
            logger.info("dnamite (NAM) not installed; skipping NAM candidate")
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
        logger.warning("Unknown model type: %s", model_name)
        return None


class ModelTrainingAgent(BaseDataScienceAgent):
    """Agent for Phase 4: Model Training.

    Determines scale tier, selects candidates via scale router,
    trains each with cross-validation, runs conformal calibration,
    probability calibration, and produces LLM behavioral analysis.

    When an augmented dataset is available (from Phase 3.5), models
    are fitted on the augmented data but CV scoring always uses the
    original dataset to prevent data leakage.
    """

    def __init__(
        self,
        llm_client: Any,
        model_router: Any,
        repository: Any,
        ds_config: DataScienceConfig,
    ):
        super().__init__(
            name="ModelTrainingAgent",
            role="ds_analyst",
            llm_client=llm_client,
            model_router=model_router,
            repository=repository,
            ds_config=ds_config,
        )

    def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Run the model training pipeline.

        Args:
            input_data: Dict with keys:
                - dataset_path: str
                - target_column: str
                - project_id: UUID
                - problem_type: str ('classification' or 'regression')
                - feature_report: dict (FeatureEngineeringReport)
                - augmentation_report: dict (AugmentationReport, optional)
                - eda_report: dict (optional, may contain imbalance_ratio)
                - task_id: UUID (optional)
                - run_id: UUID (optional)

        Returns:
            AgentResult with ModelTrainingReport in data.
        """
        dataset_path = input_data["dataset_path"]
        target_column = input_data["target_column"]
        project_id = input_data["project_id"]
        problem_type = input_data.get("problem_type", "classification")
        # feature_report consumed by experiment config for traceability
        _feature_report = input_data.get("feature_report", {})

        # Extract imbalance_ratio from EDA report
        imbalance_ratio: float = input_data.get(
            "eda_report", {},
        ).get("imbalance_ratio", 1.0)
        if imbalance_ratio is None:
            imbalance_ratio = 1.0

        # Phase H: Check for augmentation report
        augmentation_report = input_data.get("augmentation_report", {})
        augmented = augmentation_report.get("augmented", False)
        augmented_path = augmentation_report.get("augmented_dataset_path", "")

        experiment_id = self._save_experiment(
            project_id=project_id,
            phase="model_training",
            config={
                "target_column": target_column,
                "problem_type": problem_type,
                "cv_folds": self.ds_config.cv_folds,
                "conformal_alpha": self.ds_config.conformal_alpha,
                "feature_report_keys": list(_feature_report.keys()),
                "imbalance_ratio": imbalance_ratio,
                "augmented": augmented,
            },
            task_id=input_data.get("task_id"),
            run_id=input_data.get("run_id"),
        )

        try:
            # Step 1: Load original dataset (always used for CV scoring)
            df = _load_dataset(dataset_path)
            logger.info(
                "Loaded original dataset for training: %d rows x %d columns",
                df.shape[0], df.shape[1],
            )

            # Step 2: Prepare X and y from original dataset
            features, target = self._prepare_data(df, target_column)
            logger.info(
                "Prepared training data: %d samples, %d features",
                features.shape[0], features.shape[1],
            )

            # Phase H: Load augmented dataset for fitting (if available)
            aug_features = None
            aug_target = None
            if augmented and augmented_path:
                try:
                    aug_df = _load_dataset(augmented_path)
                    aug_features, aug_target = self._prepare_data(
                        aug_df, target_column,
                    )
                    logger.info(
                        "Loaded augmented dataset for fitting: "
                        "%d samples, %d features",
                        aug_features.shape[0], aug_features.shape[1],
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to load augmented dataset from %s: %s. "
                        "Falling back to original dataset for fitting.",
                        augmented_path, e,
                    )
                    aug_features = None
                    aug_target = None

            # Step 3: Determine scale tier and select candidates
            scale_tier = determine_scale_tier(
                features.shape[0], self.ds_config,
            )
            candidate_names = select_model_candidates(
                scale_tier=scale_tier,
                problem_type=problem_type,
                feature_count=features.shape[1],
                config=self.ds_config,
                imbalance_ratio=imbalance_ratio,
            )
            logger.info(
                "Scale tier: %s, selected candidates: %s",
                scale_tier, candidate_names,
            )

            # Step 4: Train each candidate with cross-validation
            # CV scoring always uses original data to prevent leakage
            scoring = (
                "balanced_accuracy"
                if problem_type == "classification"
                else "neg_mean_squared_error"
            )
            candidates: list[ModelCandidate] = []
            trained_estimators: dict[str, Any] = {}

            for name in candidate_names:
                candidate = self._train_candidate(
                    name=name,
                    problem_type=problem_type,
                    features=features,
                    target=target,
                    scoring=scoring,
                    imbalance_ratio=imbalance_ratio,
                )
                if candidate is not None:
                    candidates.append(candidate)
                    # Store fitted estimator for conformal calibration
                    # Fit on augmented data if available, else original
                    fit_features = (
                        aug_features if aug_features is not None else features
                    )
                    fit_target = (
                        aug_target if aug_target is not None else target
                    )
                    estimator = _build_sklearn_estimator(
                        name, problem_type, imbalance_ratio=imbalance_ratio,
                    )
                    if estimator is not None:
                        try:
                            estimator.fit(fit_features, fit_target)
                            trained_estimators[name] = estimator
                        except Exception as e:
                            logger.warning(
                                "Failed to fit %s for conformal: %s",
                                name, e,
                            )

            if not candidates:
                raise RuntimeError(
                    "No model candidates could be trained. Check that at "
                    "least one ML library (xgboost, catboost, lightgbm) "
                    "is installed."
                )

            # Step 5: Conformal prediction calibration
            self._calibrate_conformal(
                candidates=candidates,
                trained_estimators=trained_estimators,
                features=features,
                target=target,
                problem_type=problem_type,
                imbalance_ratio=imbalance_ratio,
            )

            # Step 6: Identify best candidate
            best_candidate_name = max(
                candidates, key=lambda c: c.mean_score,
            ).model_name

            # Step 7: Extract NAM shape functions if applicable
            nam_shape_functions = None
            if "nam" in trained_estimators:
                nam_shape_functions = self._extract_nam_shapes(
                    trained_estimators["nam"], features,
                )

            # Step 8: LLM behavioral analysis
            llm_narrative = self._behavioral_analysis(
                candidates=candidates,
                scale_tier=scale_tier,
                problem_type=problem_type,
                n_samples=features.shape[0],
                n_features=features.shape[1],
            )

            report = ModelTrainingReport(
                scale_tier=scale_tier,
                candidates=candidates,
                best_candidate=best_candidate_name,
                nam_shape_functions=nam_shape_functions,
                llm_evaluation_narrative=llm_narrative,
            )

            self._complete_experiment(
                experiment_id,
                metrics={
                    "scale_tier": scale_tier,
                    "n_candidates_trained": len(candidates),
                    "best_candidate": best_candidate_name,
                    "best_mean_score": max(
                        c.mean_score for c in candidates
                    ),
                    "augmented_fitting": augmented and aug_features is not None,
                    "imbalance_ratio": imbalance_ratio,
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
        """Prepare feature matrix and target vector.

        Encodes categorical columns via frequency encoding, selects numeric
        columns, fills NaN with median, separates target.
        """
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
                "No numeric features available for training after "
                "dropping target. Ensure feature engineering has "
                "produced numeric columns."
            )

        # Fill missing values with column medians
        features = features.fillna(features.median())

        # Handle missing target values by dropping corresponding rows
        valid_mask = target.notna()
        if not valid_mask.all():
            n_dropped = (~valid_mask).sum()
            logger.warning(
                "Dropping %d rows with missing target values", n_dropped,
            )
            features = features.loc[valid_mask].reset_index(drop=True)
            target = target.loc[valid_mask].reset_index(drop=True)

        return features, target

    def _train_candidate(
        self,
        name: str,
        problem_type: str,
        features: pd.DataFrame,
        target: pd.Series,
        scoring: str,
        imbalance_ratio: float = 1.0,
    ) -> ModelCandidate | None:
        """Train a single model candidate with cross-validation.

        Returns ModelCandidate with metrics, or None if the model
        cannot be built. When enable_calibration is set and the problem
        is classification, also runs probability calibration (Phase E)
        and stores the Brier score on the candidate.
        """
        estimator = _build_sklearn_estimator(
            name, problem_type, imbalance_ratio=imbalance_ratio,
        )
        if estimator is None:
            return None

        logger.info("Training candidate: %s", name)

        try:
            # Cross-validation scoring
            t_start = time.perf_counter()
            cv_scores = cross_val_score(
                estimator,
                features,
                target,
                cv=StratifiedKFold(
                    n_splits=self.ds_config.cv_folds,
                    shuffle=True,
                    random_state=42,
                ) if scoring == "balanced_accuracy" else self.ds_config.cv_folds,
                scoring=scoring,
                error_score="raise",
            )
            training_time = time.perf_counter() - t_start

            # For regression with neg_* scoring, convert to positive
            if scoring.startswith("neg_"):
                cv_scores = -cv_scores

            # Phase E: Probability calibration
            brier = None
            if (
                problem_type == "classification"
                and self.ds_config.enable_calibration
            ):
                try:
                    from sklearn.calibration import CalibratedClassifierCV
                    from sklearn.metrics import brier_score_loss

                    n_splits_cal = min(3, len(np.unique(target)))
                    if n_splits_cal >= 2 and len(target) >= n_splits_cal:
                        calibrated = CalibratedClassifierCV(
                            estimator=_build_sklearn_estimator(
                                name, problem_type, imbalance_ratio,
                            ),
                            method="isotonic",
                            cv=StratifiedKFold(
                                n_splits=n_splits_cal,
                                shuffle=True,
                                random_state=42,
                            ),
                        )
                        calibrated.fit(features, target)
                        if hasattr(calibrated, "predict_proba"):
                            cal_proba = calibrated.predict_proba(features)
                            if cal_proba.shape[1] == 2:
                                brier = round(
                                    float(brier_score_loss(
                                        target, cal_proba[:, 1],
                                    )),
                                    6,
                                )
                                logger.info(
                                    "Calibration Brier score for %s: %.6f",
                                    name, brier,
                                )
                except Exception as e:
                    logger.debug(
                        "Calibration failed for %s: %s", name, e,
                    )

            # Measure single-sample prediction time
            estimator_clone = _build_sklearn_estimator(
                name, problem_type, imbalance_ratio=imbalance_ratio,
            )
            if estimator_clone is not None:
                estimator_clone.fit(features, target)
                single_sample = features.iloc[:1]
                t_pred_start = time.perf_counter()
                n_pred_iters = 100
                for _ in range(n_pred_iters):
                    estimator_clone.predict(single_sample)
                prediction_time_ms = (
                    (time.perf_counter() - t_pred_start)
                    / n_pred_iters
                    * 1000
                )
            else:
                prediction_time_ms = 0.0

            # Extract hyperparameters
            hyperparams = {}
            if hasattr(estimator, "get_params"):
                raw_params = estimator.get_params(deep=False)
                # Filter to serializable types
                for k, v in raw_params.items():
                    if isinstance(v, (int, float, str, bool, type(None))):
                        hyperparams[k] = v

            candidate = ModelCandidate(
                model_name=name,
                model_type=name,
                cv_scores=[round(float(s), 6) for s in cv_scores],
                mean_score=round(float(np.mean(cv_scores)), 6),
                std_score=round(float(np.std(cv_scores)), 6),
                training_time_seconds=round(training_time, 3),
                prediction_time_ms=round(prediction_time_ms, 4),
                hyperparameters=hyperparams,
                calibration_brier_score=brier,
            )

            logger.info(
                "Candidate %s: mean=%.4f, std=%.4f, time=%.2fs%s",
                name,
                candidate.mean_score,
                candidate.std_score,
                training_time,
                f", brier={brier:.6f}" if brier is not None else "",
            )
            return candidate

        except Exception as e:
            logger.warning("Failed to train candidate %s: %s", name, e)
            return None

    def _calibrate_conformal(
        self,
        candidates: list[ModelCandidate],
        trained_estimators: dict[str, Any],
        features: pd.DataFrame,
        target: pd.Series,
        problem_type: str,
        imbalance_ratio: float = 1.0,
    ) -> None:
        """Run conformal prediction calibration for each estimator.

        Updates candidate.uncertainty_calibration and
        candidate.conformal_coverage in place.
        """
        conformal = ConformalPredictor(
            alpha=self.ds_config.conformal_alpha,
        )

        for candidate in candidates:
            estimator = trained_estimators.get(candidate.model_name)
            if estimator is None:
                continue

            try:
                # Use cross_val_predict for out-of-fold predictions
                cv_preds = cross_val_predict(
                    _build_sklearn_estimator(
                        candidate.model_name, problem_type,
                        imbalance_ratio=imbalance_ratio,
                    ),
                    features,
                    target,
                    cv=self.ds_config.cv_folds,
                )

                y_arr = np.asarray(target, dtype=float)
                preds_arr = np.asarray(cv_preds, dtype=float)

                # Calibrate on full set of out-of-fold residuals
                conformal.calibrate(y_arr, preds_arr)

                # Evaluate coverage
                coverage = conformal.evaluate_coverage(y_arr, preds_arr)
                candidate.conformal_coverage = round(
                    coverage["empirical_coverage"], 4,
                )
                candidate.uncertainty_calibration = round(
                    abs(
                        coverage["empirical_coverage"]
                        - coverage["target_coverage"]
                    ),
                    4,
                )

                logger.info(
                    "Conformal calibration for %s: "
                    "coverage=%.4f, gap=%.4f",
                    candidate.model_name,
                    coverage["empirical_coverage"],
                    candidate.uncertainty_calibration,
                )
            except Exception as e:
                logger.warning(
                    "Conformal calibration failed for %s: %s",
                    candidate.model_name, e,
                )

    def _extract_nam_shapes(
        self,
        nam_estimator: Any,
        features: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """Extract shape function data from a fitted NAM model."""
        try:
            if hasattr(nam_estimator, "get_shape_functions"):
                shapes = nam_estimator.get_shape_functions(features)
                return {
                    col: {
                        "x_values": (
                            shapes[col]["x"].tolist()
                            if hasattr(shapes[col]["x"], "tolist")
                            else shapes[col]["x"]
                        ),
                        "y_values": (
                            shapes[col]["y"].tolist()
                            if hasattr(shapes[col]["y"], "tolist")
                            else shapes[col]["y"]
                        ),
                    }
                    for col in shapes
                }
            elif hasattr(nam_estimator, "feature_importances_"):
                return {
                    "importances": {
                        col: round(float(imp), 4)
                        for col, imp in zip(
                            features.columns,
                            nam_estimator.feature_importances_,
                        )
                    }
                }
            return None
        except Exception as e:
            logger.warning("NAM shape extraction failed: %s", e)
            return None

    def _behavioral_analysis(
        self,
        candidates: list[ModelCandidate],
        scale_tier: str,
        problem_type: str,
        n_samples: int,
        n_features: int,
    ) -> str:
        """Use LLM to produce behavioral analysis of training results."""
        candidate_summaries = []
        for c in candidates:
            summary = (
                f"- {c.model_name}: mean_score={c.mean_score:.4f}, "
                f"std_score={c.std_score:.4f}, "
                f"train_time={c.training_time_seconds:.2f}s, "
                f"predict_time={c.prediction_time_ms:.3f}ms"
            )
            if c.conformal_coverage is not None:
                summary += (
                    f", conformal_coverage={c.conformal_coverage:.4f}"
                )
            if c.uncertainty_calibration is not None:
                summary += (
                    f", calibration_gap="
                    f"{c.uncertainty_calibration:.4f}"
                )
            if c.calibration_brier_score is not None:
                summary += (
                    f", brier_score={c.calibration_brier_score:.6f}"
                )
            candidate_summaries.append(summary)

        prompt = (
            "Model Training Results\n"
            "=====================\n"
            f"Dataset: {n_samples} samples, {n_features} features\n"
            f"Scale tier: {scale_tier}\n"
            f"Problem type: {problem_type}\n"
            f"CV folds: {self.ds_config.cv_folds}\n"
            f"Conformal alpha: {self.ds_config.conformal_alpha}\n\n"
            "Candidate Results:\n"
            f"{chr(10).join(candidate_summaries)}\n\n"
            "Provide a behavioral analysis of these training results."
        )

        return self._call_ds_llm(
            prompt=prompt,
            role="ds_evaluator",
            system_prompt=MODEL_TRAINING_SYSTEM_PROMPT,
        )
