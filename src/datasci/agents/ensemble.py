"""Phase 5: Ensemble Agent.

Builds a stacked ensemble from ModelTrainingReport candidates:
Level-0 base models produce out-of-fold predictions, Level-1 meta-learner
trains on the stacked predictions. Adds uncertainty-based routing,
Pareto optimization across configured objectives, and LLM overfitting analysis.
For classification with enable_calibration, the Level-1 meta-learner is
wrapped with CalibratedClassifierCV (Phase E).
Produces an EnsembleReport.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import StratifiedKFold

from src.core.config import DataScienceConfig
from src.core.models import AgentResult
from src.datasci.agents.base_ds_agent import BaseDataScienceAgent
from src.datasci.models import EnsembleReport, ModelCandidate, ModelTrainingReport
from src.datasci.pareto import ParetoOptimizer
from src.datasci.uncertainty import (
    compute_prediction_disagreement,
    route_by_uncertainty,
)

logger = logging.getLogger("associate.datasci.ensemble")

ENSEMBLE_SYSTEM_PROMPT = (
    "You are an expert ML engineer evaluating a stacked ensemble.\n"
    "Given Level-0 base model predictions, a Level-1 meta-learner, "
    "uncertainty routing stats,\n"
    "and Pareto optimization results, analyze the ensemble for:\n"
    "1. Overfitting risk: Does the meta-learner score exceed all base "
    "models significantly?\n"
    "   If so, this may indicate information leakage or overfitting.\n"
    "2. Diversity: Are base model predictions sufficiently different? "
    "Low disagreement\n"
    "   suggests redundant models that add complexity without benefit.\n"
    "3. Routing quality: What percentage of samples are routed to the "
    "uncertain path?\n"
    "   If too high (>40%), the ensemble is unreliable. "
    "If too low (<5%), routing adds no value.\n"
    "4. Pareto selection: Is the selected configuration a good "
    "trade-off across objectives?\n"
    "5. Recommendation: Should this ensemble be deployed as-is, "
    "simplified, or retrained?\n\n"
    "Be specific with numbers. Be concise but thorough."
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
                "xgboost not installed; skipping XGBoost in ensemble",
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
                "catboost not installed; skipping CatBoost in ensemble",
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
                "lightgbm not installed; skipping LightGBM in ensemble",
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
                "tabpfn not installed; skipping TabPFN in ensemble",
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
                "tabicl not installed; skipping TabICL in ensemble",
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
                "dnamite (NAM) not installed; skipping NAM in ensemble",
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
            "Unknown model type for ensemble: %s", model_name,
        )
        return None


class EnsembleAgent(BaseDataScienceAgent):
    """Agent for Phase 5: Ensemble Construction.

    Builds a stacked generalization ensemble:
    - Level-0: base models produce out-of-fold predictions
      via cross_val_predict
    - Level-1: meta-learner (LogisticRegression or Ridge) trained
      on Level-0 outputs, optionally calibrated (Phase E)
    - Uncertainty routing via prediction disagreement
    - Pareto optimization for multi-objective model selection
    - LLM overfitting check
    """

    def __init__(
        self,
        llm_client: Any,
        model_router: Any,
        repository: Any,
        ds_config: DataScienceConfig,
    ):
        super().__init__(
            name="EnsembleAgent",
            role="ds_analyst",
            llm_client=llm_client,
            model_router=model_router,
            repository=repository,
            ds_config=ds_config,
        )

    def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Run the ensemble construction pipeline.

        Args:
            input_data: Dict with keys:
                - dataset_path: str
                - target_column: str
                - project_id: UUID
                - problem_type: str ('classification' or 'regression')
                - training_report: dict (ModelTrainingReport)
                - eda_report: dict (optional, may contain imbalance_ratio)
                - task_id: UUID (optional)
                - run_id: UUID (optional)

        Returns:
            AgentResult with EnsembleReport in data.
        """
        dataset_path = input_data["dataset_path"]
        target_column = input_data["target_column"]
        project_id = input_data["project_id"]
        problem_type = input_data.get("problem_type", "classification")
        training_data = input_data.get("training_report", {})

        # Extract imbalance_ratio from EDA report
        imbalance_ratio: float = input_data.get(
            "eda_report", {},
        ).get("imbalance_ratio", 1.0)

        # Reconstruct ModelTrainingReport from dict
        training_report = ModelTrainingReport(**training_data)

        if not training_report.candidates:
            return AgentResult(
                agent_name=self.name,
                status="failure",
                data={},
                error=(
                    "No model candidates in training report; "
                    "cannot build ensemble"
                ),
            )

        experiment_id = self._save_experiment(
            project_id=project_id,
            phase="ensemble",
            config={
                "target_column": target_column,
                "problem_type": problem_type,
                "n_base_models": len(training_report.candidates),
                "cv_folds": self.ds_config.cv_folds,
                "pareto_objectives": self.ds_config.pareto_objectives,
                "imbalance_ratio": imbalance_ratio,
            },
            task_id=input_data.get("task_id"),
            run_id=input_data.get("run_id"),
        )

        try:
            # Step 1: Load dataset and prepare features/target
            df = _load_dataset(dataset_path)
            features, target = self._prepare_data(df, target_column)
            logger.info(
                "Loaded dataset for ensemble: %d samples, %d features",
                features.shape[0], features.shape[1],
            )

            # Step 2: Build Level-0 out-of-fold predictions
            level0_preds, active_candidates = self._build_level0(
                candidates=training_report.candidates,
                problem_type=problem_type,
                features=features,
                target=target,
                imbalance_ratio=imbalance_ratio,
            )

            if len(level0_preds) < 2:
                logger.warning(
                    "Only %d base models available; ensemble "
                    "requires >= 2. Falling back to single best.",
                    len(level0_preds),
                )
                # Produce minimal report with single model
                best = (
                    active_candidates[0]
                    if active_candidates
                    else training_report.candidates[0]
                )
                report = EnsembleReport(
                    stacking_architecture={
                        "level_0": [best.model_name],
                        "level_1": "none",
                        "note": (
                            "Single model fallback; insufficient "
                            "candidates for stacking"
                        ),
                    },
                    selected_configuration=best.model_name,
                    llm_ensemble_analysis=(
                        "Single model; no ensemble analysis applicable."
                    ),
                )
                self._complete_experiment(
                    experiment_id,
                    metrics={"n_base_models": 1, "fallback": True},
                )
                return AgentResult(
                    agent_name=self.name,
                    status="success",
                    data=report.model_dump(),
                )

            # Step 3: Build Level-1 meta-learner
            meta_score, meta_learner_name = self._build_level1(
                level0_preds=level0_preds,
                target=target,
                problem_type=problem_type,
            )
            logger.info(
                "Level-1 meta-learner (%s) CV score: %.4f",
                meta_learner_name, meta_score,
            )

            # Check for ensemble degradation: meta-learner worse than best base
            best_base_score = max(c.mean_score for c in active_candidates)
            degradation_threshold = 0.98  # 2% tolerance
            if meta_score < best_base_score * degradation_threshold:
                best = max(active_candidates, key=lambda c: c.mean_score)
                logger.warning(
                    "Ensemble degradation: meta_score=%.4f < "
                    "best_base=%.4f * %.2f. Falling back to %s.",
                    meta_score, best_base_score,
                    degradation_threshold, best.model_name,
                )
                report = EnsembleReport(
                    stacking_architecture={
                        "level_0": [
                            c.model_name for c in active_candidates
                        ],
                        "level_1": "none",
                        "meta_cv_score": round(meta_score, 6),
                        "best_base_score": round(best_base_score, 6),
                        "note": (
                            f"Meta-learner degradation detected "
                            f"({meta_score:.4f} < {best_base_score:.4f}); "
                            f"using best base model {best.model_name}"
                        ),
                    },
                    selected_configuration=best.model_name,
                    llm_ensemble_analysis=(
                        f"Stacking meta-learner ({meta_learner_name}) "
                        f"underperformed best base model ({best.model_name}). "
                        f"Falling back to single model."
                    ),
                )
                self._complete_experiment(
                    experiment_id,
                    metrics={
                        "n_base_models": len(active_candidates),
                        "fallback": True,
                        "reason": "degradation",
                        "meta_score": meta_score,
                        "best_base_score": best_base_score,
                    },
                )
                return AgentResult(
                    agent_name=self.name,
                    status="success",
                    data=report.model_dump(),
                )

            # Step 4: Uncertainty-based routing
            routing_info = self._compute_routing(
                level0_preds=level0_preds,
            )

            # Step 5: Pareto optimization
            pareto_result = self._run_pareto(
                active_candidates=active_candidates,
                meta_score=meta_score,
                meta_learner_name=meta_learner_name,
            )

            # Step 6: Determine selected configuration
            selected = pareto_result.get("selected")
            if selected is not None:
                selected_name = selected["name"]
            else:
                selected_name = f"stacked_{meta_learner_name}"

            # Step 7: Build stacking architecture summary
            stacking_architecture = {
                "level_0": [
                    c.model_name for c in active_candidates
                ],
                "level_1": meta_learner_name,
                "meta_cv_score": round(meta_score, 6),
                "base_model_scores": {
                    c.model_name: c.mean_score
                    for c in active_candidates
                },
            }

            # Step 8: LLM overfitting analysis
            llm_analysis = self._overfitting_analysis(
                active_candidates=active_candidates,
                meta_score=meta_score,
                meta_learner_name=meta_learner_name,
                routing_info=routing_info,
                pareto_result=pareto_result,
                problem_type=problem_type,
            )

            # Build routing rules
            routing_rules = [
                {
                    "condition": "disagreement_score >= threshold",
                    "action": "route_to_uncertain_path",
                    "threshold": routing_info["threshold"],
                    "pct_uncertain": routing_info["pct_uncertain"],
                },
                {
                    "condition": "disagreement_score < threshold",
                    "action": "route_to_confident_path",
                    "pct_confident": routing_info["pct_confident"],
                },
            ]

            # Extract Pareto front for report
            pareto_front = [
                s["scores"]
                for s in pareto_result.get("pareto_front", [])
            ]

            report = EnsembleReport(
                stacking_architecture=stacking_architecture,
                routing_rules=routing_rules,
                pareto_front=pareto_front,
                selected_configuration=selected_name,
                llm_ensemble_analysis=llm_analysis,
            )

            self._complete_experiment(
                experiment_id,
                metrics={
                    "n_base_models": len(active_candidates),
                    "meta_learner": meta_learner_name,
                    "meta_cv_score": meta_score,
                    "pct_uncertain": routing_info["pct_uncertain"],
                    "pareto_front_size": pareto_result.get(
                        "pareto_front_size", 0,
                    ),
                    "selected_configuration": selected_name,
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
                "No numeric features available for ensemble "
                "construction"
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

    def _build_level0(
        self,
        candidates: list[ModelCandidate],
        problem_type: str,
        features: pd.DataFrame,
        target: pd.Series,
        imbalance_ratio: float = 1.0,
    ) -> tuple[dict[str, np.ndarray], list[ModelCandidate]]:
        """Build Level-0 out-of-fold predictions for each base model.

        Returns:
            Tuple of (model_name -> oof_predictions, active candidates).
        """
        level0_preds: dict[str, np.ndarray] = {}
        active_candidates: list[ModelCandidate] = []

        for candidate in candidates:
            estimator = _build_sklearn_estimator(
                candidate.model_type, problem_type,
                imbalance_ratio=imbalance_ratio,
            )
            if estimator is None:
                logger.info(
                    "Skipping %s in ensemble: package not available",
                    candidate.model_name,
                )
                continue

            try:
                oof_preds = cross_val_predict(
                    estimator,
                    features,
                    target,
                    cv=StratifiedKFold(
                        n_splits=self.ds_config.cv_folds,
                        shuffle=True,
                        random_state=42,
                    ) if problem_type == "classification" else self.ds_config.cv_folds,
                )
                level0_preds[candidate.model_name] = np.asarray(
                    oof_preds, dtype=float,
                )
                active_candidates.append(candidate)
                logger.info(
                    "Level-0 OOF predictions generated for %s",
                    candidate.model_name,
                )
            except Exception as e:
                logger.warning(
                    "Failed to generate OOF predictions for %s: %s",
                    candidate.model_name, e,
                )

        return level0_preds, active_candidates

    def _build_level1(
        self,
        level0_preds: dict[str, np.ndarray],
        target: pd.Series,
        problem_type: str,
    ) -> tuple[float, str]:
        """Train Level-1 meta-learner on stacked Level-0 predictions.

        For classification with enable_calibration, wraps the meta-learner
        with CalibratedClassifierCV (Phase E) to produce well-calibrated
        probability estimates.

        Returns:
            Tuple of (meta-learner CV score, meta-learner name).
        """
        # Stack Level-0 predictions into a feature matrix
        stacked_features = np.column_stack(
            list(level0_preds.values()),
        )
        y_arr = np.asarray(target)

        if problem_type == "classification":
            meta_model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver="lbfgs",
            )
            meta_name = "LogisticRegression"
            scoring = "balanced_accuracy"
        else:
            meta_model = Ridge(alpha=1.0, random_state=42)
            meta_name = "Ridge"
            scoring = "neg_mean_squared_error"

        meta_scores = cross_val_score(
            meta_model,
            stacked_features,
            y_arr,
            cv=StratifiedKFold(
                n_splits=self.ds_config.cv_folds,
                shuffle=True,
                random_state=42,
            ) if problem_type == "classification" else self.ds_config.cv_folds,
            scoring=scoring,
        )

        if scoring.startswith("neg_"):
            meta_scores = -meta_scores

        mean_score = float(np.mean(meta_scores))

        # Phase E: Wrap meta-learner with CalibratedClassifierCV
        if (
            problem_type == "classification"
            and self.ds_config.enable_calibration
        ):
            try:
                from sklearn.calibration import CalibratedClassifierCV

                calibrated_meta = CalibratedClassifierCV(
                    estimator=LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                        solver="lbfgs",
                    ),
                    method="isotonic",
                    cv=3,
                )
                calibrated_meta.fit(stacked_features, y_arr)
                # The calibrated model is now the meta-learner
                meta_name = "CalibratedLogisticRegression"
                logger.info(
                    "Level-1 meta-learner calibrated with isotonic regression",
                )
            except Exception as e:
                logger.debug(
                    "Meta-learner calibration failed, using uncalibrated: %s",
                    e,
                )

        return round(mean_score, 6), meta_name

    def _compute_routing(
        self,
        level0_preds: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Compute uncertainty-based routing from prediction disagreement."""
        disagreement = compute_prediction_disagreement(level0_preds)
        routing = route_by_uncertainty(disagreement, threshold=0.3)

        n_samples = len(disagreement)
        n_uncertain = int(routing["uncertain"].sum())
        n_confident = int(routing["confident"].sum())

        pct_uncertain = (
            round(n_uncertain / n_samples * 100, 2)
            if n_samples > 0
            else 0.0
        )
        pct_confident = (
            round(n_confident / n_samples * 100, 2)
            if n_samples > 0
            else 0.0
        )

        mean_disagreement = round(float(np.mean(disagreement)), 4)
        max_disagreement = round(float(np.max(disagreement)), 4)

        logger.info(
            "Routing: %d confident (%.1f%%), %d uncertain (%.1f%%), "
            "mean_disagreement=%.4f, max=%.4f",
            n_confident, pct_confident,
            n_uncertain, pct_uncertain,
            mean_disagreement, max_disagreement,
        )

        return {
            "threshold": 0.3,
            "n_samples": n_samples,
            "n_uncertain": n_uncertain,
            "n_confident": n_confident,
            "pct_uncertain": pct_uncertain,
            "pct_confident": pct_confident,
            "mean_disagreement": mean_disagreement,
            "max_disagreement": max_disagreement,
        }

    def _run_pareto(
        self,
        active_candidates: list[ModelCandidate],
        meta_score: float,
        meta_learner_name: str,
    ) -> dict[str, Any]:
        """Run Pareto optimization over base models plus ensemble.

        Uses configured pareto_objectives from ds_config.
        """
        objectives = self.ds_config.pareto_objectives
        optimizer = ParetoOptimizer(objectives=objectives)

        # Add each base model as a solution
        for c in active_candidates:
            scores: dict[str, float] = {}
            for obj in objectives:
                if obj in ("accuracy", "balanced_accuracy"):
                    scores[obj] = c.mean_score
                elif obj == "interpretability":
                    # NAM highly interpretable, boosters medium
                    if c.model_type == "nam":
                        scores[obj] = 0.9
                    elif c.model_type in ("tabpfn", "tabicl"):
                        scores[obj] = 0.2
                    else:
                        scores[obj] = 0.4
                elif obj == "speed":
                    # Lower prediction time is better
                    # Inverse scaled to [0,1] with 10ms ref
                    if c.prediction_time_ms > 0:
                        scores[obj] = min(
                            1.0, 10.0 / c.prediction_time_ms,
                        )
                    else:
                        scores[obj] = 0.5
                else:
                    # Unknown objectives get neutral score
                    scores[obj] = 0.5

            optimizer.add_solution(
                name=c.model_name,
                scores=scores,
                metadata={
                    "type": "base_model",
                    "mean_score": c.mean_score,
                    "training_time": c.training_time_seconds,
                },
            )

        # Add the stacked ensemble as a solution
        ensemble_scores: dict[str, float] = {}
        for obj in objectives:
            if obj in ("accuracy", "balanced_accuracy"):
                ensemble_scores[obj] = meta_score
            elif obj == "interpretability":
                # Stacking is least interpretable
                ensemble_scores[obj] = 0.1
            elif obj == "speed":
                # Ensemble is slower than any single model
                ensemble_scores[obj] = 0.2
            else:
                ensemble_scores[obj] = 0.5
        optimizer.add_solution(
            name=f"stacked_{meta_learner_name}",
            scores=ensemble_scores,
            metadata={
                "type": "stacked_ensemble",
                "meta_score": meta_score,
            },
        )

        return optimizer.get_summary()

    def _overfitting_analysis(
        self,
        active_candidates: list[ModelCandidate],
        meta_score: float,
        meta_learner_name: str,
        routing_info: dict[str, Any],
        pareto_result: dict[str, Any],
        problem_type: str,
    ) -> str:
        """Use LLM to analyze the ensemble for overfitting risk."""
        base_summaries = []
        for c in active_candidates:
            base_summaries.append(
                f"- {c.model_name}: mean_score={c.mean_score:.4f}, "
                f"std={c.std_score:.4f}, "
                f"train_time={c.training_time_seconds:.2f}s, "
                f"predict_time={c.prediction_time_ms:.3f}ms"
            )

        pareto_front = pareto_result.get("pareto_front", [])
        pareto_selected = pareto_result.get("selected")
        pareto_summary = f"Front size: {len(pareto_front)}"
        if pareto_selected:
            pareto_summary += (
                f", selected: {pareto_selected['name']} "
                f"({pareto_selected['scores']})"
            )

        objectives_str = ", ".join(self.ds_config.pareto_objectives)

        prompt = (
            "Ensemble Analysis\n"
            "=================\n"
            f"Problem type: {problem_type}\n"
            f"CV folds: {self.ds_config.cv_folds}\n\n"
            "Base models (Level-0):\n"
            f"{chr(10).join(base_summaries)}\n\n"
            f"Meta-learner (Level-1): {meta_learner_name}\n"
            f"Meta-learner CV score: {meta_score:.4f}\n\n"
            "Uncertainty Routing:\n"
            f"- Threshold: {routing_info['threshold']}\n"
            f"- Confident samples: {routing_info['n_confident']} "
            f"({routing_info['pct_confident']}%)\n"
            f"- Uncertain samples: {routing_info['n_uncertain']} "
            f"({routing_info['pct_uncertain']}%)\n"
            f"- Mean disagreement: "
            f"{routing_info['mean_disagreement']}\n"
            f"- Max disagreement: "
            f"{routing_info['max_disagreement']}\n\n"
            f"Pareto Optimization ({objectives_str}):\n"
            f"{pareto_summary}\n\n"
            "Analyze this ensemble for overfitting risk, diversity, "
            "routing quality, and overall viability."
        )

        return self._call_ds_llm(
            prompt=prompt,
            role="ds_evaluator",
            system_prompt=ENSEMBLE_SYSTEM_PROMPT,
        )
