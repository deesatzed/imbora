"""Phase 7: Deployment Agent.

Generates a complete deployment package: final model trained on full data,
serialized artifact, FastAPI scaffold, monitoring config with drift
thresholds, data contract, and LLM completeness review.
Produces a DeploymentPackage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import DataScienceConfig
from src.core.models import AgentResult
from src.datasci.agents.base_ds_agent import BaseDataScienceAgent
from src.datasci.artifacts import ArtifactManager
from src.datasci.models import (
    DeploymentPackage,
    EvaluationReport,
    ModelTrainingReport,
)

logger = logging.getLogger("associate.datasci.deployment")

DEPLOYMENT_SYSTEM_PROMPT = (
    "You are an expert ML engineer reviewing a deployment package for completeness.\n"
    "Given a trained model artifact, API scaffold, monitoring configuration,\n"
    "data contract, and evaluation summary, assess whether the package is\n"
    "ready for production deployment. Check for:\n"
    "1. Model artifact: Is it properly serialized? Expected file format?\n"
    "2. API scaffold: Does it include input validation, health checks, error handling?\n"
    "3. Monitoring: Are drift thresholds reasonable? Is alerting configured?\n"
    "4. Data contract: Are all input features documented with types and constraints?\n"
    "5. Missing elements: Logging, versioning, rollback plan, A/B test setup?\n\n"
    "Be specific about what is present and what is missing. "
    "Provide a completeness score (0-100%) and recommendations."
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
                "xgboost not installed; skipping in deployment",
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
                "catboost not installed; skipping in deployment",
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
                "lightgbm not installed; skipping in deployment",
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
                "tabpfn not installed; skipping in deployment",
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
                "tabicl not installed; skipping in deployment",
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
                "dnamite (NAM) not installed; skipping in deployment",
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
            "Unknown model type for deployment: %s", model_name,
        )
        return None


def _build_gradient_boosting(problem_type: str) -> Any:
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


class DeploymentAgent(BaseDataScienceAgent):
    """Agent for Phase 7: Deployment Package Generation.

    Produces a complete deployment package:
    - Final model trained on full data, serialized with joblib
    - FastAPI scaffold for prediction serving
    - Monitoring configuration with drift thresholds
    - Data contract for input schema validation
    - LLM completeness review
    """

    def __init__(
        self,
        llm_client: Any,
        model_router: Any,
        repository: Any,
        ds_config: DataScienceConfig,
    ):
        super().__init__(
            name="DeploymentAgent",
            role="ds_analyst",
            llm_client=llm_client,
            model_router=model_router,
            repository=repository,
            ds_config=ds_config,
        )
        self.artifact_manager = ArtifactManager()

    def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Generate the deployment package.

        Args:
            input_data: Dict with keys:
                - dataset_path: str
                - target_column: str
                - project_id: UUID
                - problem_type: str ('classification' or 'regression')
                - training_report: dict (ModelTrainingReport)
                - evaluation_report: dict (EvaluationReport)
                - eda_report: dict (optional, may contain imbalance_ratio)
                - task_id: UUID (optional)
                - run_id: UUID (optional)

        Returns:
            AgentResult with DeploymentPackage in data.
        """
        dataset_path = input_data["dataset_path"]
        target_column = input_data["target_column"]
        project_id = input_data["project_id"]
        problem_type = input_data.get("problem_type", "classification")
        training_data = input_data.get("training_report", {})
        evaluation_data = input_data.get("evaluation_report", {})

        # Extract imbalance_ratio from EDA report
        imbalance_ratio: float = input_data.get(
            "eda_report", {},
        ).get("imbalance_ratio", 1.0)

        # Reconstruct reports from dicts
        training_report = ModelTrainingReport(**training_data)
        evaluation_report = EvaluationReport(**evaluation_data)

        experiment_id = self._save_experiment(
            project_id=project_id,
            phase="deployment",
            config={
                "target_column": target_column,
                "problem_type": problem_type,
                "best_candidate": training_report.best_candidate,
                "overall_grade": evaluation_report.overall_grade,
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
                "Loaded dataset for deployment: %d samples, "
                "%d features",
                features.shape[0], features.shape[1],
            )

            # Step 2: Train final model on full data
            best_name = training_report.best_candidate
            if not best_name and training_report.candidates:
                best_name = max(
                    training_report.candidates,
                    key=lambda c: c.mean_score,
                ).model_name

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
                logger.warning(
                    "Best candidate '%s' not available; "
                    "falling back to GradientBoosting",
                    best_name,
                )
                estimator = _build_gradient_boosting(problem_type)
                best_name = "GradientBoosting"

            estimator.fit(features, target)
            logger.info(
                "Trained final %s model on %d samples",
                best_name, features.shape[0],
            )

            # Step 3: Serialize model artifact
            project_str = str(project_id)
            artifacts_dir = Path(self.ds_config.artifacts_dir)
            model_path = artifacts_dir / project_str / "model.joblib"
            saved_path = self.artifact_manager.save_model(
                estimator, str(model_path),
            )
            logger.info("Model artifact saved: %s", saved_path)

            # Step 4: Generate FastAPI scaffold
            feature_names = list(features.columns)
            api_scaffold = self.artifact_manager.generate_api_scaffold(
                model_name=best_name,
                feature_names=feature_names,
                target_column=target_column,
            )
            logger.info("API scaffold generated (%d chars)", len(api_scaffold))

            # Step 5: Compute feature statistics for monitoring
            feature_stats = self._compute_feature_stats(features)

            # Extract training metrics from evaluation report
            training_metrics = dict(evaluation_report.predictive_scores)

            # Generate monitoring config
            monitoring_config = (
                self.artifact_manager.generate_monitoring_config(
                    feature_stats=feature_stats,
                    training_metrics=training_metrics,
                )
            )
            logger.info(
                "Monitoring config generated with %d feature thresholds",
                len(monitoring_config.get("drift_thresholds", {})),
            )

            # Step 6: Generate data contract
            feature_types = {
                col: str(features[col].dtype)
                for col in features.columns
            }
            data_contract = self.artifact_manager.generate_data_contract(
                feature_names=feature_names,
                feature_types=feature_types,
            )
            logger.info(
                "Data contract generated with %d fields",
                len(data_contract.get("fields", [])),
            )

            # Step 7: LLM completeness review
            llm_review = self._completeness_review(
                best_name=best_name,
                model_path=saved_path,
                feature_names=feature_names,
                target_column=target_column,
                evaluation_report=evaluation_report,
                monitoring_config=monitoring_config,
                data_contract=data_contract,
                problem_type=problem_type,
                n_samples=features.shape[0],
            )

            # Feature pipeline path is same directory as model
            feature_pipeline_path = str(
                artifacts_dir / project_str / "feature_pipeline",
            )

            package = DeploymentPackage(
                model_artifact_path=saved_path,
                feature_pipeline_path=feature_pipeline_path,
                api_scaffold_code=api_scaffold,
                monitoring_config=monitoring_config,
                data_contract=data_contract,
                llm_completeness_review=llm_review,
            )

            self._complete_experiment(
                experiment_id,
                metrics={
                    "model_name": best_name,
                    "model_artifact_path": saved_path,
                    "n_features": len(feature_names),
                    "evaluation_grade": evaluation_report.overall_grade,
                },
                artifacts_manifest={
                    "model_joblib": saved_path,
                    "api_scaffold_length": len(api_scaffold),
                    "monitoring_feature_count": len(
                        monitoring_config.get("drift_thresholds", {}),
                    ),
                    "data_contract_field_count": len(
                        data_contract.get("fields", []),
                    ),
                },
            )

            return AgentResult(
                agent_name=self.name,
                status="success",
                data=package.model_dump(),
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
                "No numeric features available for deployment model"
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

    def _compute_feature_stats(
        self,
        features: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """Compute per-feature statistics for monitoring config."""
        stats: dict[str, dict[str, float]] = {}
        for col in features.columns:
            col_data = features[col]
            stats[col] = {
                "mean": round(float(col_data.mean()), 6),
                "std": round(float(col_data.std()), 6),
                "min": round(float(col_data.min()), 6),
                "max": round(float(col_data.max()), 6),
                "median": round(float(col_data.median()), 6),
                "q25": round(float(col_data.quantile(0.25)), 6),
                "q75": round(float(col_data.quantile(0.75)), 6),
            }
        return stats

    def _completeness_review(
        self,
        best_name: str,
        model_path: str,
        feature_names: list[str],
        target_column: str,
        evaluation_report: EvaluationReport,
        monitoring_config: dict[str, Any],
        data_contract: dict[str, Any],
        problem_type: str,
        n_samples: int,
    ) -> str:
        """Use LLM to review deployment package completeness."""
        drift_features = list(
            monitoring_config.get("drift_thresholds", {}).keys(),
        )
        perf_thresholds = list(
            monitoring_config.get(
                "performance_thresholds", {},
            ).keys(),
        )
        contract_fields = [
            f["name"]
            for f in data_contract.get("fields", [])
        ]

        prompt = (
            "Deployment Package Review\n"
            "=========================\n"
            f"Model: {best_name}\n"
            f"Problem type: {problem_type}\n"
            f"Training samples: {n_samples}\n"
            f"Target: {target_column}\n"
            f"Evaluation grade: {evaluation_report.overall_grade}\n\n"
            f"Model artifact: {model_path} (joblib format)\n\n"
            f"API scaffold: FastAPI app with /predict and /health "
            f"endpoints, {len(feature_names)} input features\n\n"
            "Monitoring configuration:\n"
            f"  - Drift thresholds for {len(drift_features)} features: "
            f"{', '.join(drift_features[:5])}"
            f"{'...' if len(drift_features) > 5 else ''}\n"
            f"  - Performance thresholds for: "
            f"{', '.join(perf_thresholds)}\n\n"
            "Data contract:\n"
            f"  - {len(contract_fields)} fields defined: "
            f"{', '.join(contract_fields[:5])}"
            f"{'...' if len(contract_fields) > 5 else ''}\n\n"
            "Predictive scores:\n"
        )

        for k, v in evaluation_report.predictive_scores.items():
            prompt += f"  - {k}: {v:.6f}\n"

        prompt += (
            "\nReview this deployment package for completeness "
            "and production readiness."
        )

        return self._call_ds_llm(
            prompt=prompt,
            role="ds_evaluator",
            system_prompt=DEPLOYMENT_SYSTEM_PROMPT,
        )
