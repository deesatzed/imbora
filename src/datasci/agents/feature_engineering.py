"""Phase 3: Feature Engineering Agent.

Orchestrates the LLM-FE evolutionary loop, applies per-column treatments,
and produces a FeatureEngineeringReport with ranked features.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.core.config import DataScienceConfig
from src.core.models import AgentResult
from src.datasci.agents.base_ds_agent import BaseDataScienceAgent
from src.datasci.column_treatment_executor import ColumnTreatmentExecutor
from src.datasci.llm_feature_engineer import LLMFeatureEngineer
from src.datasci.models import ColumnProfile, FeatureEngineeringAssessment, FeatureEngineeringReport

logger = logging.getLogger("associate.datasci.feature_engineering")


class FeatureEngineeringAgent(BaseDataScienceAgent):
    """Agent for Phase 3: Feature Engineering.

    Orchestrates the LLM-FE evolutionary loop plus deterministic
    per-column transforms. Produces FeatureEngineeringReport.
    """

    def __init__(
        self,
        llm_client: Any,
        model_router: Any,
        repository: Any,
        ds_config: DataScienceConfig,
    ):
        super().__init__(
            name="FeatureEngineeringAgent",
            role="ds_analyst",
            llm_client=llm_client,
            model_router=model_router,
            repository=repository,
            ds_config=ds_config,
        )

    def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Run feature engineering pipeline.

        Args:
            input_data: Dict with keys:
                - dataset_path: str
                - target_column: str
                - project_id: UUID
                - problem_type: str
                - audit_report: dict (DataAuditReport)
                - task_id: UUID (optional)
                - run_id: UUID (optional)
        """
        dataset_path = input_data["dataset_path"]
        target_column = input_data["target_column"]
        project_id = input_data["project_id"]
        problem_type = input_data.get("problem_type", "classification")
        audit_data = input_data.get("audit_report", {})

        experiment_id = self._save_experiment(
            project_id=project_id,
            phase="feature_engineering",
            config={
                "target_column": target_column,
                "generations": self.ds_config.llm_fe_generations,
                "population_size": self.ds_config.llm_fe_population_size,
            },
            task_id=input_data.get("task_id"),
            run_id=input_data.get("run_id"),
        )

        try:
            # Load dataset
            if dataset_path.endswith(".csv"):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith(".tsv"):
                df = pd.read_csv(dataset_path, sep="\t")
            else:
                df = pd.read_parquet(dataset_path)

            original_feature_count = len(df.columns) - 1  # Exclude target

            # Build column profiles from audit data
            profiles = [
                ColumnProfile(**p) for p in audit_data.get("column_profiles", [])
            ]

            # Apply per-column treatments based on enriched profiles
            column_profiles = audit_data.get("column_profiles", [])
            if column_profiles:
                # Convert dict profiles to ColumnProfile objects if needed
                profile_objects = []
                for p in column_profiles:
                    if isinstance(p, dict):
                        profile_objects.append(ColumnProfile(**p))
                    else:
                        profile_objects.append(p)

                executor = ColumnTreatmentExecutor()
                df, treatment_features = executor.execute(
                    df, profile_objects, target_column,
                )
                logger.info(
                    "Applied per-column treatments, %d new features",
                    len(treatment_features),
                )

            # Recover FE assessment from upstream (Phase 2.5) if available
            fe_assessment_obj = None
            fe_assessment_data = input_data.get("fe_assessment")
            if fe_assessment_data:
                try:
                    if isinstance(fe_assessment_data, dict):
                        fe_assessment_obj = FeatureEngineeringAssessment(**fe_assessment_data)
                    elif isinstance(fe_assessment_data, FeatureEngineeringAssessment):
                        fe_assessment_obj = fe_assessment_data
                except Exception as e:
                    logger.warning("Could not parse fe_assessment: %s", e)

            # Run LLM-FE evolutionary loop
            fe_engine = LLMFeatureEngineer(
                llm_client=self.llm_client,
                model_router=self.model_router,
                generations=self.ds_config.llm_fe_generations,
                population_size=self.ds_config.llm_fe_population_size,
            )

            fe_result = fe_engine.run(
                df=df,
                target_column=target_column,
                problem_type=problem_type,
                column_profiles=profiles,
                cv_folds=self.ds_config.cv_folds,
                fe_assessment=fe_assessment_obj,
            )

            # Build feature ranking from enhanced DataFrame
            enhanced_df = fe_result["enhanced_df"]
            feature_ranking = self._rank_features(enhanced_df, target_column, problem_type)

            # Identify dropped features (those with zero importance)
            dropped = [
                r["feature"] for r in feature_ranking
                if r["importance"] < 0.001 and r["feature"] != target_column
            ]

            # Build new features list
            new_features = [
                {"name": name, "source": "llm_fe", "transform": "evolutionary"}
                for name in fe_result["new_feature_names"]
            ]
            for det_name in fe_result.get("deterministic_features", []):
                new_features.append({
                    "name": det_name, "source": "deterministic", "transform": "per_column",
                })

            # Include treatment features in the new features list
            if column_profiles:
                for tf_name in treatment_features:
                    if not any(f["name"] == tf_name for f in new_features):
                        new_features.append({
                            "name": tf_name,
                            "source": "column_treatment",
                            "transform": "per_column_enriched",
                        })

            # Transformation code summary
            transform_code = "\n---\n".join(
                t.get("code", "") for t in fe_result.get("best_transforms", [])
            )

            # Persist enhanced DataFrame so downstream agents use it
            import os
            artifacts_dir = self.ds_config.artifacts_dir
            os.makedirs(artifacts_dir, exist_ok=True)
            enhanced_path = os.path.join(artifacts_dir, "enhanced_features.parquet")
            enhanced_df.to_parquet(enhanced_path, index=False)
            logger.info(
                "Saved enhanced DataFrame (%d cols) to %s",
                len(enhanced_df.columns), enhanced_path,
            )

            report = FeatureEngineeringReport(
                original_feature_count=original_feature_count,
                new_features=new_features,
                feature_ranking=feature_ranking,
                llm_fe_generations_run=fe_result["generations_run"],
                best_generation_score=fe_result["best_score"],
                transformation_code=transform_code,
                dropped_features=dropped,
                enhanced_dataset_path=enhanced_path,
            )

            self._complete_experiment(
                experiment_id,
                metrics={
                    "original_features": original_feature_count,
                    "new_features": len(new_features),
                    "baseline_score": fe_result["baseline_score"],
                    "best_score": fe_result["best_score"],
                    "improvement": fe_result["improvement"],
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

    def _rank_features(
        self, df: pd.DataFrame, target_column: str, problem_type: str,
    ) -> list[dict[str, float]]:
        """Rank features by importance using a quick gradient booster."""
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        features = df.drop(
            columns=[target_column], errors="ignore",
        ).select_dtypes(include=["number"])
        y = df[target_column] if target_column in df.columns else None

        if y is None or len(features.columns) == 0:
            return []

        features = features.fillna(features.median())

        try:
            if problem_type == "classification":
                model = GradientBoostingClassifier(
                    n_estimators=50, max_depth=3, random_state=42,
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=50, max_depth=3, random_state=42,
                )

            model.fit(features, y)
            importances = model.feature_importances_

            ranking = []
            for col, imp in zip(features.columns, importances):
                ranking.append({"feature": col, "importance": round(float(imp), 4)})

            ranking.sort(key=lambda x: x["importance"], reverse=True)
            return ranking

        except Exception as e:
            logger.warning("Feature ranking failed: %s", e)
            return []
