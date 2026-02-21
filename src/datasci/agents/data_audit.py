"""Phase 1: Data Audit Agent.

Performs comprehensive data quality assessment:
- Column profiling (dtype detection, distributions, missing rates)
- LLM semantic enrichment (per-column treatment planning)
- Label quality assessment (Cleanlab when available)
- LLM semantic analysis (domain understanding, data issues)
- Dataset fingerprinting for reproducibility
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import DataScienceConfig
from src.core.models import AgentResult
from src.datasci.agents.base_ds_agent import BaseDataScienceAgent
from src.datasci.column_ai_enricher import ColumnAIEnricher
from src.datasci.column_profiler import (
    compute_dataset_fingerprint,
    profile_dataset,
)
from src.datasci.models import DataAuditReport

logger = logging.getLogger("associate.datasci.data_audit")

DATA_AUDIT_SYSTEM_PROMPT = """You are an expert data scientist performing a data audit.
Analyze the dataset profile provided and generate a semantic analysis covering:
1. Domain understanding: What real-world domain does this data represent?
2. Data quality issues: Missing data patterns, potential label noise, suspicious distributions.
3. Feature relationships: Which features likely relate to the target? Any redundancy?
4. Text columns: If present, what information do they encode? How should they be processed?
5. Recommended actions: Ordered list of data preparation steps.

Be specific — reference column names and statistics directly. Be concise but thorough."""


class DataAuditAgent(BaseDataScienceAgent):
    """Agent for Phase 1: Data Quality Audit.

    Loads dataset, profiles all columns, assesses label quality,
    and uses LLM for semantic analysis. Produces a DataAuditReport.
    """

    def __init__(
        self,
        llm_client: Any,
        model_router: Any,
        repository: Any,
        ds_config: DataScienceConfig,
    ):
        super().__init__(
            name="DataAuditAgent",
            role="ds_analyst",
            llm_client=llm_client,
            model_router=model_router,
            repository=repository,
            ds_config=ds_config,
        )

    def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Run the data audit pipeline.

        Args:
            input_data: Dict with keys:
                - dataset_path: str — path to dataset file
                - target_column: str — name of target variable
                - project_id: UUID — project identifier
                - problem_type: str — 'classification' or 'regression'
                - task_id: UUID (optional)
                - run_id: UUID (optional)

        Returns:
            AgentResult with DataAuditReport in data.
        """
        dataset_path = input_data["dataset_path"]
        target_column = input_data["target_column"]
        project_id = input_data["project_id"]
        problem_type = input_data.get("problem_type", "classification")

        # Save experiment record
        experiment_id = self._save_experiment(
            project_id=project_id,
            phase="data_audit",
            config={"dataset_path": dataset_path, "target_column": target_column},
            task_id=input_data.get("task_id"),
            run_id=input_data.get("run_id"),
        )

        try:
            # Step 1: Load dataset
            df = self._load_dataset(dataset_path)
            logger.info("Loaded dataset: %d rows x %d columns", df.shape[0], df.shape[1])

            # Step 2: Profile all columns
            profiles = profile_dataset(df, target_column=target_column)
            logger.info("Profiled %d columns", len(profiles))

            # Step 2.5: LLM-enrich column profiles
            enricher = ColumnAIEnricher(self.llm_client, self.model_router)
            sample_rows = df.sample(
                n=min(10, len(df)), random_state=42,
            ).to_dict(orient="records")
            profiles = enricher.enrich_profiles(
                profiles, target_column, problem_type, sample_rows,
            )
            logger.info(
                "Enriched %d column profiles with LLM semantic analysis",
                len(profiles),
            )

            # Step 3: Label quality assessment
            label_issues_count = self._assess_label_quality(df, target_column, problem_type)

            # Step 4: Dataset fingerprint
            fingerprint = compute_dataset_fingerprint(df)

            # Step 5: Compute overall quality score
            quality_score = self._compute_quality_score(profiles, label_issues_count, df.shape[0])

            # Step 6: LLM semantic analysis
            semantic_analysis = self._run_semantic_analysis(
                profiles, target_column, problem_type, df.shape, label_issues_count,
            )

            # Step 7: Generate recommended actions
            recommended_actions = self._generate_recommendations(profiles, label_issues_count)

            report = DataAuditReport(
                dataset_path=dataset_path,
                row_count=df.shape[0],
                column_count=df.shape[1],
                column_profiles=profiles,
                overall_quality_score=round(quality_score, 3),
                label_issues_count=label_issues_count,
                llm_semantic_analysis=semantic_analysis,
                recommended_actions=recommended_actions,
                dataset_fingerprint=fingerprint,
            )

            self._complete_experiment(
                experiment_id,
                metrics={
                    "row_count": df.shape[0],
                    "column_count": df.shape[1],
                    "quality_score": quality_score,
                    "label_issues": label_issues_count,
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

    def _load_dataset(self, path: str) -> pd.DataFrame:
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
            raise ValueError(f"Unsupported format: {suffix}")

    def _assess_label_quality(
        self, df: pd.DataFrame, target_column: str, problem_type: str,
    ) -> int:
        """Assess label quality using Cleanlab when available."""
        if not self.ds_config.cleanlab_enabled:
            return 0

        if target_column not in df.columns:
            logger.warning("Target column '%s' not found in dataset", target_column)
            return 0

        try:
            from cleanlab.filter import find_label_issues
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import cross_val_predict

            # Cleanlab requires predicted probabilities
            # Use a quick gradient boosting classifier for cross-val predictions
            features = df.drop(columns=[target_column]).select_dtypes(
                include=["number"],
            )
            y = df[target_column]

            if len(features.columns) == 0 or len(y.unique()) < 2:
                return 0

            clf = GradientBoostingClassifier(n_estimators=50, random_state=42)
            pred_probs = cross_val_predict(
                clf, features, y, cv=3, method="predict_proba",
            )
            issues = find_label_issues(labels=y, pred_probs=pred_probs)
            count = int(issues.sum())
            logger.info("Cleanlab found %d potential label issues", count)
            return count

        except ImportError:
            logger.info("Cleanlab not installed; skipping label quality assessment")
            return 0
        except Exception as e:
            logger.warning("Label quality assessment failed: %s", e)
            return 0

    def _compute_quality_score(
        self,
        profiles: list,
        label_issues_count: int,
        total_rows: int,
    ) -> float:
        """Compute an overall dataset quality score in [0, 1]."""
        if not profiles:
            return 0.0

        # Components:
        # 1. Completeness: inverse of average missing %
        avg_missing = sum(p.missing_pct for p in profiles) / len(profiles)
        completeness = max(0.0, 1.0 - avg_missing / 100.0)

        # 2. Label quality: fewer issues = higher score
        if total_rows > 0 and label_issues_count > 0:
            label_quality = max(0.0, 1.0 - label_issues_count / total_rows)
        else:
            label_quality = 1.0

        # 3. Diversity: having multiple dtypes suggests richer data
        dtypes = set(p.dtype for p in profiles)
        diversity = min(1.0, len(dtypes) / 3.0)

        # Weighted average
        score = 0.5 * completeness + 0.3 * label_quality + 0.2 * diversity
        return min(1.0, max(0.0, score))

    def _run_semantic_analysis(
        self,
        profiles: list,
        target_column: str,
        problem_type: str,
        shape: tuple,
        label_issues: int,
    ) -> str:
        """Use LLM to generate semantic analysis of the dataset."""
        profile_summary = []
        for p in profiles:
            line = f"- {p.name} ({p.dtype}): cardinality={p.cardinality}, missing={p.missing_pct}%"
            if p.is_target:
                line += " [TARGET]"
            if p.text_detected:
                line += " [TEXT]"
            if p.distribution_summary:
                # Include key stats
                if "mean" in p.distribution_summary:
                    line += f", mean={p.distribution_summary['mean']:.2f}"
                if "entropy" in p.distribution_summary:
                    line += f", entropy={p.distribution_summary['entropy']:.2f}"
            profile_summary.append(line)

        prompt = f"""Dataset: {shape[0]} rows x {shape[1]} columns
Problem type: {problem_type}
Target column: {target_column}
Label issues detected: {label_issues}

Column profiles:
{chr(10).join(profile_summary)}

Provide a semantic analysis of this dataset."""

        return self._call_ds_llm(
            prompt=prompt,
            role="ds_analyst",
            system_prompt=DATA_AUDIT_SYSTEM_PROMPT,
        )

    def _generate_recommendations(self, profiles: list, label_issues: int) -> list[str]:
        """Generate actionable recommendations from the audit."""
        recs = []

        # High missing columns
        high_missing = [p for p in profiles if p.missing_pct > 20]
        if high_missing:
            names = ", ".join(p.name for p in high_missing)
            recs.append(
                f"Investigate high-missing columns ({names})"
                " — consider dropping or advanced imputation"
            )

        # Moderate missing columns
        moderate_missing = [p for p in profiles if 5 < p.missing_pct <= 20]
        if moderate_missing:
            names = ", ".join(p.name for p in moderate_missing)
            recs.append(f"Impute moderately-missing columns ({names})")

        # Text columns
        text_cols = [p for p in profiles if p.text_detected]
        if text_cols:
            names = ", ".join(p.name for p in text_cols)
            recs.append(f"Embed text columns ({names}) using sentence transformers")

        # High cardinality categoricals
        high_card = [p for p in profiles if p.dtype == "categorical" and p.cardinality > 50]
        if high_card:
            names = ", ".join(p.name for p in high_card)
            recs.append(f"Target-encode high-cardinality categoricals ({names})")

        # Label issues
        if label_issues > 0:
            recs.append(f"Review {label_issues} potential label issues detected by Cleanlab")

        # Datetime columns
        dt_cols = [p for p in profiles if p.dtype == "datetime"]
        if dt_cols:
            names = ", ".join(p.name for p in dt_cols)
            recs.append(f"Extract temporal features from datetime columns ({names})")

        if not recs:
            recs.append("Dataset appears clean — proceed to EDA")

        return recs
