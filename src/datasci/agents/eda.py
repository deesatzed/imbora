"""Phase 2: Exploratory Data Analysis Agent.

Performs LLM-guided EDA:
- LLM question generation (QUIS pattern)
- Statistical analysis (correlations, outliers)
- Text column analysis
- LLM synthesis of findings
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import DataScienceConfig
from src.core.models import AgentResult
from src.datasci.agents.base_ds_agent import BaseDataScienceAgent
from src.datasci.models import EDAReport

logger = logging.getLogger("associate.datasci.eda")

EDA_SYSTEM_PROMPT = """You are an expert data scientist performing exploratory data analysis.
Given a dataset profile and audit report, generate insightful analytical questions and findings.
Focus on: feature-target relationships, interaction effects, distribution anomalies,
and actionable patterns that inform feature engineering and model selection.
Be specific and reference column names directly."""

SYNTHESIS_SYSTEM_PROMPT = """You are an expert data scientist synthesizing EDA findings.
Given statistical findings, correlations, and outlier analysis, produce a concise
narrative that highlights the most important patterns and their implications
for modeling. Prioritize actionable insights."""


class EDAAgent(BaseDataScienceAgent):
    """Agent for Phase 2: Exploratory Data Analysis.

    Uses LLM to generate analytical questions, runs statistical analysis,
    and synthesizes findings into an EDAReport.
    """

    def __init__(
        self,
        llm_client: Any,
        model_router: Any,
        repository: Any,
        ds_config: DataScienceConfig,
    ):
        super().__init__(
            name="EDAAgent",
            role="ds_analyst",
            llm_client=llm_client,
            model_router=model_router,
            repository=repository,
            ds_config=ds_config,
        )

    def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Run EDA pipeline.

        Args:
            input_data: Dict with keys:
                - dataset_path: str
                - target_column: str
                - project_id: UUID
                - problem_type: str
                - audit_report: dict (DataAuditReport as dict)
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
            phase="eda",
            config={"target_column": target_column, "problem_type": problem_type},
            task_id=input_data.get("task_id"),
            run_id=input_data.get("run_id"),
        )

        try:
            # Load data
            if dataset_path.endswith(".csv"):
                df = pd.read_csv(dataset_path)
            else:
                df = pd.read_parquet(dataset_path)

            # Step 1: LLM question generation
            questions = self._generate_questions(audit_data, target_column, problem_type)

            # Step 2: Statistical analysis
            findings = self._run_statistical_analysis(df, target_column, problem_type)

            # Step 3: Correlation analysis
            correlations = self._compute_correlations(df, target_column)

            # Step 4: Outlier detection
            outlier_summary = self._detect_outliers(df)

            # Step 5: Text column analysis
            text_analysis = self._analyze_text_columns(df, audit_data)

            # Step 5.5: Attach correlation_with_target to column profiles
            if correlations and "audit_report" in input_data:
                audit_profiles = input_data.get("audit_report", {}).get(
                    "column_profiles", [],
                )
                # Build a lookup from column name to correlation value.
                # Correlation keys are formatted as "{col}_vs_{target}".
                suffix = f"_vs_{target_column}"
                col_corr_map: dict[str, float] = {}
                for corr_key, corr_val in correlations.items():
                    if corr_key.endswith(suffix):
                        col_name = corr_key[: -len(suffix)]
                        col_corr_map[col_name] = corr_val

                for profile_dict in audit_profiles:
                    if isinstance(profile_dict, dict):
                        col_name = profile_dict.get("name", "")
                        if col_name in col_corr_map:
                            profile_dict["correlation_with_target"] = col_corr_map[
                                col_name
                            ]
                    else:
                        col_name = profile_dict.name
                        if col_name in col_corr_map:
                            profile_dict.correlation_with_target = col_corr_map[
                                col_name
                            ]

            # Step 6: LLM synthesis
            synthesis = self._synthesize_findings(
                questions, findings, correlations, outlier_summary, target_column,
            )

            # Step 7: Recommendations
            recommendations = self._generate_recommendations(
                findings, correlations, outlier_summary, text_analysis,
            )

            report = EDAReport(
                questions_generated=questions,
                findings=findings,
                correlations=correlations,
                outlier_summary=outlier_summary,
                text_column_analysis=text_analysis,
                llm_synthesis=synthesis,
                recommendations=recommendations,
            )

            # Extract structured imbalance data from findings
            for finding in findings:
                if finding.get("type") == "target_distribution" and problem_type == "classification":
                    report.imbalance_ratio = finding.get("imbalance_ratio")
                    report.is_imbalanced = bool(finding.get("is_imbalanced", False))
                    report.class_counts = finding.get("class_counts")
                    report.minority_class = finding.get("minority_class")
                    report.majority_class = finding.get("majority_class")
                    break

            self._complete_experiment(
                experiment_id,
                metrics={
                    "questions_count": len(questions),
                    "findings_count": len(findings),
                    "correlation_pairs": len(correlations),
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

    def _generate_questions(
        self, audit_data: dict, target_column: str, problem_type: str,
    ) -> list[str]:
        """Use LLM to generate analytical questions."""
        n_cols = len(audit_data.get("column_profiles", []))
        n_rows = audit_data.get("row_count", "unknown")
        prompt = f"""Given a {problem_type} dataset with target '{target_column}':
Audit summary: {n_cols} columns, {n_rows} rows.

Generate 5-8 specific analytical questions to investigate.
Format: one question per line, prefixed with "Q: "."""

        response = self._call_ds_llm(prompt, system_prompt=EDA_SYSTEM_PROMPT)
        questions = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("Q:"):
                questions.append(line[2:].strip())
            elif line and not line.startswith("#"):
                questions.append(line)
        return questions[:10]  # Cap at 10

    def _run_statistical_analysis(
        self, df: pd.DataFrame, target_column: str, problem_type: str,
    ) -> list[dict[str, Any]]:
        """Run statistical analysis: target distribution, feature-target relationships."""
        findings = []

        # Target distribution
        if target_column in df.columns:
            target = df[target_column]
            if problem_type == "classification":
                vc = target.value_counts()
                imbalance_ratio = vc.max() / vc.min() if vc.min() > 0 else float("inf")
                class_counts = {str(k): int(v) for k, v in vc.items()}
                findings.append({
                    "type": "target_distribution",
                    "detail": f"Class distribution: {vc.to_dict()}",
                    "imbalance_ratio": float(imbalance_ratio),
                    "is_imbalanced": bool(imbalance_ratio > 3.0),
                    "class_counts": class_counts,
                    "minority_class": str(vc.idxmin()),
                    "majority_class": str(vc.idxmax()),
                })
            else:
                findings.append({
                    "type": "target_distribution",
                    "detail": f"Target stats: mean={target.mean():.3f}, std={target.std():.3f}",
                    "skew": float(target.skew()),
                })

        # Numeric feature stats
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if col == target_column:
                continue
            series = df[col].dropna()
            if len(series) < 2:
                continue
            skew = float(series.skew())
            if abs(skew) > 2.0:
                findings.append({
                    "type": "high_skew",
                    "column": col,
                    "skew": skew,
                    "detail": f"Column '{col}' is highly skewed (skew={skew:.2f})",
                })

        return findings

    def _compute_correlations(
        self, df: pd.DataFrame, target_column: str,
    ) -> dict[str, float]:
        """Compute feature-target correlations for numeric columns."""
        correlations = {}
        numeric_df = df.select_dtypes(include=["number"])

        if target_column not in numeric_df.columns:
            return correlations

        for col in numeric_df.columns:
            if col == target_column:
                continue
            corr = numeric_df[col].corr(numeric_df[target_column])
            if not np.isnan(corr):
                correlations[f"{col}_vs_{target_column}"] = round(float(corr), 4)

        return correlations

    def _detect_outliers(self, df: pd.DataFrame) -> dict[str, Any]:
        """Detect outliers using IQR method for numeric columns."""
        outlier_summary = {}
        numeric_cols = df.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((series < lower) | (series > upper)).sum())
            if outlier_count > 0:
                outlier_summary[col] = {
                    "count": outlier_count,
                    "pct": round(outlier_count / len(series) * 100, 2),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                }

        return outlier_summary

    def _analyze_text_columns(
        self, df: pd.DataFrame, audit_data: dict,
    ) -> list[dict[str, Any]]:
        """Analyze text columns identified in the audit."""
        text_analysis = []
        profiles = audit_data.get("column_profiles", [])
        text_cols = [p["name"] for p in profiles if p.get("text_detected", False)]

        for col_name in text_cols:
            if col_name not in df.columns:
                continue
            series = df[col_name].dropna().astype(str)
            if len(series) == 0:
                continue
            text_analysis.append({
                "column": col_name,
                "avg_length": float(series.str.len().mean()),
                "max_length": int(series.str.len().max()),
                "avg_word_count": float(series.str.split().str.len().mean()),
                "unique_ratio": float(series.nunique() / len(series)),
                "sample": series.iloc[0][:200] if len(series) > 0 else "",
            })

        return text_analysis

    def _synthesize_findings(
        self,
        questions: list[str],
        findings: list[dict],
        correlations: dict[str, float],
        outlier_summary: dict,
        target_column: str,
    ) -> str:
        """Use LLM to synthesize all findings into a narrative."""
        # Top correlations
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        if sorted_corr:
            corr_text = "\n".join(f"  {k}: {v}" for k, v in sorted_corr)
        else:
            corr_text = "  None computed"

        # Finding summaries
        finding_text = "\n".join(f"  - {f.get('detail', str(f))}" for f in findings[:10])

        prompt = f"""Synthesize these EDA findings for target '{target_column}':

Questions investigated:
{chr(10).join(f'  - {q}' for q in questions[:5])}

Statistical findings:
{finding_text}

Top correlations with target:
{corr_text}

Outlier columns: {', '.join(outlier_summary.keys()) if outlier_summary else 'None'}

Provide a 2-3 paragraph synthesis focusing on modeling implications."""

        return self._call_ds_llm(prompt, system_prompt=SYNTHESIS_SYSTEM_PROMPT)

    def _generate_recommendations(
        self,
        findings: list[dict],
        correlations: dict[str, float],
        outlier_summary: dict,
        text_analysis: list[dict],
    ) -> list[str]:
        """Generate EDA-based recommendations for feature engineering."""
        recs = []

        # Imbalanced target
        imbalanced = [f for f in findings if f.get("is_imbalanced")]
        if imbalanced:
            recs.append("Apply class balancing (SMOTE, class weights, or stratified sampling)")

        # High skew features
        skewed = [f for f in findings if f.get("type") == "high_skew"]
        if skewed:
            cols = ", ".join(f["column"] for f in skewed)
            recs.append(f"Apply log/box-cox transform to skewed features: {cols}")

        # Strong correlations
        strong = [(k, v) for k, v in correlations.items() if abs(v) > 0.5]
        if strong:
            recs.append(f"Leverage {len(strong)} strong correlations in feature engineering")

        # Outliers
        if outlier_summary:
            cols = ", ".join(outlier_summary.keys())
            recs.append(f"Handle outliers in: {cols}")

        # Text columns
        if text_analysis:
            cols = ", ".join(t["column"] for t in text_analysis)
            recs.append(f"Embed text columns for modeling: {cols}")

        if not recs:
            recs.append("No major issues — dataset is well-suited for modeling")

        return recs
