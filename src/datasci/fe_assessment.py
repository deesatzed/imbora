"""AI-based Feature Engineering Assessment (Phase 2.5).

Combines statistical signals (MI, correlation, missingness patterns) with
LLM reasoning to assess which FE categories are applicable and propose
specific features before the expensive LLM-FE evolutionary loop runs.

The 8 FE categories assessed:
  1. cross_column_derivation  — Ratios, differences, products
  2. text_feature_extraction  — Titles, patterns, entities from text
  3. missingness_as_signal    — Missing value indicators
  4. interaction_features     — Pairwise column interactions
  5. aggregation_features     — Group-level statistics
  6. temporal_decomposition   — Year/month/day from dates
  7. binning_discretization   — Numeric to categorical bins
  8. encoding_enrichment      — Better encoding than default
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from src.llm.client import LLMMessage
from src.datasci.models import (
    ColumnProfile,
    FeatureEngineeringAssessment,
    FECategoryApplicability,
    ProposedFeature,
)

logger = logging.getLogger("associate.datasci.fe_assessment")

# Maximum rows for MI computation to keep it fast on large datasets.
MI_SAMPLE_SIZE = 5000

# Maximum number of proposed features to avoid bloat.
MAX_PROPOSED_FEATURES = 15

FE_ASSESSMENT_SYSTEM_PROMPT = (
    "You are a senior data scientist performing a Feature Engineering "
    "Assessment.\n\n"
    "Given a dataset description, column profiles with statistics, and "
    "initial statistical assessment results, your job is to:\n\n"
    "1. Refine the category assessments "
    "(upgrade confidence, add rationale, identify columns)\n"
    "2. Propose specific new features that would improve predictive "
    "performance\n"
    "3. Provide an overall FE potential rating and reasoning\n\n"
    "Respond with a JSON object matching this schema exactly:\n"
    "{\n"
    '  "category_assessments": [\n'
    "    {\n"
    '      "category": "cross_column_derivation",\n'
    '      "applicable": true,\n'
    '      "confidence": 0.85,\n'
    '      "rationale": "Age and Fare show complementary signal...",\n'
    '      "applicable_columns": ["Age", "Fare"]\n'
    "    }\n"
    "  ],\n"
    '  "proposed_features": [\n'
    "    {\n"
    '      "name": "Age_Fare_ratio",\n'
    '      "category": "cross_column_derivation",\n'
    '      "source_columns": ["Age", "Fare"],\n'
    '      "transform_description": "Ratio of Age to Fare",\n'
    '      "expected_impact": "medium",\n'
    '      "rationale": "Captures age-normalized spending..."\n'
    "    }\n"
    "  ],\n"
    '  "overall_potential": "high",\n'
    '  "reasoning": "This dataset has strong cross-column and '
    'interaction signals..."\n'
    "}\n\n"
    "Rules:\n"
    '- expected_impact must be one of: "high", "medium", "low"\n'
    '- overall_potential must be one of: "high", "medium", "low", '
    '"none"\n'
    "- Propose at most 15 features, ranked by expected impact\n"
    "- Be specific about which columns each feature uses\n"
    "- Focus on features that would ADD signal, not redundant "
    "transforms\n"
    "- Output ONLY the JSON object, no other text"
)


class FEAssessor:
    """AI-based Feature Engineering Assessment.

    Combines statistical signals (MI, correlation) with LLM reasoning
    to assess which FE categories are applicable and propose features.
    """

    def __init__(self, llm_client: Any, model_router: Any):
        self.llm_client = llm_client
        self.model_router = model_router

    def assess(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str,
        column_profiles: list[ColumnProfile],
        eda_report: dict[str, Any],
    ) -> FeatureEngineeringAssessment:
        """Run the full FE assessment.

        Steps:
        1. Compute statistical signals (MI, correlations, missingness patterns)
        2. Assess each FE category statistically (fast, no LLM)
        3. Call LLM to refine assessment with domain reasoning
        4. Merge statistical + LLM assessments
        5. Propose specific features

        Args:
            df: Input DataFrame.
            target_column: Name of the target column.
            problem_type: 'classification' or 'regression'.
            column_profiles: Column profiles from the data audit phase.
            eda_report: EDA findings dict.

        Returns:
            FeatureEngineeringAssessment with category assessments and
            proposed features.
        """
        if df.empty or target_column not in df.columns:
            logger.warning("Empty DataFrame or missing target column; returning empty assessment.")
            return FeatureEngineeringAssessment(
                dataset_summary="Empty or invalid dataset",
                overall_fe_potential="none",
            )

        # Step 1: Compute statistical signals
        signals = self._compute_statistical_signals(df, target_column, problem_type)

        # Step 2: Statistical category assessment
        stat_categories = self._assess_categories_statistically(signals, column_profiles)

        # Step 3: Generate statistical feature proposals (used as fallback too)
        stat_proposals = self._propose_features_statistically(
            signals, column_profiles, target_column,
        )

        # Step 4: LLM assessment
        llm_categories, llm_proposals, llm_reasoning = self._llm_assess(
            df, target_column, problem_type, column_profiles,
            stat_categories, eda_report,
        )

        # Step 5: Merge assessments
        final_categories = self._merge_assessments(stat_categories, llm_categories)

        # Step 6: Merge proposed features (stat + LLM, deduplicated, capped)
        final_proposals = self._merge_proposals(stat_proposals, llm_proposals)

        # Count column types
        type_counts = signals.get("column_type_counts", {})

        # Determine overall potential
        applicable_count = sum(1 for c in final_categories if c.applicable)
        if applicable_count >= 5:
            overall_potential = "high"
        elif applicable_count >= 3:
            overall_potential = "medium"
        elif applicable_count >= 1:
            overall_potential = "low"
        else:
            overall_potential = "none"

        dataset_summary = (
            f"{len(df)} rows x {len(df.columns)} columns, "
            f"target='{target_column}', problem_type='{problem_type}'"
        )

        return FeatureEngineeringAssessment(
            dataset_summary=dataset_summary,
            total_columns=len(df.columns),
            numeric_columns=type_counts.get("numeric", 0),
            categorical_columns=type_counts.get("categorical", 0),
            text_columns=type_counts.get("text", 0),
            datetime_columns=type_counts.get("datetime", 0),
            category_assessments=final_categories,
            proposed_features=final_proposals,
            overall_fe_potential=overall_potential,
            llm_reasoning=llm_reasoning,
        )

    # ------------------------------------------------------------------
    # Statistical signal computation
    # ------------------------------------------------------------------

    def _compute_statistical_signals(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str,
    ) -> dict[str, Any]:
        """Compute MI, correlations, missingness patterns, and column type counts."""
        signals: dict[str, Any] = {}

        # Column type counts from dtype inspection
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["category", "object", "bool"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        # Remove target from feature lists
        numeric_cols = [c for c in numeric_cols if c != target_column]
        categorical_cols = [c for c in categorical_cols if c != target_column]
        datetime_cols = [c for c in datetime_cols if c != target_column]

        signals["column_type_counts"] = {
            "numeric": len(numeric_cols),
            "categorical": len(categorical_cols),
            "text": 0,  # refined below from profiles
            "datetime": len(datetime_cols),
        }
        signals["numeric_cols"] = numeric_cols
        signals["categorical_cols"] = categorical_cols
        signals["datetime_cols"] = datetime_cols

        # Mutual information (sample for speed)
        mi = self._compute_mutual_information(df, target_column, problem_type, numeric_cols)
        signals["mutual_information"] = mi

        # Correlation matrix (numeric only)
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().to_dict()
            signals["correlation_matrix"] = corr
        else:
            signals["correlation_matrix"] = {}

        # Missingness MI — does the missingness pattern of each column carry signal?
        missingness_mi = self._compute_missingness_mi(df, target_column, problem_type)
        signals["missingness_mi"] = missingness_mi

        # Cardinality per categorical column
        cardinality = {}
        for c in categorical_cols:
            cardinality[c] = int(df[c].nunique())
        signals["cardinality"] = cardinality

        return signals

    def _compute_mutual_information(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str,
        numeric_cols: list[str],
    ) -> dict[str, float]:
        """Compute mutual information of each numeric feature with target."""
        if not numeric_cols or target_column not in df.columns:
            return {}

        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        except ImportError:
            logger.warning("sklearn not available for MI computation; skipping.")
            return {}

        # Sample for speed
        sample = df.sample(n=min(MI_SAMPLE_SIZE, len(df)), random_state=42)
        features = sample[numeric_cols].fillna(0)
        y = sample[target_column]

        if len(y.unique()) < 2:
            return {}

        try:
            if problem_type == "classification":
                mi_values = mutual_info_classif(features, y, random_state=42, n_neighbors=5)
            else:
                mi_values = mutual_info_regression(features, y, random_state=42, n_neighbors=5)

            return {col: float(val) for col, val in zip(numeric_cols, mi_values)}
        except Exception as e:
            logger.warning("MI computation failed: %s", e)
            return {}

    def _compute_missingness_mi(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str,
    ) -> dict[str, float]:
        """Compute MI of each column's missingness indicator vs target."""
        if target_column not in df.columns:
            return {}

        cols_with_missing = [c for c in df.columns if c != target_column and df[c].isna().any()]
        if not cols_with_missing:
            return {}

        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        except ImportError:
            return {}

        # Build missingness indicator matrix
        sample = df.sample(n=min(MI_SAMPLE_SIZE, len(df)), random_state=42)
        miss_matrix = sample[cols_with_missing].isna().astype(float)
        y = sample[target_column]

        if len(y.unique()) < 2:
            return {}

        try:
            if problem_type == "classification":
                mi_values = mutual_info_classif(miss_matrix, y, random_state=42, n_neighbors=5)
            else:
                mi_values = mutual_info_regression(miss_matrix, y, random_state=42, n_neighbors=5)

            return {col: float(val) for col, val in zip(cols_with_missing, mi_values)}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Statistical category assessment
    # ------------------------------------------------------------------

    def _assess_categories_statistically(
        self,
        signals: dict[str, Any],
        profiles: list[ColumnProfile],
    ) -> list[FECategoryApplicability]:
        """Assess each of 8 FE categories based on statistical signals only."""
        categories = []

        categories.append(self._assess_cross_column(signals, profiles))
        categories.append(self._assess_text_features(signals, profiles))
        categories.append(self._assess_missingness(signals, profiles))
        categories.append(self._assess_interactions(signals, profiles))
        categories.append(self._assess_aggregation(signals, profiles))
        categories.append(self._assess_temporal(signals, profiles))
        categories.append(self._assess_binning(signals, profiles))
        categories.append(self._assess_encoding(signals, profiles))

        return categories

    def _assess_cross_column(
        self, signals: dict[str, Any], profiles: list[ColumnProfile],
    ) -> FECategoryApplicability:
        """Cross-column derivation: ratios, differences, products."""
        numeric_cols = signals.get("numeric_cols", [])
        mi = signals.get("mutual_information", {})

        # Find pairs where both have MI > 0
        applicable_cols = []
        if len(numeric_cols) >= 2 and mi:
            cols_with_mi = [c for c in numeric_cols if mi.get(c, 0) > 0.01]
            applicable_cols = cols_with_mi[:10]  # cap for sanity

        applicable = len(applicable_cols) >= 2
        confidence = min(0.9, len(applicable_cols) * 0.15) if applicable else 0.0

        return FECategoryApplicability(
            category="cross_column_derivation",
            applicable=applicable,
            confidence=round(confidence, 2),
            rationale=(
                f"{len(applicable_cols)} numeric columns with non-trivial MI detected"
                if applicable else "Insufficient numeric columns with target signal"
            ),
            applicable_columns=applicable_cols,
        )

    def _assess_text_features(
        self, signals: dict[str, Any], profiles: list[ColumnProfile],
    ) -> FECategoryApplicability:
        """Text feature extraction: titles, patterns, entities."""
        text_cols = [
            p.name for p in profiles
            if p.text_detected or p.dtype == "text"
        ]
        # Also check for object columns with high cardinality (potential text)
        cat_cols = signals.get("categorical_cols", [])
        cardinality = signals.get("cardinality", {})
        for c in cat_cols:
            if cardinality.get(c, 0) > 50 and c not in text_cols:
                text_cols.append(c)

        # Update text count in signals
        signals["column_type_counts"]["text"] = len(text_cols)

        applicable = len(text_cols) > 0
        confidence = min(0.9, len(text_cols) * 0.3) if applicable else 0.0

        return FECategoryApplicability(
            category="text_feature_extraction",
            applicable=applicable,
            confidence=round(confidence, 2),
            rationale=(
                f"{len(text_cols)} text/high-cardinality columns detected"
                if applicable else "No text columns detected"
            ),
            applicable_columns=text_cols,
        )

    def _assess_missingness(
        self, signals: dict[str, Any], profiles: list[ColumnProfile],
    ) -> FECategoryApplicability:
        """Missingness as signal: missing value indicators."""
        missingness_mi = signals.get("missingness_mi", {})
        cols_with_high_missing = [
            p.name for p in profiles if p.missing_pct > 5
        ]

        # Columns where missingness MI is non-trivial
        mi_cols = [c for c, v in missingness_mi.items() if v > 0.005]
        applicable_cols = list(set(cols_with_high_missing + mi_cols))

        applicable = len(applicable_cols) > 0
        confidence = 0.0
        if applicable:
            # Higher confidence if MI supports it
            if mi_cols:
                confidence = min(0.9, 0.4 + len(mi_cols) * 0.15)
            else:
                confidence = min(0.6, 0.2 + len(applicable_cols) * 0.1)

        return FECategoryApplicability(
            category="missingness_as_signal",
            applicable=applicable,
            confidence=round(confidence, 2),
            rationale=(
                f"{len(applicable_cols)} columns with significant missingness "
                f"({len(mi_cols)} with MI signal)"
                if applicable else "No significant missingness patterns"
            ),
            applicable_columns=applicable_cols,
        )

    def _assess_interactions(
        self, signals: dict[str, Any], profiles: list[ColumnProfile],
    ) -> FECategoryApplicability:
        """Interaction features: pairwise column interactions."""
        numeric_cols = signals.get("numeric_cols", [])
        mi = signals.get("mutual_information", {})

        # Columns with meaningful MI
        high_mi_cols = [c for c in numeric_cols if mi.get(c, 0) > 0.01]
        applicable = len(high_mi_cols) >= 2
        confidence = min(0.85, len(high_mi_cols) * 0.12) if applicable else 0.0

        return FECategoryApplicability(
            category="interaction_features",
            applicable=applicable,
            confidence=round(confidence, 2),
            rationale=(
                f"{len(high_mi_cols)} columns with target signal available for interactions"
                if applicable else "Insufficient columns with target signal for interactions"
            ),
            applicable_columns=high_mi_cols[:10],
        )

    def _assess_aggregation(
        self, signals: dict[str, Any], profiles: list[ColumnProfile],
    ) -> FECategoryApplicability:
        """Aggregation features: group-level statistics."""
        cat_cols = signals.get("categorical_cols", [])
        cardinality = signals.get("cardinality", {})

        # Columns with repeated values (suitable for groupby aggregation)
        # Cardinality < 50% of row count means we'll get meaningful groups
        agg_cols = [c for c in cat_cols if cardinality.get(c, 0) > 1]
        applicable = len(agg_cols) > 0
        confidence = min(0.8, len(agg_cols) * 0.2) if applicable else 0.0

        return FECategoryApplicability(
            category="aggregation_features",
            applicable=applicable,
            confidence=round(confidence, 2),
            rationale=(
                f"{len(agg_cols)} categorical columns with repeated values for groupby"
                if applicable else "No categorical columns suitable for aggregation"
            ),
            applicable_columns=agg_cols,
        )

    def _assess_temporal(
        self, signals: dict[str, Any], profiles: list[ColumnProfile],
    ) -> FECategoryApplicability:
        """Temporal decomposition: year/month/day from dates."""
        dt_cols = signals.get("datetime_cols", [])
        # Also check profiles for datetime dtype
        profile_dt = [p.name for p in profiles if p.dtype == "datetime"]
        all_dt = list(set(dt_cols + profile_dt))

        applicable = len(all_dt) > 0
        confidence = min(0.95, len(all_dt) * 0.4) if applicable else 0.0

        return FECategoryApplicability(
            category="temporal_decomposition",
            applicable=applicable,
            confidence=round(confidence, 2),
            rationale=(
                f"{len(all_dt)} datetime columns detected for temporal decomposition"
                if applicable else "No datetime columns detected"
            ),
            applicable_columns=all_dt,
        )

    def _assess_binning(
        self, signals: dict[str, Any], profiles: list[ColumnProfile],
    ) -> FECategoryApplicability:
        """Binning/discretization: numeric to categorical bins."""
        numeric_cols = signals.get("numeric_cols", [])
        mi = signals.get("mutual_information", {})

        # Numeric columns with some MI signal (potential non-linear relationship)
        binning_cols = [c for c in numeric_cols if mi.get(c, 0) > 0.005]
        applicable = len(binning_cols) > 0
        confidence = min(0.7, len(binning_cols) * 0.12) if applicable else 0.0

        return FECategoryApplicability(
            category="binning_discretization",
            applicable=applicable,
            confidence=round(confidence, 2),
            rationale=(
                f"{len(binning_cols)} numeric columns with target signal suitable for binning"
                if applicable else "No numeric columns with sufficient signal for binning"
            ),
            applicable_columns=binning_cols,
        )

    def _assess_encoding(
        self, signals: dict[str, Any], profiles: list[ColumnProfile],
    ) -> FECategoryApplicability:
        """Encoding enrichment: better encoding than default."""
        cat_cols = signals.get("categorical_cols", [])
        cardinality = signals.get("cardinality", {})

        # Columns where encoding strategy might benefit from enrichment
        # High cardinality or columns with current suboptimal encoding
        encoding_cols = []
        for p in profiles:
            if p.dtype == "categorical" and p.name in cat_cols:
                # High cardinality categories benefit from target/frequency encoding
                if cardinality.get(p.name, 0) > 5:
                    encoding_cols.append(p.name)

        applicable = len(encoding_cols) > 0
        confidence = min(0.75, len(encoding_cols) * 0.15) if applicable else 0.0

        return FECategoryApplicability(
            category="encoding_enrichment",
            applicable=applicable,
            confidence=round(confidence, 2),
            rationale=(
                f"{len(encoding_cols)} categorical columns could benefit from richer encoding"
                if applicable else "No categorical columns needing encoding enrichment"
            ),
            applicable_columns=encoding_cols,
        )

    # ------------------------------------------------------------------
    # LLM assessment
    # ------------------------------------------------------------------

    def _llm_assess(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str,
        profiles: list[ColumnProfile],
        stat_categories: list[FECategoryApplicability],
        eda_report: dict[str, Any],
    ) -> tuple[list[FECategoryApplicability], list[ProposedFeature], str]:
        """Call LLM to refine category assessment and propose features.

        Returns:
            Tuple of (llm_categories, llm_proposals, llm_reasoning).
            On failure returns empty lists and empty reasoning.
        """
        try:
            models = self.model_router.get_model_chain("ds_analyst")
        except Exception as e:
            logger.warning(
                "Could not resolve ds_analyst model: %s. "
                "Using statistical-only.", e,
            )
            return [], [], ""

        # Build user prompt
        prompt = self._build_llm_prompt(
            df, target_column, problem_type,
            profiles, stat_categories, eda_report,
        )

        messages = [
            LLMMessage(role="system", content=FE_ASSESSMENT_SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt),
        ]

        try:
            response = self.llm_client.complete_with_fallback(
                messages=messages,
                models=models,
                max_tokens=4096,
                temperature=0.3,
            )
            return self._parse_llm_response(response.content)
        except Exception as e:
            logger.warning("LLM FE assessment call failed: %s. Using statistical-only.", e)
            return [], [], ""

    def _build_llm_prompt(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str,
        profiles: list[ColumnProfile],
        stat_categories: list[FECategoryApplicability],
        eda_report: dict[str, Any],
    ) -> str:
        """Build the user prompt for LLM assessment."""
        parts = []

        # Dataset summary
        parts.append(f"Dataset: {len(df)} rows x {len(df.columns)} columns")
        parts.append(f"Target: {target_column} ({problem_type})")
        parts.append(f"Target cardinality: {df[target_column].nunique()}")
        parts.append("")

        # Column profiles
        parts.append("Column Profiles:")
        for p in profiles:
            if p.is_target:
                continue
            line = (
                f"  - {p.name}: dtype={p.dtype}, "
                f"cardinality={p.cardinality}, "
                f"missing={p.missing_pct:.1f}%"
            )
            if p.correlation_with_target is not None:
                line += f", corr_target={p.correlation_with_target:.3f}"
            if p.semantic_meaning:
                line += f", semantic={p.semantic_meaning}"
            if p.text_detected:
                line += ", text_detected=True"
            parts.append(line)
        parts.append("")

        # Statistical pre-assessment
        parts.append("Statistical Pre-Assessment:")
        for cat in stat_categories:
            status = "APPLICABLE" if cat.applicable else "not applicable"
            parts.append(
                f"  - {cat.category}: {status} "
                f"(confidence={cat.confidence:.2f}) — {cat.rationale}"
            )
        parts.append("")

        # EDA findings (if available)
        if eda_report:
            parts.append("EDA Findings:")
            if "imbalance_ratio" in eda_report and eda_report["imbalance_ratio"]:
                parts.append(f"  - Imbalance ratio: {eda_report['imbalance_ratio']}")
            if "is_imbalanced" in eda_report:
                parts.append(f"  - Is imbalanced: {eda_report['is_imbalanced']}")
            if "recommendations" in eda_report:
                for rec in eda_report["recommendations"][:5]:
                    parts.append(f"  - {rec}")
            parts.append("")

        # Sample rows
        sample = df.sample(n=min(3, len(df)), random_state=42)
        parts.append("Sample rows (3):")
        for _, row in sample.iterrows():
            row_parts = []
            for c in df.columns:
                if c == target_column:
                    continue
                val = str(row[c])
                if len(val) > 50:
                    val = val[:50] + "..."
                row_parts.append(f"{c}={val}")
            parts.append("  " + ", ".join(row_parts))

        return "\n".join(parts)

    def _parse_llm_response(
        self, content: str,
    ) -> tuple[list[FECategoryApplicability], list[ProposedFeature], str]:
        """Parse LLM JSON response into structured objects.

        Returns (categories, proposals, reasoning). On parse failure,
        returns empty lists and the raw content as reasoning.
        """
        # Strip markdown code fences if present
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (code fences)
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON response; using as reasoning only.")
            return [], [], content[:500]

        categories = []
        for cat_data in data.get("category_assessments", []):
            try:
                categories.append(FECategoryApplicability(**cat_data))
            except Exception:
                continue

        proposals = []
        for prop_data in data.get("proposed_features", []):
            try:
                proposals.append(ProposedFeature(**prop_data))
            except Exception:
                continue

        reasoning = data.get("reasoning", "")
        return categories, proposals, reasoning

    # ------------------------------------------------------------------
    # Merge logic
    # ------------------------------------------------------------------

    def _merge_assessments(
        self,
        stat_categories: list[FECategoryApplicability],
        llm_categories: list[FECategoryApplicability],
    ) -> list[FECategoryApplicability]:
        """Merge statistical and LLM category assessments.

        Rules:
        - Statistical assessment is the baseline
        - LLM can upgrade confidence, add rationale, add applicable columns
        - LLM CANNOT downgrade a category below statistical evidence
          (prevents LLM hallucination from removing real signal)
        """
        if not llm_categories:
            return stat_categories

        llm_map = {c.category: c for c in llm_categories}
        merged = []

        for stat in stat_categories:
            llm = llm_map.get(stat.category)
            if llm is None:
                merged.append(stat)
                continue

            # LLM cannot downgrade applicability if statistical says True
            applicable = stat.applicable or llm.applicable

            # LLM cannot reduce confidence below statistical level
            confidence = max(stat.confidence, llm.confidence)

            # Prefer LLM rationale if it's more detailed, else keep stat
            rationale = (
                llm.rationale
                if len(llm.rationale) > len(stat.rationale)
                else stat.rationale
            )

            # Union of applicable columns
            all_cols = list(dict.fromkeys(stat.applicable_columns + llm.applicable_columns))

            merged.append(FECategoryApplicability(
                category=stat.category,
                applicable=applicable,
                confidence=round(confidence, 2),
                rationale=rationale,
                applicable_columns=all_cols,
            ))

        return merged

    def _merge_proposals(
        self,
        stat_proposals: list[ProposedFeature],
        llm_proposals: list[ProposedFeature],
    ) -> list[ProposedFeature]:
        """Merge and deduplicate feature proposals, capped at MAX_PROPOSED_FEATURES."""
        # LLM proposals first (higher quality), then stat fallbacks
        seen_names: set[str] = set()
        merged: list[ProposedFeature] = []

        for p in llm_proposals + stat_proposals:
            if p.name not in seen_names and len(merged) < MAX_PROPOSED_FEATURES:
                seen_names.add(p.name)
                merged.append(p)

        # Sort by expected impact: high > medium > low
        impact_order = {"high": 0, "medium": 1, "low": 2}
        merged.sort(key=lambda p: impact_order.get(p.expected_impact, 3))

        return merged

    # ------------------------------------------------------------------
    # Statistical feature proposals (fallback)
    # ------------------------------------------------------------------

    def _propose_features_statistically(
        self,
        signals: dict[str, Any],
        profiles: list[ColumnProfile],
        target_column: str,
    ) -> list[ProposedFeature]:
        """Generate feature proposals from statistical signals alone.

        Used as fallback when LLM is unavailable or as baseline
        proposals merged with LLM output.
        """
        proposals: list[ProposedFeature] = []
        mi = signals.get("mutual_information", {})
        missingness_mi = signals.get("missingness_mi", {})

        # Top MI pairs -> cross-column ratios
        sorted_mi = sorted(mi.items(), key=lambda x: x[1], reverse=True)
        top_mi = [c for c, v in sorted_mi if v > 0.01]

        for i, col_a in enumerate(top_mi[:5]):
            for col_b in top_mi[i + 1: i + 4]:
                proposals.append(ProposedFeature(
                    name=f"{col_a}_{col_b}_ratio",
                    category="cross_column_derivation",
                    source_columns=[col_a, col_b],
                    transform_description=f"Ratio of {col_a} to {col_b}",
                    expected_impact="medium",
                    rationale="Both columns have MI > 0.01 with target",
                ))
                if len(proposals) >= 5:
                    break
            if len(proposals) >= 5:
                break

        # Missingness indicators
        for col, mi_val in sorted(missingness_mi.items(), key=lambda x: x[1], reverse=True)[:3]:
            if mi_val > 0.005:
                proposals.append(ProposedFeature(
                    name=f"{col}_is_missing",
                    category="missingness_as_signal",
                    source_columns=[col],
                    transform_description=f"Binary indicator: is {col} missing",
                    expected_impact="medium" if mi_val > 0.02 else "low",
                    rationale=f"Missingness MI with target = {mi_val:.4f}",
                ))

        # Interaction features for top-2 MI columns
        if len(top_mi) >= 2:
            proposals.append(ProposedFeature(
                name=f"{top_mi[0]}_x_{top_mi[1]}",
                category="interaction_features",
                source_columns=[top_mi[0], top_mi[1]],
                transform_description=f"Product of {top_mi[0]} and {top_mi[1]}",
                expected_impact="medium",
                rationale="Top-2 MI columns combined for interaction signal",
            ))

        # Temporal decomposition proposals
        datetime_cols = signals.get("datetime_cols", [])
        for col in datetime_cols[:2]:
            proposals.append(ProposedFeature(
                name=f"{col}_month",
                category="temporal_decomposition",
                source_columns=[col],
                transform_description=f"Extract month from {col}",
                expected_impact="medium",
                rationale="Temporal decomposition from datetime column",
            ))

        return proposals[:MAX_PROPOSED_FEATURES]
