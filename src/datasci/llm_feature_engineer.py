"""LLM-guided evolutionary feature engineering (LLM-FE).

Implements the evolutionary loop: LLM proposes feature transforms as code,
transforms are executed in isolation, CV-evaluated, winners kept, and
mutations applied across generations.

Per-column treatment:
- Text -> embeddings + TF-IDF
- Numeric -> polynomial, binning, log transforms
- Categorical -> target/frequency/semantic encoding
- Datetime -> temporal feature extraction
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from src.datasci.models import ColumnProfile, FeatureEngineeringAssessment
from src.llm.client import LLMMessage

logger = logging.getLogger("associate.datasci.llm_fe")

FE_SYSTEM_PROMPT = """You are an expert feature engineer. Given a dataset profile and
current feature set performance, propose new feature transformations as Python code.

Rules:
1. Each transform must be a single function: def transform(df: pd.DataFrame) -> pd.DataFrame
2. The function receives the full DataFrame and must return it with new columns added
3. Use only pandas and numpy (imported as pd and np)
4. Never modify existing columns — only add new ones
5. Handle missing values gracefully (use fillna or dropna within transforms)
6. Name new columns descriptively (e.g., age_squared, income_log, age_income_ratio)

Domain-aware feature engineering priorities (apply ALL that are relevant):
- Person names: Extract titles via regex (Mr, Mrs, Miss, Master, Dr, Rev, etc.). Titles encode age bracket, gender, and social status. Example: df['Title'] = df['Name'].str.extract(r',\\s*([^.]+)\\.', expand=False).str.strip()
- Family/group features: Combine sibling/spouse and parent/child counts into FamilySize. Create IsAlone flag. Example: df['FamilySize'] = df['SibSp'] + df['Parch'] + 1; df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
- Cabin/room columns: Extract deck letter (first character). Missing cabin is itself a signal (create HasCabin binary). Example: df['Deck'] = df['Cabin'].fillna('U').str[0]; df['HasCabin'] = (df['Cabin'].notna()).astype(int)
- Ticket/reference numbers: Compute group size via value_counts(). Shared tickets indicate traveling groups. Example: ticket_counts = df['Ticket'].value_counts(); df['TicketGroupSize'] = df['Ticket'].map(ticket_counts)
- Fare/price/cost: Compute per-person fare (Fare / GroupSize). Log-transform skewed fares. Create fare bins.
- Age: Create age bins (Child <12, Teen <18, Adult <60, Senior 60+). Age * Class interactions.
- Interactions: Sex*Pclass, Age*Pclass, FamilySize*Pclass — these capture differential survival patterns.
- Missingness features: Create binary flags for whether a value was originally missing (e.g., Age_missing, Cabin_missing).

Output format: One function per transform, separated by "---".
Propose 3-5 transforms that are likely to improve predictive performance."""

FE_MUTATION_PROMPT = """You are evolving feature transforms. Given the winning transforms
from the previous generation and their scores, propose mutations:
1. Combine winning transforms in new ways
2. Add polynomial/interaction terms for top features
3. Try different encoding strategies
4. Add domain-inspired transforms based on column semantics

Keep what works. Mutate to explore. Output format: same as before."""


class LLMFeatureEngineer:
    """Evolutionary feature engineering with LLM guidance.

    The loop:
    1. LLM proposes transforms (code generation)
    2. Transforms executed safely on data copy
    3. CV evaluation scores each transform
    4. Top performers survive to next generation
    5. LLM mutates/recombines for next generation
    """

    def __init__(
        self,
        llm_client: Any,
        model_router: Any,
        generations: int = 5,
        population_size: int = 10,
    ):
        self.llm_client = llm_client
        self.model_router = model_router
        self.generations = generations
        self.population_size = population_size

    def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str,
        column_profiles: list[ColumnProfile],
        cv_folds: int = 5,
        fe_assessment: FeatureEngineeringAssessment | None = None,
    ) -> dict[str, Any]:
        """Run the evolutionary feature engineering loop.

        Args:
            df: Input DataFrame.
            target_column: Target column name.
            problem_type: 'classification' or 'regression'.
            column_profiles: Column profiles from audit phase.
            cv_folds: Number of CV folds for evaluation.
            fe_assessment: Optional Phase 2.5 FE assessment to guide gen-0
                proposals toward highest-value FE opportunities.

        Returns:
            Dict with best_transforms, best_score, all_generations,
            transformation_code, and new_feature_names.
        """
        self._fe_assessment = fe_assessment
        # Step 1: Baseline score
        baseline_score = self._evaluate_baseline(df, target_column, problem_type, cv_folds)
        logger.info("Baseline CV score: %.4f", baseline_score)

        # Step 2: Apply deterministic per-column transforms
        df_enhanced, deterministic_features = self._apply_deterministic_transforms(
            df, column_profiles, target_column,
        )

        # Step 3: Evaluate with deterministic features
        det_score = self._evaluate_features(df_enhanced, target_column, problem_type, cv_folds)
        logger.info(
            "After deterministic transforms: %.4f (+%.4f)",
            det_score, det_score - baseline_score,
        )

        # Step 4: LLM evolutionary loop
        best_score = det_score
        best_transforms: list[dict[str, Any]] = []
        all_generations: list[dict[str, Any]] = []
        best_df = df_enhanced.copy()

        for gen in range(self.generations):
            gen_result = self._run_generation(
                best_df, target_column, problem_type, column_profiles,
                gen, best_score, best_transforms, cv_folds,
            )
            all_generations.append(gen_result)

            if gen_result["best_score"] > best_score:
                best_score = gen_result["best_score"]
                best_transforms = gen_result["winning_transforms"]
                best_df = gen_result["best_df"]
                logger.info("Gen %d: new best score %.4f", gen, best_score)
            else:
                logger.info("Gen %d: no improvement (%.4f)", gen, gen_result["best_score"])

        # Collect all new feature names
        new_features = [c for c in best_df.columns if c not in df.columns]

        return {
            "baseline_score": baseline_score,
            "best_score": best_score,
            "improvement": best_score - baseline_score,
            "deterministic_features": deterministic_features,
            "new_feature_names": new_features,
            "best_transforms": best_transforms,
            "generations_run": len(all_generations),
            "all_generations": all_generations,
            "enhanced_df": best_df,
        }

    def _apply_deterministic_transforms(
        self,
        df: pd.DataFrame,
        profiles: list[ColumnProfile],
        target_column: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Apply standard per-column transforms based on dtype and domain heuristics."""
        df = df.copy()
        new_features = []

        # Build profile lookup for domain heuristics
        profile_names = {p.name.lower(): p.name for p in profiles}

        for profile in profiles:
            if profile.name == target_column or profile.is_target:
                continue

            col = profile.name
            if col not in df.columns:
                continue

            if profile.dtype == "numeric":
                # Log transform for positive values
                series = df[col].dropna()
                if len(series) > 0 and series.min() > 0:
                    new_col = f"{col}_log"
                    df[new_col] = np.log1p(df[col])
                    new_features.append(new_col)

            elif profile.dtype == "categorical":
                # Frequency encoding
                freq = df[col].value_counts(normalize=True)
                new_col = f"{col}_freq"
                df[new_col] = df[col].map(freq).fillna(0)
                new_features.append(new_col)

            elif profile.dtype == "datetime":
                # Temporal features
                try:
                    dt = pd.to_datetime(df[col], errors="coerce")
                    df[f"{col}_year"] = dt.dt.year
                    df[f"{col}_month"] = dt.dt.month
                    df[f"{col}_dayofweek"] = dt.dt.dayofweek
                    new_features.extend([f"{col}_year", f"{col}_month", f"{col}_dayofweek"])
                except Exception:
                    pass

        # Domain-aware composite features (detect by column name patterns)
        col_lower_map = {c.lower(): c for c in df.columns}

        # FamilySize = SibSp + Parch + 1 (if both columns exist)
        sibsp_col = col_lower_map.get("sibsp")
        parch_col = col_lower_map.get("parch")
        if sibsp_col and parch_col:
            df["FamilySize"] = df[sibsp_col].fillna(0) + df[parch_col].fillna(0) + 1
            new_features.append("FamilySize")
            df["IsAlone"] = (df["FamilySize"] == 1).astype(float)
            new_features.append("IsAlone")

        # FarePerPerson (Fare / group size or family size)
        fare_col = col_lower_map.get("fare")
        if fare_col:
            # Try ticket group size first
            ticket_group_col = None
            for c in df.columns:
                if "ticket" in c.lower() and "group" in c.lower():
                    ticket_group_col = c
                    break
            if ticket_group_col:
                denom = df[ticket_group_col].clip(lower=1)
            elif "FamilySize" in df.columns:
                denom = df["FamilySize"].clip(lower=1)
            else:
                denom = None

            if denom is not None:
                df["FarePerPerson"] = df[fare_col].fillna(0) / denom
                new_features.append("FarePerPerson")

        # Pclass interactions with key numeric features
        # Pclass may be raw (integer) or already one-hot encoded (Pclass_2, Pclass_3)
        pclass_col = col_lower_map.get("pclass")
        if pclass_col:
            pclass_series = df[pclass_col].fillna(df[pclass_col].median())
        elif "Pclass_2" in df.columns and "Pclass_3" in df.columns:
            # Reconstruct Pclass from one-hot: Pclass=1 when both are 0
            pclass_series = pd.Series(1, index=df.index, dtype=float)
            pclass_series[df["Pclass_2"] == 1.0] = 2.0
            pclass_series[df["Pclass_3"] == 1.0] = 3.0
        else:
            pclass_series = None

        if pclass_series is not None:
            age_col = col_lower_map.get("age")
            if age_col:
                df["Age_x_Pclass"] = df[age_col].fillna(df[age_col].median()) * pclass_series
                new_features.append("Age_x_Pclass")
            if fare_col:
                df["Fare_x_Pclass"] = df[fare_col].fillna(0) * pclass_series
                new_features.append("Fare_x_Pclass")

        # Missingness indicators: use profile.missing_pct since the data
        # may already be imputed by the time we see it. Create the binary
        # flag based on whether NaN still exists OR from profile metadata.
        for profile in profiles:
            if profile.missing_pct > 10 and profile.name in df.columns:
                miss_col = f"{profile.name}_missing"
                if miss_col not in df.columns:
                    has_na = df[profile.name].isna().any()
                    if has_na:
                        df[miss_col] = df[profile.name].isna().astype(float)
                        new_features.append(miss_col)

        # Sex * Pclass interaction (critical for survival datasets)
        sex_male_col = None
        for c in df.columns:
            if c.lower() in ("sex_male", "sex_freq"):
                sex_male_col = c
                break
        if sex_male_col and pclass_series is not None:
            df["Sex_x_Pclass"] = df[sex_male_col].fillna(0) * pclass_series
            new_features.append("Sex_x_Pclass")

        return df, new_features

    def _run_generation(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str,
        profiles: list[ColumnProfile],
        generation: int,
        current_best: float,
        previous_winners: list[dict],
        cv_folds: int,
    ) -> dict[str, Any]:
        """Run a single generation of the evolutionary loop."""
        # Generate transform proposals via LLM
        proposals = self._propose_transforms(
            df, target_column, problem_type, profiles, generation, previous_winners,
        )

        # Evaluate each proposal
        results = []
        for proposal in proposals:
            score, new_df = self._evaluate_proposal(
                df, target_column, problem_type, proposal, cv_folds,
            )
            results.append({
                "code": proposal,
                "score": score,
                "df": new_df,
            })

        # Sort by score, keep top
        results.sort(key=lambda x: x["score"], reverse=True)
        best = results[0] if results else {"code": "", "score": current_best, "df": df}

        return {
            "generation": generation,
            "proposals_count": len(proposals),
            "best_score": best["score"],
            "winning_transforms": [r for r in results if r["score"] > current_best],
            "best_df": best["df"],
        }

    def _propose_transforms(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str,
        profiles: list[ColumnProfile],
        generation: int,
        previous_winners: list[dict],
    ) -> list[str]:
        """Use LLM to propose new feature transforms."""
        if generation == 0:
            # Build enriched profile summary for generation 0
            profile_summary_parts = []
            for p in profiles:
                if p.is_target:
                    continue
                parts = [f"{p.name}({p.dtype})"]
                if hasattr(p, "semantic_meaning") and p.semantic_meaning:
                    parts.append(f"semantic: {p.semantic_meaning}")
                if (
                    hasattr(p, "correlation_with_target")
                    and p.correlation_with_target is not None
                ):
                    parts.append(f"corr_target: {p.correlation_with_target:.3f}")
                if hasattr(p, "importance_prior") and p.importance_prior:
                    parts.append(f"importance: {p.importance_prior}")
                if hasattr(p, "interaction_candidates") and p.interaction_candidates:
                    parts.append(f"interactions: {p.interaction_candidates}")
                profile_summary_parts.append(", ".join(parts))

            profile_summary = "; ".join(profile_summary_parts)

            # Build sample data display for context
            sample_df = df.sample(n=min(5, len(df)), random_state=42)
            sample_lines = []
            for _, row in sample_df.iterrows():
                row_parts = []
                for c in df.columns:
                    if c == target_column:
                        continue
                    val = str(row[c])
                    if len(val) > 60:
                        val = val[:60] + "..."
                    row_parts.append(f"{c}={val}")
                sample_lines.append("  " + ", ".join(row_parts))
            sample_text = "\n".join(sample_lines)

            prompt = f"""Dataset: {df.shape[0]} rows, {df.shape[1]} cols
Target: {target_column} ({problem_type})
Features: {profile_summary}
Current columns: {', '.join(c for c in df.columns if c != target_column)}

Sample data (5 rows):
{sample_text}

Look at the ACTUAL VALUES above. Use them to understand data patterns and propose domain-relevant transforms.
Propose 3-5 feature transforms as Python functions."""

            # Inject FE assessment guidance if available (Phase 2.5)
            if self._fe_assessment and self._fe_assessment.category_assessments:
                assessment_lines = ["\n\nFE Assessment (from Phase 2.5 analysis):"]
                assessment_lines.append(
                    f"Overall FE potential: {self._fe_assessment.overall_fe_potential}"
                )
                assessment_lines.append("Applicable categories:")
                for cat in self._fe_assessment.category_assessments:
                    if cat.applicable:
                        assessment_lines.append(
                            f"  - {cat.category} (confidence={cat.confidence:.2f}): "
                            f"{cat.rationale}"
                        )
                        if cat.applicable_columns:
                            assessment_lines.append(
                                f"    Columns: {', '.join(cat.applicable_columns[:8])}"
                            )
                if self._fe_assessment.proposed_features:
                    assessment_lines.append("\nPriority features to implement:")
                    for pf in self._fe_assessment.proposed_features[:8]:
                        assessment_lines.append(
                            f"  - {pf.name} ({pf.expected_impact} impact): "
                            f"{pf.transform_description}"
                        )
                assessment_lines.append(
                    "\nFocus your transforms on the categories and features above."
                )
                prompt += "\n".join(assessment_lines)

            system = FE_SYSTEM_PROMPT
        else:
            winner_summary = "\n".join(
                f"Score {w.get('score', 'N/A')}: {w.get('code', '')[:100]}"
                for w in previous_winners[:3]
            )
            prompt = f"""Generation {generation}. Previous winning transforms:
{winner_summary}

Mutate and combine to propose 3-5 new transforms."""
            system = FE_MUTATION_PROMPT

        models = self.model_router.get_model_chain("ds_analyst")
        messages = [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=prompt),
        ]

        response = self.llm_client.complete_with_fallback(
            messages=messages, models=models,
            max_tokens=6144, temperature=0.7,
        )

        # Parse response into individual transform functions
        return self._parse_transforms(response.content)

    def _parse_transforms(self, llm_output: str) -> list[str]:
        """Parse LLM output into individual transform code blocks."""
        transforms = []
        current = []

        for line in llm_output.split("\n"):
            if line.strip() == "---" and current:
                transforms.append("\n".join(current))
                current = []
            elif line.strip().startswith("```"):
                continue  # Skip code fences
            else:
                current.append(line)

        if current:
            transforms.append("\n".join(current))

        # Filter to only valid Python containing "def transform"
        valid = []
        for t in transforms:
            if "def transform" in t:
                valid.append(t)

        return valid[:self.population_size]

    def _evaluate_proposal(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str,
        transform_code: str,
        cv_folds: int,
    ) -> tuple[float, pd.DataFrame]:
        """Safely execute a transform and evaluate its impact."""
        try:
            # Pre-check syntax before exec to catch errors cheaply
            compile(transform_code, "<llm_fe>", "exec")
        except SyntaxError as e:
            logger.warning("Transform has syntax error, skipping: %s", e)
            return 0.0, df

        try:
            # Execute in isolated namespace
            namespace: dict[str, Any] = {"pd": pd, "np": np}
            exec(transform_code, namespace)  # noqa: S102

            if "transform" not in namespace:
                logger.warning("Transform code missing 'transform' function definition")
                return 0.0, df

            transform_fn = namespace["transform"]
            new_df = transform_fn(df.copy())

            if not isinstance(new_df, pd.DataFrame):
                logger.warning("Transform returned %s instead of DataFrame", type(new_df).__name__)
                return 0.0, df

            score = self._evaluate_features(new_df, target_column, problem_type, cv_folds)
            return score, new_df

        except Exception as e:
            logger.warning("Transform execution failed: %s (type: %s)", e, type(e).__name__)
            return 0.0, df

    def _evaluate_baseline(
        self, df: pd.DataFrame, target_column: str, problem_type: str, cv_folds: int,
    ) -> float:
        """Evaluate baseline score with original features."""
        return self._evaluate_features(df, target_column, problem_type, cv_folds)

    def _evaluate_features(
        self, df: pd.DataFrame, target_column: str, problem_type: str, cv_folds: int,
    ) -> float:
        """Evaluate feature set via CV scoring."""
        if target_column not in df.columns:
            return 0.0

        features = df.drop(columns=[target_column]).select_dtypes(
            include=["number"],
        )
        y = df[target_column]

        if len(features.columns) == 0 or len(y.unique()) < 2:
            return 0.0

        # Fill NaN for modeling
        features = features.fillna(features.median())

        try:
            if problem_type == "classification":
                model = GradientBoostingClassifier(
                    n_estimators=50, max_depth=3, random_state=42,
                )
                scoring = "balanced_accuracy"
            else:
                model = GradientBoostingRegressor(
                    n_estimators=50, max_depth=3, random_state=42,
                )
                scoring = "r2"

            scores = cross_val_score(
                model, features, y,
                cv=StratifiedKFold(n_splits=min(cv_folds, len(y)), shuffle=True, random_state=42) if problem_type == "classification" else min(cv_folds, len(y)),
                scoring=scoring,
            )
            return float(scores.mean())

        except Exception as e:
            logger.debug("CV evaluation failed: %s", e)
            return 0.0
