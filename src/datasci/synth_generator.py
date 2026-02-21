"""Scale-adaptive synthetic minority generation for imbalanced classification.

Tiered augmentation:
- Tier 1 (classical): SMOTE, ADASYN, SMOTENC via imbalanced-learn
- Tier 2 (advanced): LLM-based EPIC-style generation via OpenRouter

All strategies respect a configurable max_augmentation_ratio cap
and include adversarial validation to reject poor-quality synthetic data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.llm.client import LLMMessage

logger = logging.getLogger("associate.datasci.synth_generator")


def _compute_sampling_strategy(
    y: pd.Series,
    max_ratio: float = 5.0,
) -> dict[Any, int]:
    """Compute per-class target counts for resampling.

    Brings minority classes up toward majority count, capped by max_ratio.

    Args:
        y: Target series.
        max_ratio: Maximum ratio of augmented minority to original minority count.

    Returns:
        Dict mapping class label -> target count (after augmentation).
    """
    counts = y.value_counts()
    majority_count = int(counts.max())

    strategy: dict[Any, int] = {}
    for cls, count in counts.items():
        if count < majority_count:
            target = min(majority_count, int(count * max_ratio))
            strategy[cls] = max(target, count + 1)  # At least 1 new sample
        # Don't include majority class — no resampling needed

    return strategy


def _adversarial_validation_score(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_column: str,
) -> float:
    """Train a classifier to distinguish real vs synthetic data.

    Lower score = better quality (harder to distinguish).
    Score > 0.8 means synthetic data is too distinguishable.

    Args:
        real_df: Original DataFrame.
        synth_df: Synthetic DataFrame.
        target_column: Target column name (excluded from features).

    Returns:
        AUC score of real-vs-synthetic classifier. Lower is better.
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score

        # Prepare adversarial dataset
        feature_cols = [
            c for c in real_df.columns
            if c != target_column and real_df[c].dtype in ("float64", "int64", "float32", "int32")
        ]
        if not feature_cols:
            return 0.5  # Can't evaluate, assume OK

        real_features = real_df[feature_cols].copy().fillna(0)
        synth_features = synth_df[feature_cols].copy().fillna(0)

        x_combined = pd.concat([real_features, synth_features], ignore_index=True)
        y = np.concatenate([
            np.zeros(len(real_features)),
            np.ones(len(synth_features)),
        ])

        if len(x_combined) < 20:
            return 0.5  # Too few samples

        clf = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42,
        )

        n_folds = min(5, min(int(np.sum(y == 0)), int(np.sum(y == 1))))
        if n_folds < 2:
            return 0.5

        scores = cross_val_score(
            clf, x_combined, y, cv=n_folds, scoring="roc_auc",
        )
        return float(np.mean(scores))

    except Exception as e:
        logger.warning("Adversarial validation failed: %s", e)
        return 0.5  # Assume OK on failure


class SynthGenerator:
    """Scale-adaptive synthetic minority generation."""

    def __init__(
        self,
        llm_client: Any = None,
        model_router: Any = None,
    ):
        self.llm_client = llm_client
        self.model_router = model_router

    def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        strategy: str,
        imbalance_ratio: float,
        max_augmentation_ratio: float = 5.0,
    ) -> dict[str, Any]:
        """Execute augmentation strategy.

        Args:
            df: Input DataFrame with target column.
            target_column: Name of the target column.
            strategy: One of 'smote', 'smotenc', 'adasyn', 'llm_synth', 'none'.
            imbalance_ratio: Current class imbalance ratio.
            max_augmentation_ratio: Maximum ratio cap for augmentation.

        Returns:
            Dict with:
                - augmented_df: Augmented DataFrame
                - strategy_used: Strategy that was actually used
                - samples_generated: Number of synthetic samples
                - original_counts: Dict of original class counts
                - augmented_counts: Dict of augmented class counts
                - quality_score: Adversarial validation quality score
        """
        if strategy == "none" or target_column not in df.columns:
            return self._passthrough_result(df, target_column)

        y = df[target_column]
        original_counts = {str(k): int(v) for k, v in y.value_counts().items()}

        if strategy == "smote":
            result_df = self._apply_smote(df, target_column, max_augmentation_ratio)
        elif strategy == "smotenc":
            result_df = self._apply_smotenc(df, target_column, max_augmentation_ratio)
        elif strategy == "adasyn":
            result_df = self._apply_adasyn(df, target_column, max_augmentation_ratio)
        elif strategy == "llm_synth":
            result_df = self._apply_llm_synth(df, target_column, max_augmentation_ratio)
        else:
            logger.warning("Unknown strategy '%s'; passing through", strategy)
            return self._passthrough_result(df, target_column)

        if result_df is None:
            logger.warning(
                "Strategy '%s' failed; returning original data", strategy,
            )
            return self._passthrough_result(df, target_column)

        augmented_counts = {
            str(k): int(v) for k, v in result_df[target_column].value_counts().items()
        }
        samples_generated = len(result_df) - len(df)

        # Adversarial validation: check quality of synthetic data
        synth_rows = result_df.iloc[len(df):]
        adv_score = _adversarial_validation_score(df, synth_rows, target_column)

        # Quality score: 1.0 - adv_score (lower adv = better quality)
        quality_score = max(0.0, 1.0 - adv_score)

        if adv_score > 0.8:
            logger.warning(
                "Adversarial validation score %.3f > 0.8 — synthetic data "
                "too distinguishable from real data. Rejecting augmentation.",
                adv_score,
            )
            return self._passthrough_result(df, target_column)

        return {
            "augmented_df": result_df,
            "strategy_used": strategy,
            "samples_generated": samples_generated,
            "original_counts": original_counts,
            "augmented_counts": augmented_counts,
            "quality_score": round(quality_score, 4),
            "adversarial_validation_score": round(adv_score, 4),
        }

    def _passthrough_result(
        self, df: pd.DataFrame, target_column: str,
    ) -> dict[str, Any]:
        """Return a pass-through result (no augmentation)."""
        counts = {}
        if target_column in df.columns:
            counts = {
                str(k): int(v)
                for k, v in df[target_column].value_counts().items()
            }
        return {
            "augmented_df": df,
            "strategy_used": "none",
            "samples_generated": 0,
            "original_counts": counts,
            "augmented_counts": counts,
            "quality_score": 1.0,
            "adversarial_validation_score": 0.0,
        }

    def _apply_smote(
        self,
        df: pd.DataFrame,
        target_column: str,
        max_ratio: float,
    ) -> pd.DataFrame | None:
        """Apply SMOTE resampling on numeric features."""
        try:
            from imblearn.over_sampling import SMOTE

            y = df[target_column]
            numeric_cols = df.drop(columns=[target_column]).select_dtypes(
                include=["number"],
            ).columns.tolist()

            if not numeric_cols:
                logger.warning("No numeric columns for SMOTE")
                return None

            x_feat = df[numeric_cols].fillna(0)
            sampling_strategy = _compute_sampling_strategy(y, max_ratio)

            if not sampling_strategy:
                return None

            # Ensure k_neighbors doesn't exceed minority class size
            min_minority = min(
                int(y.value_counts()[cls]) for cls in sampling_strategy
            )
            k_neighbors = min(5, min_minority - 1)
            if k_neighbors < 1:
                k_neighbors = 1

            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=42,
            )
            x_res, y_res = smote.fit_resample(x_feat, y)

            result_df = pd.DataFrame(x_res, columns=numeric_cols)
            result_df[target_column] = y_res

            # Preserve non-numeric columns from original data
            non_numeric = [
                c for c in df.columns
                if c not in numeric_cols and c != target_column
            ]
            if non_numeric:
                # For synthetic rows, fill non-numeric with mode
                for col in non_numeric:
                    original_vals = df[col].values
                    synthetic_count = len(result_df) - len(df)
                    mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else ""
                    extended = np.concatenate([
                        original_vals,
                        np.full(synthetic_count, mode_val),
                    ])
                    result_df[col] = extended

            return result_df

        except ImportError:
            logger.info("imbalanced-learn not installed; SMOTE unavailable")
            return None
        except Exception as e:
            logger.warning("SMOTE failed: %s", e)
            return None

    def _apply_smotenc(
        self,
        df: pd.DataFrame,
        target_column: str,
        max_ratio: float,
    ) -> pd.DataFrame | None:
        """Apply SMOTENC resampling (handles categorical features)."""
        try:
            from imblearn.over_sampling import SMOTENC

            y = df[target_column]
            feature_cols = [c for c in df.columns if c != target_column]
            x_feat = df[feature_cols].copy()

            # Identify categorical feature indices
            cat_indices = [
                i for i, col in enumerate(feature_cols)
                if x_feat[col].dtype == "object" or x_feat[col].dtype.name == "category"
            ]

            if not cat_indices:
                # No categoricals; fall back to regular SMOTE
                return self._apply_smote(df, target_column, max_ratio)

            # Fill NaN
            for col in feature_cols:
                if x_feat[col].dtype in ("float64", "int64", "float32", "int32"):
                    x_feat[col] = x_feat[col].fillna(0)
                else:
                    x_feat[col] = x_feat[col].fillna("_missing_")

            sampling_strategy = _compute_sampling_strategy(y, max_ratio)
            if not sampling_strategy:
                return None

            min_minority = min(
                int(y.value_counts()[cls]) for cls in sampling_strategy
            )
            k_neighbors = min(5, min_minority - 1)
            if k_neighbors < 1:
                k_neighbors = 1

            smotenc = SMOTENC(
                categorical_features=cat_indices,
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=42,
            )
            x_res, y_res = smotenc.fit_resample(x_feat, y)

            result_df = pd.DataFrame(x_res, columns=feature_cols)
            result_df[target_column] = y_res
            return result_df

        except ImportError:
            logger.info("imbalanced-learn not installed; SMOTENC unavailable")
            return None
        except Exception as e:
            logger.warning("SMOTENC failed: %s", e)
            return None

    def _apply_adasyn(
        self,
        df: pd.DataFrame,
        target_column: str,
        max_ratio: float,
    ) -> pd.DataFrame | None:
        """Apply ADASYN resampling (adaptive synthetic sampling)."""
        try:
            from imblearn.over_sampling import ADASYN

            y = df[target_column]
            numeric_cols = df.drop(columns=[target_column]).select_dtypes(
                include=["number"],
            ).columns.tolist()

            if not numeric_cols:
                logger.warning("No numeric columns for ADASYN")
                return None

            x_feat = df[numeric_cols].fillna(0)
            sampling_strategy = _compute_sampling_strategy(y, max_ratio)

            if not sampling_strategy:
                return None

            min_minority = min(
                int(y.value_counts()[cls]) for cls in sampling_strategy
            )
            n_neighbors = min(5, min_minority - 1)
            if n_neighbors < 1:
                n_neighbors = 1

            adasyn = ADASYN(
                sampling_strategy=sampling_strategy,
                n_neighbors=n_neighbors,
                random_state=42,
            )
            x_res, y_res = adasyn.fit_resample(x_feat, y)

            result_df = pd.DataFrame(x_res, columns=numeric_cols)
            result_df[target_column] = y_res

            # Preserve non-numeric columns
            non_numeric = [
                c for c in df.columns
                if c not in numeric_cols and c != target_column
            ]
            if non_numeric:
                for col in non_numeric:
                    original_vals = df[col].values
                    synthetic_count = len(result_df) - len(df)
                    mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else ""
                    extended = np.concatenate([
                        original_vals,
                        np.full(synthetic_count, mode_val),
                    ])
                    result_df[col] = extended

            return result_df

        except ImportError:
            logger.info("imbalanced-learn not installed; ADASYN unavailable")
            return None
        except Exception as e:
            logger.warning("ADASYN failed: %s", e)
            return None

    def _apply_llm_synth(
        self,
        df: pd.DataFrame,
        target_column: str,
        max_ratio: float,
    ) -> pd.DataFrame | None:
        """Apply LLM-based EPIC-style synthetic minority generation.

        Uses the existing OpenRouter client to generate synthetic minority
        class samples by formatting minority examples as structured prompts
        and parsing LLM output back into DataFrame rows.
        """
        if self.llm_client is None or self.model_router is None:
            logger.warning("LLM client/router not available for LLM synth")
            return self._apply_smote(df, target_column, max_ratio)

        y = df[target_column]
        counts = y.value_counts()
        majority_count = int(counts.max())
        minority_class = counts.idxmin()
        minority_count = int(counts.min())

        # Target count for minority class
        target_count = min(majority_count, int(minority_count * max_ratio))
        n_to_generate = target_count - minority_count

        if n_to_generate <= 0:
            return None

        # Get minority samples for the LLM prompt
        minority_df = df[df[target_column] == minority_class]
        feature_cols = [c for c in df.columns if c != target_column]
        numeric_cols = [
            c for c in feature_cols
            if df[c].dtype in ("float64", "int64", "float32", "int32")
        ]

        # Format samples for LLM
        sample_rows = minority_df.head(5)[feature_cols].to_dict(orient="records")
        col_stats = {}
        for col in numeric_cols:
            series = minority_df[col].dropna()
            if len(series) > 0:
                col_stats[col] = {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                }

        prompt = (
            f"Generate {min(n_to_generate, 20)} synthetic data rows for the minority class.\n\n"
            f"Columns: {', '.join(numeric_cols)}\n\n"
            f"Column statistics:\n"
        )
        for col, stats in col_stats.items():
            prompt += f"  {col}: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}\n"

        prompt += "\nExample minority class rows:\n"
        for row in sample_rows[:3]:
            row_str = ", ".join(f"{k}={v}" for k, v in row.items() if k in numeric_cols)
            prompt += f"  {row_str}\n"

        prompt += (
            "\nOutput format: one row per line, values separated by commas, "
            "in the same column order as listed above. "
            "Values must be within the statistical ranges shown. "
            "No headers, no extra text."
        )

        try:
            models = self.model_router.get_model_chain("ds_analyst")
            messages = [
                LLMMessage(
                    role="system",
                    content=(
                        "You are a synthetic data generator. "
                        "Output only numeric data rows."
                    ),
                ),
                LLMMessage(role="user", content=prompt),
            ]

            response = self.llm_client.complete_with_fallback(
                messages=messages,
                models=models,
                max_tokens=4096,
                temperature=0.8,
            )

            # Parse LLM output into rows
            synthetic_rows = []
            for line in response.content.strip().split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    values = [float(v.strip()) for v in line.split(",")]
                    if len(values) == len(numeric_cols):
                        row = dict(zip(numeric_cols, values))
                        # Validate against column ranges
                        valid = True
                        for col, val in row.items():
                            if col in col_stats:
                                stats = col_stats[col]
                                margin = (stats["max"] - stats["min"]) * 0.5
                                if val < stats["min"] - margin or val > stats["max"] + margin:
                                    valid = False
                                    break
                        if valid:
                            synthetic_rows.append(row)
                except (ValueError, IndexError):
                    continue

            if not synthetic_rows:
                logger.warning("LLM generated no valid synthetic rows; falling back to SMOTE")
                return self._apply_smote(df, target_column, max_ratio)

            # Build synthetic DataFrame
            synth_df = pd.DataFrame(synthetic_rows)
            synth_df[target_column] = minority_class

            # Add non-numeric columns with mode values
            for col in feature_cols:
                if col not in numeric_cols and col not in synth_df.columns:
                    mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else ""
                    synth_df[col] = mode_val

            # Combine with original
            result_df = pd.concat([df, synth_df[df.columns]], ignore_index=True)

            logger.info(
                "LLM synth generated %d rows (requested %d)",
                len(synthetic_rows), n_to_generate,
            )
            return result_df

        except Exception as e:
            logger.warning("LLM synth failed: %s; falling back to SMOTE", e)
            return self._apply_smote(df, target_column, max_ratio)
