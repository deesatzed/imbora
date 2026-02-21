"""Column and dataset profiling utilities.

Provides per-column statistics, dtype detection, text detection, distribution
summaries, and dataset-level profiling. Pure computation — no LLM calls.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import pandas as pd

from src.datasci.models import ColumnProfile

logger = logging.getLogger("associate.datasci.profiler")

# Heuristic thresholds
TEXT_MIN_AVG_LENGTH = 20
TEXT_MIN_UNIQUE_RATIO = 0.5
CATEGORICAL_MAX_CARDINALITY_RATIO = 0.05  # If unique/total < 5%, treat as categorical


def _detect_dtype(series: pd.Series) -> str:
    """Classify a pandas series into one of our canonical types."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # For object/string columns, heuristic detection
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        non_null = series.dropna()
        if len(non_null) == 0:
            return "categorical"

        # Try datetime parsing on a sample
        sample = non_null.head(min(100, len(non_null)))
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.8:
                return "datetime"
        except Exception:
            pass

        # Check if it looks like text (long strings, high cardinality)
        avg_len = non_null.astype(str).str.len().mean()
        unique_ratio = non_null.nunique() / len(non_null) if len(non_null) > 0 else 0
        if avg_len >= TEXT_MIN_AVG_LENGTH and unique_ratio >= TEXT_MIN_UNIQUE_RATIO:
            return "text"

        return "categorical"

    return "categorical"


def _distribution_summary_numeric(series: pd.Series) -> dict[str, Any]:
    """Compute distribution stats for numeric columns."""
    clean = series.dropna()
    if len(clean) == 0:
        return {}
    desc = clean.describe()
    return {
        "mean": float(desc["mean"]),
        "std": float(desc["std"]),
        "min": float(desc["min"]),
        "25%": float(desc["25%"]),
        "50%": float(desc["50%"]),
        "75%": float(desc["75%"]),
        "max": float(desc["max"]),
        "skew": float(clean.skew()),
        "kurtosis": float(clean.kurtosis()),
    }


def _distribution_summary_categorical(series: pd.Series) -> dict[str, Any]:
    """Compute distribution stats for categorical columns."""
    clean = series.dropna()
    if len(clean) == 0:
        return {}
    vc = clean.value_counts()
    top_n = min(10, len(vc))
    return {
        "top_values": {str(k): int(v) for k, v in vc.head(top_n).items()},
        "mode": str(vc.index[0]) if len(vc) > 0 else None,
        "entropy": float(_entropy(vc.values)),
    }


def _distribution_summary_text(series: pd.Series) -> dict[str, Any]:
    """Compute distribution stats for text columns."""
    clean = series.dropna().astype(str)
    if len(clean) == 0:
        return {}
    lengths = clean.str.len()
    word_counts = clean.str.split().str.len()
    return {
        "avg_length": float(lengths.mean()),
        "max_length": int(lengths.max()),
        "min_length": int(lengths.min()),
        "avg_word_count": float(word_counts.mean()),
        "unique_ratio": float(clean.nunique() / len(clean)),
    }


def _entropy(counts) -> float:
    """Compute Shannon entropy from value counts."""
    import numpy as np

    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _recommend_treatment(dtype: str, missing_pct: float, cardinality: int, total_rows: int) -> str:
    """Suggest a treatment strategy for a column."""
    parts = []
    if missing_pct > 20:
        parts.append("high_missing_investigate")
    elif missing_pct > 5:
        parts.append("impute_missing")

    if dtype == "numeric":
        parts.append("scale_normalize")
    elif dtype == "categorical":
        if cardinality > 50:
            parts.append("target_encode_or_embed")
        else:
            parts.append("onehot_or_ordinal")
    elif dtype == "text":
        parts.append("embed_and_cluster")
    elif dtype == "datetime":
        parts.append("extract_temporal_features")
    elif dtype == "boolean":
        parts.append("keep_as_is")

    return "|".join(parts) if parts else "keep_as_is"


def profile_column(
    series: pd.Series,
    column_name: str,
    is_target: bool = False,
    total_rows: int | None = None,
) -> ColumnProfile:
    """Profile a single column and return a ColumnProfile.

    Args:
        series: The pandas Series to profile.
        column_name: Column name.
        is_target: Whether this is the target variable.
        total_rows: Total dataset rows (for ratio calculations).

    Returns:
        ColumnProfile with detected type, stats, and recommendations.
    """
    total = total_rows or len(series)
    dtype = _detect_dtype(series)
    cardinality = int(series.nunique())
    missing_pct = float(series.isna().mean() * 100)

    if dtype == "numeric":
        dist = _distribution_summary_numeric(series)
    elif dtype == "text":
        dist = _distribution_summary_text(series)
    else:
        dist = _distribution_summary_categorical(series)

    text_detected = dtype == "text"
    treatment = _recommend_treatment(dtype, missing_pct, cardinality, total)

    return ColumnProfile(
        name=column_name,
        dtype=dtype,
        cardinality=cardinality,
        missing_pct=round(missing_pct, 2),
        distribution_summary=dist,
        is_target=is_target,
        text_detected=text_detected,
        recommended_treatment=treatment,
    )


def profile_dataset(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> list[ColumnProfile]:
    """Profile all columns in a DataFrame.

    Args:
        df: The DataFrame to profile.
        target_column: Name of the target column (if any).

    Returns:
        List of ColumnProfile for each column.
    """
    profiles = []
    for col in df.columns:
        is_target = col == target_column
        profiles.append(profile_column(df[col], col, is_target=is_target, total_rows=len(df)))
    return profiles


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Compute a stable hash fingerprint for a dataset.

    Uses column names, dtypes, shape, and a sample of values to create
    a content-addressable identifier.
    """
    parts = [
        f"shape:{df.shape[0]}x{df.shape[1]}",
        f"columns:{','.join(sorted(df.columns))}",
        f"dtypes:{','.join(str(d) for d in df.dtypes.values)}",
    ]

    # Include hash of first and last 10 rows for content fingerprinting
    sample_rows = pd.concat([df.head(10), df.tail(10)]).drop_duplicates()
    parts.append(f"sample:{sample_rows.to_csv(index=False)}")

    content = "|".join(parts)
    return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
