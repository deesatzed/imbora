"""Tests for column profiler utilities.

Uses real pandas DataFrames with synthetic data (sklearn-generated where
appropriate) — no mocks, no placeholders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.datasci.column_profiler import (
    compute_dataset_fingerprint,
    profile_column,
    profile_dataset,
)
from src.datasci.models import ColumnProfile


class TestProfileColumnNumeric:
    def test_integer_column(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name="age")
        profile = profile_column(s, "age")
        assert profile.dtype == "numeric"
        assert profile.name == "age"
        assert profile.cardinality == 10
        assert profile.missing_pct == 0.0
        assert "mean" in profile.distribution_summary
        assert profile.text_detected is False

    def test_float_column(self):
        np.random.seed(42)
        s = pd.Series(np.random.normal(100, 15, 500), name="score")
        profile = profile_column(s, "score")
        assert profile.dtype == "numeric"
        assert profile.distribution_summary["mean"] == pytest.approx(100, abs=5)
        assert "skew" in profile.distribution_summary
        assert "kurtosis" in profile.distribution_summary

    def test_numeric_with_missing(self):
        data = [1.0, 2.0, None, 4.0, None, 6.0, 7.0, 8.0, 9.0, 10.0]
        s = pd.Series(data, name="val")
        profile = profile_column(s, "val")
        assert profile.dtype == "numeric"
        assert profile.missing_pct == pytest.approx(20.0, abs=0.1)
        # 20% exactly is at the boundary (> 20 triggers high_missing)
        assert "impute_missing" in profile.recommended_treatment

    def test_numeric_very_high_missing(self):
        data = [1.0, None, None, None, None, None, None, None, None, 10.0]
        s = pd.Series(data, name="sparse")
        profile = profile_column(s, "sparse")
        assert profile.missing_pct == pytest.approx(80.0, abs=0.1)
        assert "high_missing_investigate" in profile.recommended_treatment

    def test_target_flag(self):
        s = pd.Series([0, 1, 0, 1, 1], name="target")
        profile = profile_column(s, "target", is_target=True)
        assert profile.is_target is True

    def test_treatment_scale_normalize(self):
        s = pd.Series(range(100), name="x")
        profile = profile_column(s, "x")
        assert "scale_normalize" in profile.recommended_treatment


class TestProfileColumnCategorical:
    def test_low_cardinality_string(self):
        s = pd.Series(["A", "B", "C", "A", "B", "A"] * 100, name="grade")
        profile = profile_column(s, "grade")
        assert profile.dtype == "categorical"
        assert profile.cardinality == 3
        assert "top_values" in profile.distribution_summary
        assert "onehot_or_ordinal" in profile.recommended_treatment

    def test_high_cardinality_string(self):
        s = pd.Series([f"cat_{i}" for i in range(100)], name="id_col")
        profile = profile_column(s, "id_col")
        assert profile.dtype == "categorical"
        assert profile.cardinality == 100
        assert "target_encode_or_embed" in profile.recommended_treatment


class TestProfileColumnText:
    def test_text_detection(self):
        texts = [
            "This is a long descriptive text about a product that has many words.",
            "Another detailed review explaining the features and benefits of the item.",
            "A third piece of text that is sufficiently long to be detected as text.",
        ] * 50  # Repeat to ensure high unique ratio
        # Make them unique enough
        texts = [f"{t} (variant {i})" for i, t in enumerate(texts)]
        s = pd.Series(texts, name="description")
        profile = profile_column(s, "description")
        assert profile.dtype == "text"
        assert profile.text_detected is True
        assert "avg_length" in profile.distribution_summary
        assert "avg_word_count" in profile.distribution_summary
        assert "embed_and_cluster" in profile.recommended_treatment


class TestProfileColumnBoolean:
    def test_boolean_column(self):
        s = pd.Series([True, False, True, True, False], name="is_active")
        profile = profile_column(s, "is_active")
        assert profile.dtype == "boolean"
        assert "keep_as_is" in profile.recommended_treatment


class TestProfileColumnDatetime:
    def test_datetime_column(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        s = pd.Series(dates, name="created_at")
        profile = profile_column(s, "created_at")
        assert profile.dtype == "datetime"
        assert "extract_temporal_features" in profile.recommended_treatment


class TestProfileColumnEdgeCases:
    def test_all_missing(self):
        s = pd.Series([None, None, None], dtype="object", name="empty")
        profile = profile_column(s, "empty")
        assert profile.missing_pct == pytest.approx(100.0, abs=0.1)

    def test_single_value(self):
        s = pd.Series([42, 42, 42, 42], name="constant")
        profile = profile_column(s, "constant")
        assert profile.cardinality == 1

    def test_empty_series(self):
        s = pd.Series([], dtype="float64", name="empty_numeric")
        profile = profile_column(s, "empty_numeric")
        assert profile.name == "empty_numeric"


class TestProfileDataset:
    def test_basic_dataset(self):
        df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "grade": ["A", "B", "A", "C", "B"],
            "score": [85.5, 90.1, 78.3, 92.0, 88.7],
            "target": [0, 1, 0, 1, 1],
        })
        profiles = profile_dataset(df, target_column="target")
        assert len(profiles) == 4
        names = [p.name for p in profiles]
        assert "age" in names
        assert "grade" in names
        assert "score" in names
        assert "target" in names

        target_profile = [p for p in profiles if p.name == "target"][0]
        assert target_profile.is_target is True

    def test_no_target(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        profiles = profile_dataset(df)
        assert len(profiles) == 2
        assert all(p.is_target is False for p in profiles)

    def test_all_column_types(self):
        """Dataset with multiple column types."""
        df = pd.DataFrame({
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat": ["a", "b", "a", "c", "b"],
            "bool_col": [True, False, True, True, False],
            "date": pd.date_range("2020-01-01", periods=5),
        })
        profiles = profile_dataset(df)
        assert len(profiles) == 4
        dtypes = {p.name: p.dtype for p in profiles}
        assert dtypes["num"] == "numeric"
        assert dtypes["cat"] == "categorical"
        assert dtypes["bool_col"] == "boolean"
        assert dtypes["date"] == "datetime"


class TestDatasetFingerprint:
    def test_deterministic(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        fp1 = compute_dataset_fingerprint(df)
        fp2 = compute_dataset_fingerprint(df)
        assert fp1 == fp2
        assert fp1.startswith("sha256:")

    def test_different_data_different_fingerprint(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})
        fp1 = compute_dataset_fingerprint(df1)
        fp2 = compute_dataset_fingerprint(df2)
        assert fp1 != fp2

    def test_different_columns_different_fingerprint(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [1, 2, 3]})
        fp1 = compute_dataset_fingerprint(df1)
        fp2 = compute_dataset_fingerprint(df2)
        assert fp1 != fp2

    def test_fingerprint_format(self):
        df = pd.DataFrame({"x": [1]})
        fp = compute_dataset_fingerprint(df)
        assert fp.startswith("sha256:")
        assert len(fp) == 7 + 64  # "sha256:" + 64 hex chars
