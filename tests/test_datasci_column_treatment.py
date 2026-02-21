"""Tests for ColumnTreatmentExecutor: per-column treatment execution.

Uses real pandas DataFrames, real sklearn transformers, and real numpy
computations. No mocks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.datasci.column_treatment_executor import ColumnTreatmentExecutor
from src.datasci.models import ColumnProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_executor() -> ColumnTreatmentExecutor:
    return ColumnTreatmentExecutor()


def _make_profile(
    name: str,
    dtype: str = "numeric",
    imputation_strategy: str = "",
    encoding_strategy: str = "",
    outlier_strategy: str = "",
    text_processing_strategy: str = "",
    interaction_candidates: list[str] | None = None,
    importance_prior: str = "",
    **kwargs: Any,
) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype=dtype,
        imputation_strategy=imputation_strategy,
        encoding_strategy=encoding_strategy,
        outlier_strategy=outlier_strategy,
        text_processing_strategy=text_processing_strategy,
        interaction_candidates=interaction_candidates or [],
        importance_prior=importance_prior,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests: _apply_imputation
# ---------------------------------------------------------------------------

class TestApplyImputation:
    def test_median_imputation_numeric(self):
        executor = _make_executor()
        df = pd.DataFrame({"val": [1.0, 2.0, np.nan, 4.0, 5.0]})
        result = executor._apply_imputation(df, "val", "median")
        assert not result["val"].isna().any()
        # Median of [1, 2, 4, 5] = 3.0
        assert result["val"].iloc[2] == 3.0

    def test_mean_imputation_numeric(self):
        executor = _make_executor()
        df = pd.DataFrame({"val": [1.0, 2.0, np.nan, 4.0, 5.0]})
        result = executor._apply_imputation(df, "val", "mean")
        assert not result["val"].isna().any()
        # Mean of [1, 2, 4, 5] = 3.0
        assert result["val"].iloc[2] == 3.0

    def test_mode_imputation(self):
        executor = _make_executor()
        df = pd.DataFrame({"val": ["a", "b", "a", np.nan, "a"]})
        result = executor._apply_imputation(df, "val", "mode")
        assert not result["val"].isna().any()
        assert result["val"].iloc[3] == "a"

    def test_domain_zero_imputation(self):
        executor = _make_executor()
        df = pd.DataFrame({"val": [1.0, np.nan, 3.0]})
        result = executor._apply_imputation(df, "val", "domain_zero")
        assert result["val"].iloc[1] == 0

    def test_fill_empty_string(self):
        executor = _make_executor()
        df = pd.DataFrame({"val": ["hello", np.nan, "world"]})
        result = executor._apply_imputation(df, "val", "fill_empty_string")
        assert result["val"].iloc[1] == ""

    def test_forward_fill(self):
        executor = _make_executor()
        df = pd.DataFrame({"val": [1.0, np.nan, np.nan, 4.0, np.nan]})
        result = executor._apply_imputation(df, "val", "forward_fill")
        assert not result["val"].isna().any()
        assert result["val"].iloc[1] == 1.0  # forward filled
        assert result["val"].iloc[2] == 1.0  # forward filled
        assert result["val"].iloc[4] == 4.0  # forward filled

    def test_forward_fill_leading_nan_backfilled(self):
        executor = _make_executor()
        df = pd.DataFrame({"val": [np.nan, np.nan, 3.0, 4.0]})
        result = executor._apply_imputation(df, "val", "forward_fill")
        assert not result["val"].isna().any()
        # Leading NaN should be backfilled
        assert result["val"].iloc[0] == 3.0

    def test_no_missing_values_unchanged(self):
        executor = _make_executor()
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        result = executor._apply_imputation(df.copy(), "val", "median")
        pd.testing.assert_frame_equal(result, df)

    def test_median_on_non_numeric_uses_mode(self):
        executor = _make_executor()
        df = pd.DataFrame({"cat": ["a", "a", "b", np.nan]})
        result = executor._apply_imputation(df, "cat", "median")
        assert result["cat"].iloc[3] == "a"

    def test_mean_on_non_numeric_uses_mode(self):
        executor = _make_executor()
        df = pd.DataFrame({"cat": ["x", "x", "y", np.nan]})
        result = executor._apply_imputation(df, "cat", "mean")
        assert result["cat"].iloc[3] == "x"

    def test_drop_row_strategy_fills_instead(self):
        """drop_row strategy fills with median to avoid row misalignment."""
        executor = _make_executor()
        df = pd.DataFrame({"val": [1.0, 2.0, np.nan, 4.0, 5.0]})
        result = executor._apply_imputation(df, "val", "drop_row")
        assert len(result) == 5  # No rows dropped
        assert not result["val"].isna().any()

    def test_unknown_strategy_uses_default(self):
        executor = _make_executor()
        df = pd.DataFrame({"val": [1.0, np.nan, 3.0]})
        result = executor._apply_imputation(df, "val", "nonexistent_strategy")
        assert not result["val"].isna().any()


# ---------------------------------------------------------------------------
# Tests: _apply_outlier_treatment
# ---------------------------------------------------------------------------

class TestApplyOutlierTreatment:
    def test_clip_to_iqr(self):
        executor = _make_executor()
        # Values with clear outliers
        data = list(range(1, 21)) + [1000]  # 1-20 plus outlier at 1000
        df = pd.DataFrame({"val": [float(x) for x in data]})
        result = executor._apply_outlier_treatment(df.copy(), "val", "clip_to_iqr")
        # Outlier should be clipped
        assert result["val"].max() < 1000

    def test_winsorize(self):
        executor = _make_executor()
        data = list(range(1, 101))
        df = pd.DataFrame({"val": [float(x) for x in data]})
        result = executor._apply_outlier_treatment(df.copy(), "val", "winsorize")
        # Extreme values should be winsorized
        assert result["val"].min() >= df["val"].quantile(0.05) - 1
        assert result["val"].max() <= df["val"].quantile(0.95) + 1

    def test_log_transform_positive_values(self):
        executor = _make_executor()
        df = pd.DataFrame({"val": [0.0, 1.0, 10.0, 100.0, 1000.0]})
        result = executor._apply_outlier_treatment(df.copy(), "val", "log_transform")
        # log1p(0) = 0, log1p(1000) ~= 6.9
        assert result["val"].iloc[0] == 0.0
        assert result["val"].iloc[4] < 1000.0

    def test_log_transform_negative_values_shifted(self):
        executor = _make_executor()
        df = pd.DataFrame({"val": [-5.0, -1.0, 0.0, 10.0]})
        result = executor._apply_outlier_treatment(df.copy(), "val", "log_transform")
        # Should shift to positive before log
        assert not result["val"].isna().any()
        # All values should be non-negative after log1p of shifted values
        assert (result["val"] >= 0).all()

    def test_non_numeric_column_unchanged(self):
        executor = _make_executor()
        df = pd.DataFrame({"cat": ["a", "b", "c", "d"]})
        result = executor._apply_outlier_treatment(df.copy(), "cat", "clip_to_iqr")
        pd.testing.assert_frame_equal(result, df)

    def test_too_few_values_unchanged(self):
        executor = _make_executor()
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        original = df.copy()
        result = executor._apply_outlier_treatment(df, "val", "clip_to_iqr")
        pd.testing.assert_frame_equal(result, original)

    def test_keep_strategy_no_change(self):
        executor = _make_executor()
        data = [float(x) for x in range(1, 21)] + [1000.0]
        df = pd.DataFrame({"val": data})
        original = df.copy()
        result = executor._apply_outlier_treatment(df, "val", "keep")
        pd.testing.assert_frame_equal(result, original)


# ---------------------------------------------------------------------------
# Tests: _apply_encoding
# ---------------------------------------------------------------------------

class TestApplyEncoding:
    def test_keep_numeric_no_change(self):
        executor = _make_executor()
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
        result_df, new_feats = executor._apply_encoding(df.copy(), "val", "keep_numeric", "target")
        assert new_feats == []
        assert "val" in result_df.columns

    def test_onehot_encoding(self):
        executor = _make_executor()
        df = pd.DataFrame({
            "color": ["red", "blue", "green", "red", "blue"],
            "target": [0, 1, 0, 1, 0],
        })
        result_df, new_feats = executor._apply_encoding(df.copy(), "color", "onehot", "target")
        # Original column should be dropped
        assert "color" not in result_df.columns
        # New one-hot columns should be created (drop_first=True -> 2 columns)
        assert len(new_feats) > 0
        # Target should be preserved
        assert "target" in result_df.columns

    def test_frequency_encoding(self):
        executor = _make_executor()
        df = pd.DataFrame({
            "brand": ["A", "A", "B", "B", "B", "C"],
            "target": [0, 1, 0, 1, 0, 1],
        })
        result_df, new_feats = executor._apply_encoding(df.copy(), "brand", "frequency", "target")
        assert "brand" not in result_df.columns
        assert "brand_freq" in result_df.columns
        assert "brand_freq" in new_feats
        # B appears 3/6 times = 0.5
        assert abs(result_df["brand_freq"].iloc[2] - 0.5) < 0.01

    def test_hash_encoding(self):
        executor = _make_executor()
        df = pd.DataFrame({
            "item": ["abc", "def", "ghi", "jkl"],
            "target": [0, 1, 0, 1],
        })
        result_df, new_feats = executor._apply_encoding(df.copy(), "item", "hash", "target")
        assert "item" not in result_df.columns
        assert "item_hash" in result_df.columns
        assert "item_hash" in new_feats
        # Hash values should be between 0 and 31
        assert result_df["item_hash"].min() >= 0
        assert result_df["item_hash"].max() < 32

    def test_target_encode(self):
        executor = _make_executor()
        df = pd.DataFrame({
            "category": ["A"] * 50 + ["B"] * 50,
            "target": [0] * 25 + [1] * 25 + [0] * 10 + [1] * 40,
        })
        result_df, new_feats = executor._apply_encoding(
            df.copy(), "category", "target_encode", "target",
        )
        # Column should still exist (target encoding replaces in-place)
        assert "category" in result_df.columns
        # Values should be numeric
        assert pd.api.types.is_numeric_dtype(result_df["category"])


# ---------------------------------------------------------------------------
# Tests: _apply_text_processing
# ---------------------------------------------------------------------------

class TestApplyTextProcessing:
    def test_tfidf_processing(self):
        executor = _make_executor()
        df = pd.DataFrame({
            "text": [
                "the quick brown fox",
                "jumps over the lazy dog",
                "the dog is quick and brown",
                "fox jumps over dog",
                "quick fox lazy dog",
                "brown fox quick dog",
                "lazy dog jumps fox",
                "quick brown lazy fox",
            ],
        })
        result_df, new_feats = executor._apply_text_processing(df.copy(), "text", "tfidf")
        # Original text column should be dropped
        assert "text" not in result_df.columns
        # TF-IDF features should be created
        assert len(new_feats) > 0
        for feat in new_feats:
            assert feat.startswith("text_tfidf_")

    def test_tfidf_too_few_unique_documents(self):
        executor = _make_executor()
        df = pd.DataFrame({"text": ["same text"] * 10})
        result_df, new_feats = executor._apply_text_processing(df.copy(), "text", "tfidf")
        # Should return unchanged since only 1 unique document
        assert "text" in result_df.columns
        assert new_feats == []

    def test_none_strategy_no_change(self):
        executor = _make_executor()
        df = pd.DataFrame({"text": ["hello", "world"]})
        result_df, new_feats = executor._apply_text_processing(df.copy(), "text", "none")
        assert "text" in result_df.columns
        assert new_feats == []


# ---------------------------------------------------------------------------
# Tests: _apply_interactions
# ---------------------------------------------------------------------------

class TestApplyInteractions:
    def test_numeric_interactions(self):
        executor = _make_executor()
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
        })
        profiles = [
            _make_profile("a", interaction_candidates=["b"]),
            _make_profile("b", interaction_candidates=["a"]),
        ]
        result = executor._apply_interactions(df, profiles, "target")
        # Should create a_x_b (sorted pair)
        assert "a_x_b" in result
        expected = df["a"] * df["b"]
        pd.testing.assert_series_equal(result["a_x_b"], expected, check_names=False)

    def test_no_duplicate_interactions(self):
        """a*b and b*a should produce only one interaction."""
        executor = _make_executor()
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
        })
        profiles = [
            _make_profile("a", interaction_candidates=["b"]),
            _make_profile("b", interaction_candidates=["a"]),
        ]
        result = executor._apply_interactions(df, profiles, "target")
        assert len(result) == 1

    def test_skip_target_column_interactions(self):
        executor = _make_executor()
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        })
        profiles = [
            _make_profile("a", interaction_candidates=["target"]),
        ]
        result = executor._apply_interactions(df, profiles, "target")
        assert len(result) == 0

    def test_skip_dropped_columns(self):
        executor = _make_executor()
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
        })
        profiles = [
            _make_profile("a", interaction_candidates=["b"], importance_prior="drop"),
            _make_profile("b"),
        ]
        result = executor._apply_interactions(df, profiles, "target")
        assert len(result) == 0

    def test_skip_non_numeric_interaction(self):
        executor = _make_executor()
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": ["x", "y", "z"],
            "target": [0, 1, 0],
        })
        profiles = [
            _make_profile("a", interaction_candidates=["b"]),
        ]
        result = executor._apply_interactions(df, profiles, "target")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: execute() end-to-end
# ---------------------------------------------------------------------------

class TestExecuteEndToEnd:
    def test_mixed_column_types(self):
        executor = _make_executor()
        df = pd.DataFrame({
            "age": [25.0, np.nan, 35.0, 40.0, 50.0],
            "income": [50000.0, 60000.0, np.nan, 80000.0, 90000.0],
            "city": ["NYC", "LA", "NYC", "SF", "LA"],
            "target": [0, 1, 0, 1, 0],
        })
        profiles = [
            _make_profile(
                "age",
                imputation_strategy="median",
                encoding_strategy="keep_numeric",
                outlier_strategy="none",
            ),
            _make_profile(
                "income",
                imputation_strategy="mean",
                encoding_strategy="keep_numeric",
                outlier_strategy="clip_to_iqr",
            ),
            _make_profile(
                "city",
                dtype="categorical",
                imputation_strategy="mode",
                encoding_strategy="onehot",
                outlier_strategy="none",
            ),
        ]
        result_df, new_features = executor.execute(df, profiles, "target")

        # Target should be preserved
        assert "target" in result_df.columns
        # No NaN in age or income
        assert not result_df["age"].isna().any()
        assert not result_df["income"].isna().any()
        # city should be one-hot encoded (dropped, replaced by dummies)
        assert "city" not in result_df.columns

    def test_drop_column_with_importance_prior(self):
        executor = _make_executor()
        df = pd.DataFrame({
            "row_id": [1, 2, 3, 4],
            "feature": [10.0, 20.0, 30.0, 40.0],
            "target": [0, 1, 0, 1],
        })
        profiles = [
            _make_profile("row_id", importance_prior="drop"),
            _make_profile("feature", imputation_strategy="median", encoding_strategy="keep_numeric"),
        ]
        result_df, _ = executor.execute(df, profiles, "target")
        assert "row_id" not in result_df.columns
        assert "feature" in result_df.columns
        assert "target" in result_df.columns

    def test_target_column_preserved(self):
        executor = _make_executor()
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        })
        profiles = [
            _make_profile("x", imputation_strategy="median", encoding_strategy="keep_numeric"),
            _make_profile("target"),  # target profile exists but should not be treated
        ]
        result_df, _ = executor.execute(df, profiles, "target")
        assert "target" in result_df.columns
        assert list(result_df["target"]) == [0, 1, 0]

    def test_column_not_in_profiles_skipped(self):
        executor = _make_executor()
        df = pd.DataFrame({
            "known": [1.0, 2.0, 3.0],
            "unknown": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
        })
        profiles = [
            _make_profile("known", imputation_strategy="median", encoding_strategy="keep_numeric"),
        ]
        result_df, _ = executor.execute(df, profiles, "target")
        # unknown column should still be present (not treated, not dropped)
        assert "unknown" in result_df.columns

    def test_interaction_features_added(self):
        executor = _make_executor()
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "target": [0, 1, 0, 1, 0],
        })
        profiles = [
            _make_profile(
                "a",
                imputation_strategy="median",
                encoding_strategy="keep_numeric",
                interaction_candidates=["b"],
            ),
            _make_profile(
                "b",
                imputation_strategy="median",
                encoding_strategy="keep_numeric",
            ),
        ]
        result_df, new_features = executor.execute(df, profiles, "target")
        assert "a_x_b" in result_df.columns
        assert "a_x_b" in new_features
        expected = df["a"] * df["b"]
        pd.testing.assert_series_equal(
            result_df["a_x_b"], expected, check_names=False,
        )

    def test_complete_pipeline_with_nan_and_outliers(self):
        """Full pipeline: imputation -> outlier treatment -> encoding."""
        executor = _make_executor()
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "feat": np.concatenate([
                np.random.normal(10, 2, n - 5),
                [np.nan, np.nan, np.nan, 100.0, -50.0],  # NaN + outliers
            ]),
            "target": np.random.randint(0, 2, n),
        })
        profiles = [
            _make_profile(
                "feat",
                imputation_strategy="median",
                encoding_strategy="keep_numeric",
                outlier_strategy="clip_to_iqr",
            ),
        ]
        result_df, _ = executor.execute(df, profiles, "target")
        assert not result_df["feat"].isna().any()
        # Outliers should be clipped (no more 100 or -50)
        assert result_df["feat"].max() < 100.0
        assert result_df["feat"].min() > -50.0
