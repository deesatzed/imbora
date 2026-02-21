"""Tests for scale-adaptive model routing."""

from __future__ import annotations

import pytest

from src.core.config import DataScienceConfig
from src.datasci.scale_router import (
    determine_scale_tier,
    get_interpretable_models,
    get_model_metadata,
    select_model_candidates,
)


@pytest.fixture
def default_config():
    return DataScienceConfig()


class TestDetermineScaleTier:
    def test_small_dataset(self, default_config):
        assert determine_scale_tier(5000, default_config) == "small"

    def test_small_boundary(self, default_config):
        assert determine_scale_tier(10_000, default_config) == "small"

    def test_medium_dataset(self, default_config):
        assert determine_scale_tier(25_000, default_config) == "medium"

    def test_medium_boundary(self, default_config):
        assert determine_scale_tier(50_000, default_config) == "medium"

    def test_large_dataset(self, default_config):
        assert determine_scale_tier(100_000, default_config) == "large"

    def test_very_large_dataset(self, default_config):
        assert determine_scale_tier(1_000_000, default_config) == "large"

    def test_single_row(self, default_config):
        assert determine_scale_tier(1, default_config) == "small"

    def test_custom_thresholds(self):
        config = DataScienceConfig(
            small_dataset_threshold=1000,
            medium_dataset_threshold=5000,
        )
        assert determine_scale_tier(500, config) == "small"
        assert determine_scale_tier(3000, config) == "medium"
        assert determine_scale_tier(10_000, config) == "large"


class TestSelectModelCandidates:
    def test_small_classification(self, default_config):
        candidates = select_model_candidates("small", "classification", 20, default_config)
        assert "tabpfn" in candidates
        assert "tabicl" in candidates
        assert "xgboost" in candidates
        assert "nam" in candidates

    def test_small_regression(self, default_config):
        candidates = select_model_candidates("small", "regression", 20, default_config)
        assert "tabpfn" in candidates
        assert "xgboost" in candidates

    def test_medium_excludes_tabpfn(self, default_config):
        candidates = select_model_candidates("medium", "classification", 20, default_config)
        assert "tabpfn" not in candidates
        # TabICL supports up to 600K, so included in medium if max_rows >= 50K
        assert "xgboost" in candidates
        assert "catboost" in candidates
        assert "lightgbm" in candidates

    def test_large_excludes_foundation_models(self, default_config):
        candidates = select_model_candidates("large", "classification", 20, default_config)
        assert "tabpfn" not in candidates
        assert "tabicl" not in candidates
        assert "xgboost" in candidates
        assert "catboost" in candidates
        assert "lightgbm" in candidates

    def test_large_includes_nam(self, default_config):
        candidates = select_model_candidates("large", "classification", 20, default_config)
        assert "nam" in candidates

    def test_tabpfn_disabled(self):
        config = DataScienceConfig(enable_tabpfn=False)
        candidates = select_model_candidates("small", "classification", 20, config)
        assert "tabpfn" not in candidates

    def test_tabicl_disabled(self):
        config = DataScienceConfig(enable_tabicl=False)
        candidates = select_model_candidates("small", "classification", 20, config)
        assert "tabicl" not in candidates

    def test_nam_disabled(self):
        config = DataScienceConfig(enable_nam=False)
        candidates = select_model_candidates("small", "classification", 20, config)
        assert "nam" not in candidates

    def test_high_feature_count_excludes_limited_models(self, default_config):
        # TabPFN, TabICL, NAM all limited to 100 features
        candidates = select_model_candidates("small", "classification", 200, default_config)
        assert "tabpfn" not in candidates
        assert "tabicl" not in candidates
        assert "nam" not in candidates
        # Gradient boosters have no feature limit
        assert "xgboost" in candidates
        assert "catboost" in candidates
        assert "lightgbm" in candidates

    def test_always_has_gradient_booster(self):
        # Even with everything disabled, ensure at least one gradient booster
        config = DataScienceConfig(
            enable_tabpfn=False,
            enable_tabicl=False,
            enable_nam=False,
        )
        candidates = select_model_candidates("small", "classification", 20, config)
        assert any(c in candidates for c in ("xgboost", "catboost", "lightgbm"))

    def test_all_disabled_fallback(self):
        # Feature count too high for everything with limits disabled
        config = DataScienceConfig(
            enable_tabpfn=False,
            enable_tabicl=False,
            enable_nam=False,
        )
        candidates = select_model_candidates("large", "regression", 200, config)
        assert len(candidates) >= 1
        assert "xgboost" in candidates


class TestGetModelMetadata:
    def test_known_model(self):
        meta = get_model_metadata("xgboost")
        assert meta is not None
        assert meta["type"] == "xgboost"
        assert meta["supports_classification"] is True

    def test_unknown_model(self):
        assert get_model_metadata("nonexistent") is None

    def test_tabpfn_metadata(self):
        meta = get_model_metadata("tabpfn")
        assert meta is not None
        assert meta["max_rows"] == 10_000
        assert "zero-shot" in meta["description"]

    def test_nam_interpretable(self):
        meta = get_model_metadata("nam")
        assert meta is not None
        assert meta["interpretable"] is True


class TestGetInterpretableModels:
    def test_filters_correctly(self):
        candidates = ["xgboost", "nam", "tabpfn", "catboost"]
        interpretable = get_interpretable_models(candidates)
        assert interpretable == ["nam"]

    def test_no_interpretable(self):
        candidates = ["xgboost", "catboost", "lightgbm"]
        interpretable = get_interpretable_models(candidates)
        assert interpretable == []

    def test_empty_input(self):
        assert get_interpretable_models([]) == []
