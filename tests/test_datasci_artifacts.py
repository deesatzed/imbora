"""Tests for artifact management utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.datasci.artifacts import ArtifactManager


class TestArtifactManager:
    @pytest.fixture
    def mgr(self):
        return ArtifactManager()

    def test_save_and_load_model(self, mgr, tmp_path):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()

        path = tmp_path / "model.joblib"
        mgr.save_model(model, str(path))
        assert path.exists()

        loaded = mgr.load_model(str(path))
        assert isinstance(loaded, LogisticRegression)

    def test_load_nonexistent_raises(self, mgr, tmp_path):
        with pytest.raises(FileNotFoundError):
            mgr.load_model(str(tmp_path / "nope.joblib"))

    def test_save_creates_parent_dirs(self, mgr, tmp_path):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

        path = tmp_path / "nested" / "dir" / "model.joblib"
        mgr.save_model(model, str(path))
        assert path.exists()

    def test_generate_api_scaffold(self, mgr):
        scaffold = mgr.generate_api_scaffold(
            model_name="xgboost_classifier",
            feature_names=["age", "income", "score"],
            target_column="target",
        )
        assert "FastAPI" in scaffold or "fastapi" in scaffold
        assert "age" in scaffold
        assert "predict" in scaffold

    def test_generate_monitoring_config(self, mgr):
        feature_stats = {
            "age": {"mean": 35.0, "std": 10.0, "min": 18.0, "max": 80.0},
            "income": {"mean": 50000.0, "std": 15000.0, "min": 10000.0, "max": 200000.0},
        }
        training_metrics = {"accuracy": 0.90, "f1": 0.88}

        config = mgr.generate_monitoring_config(feature_stats, training_metrics)
        assert len(config) > 0
        # Should be JSON-serializable
        json.dumps(config)

    def test_generate_data_contract(self, mgr):
        feature_names = ["age", "income", "name"]
        feature_types = {"age": "int64", "income": "float64", "name": "object"}

        contract = mgr.generate_data_contract(feature_names, feature_types)
        assert len(contract) > 0
        # Should be JSON-serializable
        json.dumps(contract)
