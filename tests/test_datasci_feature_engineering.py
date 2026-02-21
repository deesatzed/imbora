"""Tests for Phase 3: Feature Engineering Agent.

Uses real sklearn synthetic datasets. No mocks.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.core.config import DataScienceConfig
from src.datasci.agents.feature_engineering import FeatureEngineeringAgent
from src.datasci.models import FeatureEngineeringReport


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLMClient:
    def __init__(self):
        self.calls = []

    def complete_with_fallback(self, **kwargs):
        self.calls.append(True)
        return FakeLLMResponse("""
def transform(df: pd.DataFrame) -> pd.DataFrame:
    if 'x0' in df.columns and 'x1' in df.columns:
        df['x0_x1_interact'] = df['x0'] * df['x1']
    return df
""")


class FakeModelRouter:
    def get_model(self, role): return "test/model"
    def get_fallback_models(self, role): return []

    def get_model_chain(self, role: str) -> list[str]:
        return [self.get_model(role)]


class FakeRepository:
    def __init__(self):
        self.experiments = {}

    def create_ds_experiment(self, **kwargs):
        eid = uuid.uuid4()
        self.experiments[eid] = {"status": "RUNNING"}
        return eid

    def update_ds_experiment(self, experiment_id, **kwargs):
        if experiment_id in self.experiments:
            self.experiments[experiment_id].update(kwargs)


@pytest.fixture
def test_csv(tmp_path):
    X, y = make_classification(
        n_samples=150, n_features=5, n_informative=3,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    df["target"] = y
    path = tmp_path / "fe_test.csv"
    df.to_csv(path, index=False)
    audit_report = {
        "column_profiles": [
            {"name": f"x{i}", "dtype": "numeric", "text_detected": False, "is_target": False}
            for i in range(5)
        ] + [{"name": "target", "dtype": "numeric", "text_detected": False, "is_target": True}],
        "row_count": 150,
    }
    return str(path), audit_report


@pytest.fixture
def agent():
    config = DataScienceConfig(llm_fe_generations=1, llm_fe_population_size=3)
    return FeatureEngineeringAgent(
        llm_client=FakeLLMClient(),
        model_router=FakeModelRouter(),
        repository=FakeRepository(),
        ds_config=config,
    )


class TestFeatureEngineeringAgent:
    def test_produces_valid_report(self, agent, test_csv):
        path, audit = test_csv
        result = agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "audit_report": audit,
        })
        assert result.status == "success"
        report = FeatureEngineeringReport(**result.data)
        assert report.original_feature_count == 5
        assert report.llm_fe_generations_run >= 1

    def test_feature_ranking_generated(self, agent, test_csv):
        path, audit = test_csv
        result = agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "audit_report": audit,
        })
        report = FeatureEngineeringReport(**result.data)
        assert len(report.feature_ranking) > 0
        # Rankings should be sorted descending
        importances = [r["importance"] for r in report.feature_ranking]
        assert importances == sorted(importances, reverse=True)

    def test_best_score_positive(self, agent, test_csv):
        path, audit = test_csv
        result = agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "audit_report": audit,
        })
        report = FeatureEngineeringReport(**result.data)
        assert report.best_generation_score > 0.0

    def test_experiment_tracked(self, agent, test_csv):
        path, audit = test_csv
        agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "audit_report": audit,
        })
        repo = agent.repository
        assert len(repo.experiments) == 1
        exp = list(repo.experiments.values())[0]
        assert exp["status"] == "COMPLETED"
