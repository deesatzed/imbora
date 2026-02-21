"""Tests for Phase 2: EDA Agent.

Uses real CSV data and real-shaped fakes. No mocks.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.core.config import DataScienceConfig
from src.datasci.agents.eda import EDAAgent
from src.datasci.models import EDAReport


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLMClient:
    def __init__(self, responses=None):
        self.calls: list[dict] = []
        self._call_count = 0
        self._responses = responses

    def complete_with_fallback(self, **kwargs):
        self._call_count += 1
        self.calls.append(kwargs)

        if self._responses and self._call_count <= len(self._responses):
            return FakeLLMResponse(self._responses[self._call_count - 1])

        # Default responses based on call order
        if self._call_count == 1:
            return FakeLLMResponse(
                "Q: What is the relationship between age and target?\n"
                "Q: Are there non-linear effects in income?\n"
                "Q: Does the text description contain predictive signal?\n"
                "Q: Is class imbalance a concern?\n"
                "Q: What interactions between features exist?"
            )
        else:
            return FakeLLMResponse(
                "The dataset shows strong age-income interactions. "
                "Text descriptions may contain latent signal worth embedding. "
                "The target distribution is reasonably balanced."
            )


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
        self.experiments[eid] = {"status": "RUNNING", **kwargs}
        return eid

    def update_ds_experiment(self, experiment_id, **kwargs):
        if experiment_id in self.experiments:
            self.experiments[experiment_id].update(kwargs)


def _make_test_csv(tmp_path):
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(18, 80, n),
        "income": np.random.normal(50000, 15000, n),
        "grade": np.random.choice(["A", "B", "C"], n),
        "target": np.random.choice([0, 1], n),
    })
    path = tmp_path / "eda_test.csv"
    df.to_csv(path, index=False)
    return str(path), df


@pytest.fixture
def test_csv(tmp_path):
    return _make_test_csv(tmp_path)


@pytest.fixture
def agent():
    return EDAAgent(
        llm_client=FakeLLMClient(),
        model_router=FakeModelRouter(),
        repository=FakeRepository(),
        ds_config=DataScienceConfig(),
    )


class TestEDAAgentProcess:
    def test_produces_valid_report(self, agent, test_csv):
        path, df = test_csv
        result = agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "audit_report": {
                "column_profiles": [
                    {"name": "age", "dtype": "numeric", "text_detected": False},
                    {"name": "income", "dtype": "numeric", "text_detected": False},
                    {"name": "grade", "dtype": "categorical", "text_detected": False},
                    {"name": "target", "dtype": "numeric", "is_target": True, "text_detected": False},
                ],
                "row_count": 100,
            },
        })
        assert result.status == "success"
        report = EDAReport(**result.data)
        assert len(report.questions_generated) > 0
        assert len(report.findings) > 0
        assert len(report.correlations) > 0

    def test_correlations_computed(self, agent, test_csv):
        path, _ = test_csv
        result = agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "audit_report": {"column_profiles": [], "row_count": 100},
        })
        report = EDAReport(**result.data)
        # Should have correlations for age and income vs target
        assert any("age" in k for k in report.correlations.keys())
        assert any("income" in k for k in report.correlations.keys())

    def test_target_distribution_analyzed(self, agent, test_csv):
        path, _ = test_csv
        result = agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "audit_report": {"column_profiles": [], "row_count": 100},
        })
        report = EDAReport(**result.data)
        target_findings = [f for f in report.findings if f.get("type") == "target_distribution"]
        assert len(target_findings) == 1

    def test_outlier_detection(self, agent, test_csv):
        path, _ = test_csv
        result = agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "audit_report": {"column_profiles": [], "row_count": 100},
        })
        report = EDAReport(**result.data)
        # Outlier summary is a dict of column->stats
        assert isinstance(report.outlier_summary, dict)

    def test_llm_called_twice(self, agent, test_csv):
        path, _ = test_csv
        agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "audit_report": {"column_profiles": [], "row_count": 100},
        })
        # Should call LLM for question generation + synthesis
        assert len(agent.llm_client.calls) == 2

    def test_recommendations_generated(self, agent, test_csv):
        path, _ = test_csv
        result = agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "audit_report": {"column_profiles": [], "row_count": 100},
        })
        report = EDAReport(**result.data)
        assert len(report.recommendations) > 0

    def test_synthesis_non_empty(self, agent, test_csv):
        path, _ = test_csv
        result = agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "audit_report": {"column_profiles": [], "row_count": 100},
        })
        report = EDAReport(**result.data)
        assert len(report.llm_synthesis) > 0


# ---------------------------------------------------------------------------
# Additional tests for coverage gaps
# ---------------------------------------------------------------------------

class TestEDAAgentEdgeCases:
    def test_regression_target_stats(self, tmp_path):
        """Test regression target produces target stats with skew instead of class counts."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "x0": np.random.normal(0, 1, n),
            "target": np.random.normal(50, 10, n),
        })
        path = tmp_path / "eda_reg.csv"
        df.to_csv(path, index=False)

        agent = EDAAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=DataScienceConfig(),
        )
        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "regression",
            "audit_report": {"column_profiles": [], "row_count": n},
        })
        assert result.status == "success"
        report = EDAReport(**result.data)
        target_findings = [f for f in report.findings if f.get("type") == "target_distribution"]
        assert len(target_findings) == 1
        assert "skew" in target_findings[0]

    def test_high_skew_detection(self, tmp_path):
        """Test that highly skewed features are detected (abs(skew) > 2.0)."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "skewed": np.exp(np.random.normal(0, 2, n)),
            "target": np.random.choice([0, 1], n),
        })
        path = tmp_path / "eda_skew.csv"
        df.to_csv(path, index=False)

        agent = EDAAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=DataScienceConfig(),
        )
        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "audit_report": {"column_profiles": [], "row_count": n},
        })
        report = EDAReport(**result.data)
        skew_findings = [f for f in report.findings if f.get("type") == "high_skew"]
        assert len(skew_findings) >= 1
        assert skew_findings[0]["column"] == "skewed"
        assert abs(skew_findings[0]["skew"]) > 2.0

    def test_text_column_analysis(self, tmp_path):
        """Test text column analysis when text_detected is True in audit_report."""
        n = 50
        df = pd.DataFrame({
            "text_col": [f"Some text content for row {i} with enough words" for i in range(n)],
            "target": np.random.choice([0, 1], n),
        })
        path = tmp_path / "eda_text.csv"
        df.to_csv(path, index=False)

        agent = EDAAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=DataScienceConfig(),
        )
        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "audit_report": {
                "column_profiles": [
                    {"name": "text_col", "dtype": "text", "text_detected": True},
                    {"name": "target", "dtype": "numeric", "is_target": True, "text_detected": False},
                ],
                "row_count": n,
            },
        })
        report = EDAReport(**result.data)
        assert len(report.text_column_analysis) >= 1
        assert report.text_column_analysis[0]["column"] == "text_col"
        assert "avg_length" in report.text_column_analysis[0]
        assert "avg_word_count" in report.text_column_analysis[0]
        assert "unique_ratio" in report.text_column_analysis[0]

    def test_recommendations_with_imbalanced_target(self, tmp_path):
        """Test that class imbalance (ratio > 3.0) triggers SMOTE recommendation."""
        n = 100
        df = pd.DataFrame({
            "x0": np.random.normal(0, 1, n),
            "target": [0] * 90 + [1] * 10,  # Highly imbalanced (ratio=9)
        })
        path = tmp_path / "eda_imb.csv"
        df.to_csv(path, index=False)

        agent = EDAAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=DataScienceConfig(),
        )
        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "audit_report": {"column_profiles": [], "row_count": n},
        })
        report = EDAReport(**result.data)
        assert any("balanc" in r.lower() or "smote" in r.lower() for r in report.recommendations)

    def test_strong_correlations_recommendation(self, tmp_path):
        """Test that strong correlations (abs > 0.5) trigger recommendation."""
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)
        df = pd.DataFrame({
            "x0": x,
            "x1": x * 0.9 + np.random.normal(0, 0.1, n),  # strongly correlated with x0
            "target": (x > 0).astype(int),
        })
        path = tmp_path / "eda_corr.csv"
        df.to_csv(path, index=False)

        agent = EDAAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=DataScienceConfig(),
        )
        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "audit_report": {"column_profiles": [], "row_count": n},
        })
        report = EDAReport(**result.data)
        assert any("correlation" in r.lower() for r in report.recommendations)

    def test_text_column_recommendation(self, tmp_path):
        """Test that text columns in analysis trigger embed recommendation."""
        n = 50
        df = pd.DataFrame({
            "description": [f"Sample text for row {i} with enough words" for i in range(n)],
            "target": np.random.choice([0, 1], n),
        })
        path = tmp_path / "eda_textrec.csv"
        df.to_csv(path, index=False)

        agent = EDAAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=DataScienceConfig(),
        )
        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "audit_report": {
                "column_profiles": [
                    {"name": "description", "dtype": "text", "text_detected": True},
                    {"name": "target", "dtype": "numeric", "is_target": True, "text_detected": False},
                ],
                "row_count": n,
            },
        })
        report = EDAReport(**result.data)
        assert any("embed" in r.lower() or "text" in r.lower() for r in report.recommendations)

    def test_exception_marks_experiment_failed(self, tmp_path):
        """Test that process exception marks experiment FAILED and re-raises."""
        repo = FakeRepository()
        agent = EDAAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=repo,
            ds_config=DataScienceConfig(),
        )
        with pytest.raises(Exception):
            agent.process({
                "dataset_path": "/nonexistent/file.csv",
                "target_column": "target",
                "project_id": uuid.uuid4(),
            })
        # Experiment should have been created and marked FAILED
        assert len(repo.experiments) == 1
        exp = list(repo.experiments.values())[0]
        assert exp["status"] == "FAILED"

    def test_parquet_loading(self, tmp_path):
        """Test loading data from parquet format (line 92)."""
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "x0": np.random.normal(0, 1, n),
            "target": np.random.choice([0, 1], n),
        })
        path = tmp_path / "eda_test.parquet"
        df.to_parquet(path, index=False)

        agent = EDAAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=DataScienceConfig(),
        )
        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "audit_report": {"column_profiles": [], "row_count": n},
        })
        assert result.status == "success"
        report = EDAReport(**result.data)
        assert len(report.findings) > 0

    def test_non_numeric_target_skips_correlations(self, tmp_path):
        """When target column is not numeric, correlations return empty (line 222)."""
        n = 50
        df = pd.DataFrame({
            "x0": np.random.normal(0, 1, n),
            "x1": np.random.normal(0, 1, n),
            "target": np.random.choice(["cat_a", "cat_b"], n),
        })
        path = tmp_path / "eda_cat_target.csv"
        df.to_csv(path, index=False)

        agent = EDAAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=DataScienceConfig(),
        )
        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "audit_report": {"column_profiles": [], "row_count": n},
        })
        report = EDAReport(**result.data)
        # Correlations should be empty because target is not numeric
        assert len(report.correlations) == 0

    def test_text_column_not_in_df_skipped(self, tmp_path):
        """Text column listed in audit_report but not in DataFrame is skipped (line 268)."""
        n = 30
        df = pd.DataFrame({
            "x0": np.random.normal(0, 1, n),
            "target": np.random.choice([0, 1], n),
        })
        path = tmp_path / "eda_ghost_col.csv"
        df.to_csv(path, index=False)

        agent = EDAAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=DataScienceConfig(),
        )
        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "audit_report": {
                "column_profiles": [
                    {"name": "nonexistent_text", "dtype": "text", "text_detected": True},
                ],
                "row_count": n,
            },
        })
        report = EDAReport(**result.data)
        # text column not in df should be skipped
        assert len(report.text_column_analysis) == 0

    def test_text_column_all_null_skipped(self, tmp_path):
        """Text column that is entirely NaN is skipped (line 271)."""
        n = 30
        df = pd.DataFrame({
            "text_col": [np.nan] * n,
            "x0": np.random.normal(0, 1, n),
            "target": np.random.choice([0, 1], n),
        })
        path = tmp_path / "eda_null_text.csv"
        df.to_csv(path, index=False)

        agent = EDAAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=DataScienceConfig(),
        )
        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "audit_report": {
                "column_profiles": [
                    {"name": "text_col", "dtype": "text", "text_detected": True},
                ],
                "row_count": n,
            },
        })
        report = EDAReport(**result.data)
        # All-null text column should be skipped
        assert len(report.text_column_analysis) == 0

    def test_question_parsing_non_q_prefix_lines(self, tmp_path):
        """LLM response with non-Q: lines should still be parsed as questions (lines 166-167)."""
        n = 30
        df = pd.DataFrame({
            "x0": np.random.normal(0, 1, n),
            "target": np.random.choice([0, 1], n),
        })
        path = tmp_path / "eda_qparse.csv"
        df.to_csv(path, index=False)

        # Return responses where the first call returns lines without Q: prefix
        llm = FakeLLMClient(responses=[
            "What are the key feature distributions?\n"
            "How does x0 relate to target?\n"
            "# This is a comment that should be ignored\n"
            "Are there any outlier patterns?\n"
            "\n"
            "What encoding strategies work best?",
            "Synthesis: the data shows patterns.",
        ])
        agent = EDAAgent(
            llm_client=llm,
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=DataScienceConfig(),
        )
        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "audit_report": {"column_profiles": [], "row_count": n},
        })
        report = EDAReport(**result.data)
        # Non-Q: non-empty non-# lines should be parsed as questions
        assert len(report.questions_generated) >= 4
        # The comment line starting with # should be excluded
        assert not any(q.startswith("#") for q in report.questions_generated)

    def test_low_sample_numeric_column_skipped(self, tmp_path):
        """Test that numeric columns with fewer than 2 non-null values are skipped in stat analysis."""
        n = 5
        df = pd.DataFrame({
            "sparse_col": [np.nan, np.nan, np.nan, np.nan, 1.0],
            "target": [0, 1, 0, 1, 0],
        })
        path = tmp_path / "eda_sparse.csv"
        df.to_csv(path, index=False)

        agent = EDAAgent(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
            repository=FakeRepository(),
            ds_config=DataScienceConfig(),
        )
        result = agent.process({
            "dataset_path": str(path),
            "target_column": "target",
            "project_id": uuid.uuid4(),
            "problem_type": "classification",
            "audit_report": {"column_profiles": [], "row_count": n},
        })
        assert result.status == "success"
        report = EDAReport(**result.data)
        # sparse_col should not appear in high_skew findings (only 1 non-null value)
        skew_findings = [f for f in report.findings if f.get("type") == "high_skew" and f.get("column") == "sparse_col"]
        assert len(skew_findings) == 0
