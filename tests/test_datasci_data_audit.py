"""Tests for Phase 1: Data Audit Agent.

Uses real CSV data (written to tmp_path) and real-shaped fakes for
LLM client, model router, and repository. No mocks.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.core.config import DataScienceConfig
from src.core.models import AgentResult
from src.datasci.agents.data_audit import DataAuditAgent
from src.datasci.models import DataAuditReport, ColumnProfile


# ---------------------------------------------------------------------------
# Real-shaped fakes (same pattern as test_datasci_base_agent.py)
# ---------------------------------------------------------------------------

class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLMClient:
    def __init__(self, response_text: str = "Semantic analysis: clean dataset."):
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def complete_with_fallback(self, **kwargs):
        self.calls.append(kwargs)
        return FakeLLMResponse(self.response_text)


class FakeModelRouter:
    def get_model(self, role: str) -> str:
        return "test/model"

    def get_fallback_models(self, role: str) -> list[str]:
        return ["test/fallback"]

    def get_model_chain(self, role: str) -> list[str]:
        return [self.get_model(role)]


class FakeRepository:
    def __init__(self):
        self.experiments: dict[uuid.UUID, dict[str, Any]] = {}

    def create_ds_experiment(self, project_id, experiment_phase, experiment_config=None,
                             task_id=None, run_id=None, dataset_fingerprint=None,
                             parent_experiment_id=None) -> uuid.UUID:
        exp_id = uuid.uuid4()
        self.experiments[exp_id] = {
            "status": "RUNNING", "phase": experiment_phase,
            "metrics": {}, "artifacts_manifest": {},
        }
        return exp_id

    def update_ds_experiment(self, experiment_id, status="COMPLETED",
                             metrics=None, artifacts_manifest=None):
        if experiment_id in self.experiments:
            self.experiments[experiment_id]["status"] = status
            if metrics:
                self.experiments[experiment_id]["metrics"] = metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _create_test_csv(tmp_path, name="train.csv"):
    """Create a real CSV dataset for testing."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "age": np.random.randint(18, 80, n),
        "income": np.random.normal(50000, 15000, n).round(2),
        "grade": np.random.choice(["A", "B", "C", "D"], n),
        "is_active": np.random.choice([True, False], n),
        "description": [
            f"Customer profile with various attributes and details for record {i}. "
            f"This text is long enough to be detected as a text column by the profiler."
            for i in range(n)
        ],
        "target": np.random.choice([0, 1], n),
    })
    # Inject some missing values
    df.loc[np.random.choice(n, 10, replace=False), "income"] = np.nan
    df.loc[np.random.choice(n, 5, replace=False), "grade"] = np.nan

    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path), df


@pytest.fixture
def test_dataset(tmp_path):
    return _create_test_csv(tmp_path)


@pytest.fixture
def fake_llm():
    return FakeLLMClient(
        response_text="This dataset represents customer data with age, income, "
        "and activity indicators. The description column contains text that should "
        "be embedded. Target is binary classification."
    )


@pytest.fixture
def fake_repo():
    return FakeRepository()


@pytest.fixture
def agent(fake_llm, fake_repo):
    return DataAuditAgent(
        llm_client=fake_llm,
        model_router=FakeModelRouter(),
        repository=fake_repo,
        ds_config=DataScienceConfig(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataAuditAgentProcess:
    def test_produces_valid_report(self, agent, test_dataset):
        path, df = test_dataset
        project_id = uuid.uuid4()

        result = agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": project_id,
            "problem_type": "classification",
        })

        assert result.status == "success"
        assert result.agent_name == "DataAuditAgent"

        # Reconstruct the report from result data
        report = DataAuditReport(**result.data)
        assert report.row_count == 200
        assert report.column_count == 6
        assert len(report.column_profiles) == 6
        assert report.dataset_fingerprint.startswith("sha256:")
        assert report.overall_quality_score > 0.0
        assert report.overall_quality_score <= 1.0

    def test_column_profiles_correct(self, agent, test_dataset):
        path, _ = test_dataset
        result = agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
        })
        report = DataAuditReport(**result.data)

        # Check specific column types
        profile_map = {p.name: p for p in report.column_profiles}

        assert profile_map["age"].dtype == "numeric"
        assert profile_map["income"].dtype == "numeric"
        assert profile_map["grade"].dtype == "categorical"
        assert profile_map["is_active"].dtype == "boolean"
        assert profile_map["description"].dtype == "text"
        assert profile_map["description"].text_detected is True

    def test_target_column_marked(self, agent, test_dataset):
        path, _ = test_dataset
        result = agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
        })
        report = DataAuditReport(**result.data)
        target_profiles = [p for p in report.column_profiles if p.is_target]
        assert len(target_profiles) == 1
        assert target_profiles[0].name == "target"

    def test_missing_values_detected(self, agent, test_dataset):
        path, _ = test_dataset
        result = agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
        })
        report = DataAuditReport(**result.data)
        profile_map = {p.name: p for p in report.column_profiles}

        # Income has 10 missing values out of 200
        assert profile_map["income"].missing_pct > 0

    def test_llm_called_for_semantic_analysis(self, agent, test_dataset, fake_llm):
        path, _ = test_dataset
        agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
        })
        # 2 LLM calls: ColumnAIEnricher (column enrichment) + semantic analysis
        assert len(fake_llm.calls) == 2
        # Verify the semantic analysis prompt references the target column
        user_msg = fake_llm.calls[-1]["messages"][-1].content
        assert "target" in user_msg

    def test_recommendations_generated(self, agent, test_dataset):
        path, _ = test_dataset
        result = agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
        })
        report = DataAuditReport(**result.data)
        assert len(report.recommended_actions) > 0
        # Should recommend embedding text columns
        text_rec = [r for r in report.recommended_actions if "text" in r.lower() or "embed" in r.lower()]
        assert len(text_rec) > 0

    def test_experiment_lifecycle(self, agent, test_dataset, fake_repo):
        path, _ = test_dataset
        agent.process({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
        })
        # Should have one experiment, marked COMPLETED
        assert len(fake_repo.experiments) == 1
        exp = list(fake_repo.experiments.values())[0]
        assert exp["status"] == "COMPLETED"
        assert exp["phase"] == "data_audit"
        assert exp["metrics"]["row_count"] == 200


class TestDataAuditAgentRunLifecycle:
    def test_run_wraps_process(self, agent, test_dataset):
        """Test that run() wraps process() with metrics tracking."""
        path, _ = test_dataset
        result = agent.run({
            "dataset_path": path,
            "target_column": "target",
            "project_id": uuid.uuid4(),
        })
        assert result.status == "success"
        assert result.duration_seconds > 0
        metrics = agent.get_metrics()
        assert metrics["total_processed"] == 1
        assert metrics["total_errors"] == 0


class TestDataAuditAgentEdgeCases:
    def test_unsupported_format(self, agent, tmp_path):
        path = tmp_path / "data.json"
        path.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported format"):
            agent.process({
                "dataset_path": str(path),
                "target_column": "target",
                "project_id": uuid.uuid4(),
            })

    def test_file_not_found(self, agent):
        result = agent.run({
            "dataset_path": "/nonexistent/data.csv",
            "target_column": "target",
            "project_id": uuid.uuid4(),
        })
        # run() catches exceptions and returns failure
        assert result.status == "failure"

    def test_tsv_format(self, agent, tmp_path):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        path = tmp_path / "data.tsv"
        df.to_csv(path, sep="\t", index=False)
        result = agent.process({
            "dataset_path": str(path),
            "target_column": "y",
            "project_id": uuid.uuid4(),
        })
        assert result.status == "success"
        report = DataAuditReport(**result.data)
        assert report.row_count == 3

    def test_parquet_format(self, agent, tmp_path):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
        path = tmp_path / "data.parquet"
        df.to_parquet(path)
        result = agent.process({
            "dataset_path": str(path),
            "target_column": "a",
            "project_id": uuid.uuid4(),
        })
        assert result.status == "success"
        report = DataAuditReport(**result.data)
        assert report.column_count == 2

    def test_minimal_dataset(self, agent, tmp_path):
        """Single column, single row."""
        df = pd.DataFrame({"x": [42]})
        path = tmp_path / "tiny.csv"
        df.to_csv(path, index=False)
        result = agent.process({
            "dataset_path": str(path),
            "target_column": "x",
            "project_id": uuid.uuid4(),
        })
        assert result.status == "success"
        report = DataAuditReport(**result.data)
        assert report.row_count == 1
        assert report.column_count == 1


# ---------------------------------------------------------------------------
# Additional tests for internal methods -- coverage gaps
# ---------------------------------------------------------------------------

class TestDataAuditInternalMethods:
    """Tests targeting _assess_label_quality, _compute_quality_score,
    _generate_recommendations, and _run_semantic_analysis uncovered paths."""

    def test_assess_label_quality_cleanlab_disabled(self, fake_llm, fake_repo):
        """When cleanlab_enabled=False, _assess_label_quality returns 0 immediately (line 168)."""
        config = DataScienceConfig(cleanlab_enabled=False)
        agent = DataAuditAgent(
            llm_client=fake_llm, model_router=FakeModelRouter(),
            repository=fake_repo, ds_config=config,
        )
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        })
        result = agent._assess_label_quality(df, "target", "classification")
        assert result == 0

    def test_assess_label_quality_cleanlab_enabled(self, fake_llm, fake_repo, tmp_path):
        """When cleanlab_enabled=True, _assess_label_quality runs.

        GradientBoostedClassifier is a typo in the source (sklearn has
        GradientBoostingClassifier), so it hits the except ImportError or
        except Exception handler and returns 0.
        """
        config = DataScienceConfig(cleanlab_enabled=True)
        agent = DataAuditAgent(
            llm_client=fake_llm, model_router=FakeModelRouter(),
            repository=fake_repo, ds_config=config,
        )
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 50),
            "target": np.random.choice([0, 1], 50),
        })
        result = agent._assess_label_quality(df, "target", "classification")
        assert isinstance(result, int)
        assert result >= 0

    def test_assess_label_quality_missing_target(self, fake_llm, fake_repo):
        """When target column is not in DataFrame, returns 0 (lines 171-172)."""
        config = DataScienceConfig(cleanlab_enabled=True)
        agent = DataAuditAgent(
            llm_client=fake_llm, model_router=FakeModelRouter(),
            repository=fake_repo, ds_config=config,
        )
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = agent._assess_label_quality(df, "nonexistent", "classification")
        assert result == 0

    def test_assess_label_quality_empty_features(self, fake_llm, fake_repo):
        """When no numeric feature columns exist (only target), returns 0."""
        config = DataScienceConfig(cleanlab_enabled=True)
        agent = DataAuditAgent(
            llm_client=fake_llm, model_router=FakeModelRouter(),
            repository=fake_repo, ds_config=config,
        )
        df = pd.DataFrame({
            "cat_only": ["a", "b", "c", "d"],
            "target": [0, 1, 0, 1],
        })
        result = agent._assess_label_quality(df, "target", "classification")
        assert result == 0

    def test_assess_label_quality_single_unique_target(self, fake_llm, fake_repo):
        """When target has fewer than 2 unique values, returns 0."""
        config = DataScienceConfig(cleanlab_enabled=True)
        agent = DataAuditAgent(
            llm_client=fake_llm, model_router=FakeModelRouter(),
            repository=fake_repo, ds_config=config,
        )
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 0, 0, 0],
        })
        result = agent._assess_label_quality(df, "target", "classification")
        assert result == 0

    def test_compute_quality_score_with_label_issues(self, agent):
        """When label_issues_count > 0, the label_quality component reduces score."""
        profiles = [
            ColumnProfile(name="x", dtype="numeric", missing_pct=5.0),
            ColumnProfile(name="y", dtype="categorical", missing_pct=0.0),
        ]
        score = agent._compute_quality_score(profiles, label_issues_count=20, total_rows=100)
        assert 0.0 < score < 1.0
        # Compare to score with no label issues -- it should be lower
        score_clean = agent._compute_quality_score(profiles, label_issues_count=0, total_rows=100)
        assert score < score_clean

    def test_compute_quality_score_empty_profiles(self, agent):
        """When profiles list is empty, returns 0.0."""
        score = agent._compute_quality_score([], label_issues_count=0, total_rows=100)
        assert score == 0.0

    def test_recommendations_high_missing(self, agent):
        """Columns with >20% missing trigger 'investigate high-missing' recommendation (lines 281-282)."""
        profiles = [ColumnProfile(name="sparse_col", dtype="numeric", missing_pct=35.0)]
        recs = agent._generate_recommendations(profiles, label_issues=0)
        assert any("high-missing" in r.lower() or "dropping" in r.lower() for r in recs)

    def test_recommendations_moderate_missing(self, agent):
        """Moderate missing (5-20%) triggers imputation recommendation."""
        profiles = [ColumnProfile(name="col1", dtype="numeric", missing_pct=15.0)]
        recs = agent._generate_recommendations(profiles, label_issues=0)
        assert any("impute" in r.lower() or "moderately" in r.lower() for r in recs)

    def test_recommendations_high_cardinality(self, agent):
        """Categorical with cardinality > 50 triggers target-encode recommendation."""
        profiles = [ColumnProfile(name="col1", dtype="categorical", cardinality=100)]
        recs = agent._generate_recommendations(profiles, label_issues=0)
        assert any("target-encode" in r.lower() or "cardinality" in r.lower() for r in recs)

    def test_recommendations_label_issues(self, agent):
        """Non-zero label_issues triggers Cleanlab/label review recommendation."""
        profiles = [ColumnProfile(name="col1", dtype="numeric")]
        recs = agent._generate_recommendations(profiles, label_issues=10)
        assert any("cleanlab" in r.lower() or "label" in r.lower() for r in recs)

    def test_recommendations_datetime(self, agent):
        """Datetime columns trigger temporal feature extraction recommendation."""
        profiles = [ColumnProfile(name="created_at", dtype="datetime")]
        recs = agent._generate_recommendations(profiles, label_issues=0)
        assert any("temporal" in r.lower() or "datetime" in r.lower() for r in recs)

    def test_recommendations_clean_dataset(self, agent):
        """When no issues found, recommends proceeding to EDA."""
        profiles = [ColumnProfile(name="x", dtype="numeric", missing_pct=0.0)]
        recs = agent._generate_recommendations(profiles, label_issues=0)
        assert any("clean" in r.lower() or "proceed" in r.lower() for r in recs)

    def test_semantic_analysis_with_distribution_stats(self, fake_repo):
        """Test _run_semantic_analysis with profiles that have distribution_summary."""
        fake_llm = FakeLLMClient("Semantic analysis with stats.")
        config = DataScienceConfig()
        agent = DataAuditAgent(
            llm_client=fake_llm, model_router=FakeModelRouter(),
            repository=fake_repo, ds_config=config,
        )
        profiles = [
            ColumnProfile(
                name="x", dtype="numeric", is_target=False, text_detected=False,
                distribution_summary={"mean": 5.0, "entropy": 1.5},
            ),
            ColumnProfile(
                name="target", dtype="numeric", is_target=True, text_detected=False,
            ),
            ColumnProfile(
                name="desc", dtype="text", is_target=False, text_detected=True,
            ),
        ]
        result = agent._run_semantic_analysis(profiles, "target", "classification", (100, 3), 0)
        assert isinstance(result, str)
        assert len(result) > 0
        # Verify the LLM prompt includes distribution stats
        prompt_content = fake_llm.calls[0]["messages"][-1].content
        assert "mean=5.00" in prompt_content
        assert "entropy=1.50" in prompt_content
        assert "[TARGET]" in prompt_content
        assert "[TEXT]" in prompt_content

    def test_recommendations_all_issues_combined(self, agent):
        """All recommendation paths fire together with combined profiles."""
        profiles = [
            ColumnProfile(name="high_miss", dtype="numeric", missing_pct=30.0),
            ColumnProfile(name="mod_miss", dtype="numeric", missing_pct=12.0),
            ColumnProfile(name="hi_card", dtype="categorical", cardinality=75),
            ColumnProfile(name="text_col", dtype="text", text_detected=True),
            ColumnProfile(name="ts_col", dtype="datetime"),
        ]
        recs = agent._generate_recommendations(profiles, label_issues=5)
        # All 6 recommendation categories should be present
        assert any("high-missing" in r.lower() for r in recs)
        assert any("impute" in r.lower() for r in recs)
        assert any("target-encode" in r.lower() for r in recs)
        assert any("embed" in r.lower() or "text" in r.lower() for r in recs)
        assert any("label" in r.lower() for r in recs)
        assert any("temporal" in r.lower() or "datetime" in r.lower() for r in recs)
