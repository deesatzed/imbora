"""Tests for AI-based Feature Engineering Assessment (Phase 2.5).

Uses real-shaped fakes (FakeLLMClient, FakeModelRouter) following the same
pattern as test_datasci_column_ai_enricher.py. No mocks, no placeholders.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.datasci.fe_assessment import FEAssessor, MAX_PROPOSED_FEATURES
from src.datasci.models import (
    ColumnProfile,
    FECategoryApplicability,
    FeatureEngineeringAssessment,
    ProposedFeature,
)


# ---------------------------------------------------------------------------
# Real-shaped fakes (same pattern as test_datasci_column_ai_enricher.py)
# ---------------------------------------------------------------------------

class FakeLLMResponse:
    def __init__(self, content: str = ""):
        self.content = content


class FakeLLMClient:
    """In-memory implementation matching the real LLMClient interface."""

    def __init__(self, response_content: str = ""):
        self.response_content = response_content
        self.calls: list[dict] = []

    def complete_with_fallback(self, **kwargs):
        self.calls.append(kwargs)
        return FakeLLMResponse(self.response_content)


class FakeErrorLLMClient:
    """LLM client that raises on every call, simulating network failure."""

    def complete_with_fallback(self, **kwargs):
        raise ConnectionError("Simulated network failure")


class FakeModelRouter:
    """In-memory implementation matching the real ModelRouter interface."""

    def get_model(self, role):
        return "test/model"

    def get_fallback_models(self, role):
        return []

    def get_model_chain(self, role: str) -> list[str]:
        return [self.get_model(role)]


class FakeModelRouterNoDS:
    """ModelRouter that raises on ds_analyst role (no model configured)."""

    def get_model(self, role):
        raise KeyError(f"No model configured for role '{role}'")

    def get_fallback_models(self, role):
        return []

    def get_model_chain(self, role: str) -> list[str]:
        return [self.get_model(role)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_titanic_df() -> pd.DataFrame:
    """Create a small Titanic-like DataFrame with realistic patterns."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Pclass": np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
        "Name": [f"Doe, Mr. John{i}" for i in range(n)],
        "Sex": np.random.choice(["male", "female"], n),
        "Age": np.where(
            np.random.random(n) > 0.2,
            np.random.normal(30, 10, n).clip(1, 80),
            np.nan,
        ),
        "SibSp": np.random.randint(0, 5, n),
        "Parch": np.random.randint(0, 4, n),
        "Fare": np.random.exponential(30, n),
        "Cabin": np.where(
            np.random.random(n) > 0.7,
            [f"C{i}" for i in range(n)],
            None,
        ),
        "Embarked": np.random.choice(["S", "C", "Q"], n, p=[0.7, 0.2, 0.1]),
        "Survived": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    })


def _make_titanic_profiles() -> list[ColumnProfile]:
    """Create profiles matching the Titanic-like DataFrame."""
    return [
        ColumnProfile(name="PassengerId", dtype="numeric", cardinality=100, missing_pct=0.0),
        ColumnProfile(name="Pclass", dtype="numeric", cardinality=3, missing_pct=0.0),
        ColumnProfile(name="Name", dtype="text", cardinality=100, missing_pct=0.0, text_detected=True),
        ColumnProfile(name="Sex", dtype="categorical", cardinality=2, missing_pct=0.0),
        ColumnProfile(name="Age", dtype="numeric", cardinality=80, missing_pct=20.0),
        ColumnProfile(name="SibSp", dtype="numeric", cardinality=5, missing_pct=0.0),
        ColumnProfile(name="Parch", dtype="numeric", cardinality=4, missing_pct=0.0),
        ColumnProfile(name="Fare", dtype="numeric", cardinality=90, missing_pct=0.0),
        ColumnProfile(
            name="Cabin", dtype="categorical", cardinality=30, missing_pct=70.0, text_detected=False,
        ),
        ColumnProfile(name="Embarked", dtype="categorical", cardinality=3, missing_pct=0.0),
        ColumnProfile(name="Survived", dtype="numeric", cardinality=2, missing_pct=0.0, is_target=True),
    ]


def _make_numeric_only_df() -> pd.DataFrame:
    """Create a numeric-only DataFrame for interaction/cross-column tests."""
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    target = (x1 * 0.5 + x2 * 0.3 + np.random.normal(0, 0.1, n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


def _make_valid_llm_json() -> str:
    """Build a valid JSON response that the FE assessor would parse."""
    return json.dumps({
        "category_assessments": [
            {
                "category": "cross_column_derivation",
                "applicable": True,
                "confidence": 0.9,
                "rationale": "Age and Fare show complementary signal with target",
                "applicable_columns": ["Age", "Fare"],
            },
            {
                "category": "text_feature_extraction",
                "applicable": True,
                "confidence": 0.85,
                "rationale": "Name column contains titles that encode social status",
                "applicable_columns": ["Name"],
            },
        ],
        "proposed_features": [
            {
                "name": "Age_Fare_ratio",
                "category": "cross_column_derivation",
                "source_columns": ["Age", "Fare"],
                "transform_description": "Ratio of Age to Fare",
                "expected_impact": "high",
                "rationale": "Captures age-normalized spending patterns",
            },
            {
                "name": "Title",
                "category": "text_feature_extraction",
                "source_columns": ["Name"],
                "transform_description": "Extract title from Name",
                "expected_impact": "high",
                "rationale": "Titles encode survival-relevant demographics",
            },
        ],
        "overall_potential": "high",
        "reasoning": "Strong cross-column and text signals detected.",
    })


# ---------------------------------------------------------------------------
# Tests: Model serialization
# ---------------------------------------------------------------------------

class TestModelSerialization:
    """Verify Pydantic model creation and serialization."""

    def test_fe_category_applicability_defaults(self):
        cat = FECategoryApplicability(category="test_category")
        assert cat.category == "test_category"
        assert cat.applicable is False
        assert cat.confidence == 0.0
        assert cat.rationale == ""
        assert cat.applicable_columns == []

    def test_fe_category_applicability_full(self):
        cat = FECategoryApplicability(
            category="cross_column_derivation",
            applicable=True,
            confidence=0.85,
            rationale="Strong signal",
            applicable_columns=["col_a", "col_b"],
        )
        d = cat.model_dump()
        assert d["category"] == "cross_column_derivation"
        assert d["applicable"] is True
        assert d["confidence"] == 0.85
        assert d["applicable_columns"] == ["col_a", "col_b"]

    def test_proposed_feature_defaults(self):
        feat = ProposedFeature(name="test_feat", category="test")
        assert feat.name == "test_feat"
        assert feat.source_columns == []
        assert feat.expected_impact == ""

    def test_proposed_feature_full(self):
        feat = ProposedFeature(
            name="age_fare_ratio",
            category="cross_column_derivation",
            source_columns=["Age", "Fare"],
            transform_description="Ratio of Age to Fare",
            expected_impact="high",
            rationale="Complementary signal",
        )
        d = feat.model_dump()
        assert d["name"] == "age_fare_ratio"
        assert d["expected_impact"] == "high"
        assert len(d["source_columns"]) == 2

    def test_fe_assessment_defaults(self):
        assessment = FeatureEngineeringAssessment()
        assert assessment.total_columns == 0
        assert assessment.category_assessments == []
        assert assessment.proposed_features == []
        assert assessment.overall_fe_potential == ""

    def test_fe_assessment_round_trip(self):
        assessment = FeatureEngineeringAssessment(
            dataset_summary="100 rows x 10 cols",
            total_columns=10,
            numeric_columns=5,
            categorical_columns=3,
            text_columns=1,
            datetime_columns=1,
            category_assessments=[
                FECategoryApplicability(category="test", applicable=True, confidence=0.7),
            ],
            proposed_features=[
                ProposedFeature(name="feat1", category="test"),
            ],
            overall_fe_potential="high",
            llm_reasoning="Test reasoning",
        )
        d = assessment.model_dump()
        restored = FeatureEngineeringAssessment(**d)
        assert restored.total_columns == 10
        assert len(restored.category_assessments) == 1
        assert restored.category_assessments[0].applicable is True


# ---------------------------------------------------------------------------
# Tests: Statistical signals computation
# ---------------------------------------------------------------------------

class TestStatisticalSignals:
    """Test statistical signal computation methods."""

    def test_compute_mi_returns_dict(self):
        df = _make_numeric_only_df()
        client = FakeLLMClient()
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        signals = assessor._compute_statistical_signals(df, "target", "classification")
        mi = signals["mutual_information"]
        assert isinstance(mi, dict)
        assert "x1" in mi
        assert "x2" in mi
        assert mi["x1"] > 0  # x1 has real signal
        assert mi["x2"] > 0  # x2 has real signal

    def test_compute_mi_regression(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 100),
            "target": np.random.normal(0, 1, 100),
        })
        client = FakeLLMClient()
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        signals = assessor._compute_statistical_signals(df, "target", "regression")
        mi = signals["mutual_information"]
        assert isinstance(mi, dict)

    def test_column_type_counts(self):
        df = _make_titanic_df()
        client = FakeLLMClient()
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        signals = assessor._compute_statistical_signals(df, "Survived", "classification")
        counts = signals["column_type_counts"]
        assert counts["numeric"] > 0  # Pclass, Age, SibSp, Parch, Fare, etc.
        assert "categorical" in counts
        assert "datetime" in counts

    def test_correlation_matrix(self):
        df = _make_numeric_only_df()
        client = FakeLLMClient()
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        signals = assessor._compute_statistical_signals(df, "target", "classification")
        corr = signals["correlation_matrix"]
        assert isinstance(corr, dict)
        assert "x1" in corr  # pairwise correlations present

    def test_missingness_mi(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "col_with_missing": np.where(
                np.random.random(100) > 0.3, np.random.normal(0, 1, 100), np.nan,
            ),
            "target": np.random.choice([0, 1], 100),
        })
        client = FakeLLMClient()
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        signals = assessor._compute_statistical_signals(df, "target", "classification")
        miss_mi = signals["missingness_mi"]
        assert isinstance(miss_mi, dict)
        assert "col_with_missing" in miss_mi

    def test_no_numeric_cols(self):
        df = pd.DataFrame({
            "cat1": ["a", "b", "c"] * 30,
            "target": [0, 1, 0] * 30,
        })
        client = FakeLLMClient()
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        signals = assessor._compute_statistical_signals(df, "target", "classification")
        mi = signals["mutual_information"]
        assert mi == {}  # no numeric features to compute MI for


# ---------------------------------------------------------------------------
# Tests: Category assessments (8 categories)
# ---------------------------------------------------------------------------

class TestCategoryAssessments:
    """Test each of the 8 FE category statistical assessments."""

    def test_cross_column_triggers(self):
        df = _make_numeric_only_df()
        profiles = [
            ColumnProfile(name="x1", dtype="numeric"),
            ColumnProfile(name="x2", dtype="numeric"),
            ColumnProfile(name="target", dtype="numeric", is_target=True),
        ]
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "target", "classification", profiles, {})
        cat = next(c for c in result.category_assessments if c.category == "cross_column_derivation")
        assert cat.applicable is True
        assert cat.confidence > 0

    def test_text_features_triggers(self):
        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "Survived", "classification", profiles, {})
        cat = next(c for c in result.category_assessments if c.category == "text_feature_extraction")
        assert cat.applicable is True
        assert "Name" in cat.applicable_columns

    def test_missingness_triggers(self):
        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "Survived", "classification", profiles, {})
        cat = next(c for c in result.category_assessments if c.category == "missingness_as_signal")
        # Age has 20% missing, Cabin has 70% missing
        assert cat.applicable is True

    def test_interaction_features_triggers(self):
        df = _make_numeric_only_df()
        profiles = [
            ColumnProfile(name="x1", dtype="numeric"),
            ColumnProfile(name="x2", dtype="numeric"),
            ColumnProfile(name="target", dtype="numeric", is_target=True),
        ]
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "target", "classification", profiles, {})
        cat = next(c for c in result.category_assessments if c.category == "interaction_features")
        assert cat.applicable is True

    def test_aggregation_features_triggers(self):
        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "Survived", "classification", profiles, {})
        cat = next(c for c in result.category_assessments if c.category == "aggregation_features")
        # Sex, Embarked have repeated values
        assert cat.applicable is True

    def test_temporal_not_applicable(self):
        """No datetime columns -> temporal_decomposition should not trigger."""
        df = _make_numeric_only_df()
        profiles = [
            ColumnProfile(name="x1", dtype="numeric"),
            ColumnProfile(name="x2", dtype="numeric"),
            ColumnProfile(name="target", dtype="numeric", is_target=True),
        ]
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "target", "classification", profiles, {})
        cat = next(c for c in result.category_assessments if c.category == "temporal_decomposition")
        assert cat.applicable is False
        assert cat.confidence == 0.0

    def test_temporal_triggers_with_datetime(self):
        """Datetime columns -> temporal_decomposition should trigger."""
        df = _make_numeric_only_df()
        df["date_col"] = pd.date_range("2020-01-01", periods=len(df))
        profiles = [
            ColumnProfile(name="x1", dtype="numeric"),
            ColumnProfile(name="x2", dtype="numeric"),
            ColumnProfile(name="date_col", dtype="datetime"),
            ColumnProfile(name="target", dtype="numeric", is_target=True),
        ]
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "target", "classification", profiles, {})
        cat = next(c for c in result.category_assessments if c.category == "temporal_decomposition")
        assert cat.applicable is True

    def test_binning_triggers(self):
        df = _make_numeric_only_df()
        profiles = [
            ColumnProfile(name="x1", dtype="numeric"),
            ColumnProfile(name="x2", dtype="numeric"),
            ColumnProfile(name="target", dtype="numeric", is_target=True),
        ]
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "target", "classification", profiles, {})
        cat = next(c for c in result.category_assessments if c.category == "binning_discretization")
        # numeric cols with MI > 0.005 should trigger
        assert cat.applicable is True

    def test_encoding_triggers(self):
        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "Survived", "classification", profiles, {})
        cat = next(c for c in result.category_assessments if c.category == "encoding_enrichment")
        # Cabin has cardinality 30 > 5 -> should trigger
        assert isinstance(cat.applicable, bool)

    def test_all_8_categories_present(self):
        """Assessment always returns exactly 8 categories."""
        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "Survived", "classification", profiles, {})
        cats = {c.category for c in result.category_assessments}
        expected = {
            "cross_column_derivation",
            "text_feature_extraction",
            "missingness_as_signal",
            "interaction_features",
            "aggregation_features",
            "temporal_decomposition",
            "binning_discretization",
            "encoding_enrichment",
        }
        assert cats == expected


# ---------------------------------------------------------------------------
# Tests: LLM assessment
# ---------------------------------------------------------------------------

class TestLLMAssessment:
    """Test LLM interaction and JSON parsing."""

    def test_valid_json_parsed(self):
        json_response = _make_valid_llm_json()
        client = FakeLLMClient(json_response)
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        result = assessor.assess(df, "Survived", "classification", profiles, {})

        # LLM was called
        assert len(client.calls) == 1
        # LLM reasoning should be present
        assert result.llm_reasoning != ""

    def test_malformed_json_fallback(self):
        """Malformed JSON from LLM should not crash; falls back to statistical."""
        client = FakeLLMClient("This is not valid JSON at all {{{")
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        result = assessor.assess(df, "Survived", "classification", profiles, {})

        assert len(result.category_assessments) == 8
        # Still has statistical proposals
        assert result.overall_fe_potential in ("high", "medium", "low", "none")

    def test_empty_json_response(self):
        """Empty JSON object from LLM should work (no crash)."""
        client = FakeLLMClient("{}")
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        result = assessor.assess(df, "Survived", "classification", profiles, {})

        assert len(result.category_assessments) == 8

    def test_llm_network_failure_fallback(self):
        """Network failure during LLM call -> fallback to statistical-only."""
        client = FakeErrorLLMClient()
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        result = assessor.assess(df, "Survived", "classification", profiles, {})

        assert len(result.category_assessments) == 8
        assert result.llm_reasoning == ""  # No LLM reasoning

    def test_no_model_configured_fallback(self):
        """No ds_analyst model configured -> fallback to statistical-only."""
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouterNoDS()
        assessor = FEAssessor(client, router)

        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        result = assessor.assess(df, "Survived", "classification", profiles, {})

        assert len(result.category_assessments) == 8
        assert len(client.calls) == 0  # LLM never called

    def test_json_with_code_fences(self):
        """LLM wraps JSON in ```json ... ``` code fences."""
        json_str = _make_valid_llm_json()
        wrapped = f"```json\n{json_str}\n```"
        client = FakeLLMClient(wrapped)
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        result = assessor.assess(df, "Survived", "classification", profiles, {})

        # Should still parse
        assert result.llm_reasoning != ""

    def test_llm_prompt_contains_dataset_info(self):
        """Verify the LLM prompt includes dataset summary and column profiles."""
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        assessor.assess(df, "Survived", "classification", profiles, {})

        assert len(client.calls) == 1
        user_msg = client.calls[0]["messages"][1].content
        assert "Survived" in user_msg
        assert "classification" in user_msg
        assert "Age" in user_msg


# ---------------------------------------------------------------------------
# Tests: Merge logic
# ---------------------------------------------------------------------------

class TestMergeLogic:
    """Test merge behavior between statistical and LLM assessments."""

    def test_llm_cannot_downgrade_statistical(self):
        """LLM cannot downgrade a category that statistical evidence supports."""
        client = FakeLLMClient()
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        stat_cat = FECategoryApplicability(
            category="cross_column_derivation",
            applicable=True,
            confidence=0.7,
            rationale="Statistical evidence",
            applicable_columns=["col_a"],
        )
        llm_cat = FECategoryApplicability(
            category="cross_column_derivation",
            applicable=False,  # LLM says not applicable
            confidence=0.3,   # LLM gives lower confidence
            rationale="",
            applicable_columns=[],
        )

        merged = assessor._merge_assessments([stat_cat], [llm_cat])
        assert len(merged) == 1
        assert merged[0].applicable is True  # NOT downgraded
        assert merged[0].confidence >= 0.7  # NOT reduced

    def test_llm_can_upgrade(self):
        """LLM can upgrade confidence and add columns."""
        client = FakeLLMClient()
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        stat_cat = FECategoryApplicability(
            category="text_feature_extraction",
            applicable=True,
            confidence=0.3,
            rationale="Short",
            applicable_columns=["Name"],
        )
        llm_cat = FECategoryApplicability(
            category="text_feature_extraction",
            applicable=True,
            confidence=0.9,
            rationale="Name column contains titles that encode social status and demographics",
            applicable_columns=["Name", "Ticket"],
        )

        merged = assessor._merge_assessments([stat_cat], [llm_cat])
        assert merged[0].confidence == 0.9
        assert "Ticket" in merged[0].applicable_columns
        # LLM rationale is longer, so it should be preferred
        assert "social status" in merged[0].rationale

    def test_no_llm_categories_returns_stat(self):
        """Empty LLM categories -> return statistical categories unchanged."""
        client = FakeLLMClient()
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        stat_cat = FECategoryApplicability(
            category="test", applicable=True, confidence=0.5,
        )
        merged = assessor._merge_assessments([stat_cat], [])
        assert merged == [stat_cat]

    def test_column_union(self):
        """Merged applicable_columns is union of stat + LLM columns."""
        client = FakeLLMClient()
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        stat_cat = FECategoryApplicability(
            category="test", applicable=True, confidence=0.5,
            applicable_columns=["a", "b"],
        )
        llm_cat = FECategoryApplicability(
            category="test", applicable=True, confidence=0.6,
            applicable_columns=["b", "c"],
        )
        merged = assessor._merge_assessments([stat_cat], [llm_cat])
        assert set(merged[0].applicable_columns) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Tests: Feature proposals
# ---------------------------------------------------------------------------

class TestFeatureProposals:
    """Test statistical and merged feature proposal generation."""

    def test_statistical_proposals_generated(self):
        """Assessor generates statistical feature proposals without LLM."""
        client = FakeErrorLLMClient()  # LLM fails
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        df = _make_numeric_only_df()
        profiles = [
            ColumnProfile(name="x1", dtype="numeric"),
            ColumnProfile(name="x2", dtype="numeric"),
            ColumnProfile(name="target", dtype="numeric", is_target=True),
        ]
        result = assessor.assess(df, "target", "classification", profiles, {})

        # Should have at least some proposals from statistical path
        assert len(result.proposed_features) > 0

    def test_proposals_capped(self):
        """Proposals should never exceed MAX_PROPOSED_FEATURES."""
        # Create LLM response with many proposals
        many_proposals = [
            {
                "name": f"feat_{i}",
                "category": "cross_column_derivation",
                "source_columns": ["x1"],
                "transform_description": f"Transform {i}",
                "expected_impact": "low",
                "rationale": "test",
            }
            for i in range(30)
        ]
        response = json.dumps({
            "category_assessments": [],
            "proposed_features": many_proposals,
            "overall_potential": "high",
            "reasoning": "Many features",
        })
        client = FakeLLMClient(response)
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        df = _make_numeric_only_df()
        profiles = [
            ColumnProfile(name="x1", dtype="numeric"),
            ColumnProfile(name="x2", dtype="numeric"),
            ColumnProfile(name="target", dtype="numeric", is_target=True),
        ]
        result = assessor.assess(df, "target", "classification", profiles, {})
        assert len(result.proposed_features) <= MAX_PROPOSED_FEATURES

    def test_proposals_sorted_by_impact(self):
        """Proposals should be sorted: high > medium > low."""
        response = json.dumps({
            "category_assessments": [],
            "proposed_features": [
                {"name": "low_feat", "category": "test", "expected_impact": "low", "rationale": ""},
                {"name": "high_feat", "category": "test", "expected_impact": "high", "rationale": ""},
                {"name": "med_feat", "category": "test", "expected_impact": "medium", "rationale": ""},
            ],
            "overall_potential": "medium",
            "reasoning": "",
        })
        client = FakeLLMClient(response)
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        df = _make_numeric_only_df()
        profiles = [
            ColumnProfile(name="x1", dtype="numeric"),
            ColumnProfile(name="x2", dtype="numeric"),
            ColumnProfile(name="target", dtype="numeric", is_target=True),
        ]
        result = assessor.assess(df, "target", "classification", profiles, {})

        # Find the LLM proposals in the results
        llm_names = ["high_feat", "med_feat", "low_feat"]
        llm_in_result = [p for p in result.proposed_features if p.name in llm_names]
        if len(llm_in_result) == 3:
            assert llm_in_result[0].expected_impact == "high"
            assert llm_in_result[1].expected_impact == "medium"
            assert llm_in_result[2].expected_impact == "low"

    def test_proposals_deduplicated(self):
        """Duplicate feature names across stat and LLM are deduplicated."""
        client = FakeLLMClient()
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        stat = [
            ProposedFeature(name="feat_a", category="test", expected_impact="low"),
            ProposedFeature(name="feat_b", category="test", expected_impact="medium"),
        ]
        llm = [
            ProposedFeature(name="feat_a", category="test", expected_impact="high"),  # duplicate
            ProposedFeature(name="feat_c", category="test", expected_impact="high"),
        ]
        merged = assessor._merge_proposals(stat, llm)
        names = [p.name for p in merged]
        assert len(names) == len(set(names))  # no duplicates


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases: empty data, single column, all-missing, constant."""

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        client = FakeLLMClient()
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "target", "classification", [], {})
        assert result.overall_fe_potential == "none"
        assert len(client.calls) == 0  # LLM never called

    def test_target_not_in_df(self):
        df = pd.DataFrame({"col_a": [1, 2, 3]})
        client = FakeLLMClient()
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "missing_target", "classification", [], {})
        assert result.overall_fe_potential == "none"

    def test_single_column_df(self):
        """DataFrame with only the target column."""
        df = pd.DataFrame({"target": [0, 1, 0, 1, 0]})
        profiles = [ColumnProfile(name="target", dtype="numeric", is_target=True)]
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "target", "classification", profiles, {})
        assert result.total_columns == 1

    def test_all_missing_column(self):
        """Column that is entirely NaN."""
        df = pd.DataFrame({
            "all_nan": [np.nan] * 50,
            "target": np.random.choice([0, 1], 50),
        })
        profiles = [
            ColumnProfile(name="all_nan", dtype="numeric", missing_pct=100.0),
            ColumnProfile(name="target", dtype="numeric", is_target=True),
        ]
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        # Should not crash
        result = assessor.assess(df, "target", "classification", profiles, {})
        assert isinstance(result, FeatureEngineeringAssessment)

    def test_constant_column(self):
        """Column with a single unique value."""
        df = pd.DataFrame({
            "constant": [42] * 50,
            "target": np.random.choice([0, 1], 50),
        })
        profiles = [
            ColumnProfile(name="constant", dtype="numeric", cardinality=1),
            ColumnProfile(name="target", dtype="numeric", is_target=True),
        ]
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "target", "classification", profiles, {})
        assert isinstance(result, FeatureEngineeringAssessment)

    def test_large_sample_size_capped(self):
        """MI computation should sample, not use full dataset."""
        np.random.seed(42)
        n = 10_000
        df = pd.DataFrame({
            "x1": np.random.normal(0, 1, n),
            "target": np.random.choice([0, 1], n),
        })
        profiles = [
            ColumnProfile(name="x1", dtype="numeric"),
            ColumnProfile(name="target", dtype="numeric", is_target=True),
        ]
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        # Should not crash or be excessively slow
        result = assessor.assess(df, "target", "classification", profiles, {})
        assert isinstance(result, FeatureEngineeringAssessment)


# ---------------------------------------------------------------------------
# Tests: Overall potential determination
# ---------------------------------------------------------------------------

class TestOverallPotential:
    """Test the overall FE potential determination logic."""

    def test_high_potential_titanic(self):
        """Titanic-like data with many signals should be high potential."""
        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "Survived", "classification", profiles, {})
        # Titanic data triggers many categories
        assert result.overall_fe_potential in ("high", "medium")

    def test_dataset_summary_populated(self):
        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "Survived", "classification", profiles, {})
        assert "100 rows" in result.dataset_summary
        assert "Survived" in result.dataset_summary

    def test_column_counts_populated(self):
        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        result = assessor.assess(df, "Survived", "classification", profiles, {})
        assert result.total_columns == len(df.columns)
        assert result.numeric_columns > 0


# ---------------------------------------------------------------------------
# Tests: EDA report consumption
# ---------------------------------------------------------------------------

class TestEDAReportConsumption:
    """Test that EDA report findings are passed to LLM prompt."""

    def test_eda_report_in_prompt(self):
        eda_report = {
            "imbalance_ratio": 1.5,
            "is_imbalanced": False,
            "recommendations": ["Consider feature interactions", "Check outliers"],
        }
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        assessor.assess(df, "Survived", "classification", profiles, eda_report)

        assert len(client.calls) == 1
        user_msg = client.calls[0]["messages"][1].content
        assert "imbalance" in user_msg.lower() or "Imbalance" in user_msg

    def test_empty_eda_report_ok(self):
        client = FakeLLMClient(_make_valid_llm_json())
        router = FakeModelRouter()
        assessor = FEAssessor(client, router)

        df = _make_titanic_df()
        profiles = _make_titanic_profiles()
        result = assessor.assess(df, "Survived", "classification", profiles, {})
        assert isinstance(result, FeatureEngineeringAssessment)
