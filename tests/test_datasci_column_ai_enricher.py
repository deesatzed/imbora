"""Tests for ColumnAIEnricher: LLM-guided per-column semantic analysis.

Uses real-shaped fakes for LLM client and model router. No mocks.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.datasci.column_ai_enricher import BATCH_SIZE, ColumnAIEnricher
from src.datasci.models import ColumnProfile


# ---------------------------------------------------------------------------
# Real-shaped fakes
# ---------------------------------------------------------------------------

class FakeLLMResponse:
    def __init__(self, content: str = ""):
        self.content = content


class FakeLLMClient:
    def __init__(self, response_content: str = ""):
        self.response_content = response_content
        self.calls = []

    def complete_with_fallback(self, **kwargs):
        self.calls.append(kwargs)
        return FakeLLMResponse(self.response_content)


class FakeModelRouter:
    def get_model(self, role):
        return "test/model"

    def get_fallback_models(self, role):
        return []

    def get_model_chain(self, role: str) -> list[str]:
        return [self.get_model(role)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_enrichment_json(profiles: list[ColumnProfile]) -> str:
    """Build a valid JSON enrichment array for the given profiles."""
    items = []
    for p in profiles:
        items.append({
            "column": p.name,
            "semantic_meaning": f"{p.name} meaning",
            "semantic_dtype": "count",
            "imputation_strategy": "median",
            "encoding_strategy": "keep_numeric",
            "outlier_strategy": "clip_to_iqr",
            "text_processing_strategy": "none",
            "interaction_candidates": [],
            "data_quality_flags": [],
            "importance_prior": "high",
        })
    return json.dumps(items)


# ---------------------------------------------------------------------------
# Tests: enrich_profiles()
# ---------------------------------------------------------------------------

class TestEnrichProfiles:
    def test_valid_json_enrichment(self):
        """LLM returning valid JSON should enrich all profiles."""
        profiles = [
            ColumnProfile(name="age", dtype="numeric", cardinality=50, missing_pct=5.0),
            ColumnProfile(name="income", dtype="numeric", cardinality=1000, missing_pct=0.0),
        ]
        target_profile = ColumnProfile(name="target", dtype="numeric", is_target=True)

        all_profiles = profiles + [target_profile]
        json_response = _make_valid_enrichment_json(profiles)

        client = FakeLLMClient(response_content=json_response)
        router = FakeModelRouter()
        enricher = ColumnAIEnricher(llm_client=client, model_router=router)

        result = enricher.enrich_profiles(
            profiles=all_profiles,
            target_column="target",
            problem_type="classification",
            sample_rows=[{"age": 25, "income": 50000, "target": 1}],
        )

        # Target should not be enriched
        target_p = [p for p in result if p.name == "target"][0]
        assert target_p.semantic_meaning == ""

        # Non-target should be enriched
        age_p = [p for p in result if p.name == "age"][0]
        assert age_p.semantic_meaning == "age meaning"
        assert age_p.imputation_strategy == "median"
        assert age_p.importance_prior == "high"

    def test_empty_non_target_profiles_returns_unchanged(self):
        """When only target exists, return profiles unchanged."""
        profiles = [
            ColumnProfile(name="target", dtype="numeric", is_target=True),
        ]
        client = FakeLLMClient(response_content="[]")
        enricher = ColumnAIEnricher(llm_client=client, model_router=FakeModelRouter())

        result = enricher.enrich_profiles(
            profiles=profiles,
            target_column="target",
            problem_type="classification",
            sample_rows=[],
        )
        assert len(result) == 1
        assert result[0].name == "target"

    def test_llm_failure_falls_back_to_heuristic(self):
        """When LLM raises an error, heuristic enrichment is used."""

        class FailingLLMClient:
            def complete_with_fallback(self, **kwargs):
                raise RuntimeError("LLM unavailable")

        profiles = [
            ColumnProfile(name="age", dtype="numeric", cardinality=50, missing_pct=5.0),
        ]
        enricher = ColumnAIEnricher(
            llm_client=FailingLLMClient(),
            model_router=FakeModelRouter(),
        )

        result = enricher.enrich_profiles(
            profiles=profiles,
            target_column="target",
            problem_type="classification",
            sample_rows=[{"age": 25}],
        )
        age_p = [p for p in result if p.name == "age"][0]
        # Heuristic should set median for numeric
        assert age_p.imputation_strategy == "median"
        assert age_p.encoding_strategy == "keep_numeric"


# ---------------------------------------------------------------------------
# Tests: _parse_enrichments()
# ---------------------------------------------------------------------------

class TestParseEnrichments:
    def _make_enricher(self) -> ColumnAIEnricher:
        return ColumnAIEnricher(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
        )

    def test_valid_json(self):
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="age", dtype="numeric", cardinality=50),
        ]
        json_str = json.dumps([{
            "column": "age",
            "semantic_meaning": "patient age",
            "semantic_dtype": "age",
            "imputation_strategy": "median",
            "encoding_strategy": "keep_numeric",
            "outlier_strategy": "clip_to_iqr",
            "text_processing_strategy": "none",
            "interaction_candidates": ["income"],
            "data_quality_flags": [],
            "importance_prior": "high",
        }])
        result = enricher._parse_enrichments(json_str, profiles)
        assert "age" in result
        assert result["age"]["semantic_meaning"] == "patient age"
        assert result["age"]["importance_prior"] == "high"
        assert result["age"]["interaction_candidates"] == ["income"]

    def test_markdown_fenced_json(self):
        """JSON wrapped in markdown code fences should be parsed."""
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="temp", dtype="numeric", cardinality=30),
        ]
        inner_json = json.dumps([{
            "column": "temp",
            "semantic_meaning": "temperature",
            "semantic_dtype": "ratio",
            "imputation_strategy": "mean",
            "encoding_strategy": "keep_numeric",
            "outlier_strategy": "none",
            "text_processing_strategy": "none",
            "interaction_candidates": [],
            "data_quality_flags": [],
            "importance_prior": "medium",
        }])
        markdown_wrapped = f"```json\n{inner_json}\n```"
        result = enricher._parse_enrichments(markdown_wrapped, profiles)
        assert "temp" in result
        assert result["temp"]["imputation_strategy"] == "mean"

    def test_invalid_json_falls_back_to_heuristic(self):
        """Completely invalid JSON should fall back to heuristic."""
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="score", dtype="numeric", cardinality=100),
        ]
        result = enricher._parse_enrichments("This is not JSON at all!", profiles)
        assert "score" in result
        # Heuristic fallback for numeric
        assert result["score"]["imputation_strategy"] == "median"

    def test_non_list_json_falls_back_to_heuristic(self):
        """JSON that is a dict (not a list) should fall back to heuristic."""
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="val", dtype="numeric", cardinality=50),
        ]
        result = enricher._parse_enrichments('{"not": "a list"}', profiles)
        assert "val" in result
        assert result["val"]["imputation_strategy"] == "median"

    def test_invalid_field_values_get_defaults(self):
        """Invalid enum values in LLM output should get default values."""
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="col1", dtype="numeric", cardinality=20),
        ]
        json_str = json.dumps([{
            "column": "col1",
            "semantic_meaning": "some column",
            "semantic_dtype": "whatever",
            "imputation_strategy": "invalid_strategy",
            "encoding_strategy": "magic_encode",
            "outlier_strategy": "destroy_outliers",
            "text_processing_strategy": "do_something_weird",
            "interaction_candidates": [],
            "data_quality_flags": [],
            "importance_prior": "ultra_high",
        }])
        result = enricher._parse_enrichments(json_str, profiles)
        assert result["col1"]["imputation_strategy"] == "median"  # default
        assert result["col1"]["encoding_strategy"] == "keep_numeric"  # default
        assert result["col1"]["outlier_strategy"] == "none"  # default
        assert result["col1"]["text_processing_strategy"] == "none"  # default
        assert result["col1"]["importance_prior"] == "medium"  # default

    def test_missing_columns_filled_with_heuristic(self):
        """Columns not in LLM output get heuristic enrichment."""
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="col_a", dtype="numeric", cardinality=50),
            ColumnProfile(name="col_b", dtype="categorical", cardinality=5),
        ]
        # Only col_a is in the JSON
        json_str = json.dumps([{
            "column": "col_a",
            "semantic_meaning": "first column",
            "semantic_dtype": "count",
            "imputation_strategy": "median",
            "encoding_strategy": "keep_numeric",
            "outlier_strategy": "clip_to_iqr",
            "text_processing_strategy": "none",
            "interaction_candidates": [],
            "data_quality_flags": [],
            "importance_prior": "high",
        }])
        result = enricher._parse_enrichments(json_str, profiles)
        assert "col_a" in result
        assert "col_b" in result
        # col_b should get heuristic enrichment for categorical
        assert result["col_b"]["encoding_strategy"] == "onehot"

    def test_json_embedded_in_text(self):
        """JSON array embedded in explanatory text should be extracted."""
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="x", dtype="numeric", cardinality=20),
        ]
        text = 'Here is the analysis:\n[{"column": "x", "semantic_meaning": "feature x", "semantic_dtype": "ratio", "imputation_strategy": "mean", "encoding_strategy": "keep_numeric", "outlier_strategy": "none", "text_processing_strategy": "none", "interaction_candidates": [], "data_quality_flags": [], "importance_prior": "medium"}]\nDone.'
        result = enricher._parse_enrichments(text, profiles)
        assert "x" in result
        assert result["x"]["semantic_meaning"] == "feature x"


# ---------------------------------------------------------------------------
# Tests: _heuristic_enrichment()
# ---------------------------------------------------------------------------

class TestHeuristicEnrichment:
    def _make_enricher(self) -> ColumnAIEnricher:
        return ColumnAIEnricher(
            llm_client=FakeLLMClient(),
            model_router=FakeModelRouter(),
        )

    def test_numeric_column(self):
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="value", dtype="numeric", cardinality=100),
        ]
        result = enricher._heuristic_enrichment(profiles)
        assert result["value"]["imputation_strategy"] == "median"
        assert result["value"]["encoding_strategy"] == "keep_numeric"
        assert result["value"]["outlier_strategy"] == "clip_to_iqr"
        assert result["value"]["text_processing_strategy"] == "none"
        assert result["value"]["importance_prior"] == "medium"

    def test_categorical_low_cardinality(self):
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="color", dtype="categorical", cardinality=5),
        ]
        result = enricher._heuristic_enrichment(profiles)
        assert result["color"]["encoding_strategy"] == "onehot"
        assert result["color"]["imputation_strategy"] == "mode"

    def test_categorical_medium_cardinality(self):
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="city", dtype="categorical", cardinality=25),
        ]
        result = enricher._heuristic_enrichment(profiles)
        assert result["city"]["encoding_strategy"] == "target_encode"

    def test_categorical_high_cardinality(self):
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="zipcode", dtype="categorical", cardinality=100),
        ]
        result = enricher._heuristic_enrichment(profiles)
        assert result["zipcode"]["encoding_strategy"] == "frequency"

    def test_text_column(self):
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="review", dtype="text", cardinality=500, text_detected=True),
        ]
        result = enricher._heuristic_enrichment(profiles)
        assert result["review"]["encoding_strategy"] == "tfidf"
        assert result["review"]["text_processing_strategy"] == "tfidf"
        assert result["review"]["imputation_strategy"] == "fill_empty_string"

    def test_datetime_column(self):
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="created_at", dtype="datetime", cardinality=1000),
        ]
        result = enricher._heuristic_enrichment(profiles)
        assert result["created_at"]["imputation_strategy"] == "forward_fill"
        assert result["created_at"]["encoding_strategy"] == "keep_numeric"

    def test_boolean_column(self):
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="is_active", dtype="boolean", cardinality=2),
        ]
        result = enricher._heuristic_enrichment(profiles)
        assert result["is_active"]["imputation_strategy"] == "mode"
        assert result["is_active"]["encoding_strategy"] == "keep_numeric"

    def test_high_missing_flag(self):
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="sparse_col", dtype="numeric", cardinality=10, missing_pct=75.0),
        ]
        result = enricher._heuristic_enrichment(profiles)
        assert "high_missing" in result["sparse_col"]["data_quality_flags"]

    def test_constant_column_flagged_for_drop(self):
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="const_col", dtype="numeric", cardinality=1),
        ]
        result = enricher._heuristic_enrichment(profiles)
        assert "constant" in result["const_col"]["data_quality_flags"]
        assert result["const_col"]["importance_prior"] == "drop"

    def test_id_column_flagged_for_drop(self):
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="patient_id", dtype="numeric", cardinality=1000),
        ]
        result = enricher._heuristic_enrichment(profiles)
        assert "id_column" in result["patient_id"]["data_quality_flags"]
        assert result["patient_id"]["importance_prior"] == "drop"

    def test_index_column_flagged(self):
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="row_index", dtype="numeric", cardinality=500),
        ]
        result = enricher._heuristic_enrichment(profiles)
        assert "id_column" in result["row_index"]["data_quality_flags"]

    def test_unnamed_column_flagged(self):
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="Unnamed_0", dtype="numeric", cardinality=100),
        ]
        result = enricher._heuristic_enrichment(profiles)
        assert "id_column" in result["Unnamed_0"]["data_quality_flags"]

    def test_unknown_dtype_gets_defaults(self):
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="mystery", dtype="unknown_type", cardinality=50),
        ]
        result = enricher._heuristic_enrichment(profiles)
        assert result["mystery"]["imputation_strategy"] == "median"
        assert result["mystery"]["encoding_strategy"] == "keep_numeric"

    def test_text_detected_triggers_text_path_on_other_dtype(self):
        """text_detected=True on a non-standard dtype triggers the text path.

        The heuristic uses elif chain: numeric -> categorical -> text/text_detected
        -> datetime -> boolean -> else. For dtype='other' with text_detected=True,
        the 'text or text_detected' elif fires.
        """
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="notes", dtype="other", cardinality=200, text_detected=True),
        ]
        result = enricher._heuristic_enrichment(profiles)
        assert result["notes"]["text_processing_strategy"] == "tfidf"
        assert result["notes"]["encoding_strategy"] == "tfidf"
        assert result["notes"]["imputation_strategy"] == "fill_empty_string"

    def test_categorical_with_text_detected_follows_categorical_path(self):
        """Categorical dtype takes precedence over text_detected in heuristic.

        The elif chain means dtype='categorical' matches before text_detected.
        """
        enricher = self._make_enricher()
        profiles = [
            ColumnProfile(name="notes", dtype="categorical", cardinality=200, text_detected=True),
        ]
        result = enricher._heuristic_enrichment(profiles)
        # Categorical path sets text_processing_strategy to "none"
        assert result["notes"]["text_processing_strategy"] == "none"
        assert result["notes"]["imputation_strategy"] == "mode"


# ---------------------------------------------------------------------------
# Tests: batch processing
# ---------------------------------------------------------------------------

class TestBatchProcessing:
    def test_more_than_batch_size_splits_into_batches(self):
        """More than BATCH_SIZE columns should trigger multiple LLM calls."""
        n_columns = BATCH_SIZE + 5
        profiles = [
            ColumnProfile(name=f"col_{i}", dtype="numeric", cardinality=50)
            for i in range(n_columns)
        ]
        json_items = []
        for p in profiles:
            json_items.append({
                "column": p.name,
                "semantic_meaning": f"{p.name} meaning",
                "semantic_dtype": "count",
                "imputation_strategy": "median",
                "encoding_strategy": "keep_numeric",
                "outlier_strategy": "none",
                "text_processing_strategy": "none",
                "interaction_candidates": [],
                "data_quality_flags": [],
                "importance_prior": "medium",
            })

        # FakeLLMClient returns all items each call, but _parse_enrichments
        # only picks items matching the batch column names
        client = FakeLLMClient(response_content=json.dumps(json_items))
        enricher = ColumnAIEnricher(llm_client=client, model_router=FakeModelRouter())

        result = enricher.enrich_profiles(
            profiles=profiles,
            target_column="target",
            problem_type="classification",
            sample_rows=[],
        )

        # Should have made 2 LLM calls (BATCH_SIZE + remainder)
        assert len(client.calls) == 2
        # All profiles should be enriched
        for p in result:
            assert p.imputation_strategy == "median"

    def test_exact_batch_size_single_call(self):
        """Exactly BATCH_SIZE columns should use a single LLM call."""
        profiles = [
            ColumnProfile(name=f"col_{i}", dtype="numeric", cardinality=50)
            for i in range(BATCH_SIZE)
        ]
        json_response = _make_valid_enrichment_json(profiles)
        client = FakeLLMClient(response_content=json_response)
        enricher = ColumnAIEnricher(llm_client=client, model_router=FakeModelRouter())

        enricher.enrich_profiles(
            profiles=profiles,
            target_column="target",
            problem_type="classification",
            sample_rows=[],
        )
        assert len(client.calls) == 1
