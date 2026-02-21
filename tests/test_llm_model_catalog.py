"""Tests for live-model catalog ranking utilities."""

from __future__ import annotations

import os

import pytest

from src.llm.model_catalog import (
    OpenRouterModelCatalog,
    _as_float,
    _as_int,
    _to_datetime,
)


class FakeCatalog(OpenRouterModelCatalog):
    def __init__(self):
        super().__init__(api_key="x")
        self.latency_calls: list[str] = []

    def _fetch_models(self):
        return [
            {
                "id": "openai/gpt-4o",
                "name": "GPT-4o",
                "description": "general",
                "created": 1730000000,
                "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
                "context_length": 128000,
                "top_provider": {"max_completion_tokens": 16384},
            },
            {
                "id": "google/gemini-3-flash-preview",
                "name": "Gemini Flash",
                "description": "fast",
                "created": 1740000000,
                "pricing": {"prompt": "0.0000002", "completion": "0.0000008"},
                "context_length": 1000000,
                "top_provider": {"max_completion_tokens": 65536},
            },
            {
                "id": "anthropic/claude-sonnet-4",
                "name": "Sonnet",
                "description": "quality",
                "created": 1720000000,
                "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                "context_length": 200000,
                "top_provider": {"max_completion_tokens": 32768},
            },
        ]

    def _fetch_latency_metrics(self, model_id: str):
        self.latency_calls.append(model_id)
        data = {
            "openai/gpt-4o": {"latency_ms": 950.0, "throughput": 35.0, "uptime": 99.8},
            "google/gemini-3-flash-preview": {
                "latency_ms": 420.0,
                "throughput": 80.0,
                "uptime": 99.9,
            },
            "anthropic/claude-sonnet-4": {
                "latency_ms": 1250.0,
                "throughput": 20.0,
                "uptime": 99.7,
            },
        }
        return data.get(model_id, {"latency_ms": None, "throughput": None, "uptime": None})


def test_cost_ranking_prefers_cheaper_models():
    catalog = FakeCatalog()
    models = catalog.list_models(criteria="cost", limit=3)

    assert len(models) == 3
    assert models[0]["id"] == "google/gemini-3-flash-preview"
    assert models[1]["id"] == "openai/gpt-4o"


def test_newness_ranking_prefers_recent_models():
    catalog = FakeCatalog()
    models = catalog.list_models(criteria="newness", limit=3)

    assert [m["id"] for m in models] == [
        "google/gemini-3-flash-preview",
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4",
    ]


def test_latency_ranking_uses_endpoint_metrics():
    catalog = FakeCatalog()
    models = catalog.list_models(criteria="latency", limit=3, latency_probe_limit=3)

    assert models[0]["id"] == "google/gemini-3-flash-preview"
    assert models[0]["latency_ms"] == 420.0
    assert set(catalog.latency_calls) == {
        "openai/gpt-4o",
        "google/gemini-3-flash-preview",
        "anthropic/claude-sonnet-4",
    }


def test_provider_and_search_filters_reduce_results():
    catalog = FakeCatalog()
    models = catalog.list_models(
        criteria="newness",
        limit=5,
        provider="openai",
        search="gpt",
    )

    assert len(models) == 1
    assert models[0]["id"] == "openai/gpt-4o"


def test_negative_prices_are_treated_as_unknown():
    class NegativePriceCatalog(FakeCatalog):
        def _fetch_models(self):
            rows = super()._fetch_models()
            rows.append(
                {
                    "id": "openrouter/auto",
                    "name": "Auto",
                    "description": "router",
                    "created": 1750000000,
                    "pricing": {"prompt": "-1", "completion": "-1"},
                    "context_length": 100000,
                    "top_provider": {"max_completion_tokens": 8192},
                }
            )
            return rows

    catalog = NegativePriceCatalog()
    models = catalog.list_models(criteria="cost", limit=10)

    priced = [m for m in models if m["id"] == "openrouter/auto"]
    assert priced
    assert priced[0]["prompt_cost"] is None
    assert priced[0]["completion_cost"] is None


class TestListModelFilters:
    """Cost and free-model filter branches."""

    def test_max_prompt_cost_filters(self):
        catalog = FakeCatalog()
        # Gemini prompt cost is 0.0000002, GPT-4o is 0.0000025, Sonnet is 0.000003
        models = catalog.list_models(criteria="newness", limit=10, max_prompt_cost=0.000001)
        ids = [m["id"] for m in models]
        assert "google/gemini-3-flash-preview" in ids
        assert "openai/gpt-4o" not in ids
        assert "anthropic/claude-sonnet-4" not in ids

    def test_max_completion_cost_filters(self):
        catalog = FakeCatalog()
        # Gemini completion cost is 0.0000008, GPT-4o is 0.00001, Sonnet is 0.000015
        models = catalog.list_models(criteria="newness", limit=10, max_completion_cost=0.000005)
        ids = [m["id"] for m in models]
        assert "google/gemini-3-flash-preview" in ids
        assert "openai/gpt-4o" not in ids

    def test_include_free_false_removes_zero_cost(self):
        class FreeCatalog(FakeCatalog):
            def _fetch_models(self):
                rows = super()._fetch_models()
                rows.append(
                    {
                        "id": "free/model-zero",
                        "name": "Free Model",
                        "description": "no cost",
                        "created": 1750000000,
                        "pricing": {"prompt": "0", "completion": "0"},
                        "context_length": 8192,
                        "top_provider": {"max_completion_tokens": 1024},
                    }
                )
                return rows

        catalog = FreeCatalog()
        models_all = catalog.list_models(criteria="newness", limit=10, include_free=True)
        models_paid = catalog.list_models(criteria="newness", limit=10, include_free=False)

        all_ids = [m["id"] for m in models_all]
        paid_ids = [m["id"] for m in models_paid]

        assert "free/model-zero" in all_ids
        assert "free/model-zero" not in paid_ids


class TestLatencyProbeSkip:
    """Test that _attach_latency_metrics skips models outside the probe limit."""

    def test_latency_probe_limit_skips_old_models(self):
        catalog = FakeCatalog()
        # probe_limit=1 means only the newest model is probed
        models = catalog.list_models(criteria="latency", limit=3, latency_probe_limit=1)

        # All 3 models returned but only 1 should have been probed
        assert len(models) == 3
        assert len(catalog.latency_calls) == 1
        # The newest model (Gemini, created=1740000000) should be probed
        assert "google/gemini-3-flash-preview" in catalog.latency_calls

        # Non-probed models should have None latency
        gpt4o = next(m for m in models if m["id"] == "openai/gpt-4o")
        assert gpt4o["latency_ms"] is None
        assert gpt4o["latency_source"] == "heuristic"


class TestHelperFunctions:
    """Edge cases for _as_float, _as_int, _to_datetime."""

    def test_as_float_success(self):
        assert _as_float("3.14") == pytest.approx(3.14)

    def test_as_float_none(self):
        assert _as_float(None) is None

    def test_as_float_invalid(self):
        assert _as_float("abc") is None

    def test_as_float_negative(self):
        assert _as_float("-1.0") is None

    def test_as_int_none(self):
        assert _as_int(None) is None

    def test_as_int_invalid(self):
        assert _as_int("abc") is None

    def test_to_datetime_zero(self):
        assert _to_datetime(0) is None

    def test_to_datetime_negative(self):
        assert _to_datetime(-1) is None

    def test_to_datetime_milliseconds(self):
        # 1700000000000 ms → should be converted to 1700000000 seconds
        result = _to_datetime(1700000000000)
        assert result is not None
        assert result.year == 2023

    def test_to_datetime_overflow(self):
        assert _to_datetime(99999999999999999) is None


requires_openrouter = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)


@requires_openrouter
class TestFetchModelsIntegration:
    """Live API tests — skipped unless OPENROUTER_API_KEY is present."""

    def test_fetch_models_returns_list(self):
        catalog = OpenRouterModelCatalog()
        raw = catalog._fetch_models()
        assert isinstance(raw, list)
        assert len(raw) > 0
        assert "id" in raw[0]

    def test_fetch_latency_metrics_returns_dict(self):
        catalog = OpenRouterModelCatalog()
        metrics = catalog._fetch_latency_metrics("openai/gpt-4o")
        assert isinstance(metrics, dict)
        assert "latency_ms" in metrics
        assert "throughput" in metrics
        assert "uptime" in metrics
