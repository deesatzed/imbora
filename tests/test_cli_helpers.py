"""Tests for CLI helper functions — pure function tests, no CLI runner needed."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import click
import pytest
import yaml

from src.cli import (
    _archive_report,
    _format_cost,
    _format_latency,
    _filter_runs_by_status,
    _is_budget_pause,
    _iso,
    _load_component_factory,
    _load_openrouter_model_catalog,
    _load_sotappr_executor,
    _load_spec,
    _parse_status_filters,
    _parse_csv_items,
    _parse_uuid,
    _persist_model_role,
    _runtime_budget_contract,
    _setup_logging,
    _sigint_handler,
    _update_role_mapping_in_lines,
)


class TestSigintHandler:
    def test_sigint_handler_exits_130(self, monkeypatch):
        import src.cli as cli_module

        monkeypatch.setattr(cli_module, "_active_run_id", "abc-123")
        monkeypatch.setattr(cli_module, "_active_project_id", "proj-456")
        monkeypatch.setattr(cli_module, "_tasks_processed", 7)

        with pytest.raises(SystemExit) as exc_info:
            _sigint_handler(2, None)
        assert exc_info.value.code == 130


class TestSetupLogging:
    def test_setup_logging_fallback(self, monkeypatch):
        """When load_config raises, _setup_logging should not propagate the error."""
        # load_config is imported locally inside _setup_logging from src.core.config
        monkeypatch.setattr(
            "src.core.config.load_config",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no config")),
        )
        # Should not raise — gracefully falls back
        _setup_logging(verbose=False)


class TestParseUuid:
    def test_parse_uuid_none_raises(self):
        with pytest.raises(click.ClickException, match="Missing required UUID"):
            _parse_uuid(None, "test")

    def test_parse_uuid_invalid_raises(self):
        with pytest.raises(click.ClickException, match="Invalid UUID"):
            _parse_uuid("not-a-uuid", "test")

    def test_parse_uuid_valid(self):
        raw = str(uuid4())
        result = _parse_uuid(raw, "test")
        assert str(result) == raw


class TestIso:
    def test_iso_none_returns_none(self):
        assert _iso(None) is None

    def test_iso_datetime_returns_isoformat(self):
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        result = _iso(dt)
        assert "2024-01-15" in result
        assert isinstance(result, str)

    def test_iso_string_returns_str(self):
        assert _iso("raw") == "raw"


class TestArchiveReport:
    def test_archive_report_noop_when_run_id_none(self, tmp_path):
        report = tmp_path / "report.json"
        report.write_text("{}", encoding="utf-8")
        archive_dir = tmp_path / "archives"
        # Should not create archive dir
        _archive_report(report_out=report, archive_dir=archive_dir, run_id=None)
        assert not archive_dir.exists()


class TestFormatCost:
    def test_format_cost_none(self):
        assert _format_cost(None) == "n/a"

    def test_format_cost_invalid(self):
        assert _format_cost("bad") == "n/a"

    def test_format_cost_valid(self):
        result = _format_cost(0.001)
        assert result == "0.00100000"


class TestFormatLatency:
    def test_format_latency_none(self):
        assert _format_latency(None) == "n/a"

    def test_format_latency_invalid(self):
        assert _format_latency("bad") == "n/a"

    def test_format_latency_valid(self):
        assert _format_latency(42.5) == "42.5"


class TestRuntimeBudgetContract:
    def test_returns_none_without_caps(self):
        assert _runtime_budget_contract(
            max_hours=None,
            max_cost_usd=None,
            estimated_cost_per_1k_tokens_usd=0.02,
        ) is None

    def test_builds_contract_with_values(self):
        result = _runtime_budget_contract(
            max_hours=3.0,
            max_cost_usd=1.25,
            estimated_cost_per_1k_tokens_usd=0.02,
        )
        assert result is not None
        assert result["max_hours"] == 3.0
        assert result["max_cost_usd"] == 1.25
        assert result["estimated_cost_per_1k_tokens_usd"] == 0.02


class TestParseCsvItems:
    def test_parse_csv_items_filters_empty_entries(self):
        assert _parse_csv_items("one, , two ,,three") == ["one", "two", "three"]


class TestStatusFilters:
    def test_parse_status_filters(self):
        statuses = _parse_status_filters("completed, paused ,FAILED")
        assert statuses == {"completed", "paused", "failed"}

    def test_filter_runs_by_status(self):
        runs = [
            {"id": "1", "status": "completed"},
            {"id": "2", "status": "paused"},
            {"id": "3", "status": "failed"},
        ]
        filtered = _filter_runs_by_status(runs, {"completed", "paused"})
        assert len(filtered) == 2
        assert {r["id"] for r in filtered} == {"1", "2"}


class TestBudgetPause:
    def test_detects_budget_pause_from_reason(self):
        assert _is_budget_pause("budget_cost_exceeded", None) is True

    def test_detects_budget_pause_from_last_error(self):
        assert _is_budget_pause(None, "Run paused: budget_time_exceeded") is True

    def test_returns_false_when_not_budget(self):
        assert _is_budget_pause("completed", "none") is False


class TestLoadFunctions:
    def test_load_component_factory_returns_class(self):
        cls = _load_component_factory()
        assert cls.__name__ == "ComponentFactory"

    def test_load_sotappr_executor_returns_class(self):
        cls = _load_sotappr_executor()
        assert cls.__name__ == "SOTAppRExecutor"

    def test_load_openrouter_model_catalog_returns_class(self):
        cls = _load_openrouter_model_catalog()
        assert cls.__name__ == "OpenRouterModelCatalog"


class TestLoadSpec:
    def test_load_spec_yaml(self, tmp_path):
        spec = tmp_path / "spec.yaml"
        spec.write_text(
            yaml.dump({"organism_name": "test", "root_need": "need"}),
            encoding="utf-8",
        )
        result = _load_spec(spec)
        assert result["organism_name"] == "test"

    def test_load_spec_json(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        result = _load_spec(spec)
        assert result["key"] == "value"

    def test_load_spec_invalid_data(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(click.ClickException, match="Spec must be a JSON/YAML object"):
            _load_spec(spec)


class TestPersistModelRole:
    def test_persist_model_role_creates_new_file(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        result = _persist_model_role(
            role="builder", model_id="openai/gpt-4o", config_dir=config_dir
        )
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "builder: openai/gpt-4o" in content


class TestUpdateRoleMappingInLines:
    def test_empty_lines(self):
        result = _update_role_mapping_in_lines([], "builder", "m")
        assert "roles:" in result
        assert "  builder: m" in result

    def test_no_roles_section(self):
        lines = ["other_key: value", "another: x"]
        result = _update_role_mapping_in_lines(lines, "builder", "m")
        assert "roles:" in result
        assert "  builder: m" in result

    def test_new_role_inserted(self):
        lines = ["roles:", "  sentinel: old-model"]
        result = _update_role_mapping_in_lines(lines, "builder", "new-model")
        text = "\n".join(result)
        assert "builder: new-model" in text
        assert "sentinel: old-model" in text

    def test_existing_role_updated(self):
        lines = ["roles:", "  builder: old-model", "  sentinel: keep"]
        result = _update_role_mapping_in_lines(lines, "builder", "new-model")
        text = "\n".join(result)
        assert "builder: new-model" in text
        assert "builder: old-model" not in text
        assert "sentinel: keep" in text

    def test_comment_lines_skipped(self):
        lines = ["roles:", "  # This is a comment", "  builder: old"]
        result = _update_role_mapping_in_lines(lines, "builder", "new")
        text = "\n".join(result)
        assert "# This is a comment" in text
        assert "builder: new" in text
