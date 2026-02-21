"""CLI tests for OpenRouter model catalog workflows."""

from __future__ import annotations

import json

from click.testing import CliRunner

from src.cli import _update_role_mapping_in_lines, cli


def test_cli_models_json_output(monkeypatch):
    class FakeCatalog:
        def __init__(self, api_key=None):
            self.api_key = api_key

        def list_models(self, **kwargs):
            assert kwargs["criteria"] == "cost"
            assert kwargs["limit"] == 2
            return [
                {
                    "id": "google/gemini-3-flash-preview",
                    "provider": "google",
                    "prompt_cost": 0.0000002,
                    "completion_cost": 0.0000008,
                    "created_iso": "2025-02-19T00:00:00+00:00",
                    "latency_ms": 420.0,
                },
                {
                    "id": "openai/gpt-4o",
                    "provider": "openai",
                    "prompt_cost": 0.0000025,
                    "completion_cost": 0.00001,
                    "created_iso": "2024-10-27T00:00:00+00:00",
                    "latency_ms": 950.0,
                },
            ]

    monkeypatch.setattr("src.cli._load_openrouter_model_catalog", lambda: FakeCatalog)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "sotappr-models",
            "--criteria",
            "cost",
            "--limit",
            "2",
            "--json-out",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["criteria"] == "cost"
    assert payload["count"] == 2
    assert payload["models"][0]["id"] == "google/gemini-3-flash-preview"


def test_cli_models_choose_updates_models_yaml(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    models_path = config_dir / "models.yaml"
    models_path.write_text(
        "roles:\n"
        "  builder: openai/gpt-4o\n"
        "  sentinel: anthropic/claude-sonnet-4\n"
        "\n"
        "fallbacks:\n"
        "  builder:\n"
        "    - google/gemini-3-flash-preview\n",
        encoding="utf-8",
    )

    class FakeCatalog:
        def __init__(self, api_key=None):
            self.api_key = api_key

        def list_models(self, **kwargs):
            _ = kwargs
            return [
                {
                    "id": "openai/gpt-4o",
                    "provider": "openai",
                    "prompt_cost": 0.0000025,
                    "completion_cost": 0.00001,
                    "created_iso": "2024-10-27T00:00:00+00:00",
                    "latency_ms": 950.0,
                },
                {
                    "id": "google/gemini-3-flash-preview",
                    "provider": "google",
                    "prompt_cost": 0.0000002,
                    "completion_cost": 0.0000008,
                    "created_iso": "2025-02-19T00:00:00+00:00",
                    "latency_ms": 420.0,
                },
            ]

    monkeypatch.setattr("src.cli._load_openrouter_model_catalog", lambda: FakeCatalog)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "sotappr-models",
            "--choose",
            "--role",
            "builder",
            "--config-dir",
            str(config_dir),
        ],
        input="2\n",
    )

    assert result.exit_code == 0, result.output
    updated = models_path.read_text(encoding="utf-8")
    assert "builder: google/gemini-3-flash-preview" in updated
    assert "Updated role 'builder'" in result.output


def test_update_role_mapping_preserves_non_role_sections():
    lines = [
        "# comment",
        "roles:",
        "  builder: openai/gpt-4o",
        "  sentinel: anthropic/claude-sonnet-4",
        "",
        "fallbacks:",
        "  builder:",
        "    - google/gemini-3-flash-preview",
    ]

    updated = _update_role_mapping_in_lines(
        lines=lines,
        role="builder",
        model_id="google/gemini-3-flash-preview",
    )

    assert any(line.strip() == "builder: google/gemini-3-flash-preview" for line in updated)
    assert any(line.strip() == "fallbacks:" for line in updated)
