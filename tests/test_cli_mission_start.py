"""CLI tests for mission-start guided workflow."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from src.cli import cli


def test_mission_start_plan_only_writes_session_artifacts(tmp_path, config_dir):
    target_path = tmp_path / "new_app"
    knowledge_file = tmp_path / "notes.md"
    knowledge_file.write_text("notes", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "mission-start",
            "--task",
            "Build a notes app",
            "--target-path",
            str(target_path),
            "--feature",
            "Create notes",
            "--feature",
            "Search notes",
            "--knowledge-local",
            str(knowledge_file),
            "--knowledge-url",
            "https://example.com/api",
            "--knowledge-query",
            "notes app sqlite best practices",
            "--config-dir",
            str(config_dir),
            "--non-interactive",
        ],
    )

    assert result.exit_code == 0, result.output
    sessions_root = target_path / ".associate" / "sessions"
    session_dirs = [p for p in sessions_root.iterdir() if p.is_dir()]
    assert session_dirs, "Expected at least one session directory"

    session_file = session_dirs[0] / "session_plan.json"
    assert session_file.exists()
    payload = json.loads(session_file.read_text(encoding="utf-8"))
    assert payload["status"] == "approved"
    assert payload["alignment"]["approved"] is True
    assert payload["budget"]["approved"] is True
    assert payload["charter"]["approved"] is True


def test_mission_start_allows_loop_back_to_budget_gate(tmp_path, config_dir):
    target_path = tmp_path / "loopback_app"
    runner = CliRunner()
    user_input = "\n".join(
        [
            "n",              # alignment approved?
            "methodology",    # edit section
            "stepwise method",  # new methodology
            "y",              # approve alignment
            "4",              # max hours
            "1.5",            # max cost
            "y",              # approve budget
            "n",              # approve charter?
            "budget",         # loop back target
            "6",              # new max hours
            "2.0",            # new max cost
            "y",              # approve budget
            "y",              # approve charter
        ]
    ) + "\n"

    result = runner.invoke(
        cli,
        [
            "mission-start",
            "--task",
            "Build reporting dashboard",
            "--target-path",
            str(target_path),
            "--feature",
            "Daily reports",
            "--config-dir",
            str(config_dir),
        ],
        input=user_input,
    )

    assert result.exit_code == 0, result.output
    sessions_root = target_path / ".associate" / "sessions"
    session_dirs = [p for p in sessions_root.iterdir() if p.is_dir()]
    assert session_dirs
    payload = json.loads((session_dirs[0] / "session_plan.json").read_text(encoding="utf-8"))
    assert payload["alignment"]["methodology"] == "stepwise method"
    assert payload["budget"]["max_hours"] == 6.0
    assert payload["budget"]["max_cost_usd"] == 2.0
