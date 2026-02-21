"""Tests for src/agents/sentinel_rubrics.py — Task-type-specific Sentinel rubrics."""

import json
import tempfile
from pathlib import Path

from src.agents.sentinel_rubrics import (
    DEFAULT_RUBRIC,
    TASK_TYPE_KEYWORDS,
    classify_task_type,
    format_rubric_for_prompt,
    load_rubric,
)


# ---------------------------------------------------------------------------
# classify_task_type
# ---------------------------------------------------------------------------

def test_classify_db_migration():
    result = classify_task_type("Add migration", "Create a database schema migration for users table")
    assert result == "db_migration"


def test_classify_api_integration():
    result = classify_task_type("Add REST endpoint", "Create an API route for user profile with HTTP response")
    assert result == "api_integration"


def test_classify_ui_component():
    result = classify_task_type("Build component", "Create a frontend UI component to render user display")
    assert result == "ui_component"


def test_classify_testing():
    result = classify_task_type("Add tests", "Write pytest coverage with fixtures and assert checks")
    assert result == "testing"


def test_classify_refactoring():
    result = classify_task_type("Refactor auth", "Restructure and simplify the authentication module")
    assert result == "refactoring"


def test_classify_configuration():
    result = classify_task_type("Config update", "Update the yaml config settings for environment")
    assert result == "configuration"


def test_classify_security():
    result = classify_task_type("Auth hardening", "Add security token encryption and permission validation")
    assert result == "security"


def test_classify_returns_general_for_unknown():
    result = classify_task_type("Do something", "This is a completely unrelated task about cooking recipes")
    assert result == "general"


def test_classify_picks_highest_score():
    # "database migration sql" hits db_migration 3 times, stronger than others
    result = classify_task_type("Database migration", "Write SQL for the database schema migration")
    assert result == "db_migration"


def test_classify_is_case_insensitive():
    result = classify_task_type("DATABASE MIGRATION", "SQL SCHEMA")
    assert result == "db_migration"


def test_classify_empty_strings():
    result = classify_task_type("", "")
    assert result == "general"


# ---------------------------------------------------------------------------
# DEFAULT_RUBRIC
# ---------------------------------------------------------------------------

def test_default_rubric_has_task_type():
    assert DEFAULT_RUBRIC["task_type"] == "general"


def test_default_rubric_has_four_dimensions():
    dims = DEFAULT_RUBRIC["dimensions"]
    assert len(dims) == 4


def test_default_rubric_weights_sum_to_one():
    total_weight = sum(d["weight"] for d in DEFAULT_RUBRIC["dimensions"])
    assert abs(total_weight - 1.0) < 1e-9


def test_default_rubric_dimension_names():
    names = {d["name"] for d in DEFAULT_RUBRIC["dimensions"]}
    assert names == {"completeness", "correctness", "quality", "test_coverage"}


def test_default_rubric_dimensions_have_descriptions():
    for dim in DEFAULT_RUBRIC["dimensions"]:
        assert "description" in dim
        assert len(dim["description"]) > 0


# ---------------------------------------------------------------------------
# load_rubric
# ---------------------------------------------------------------------------

def test_load_rubric_returns_default_when_dir_is_none():
    result = load_rubric(None, "db_migration")
    assert result == DEFAULT_RUBRIC


def test_load_rubric_returns_default_when_file_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_rubric(tmpdir, "nonexistent_type")
        assert result == DEFAULT_RUBRIC


def test_load_rubric_loads_json_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        rubric_data = {
            "task_type": "db_migration",
            "dimensions": [
                {"name": "schema_correctness", "weight": 0.50, "description": "Schema is valid"},
                {"name": "rollback_safety", "weight": 0.50, "description": "Migration is reversible"},
            ],
        }
        rubric_path = Path(tmpdir) / "db_migration.json"
        rubric_path.write_text(json.dumps(rubric_data))

        result = load_rubric(tmpdir, "db_migration")
        assert result["task_type"] == "db_migration"
        assert len(result["dimensions"]) == 2
        assert result["dimensions"][0]["name"] == "schema_correctness"


def test_load_rubric_returns_default_on_invalid_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        rubric_path = Path(tmpdir) / "broken.json"
        rubric_path.write_text("this is not json {{{{")

        result = load_rubric(tmpdir, "broken")
        assert result == DEFAULT_RUBRIC


def test_load_rubric_uses_task_type_as_filename():
    with tempfile.TemporaryDirectory() as tmpdir:
        rubric_data = {"task_type": "testing", "dimensions": []}
        rubric_path = Path(tmpdir) / "testing.json"
        rubric_path.write_text(json.dumps(rubric_data))

        result = load_rubric(tmpdir, "testing")
        assert result["task_type"] == "testing"


# ---------------------------------------------------------------------------
# format_rubric_for_prompt
# ---------------------------------------------------------------------------

def test_format_rubric_for_prompt_contains_task_type():
    output = format_rubric_for_prompt(DEFAULT_RUBRIC)
    assert "general" in output


def test_format_rubric_for_prompt_contains_dimensions():
    output = format_rubric_for_prompt(DEFAULT_RUBRIC)
    assert "completeness" in output
    assert "correctness" in output
    assert "quality" in output
    assert "test_coverage" in output


def test_format_rubric_for_prompt_contains_weights():
    output = format_rubric_for_prompt(DEFAULT_RUBRIC)
    assert "35%" in output
    assert "30%" in output
    assert "20%" in output
    assert "15%" in output


def test_format_rubric_for_prompt_contains_descriptions():
    output = format_rubric_for_prompt(DEFAULT_RUBRIC)
    assert "All task requirements addressed" in output
    assert "Logic correct, no bugs" in output


def test_format_rubric_for_prompt_contains_scoring_instruction():
    output = format_rubric_for_prompt(DEFAULT_RUBRIC)
    assert "Score each dimension 0.0-1.0" in output


def test_format_rubric_for_prompt_custom_rubric():
    custom = {
        "task_type": "security_audit",
        "dimensions": [
            {"name": "threat_coverage", "weight": 1.0, "description": "All threats analyzed"},
        ],
    }
    output = format_rubric_for_prompt(custom)
    assert "security_audit" in output
    assert "threat_coverage" in output
    assert "100%" in output


def test_format_rubric_for_prompt_empty_dimensions():
    rubric = {"task_type": "empty", "dimensions": []}
    output = format_rubric_for_prompt(rubric)
    assert "empty" in output
    assert "Score each dimension" in output


def test_format_rubric_for_prompt_missing_fields_uses_defaults():
    rubric = {
        "dimensions": [
            {"weight": 0.5},  # no name, no description
        ],
    }
    output = format_rubric_for_prompt(rubric)
    assert "general" in output  # default task_type
    assert "unknown" in output  # default name


# ---------------------------------------------------------------------------
# TASK_TYPE_KEYWORDS completeness
# ---------------------------------------------------------------------------

def test_all_keyword_lists_are_non_empty():
    for task_type, keywords in TASK_TYPE_KEYWORDS.items():
        assert len(keywords) > 0, f"Task type '{task_type}' has no keywords"
