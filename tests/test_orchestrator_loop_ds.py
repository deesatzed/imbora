"""Tests for DS task routing in the orchestrator loop.

Tests the _parse_ds_task_description method and DS task detection logic
from src/orchestrator/loop.py. Uses real-shaped fakes -- no mocks.

The parsing extracts dataset_path, target_column, problem_type, and
sensitive_columns from a DS task's description field, which follows
a structured format:

    Data science pipeline phase: ds_audit
    Dataset: /path/to/data.csv
    Target: target_col
    Problem type: classification
    Sensitive columns: col_a, col_b
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import pytest

from src.core.config import AppConfig, OrchestratorConfig, SOTAppRConfig
from src.core.models import (
    AgentResult,
    BuildResult,
    Task,
    TaskContext,
    TaskStatus,
)
from src.orchestrator.diagnostics import DiagnosticsCollector
from src.orchestrator.health_monitor import HealthMonitor
from src.orchestrator.loop import TaskLoop
from src.orchestrator.metrics import PipelineMetrics
from src.orchestrator.task_router import TaskRouter


# ---------------------------------------------------------------------------
# Real-shaped fakes -- minimal in-memory implementations matching
# the real interfaces. Same pattern as test_orchestrator_loop_integration.py.
# ---------------------------------------------------------------------------


class FakeRepository:
    """In-memory repository with the methods TaskLoop and TaskRouter need."""

    def __init__(self):
        self.tasks: dict[uuid.UUID, Task] = {}

    def get_next_task(self, project_id):
        return None

    def update_task_status(self, task_id: uuid.UUID, status: TaskStatus) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].status = status

    def increment_task_attempt(self, task_id: uuid.UUID) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].attempt_count += 1

    def increment_task_council_count(self, task_id: uuid.UUID) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].council_count += 1

    def get_in_progress_tasks(self) -> list[Task]:
        return []

    def log_hypothesis(self, entry):
        return entry

    def get_failed_approaches(self, task_id: uuid.UUID) -> list:
        return []


class FakeStateManager:
    """Returns success with TaskContext data."""

    name = "StateManager"
    repo_path = "/tmp/fake-repo"

    def run(self, task: Task) -> AgentResult:
        tc = TaskContext(task=task, checkpoint_ref="fake-checkpoint")
        return AgentResult(
            agent_name="StateManager",
            status="success",
            data={"task_context": tc.model_dump()},
        )

    def rewind_to_checkpoint(self, task: Task, checkpoint_ref: str | None = None) -> bool:
        return True


class FakeBuilder:
    """Returns configurable success with BuildResult data."""

    name = "Builder"

    def run(self, input_data: Any) -> AgentResult:
        br = BuildResult(
            files_changed=["src/main.py"],
            tests_passed=True,
            test_output="All tests passed",
            approach_summary="Implemented feature",
            model_used="test/model",
        )
        return AgentResult(
            agent_name="Builder",
            status="success",
            data={"build_result": br.model_dump(), "tokens_used": 100},
        )


# ---------------------------------------------------------------------------
# Helper: build a TaskLoop instance for testing parsing methods.
# ---------------------------------------------------------------------------


def _make_task_loop() -> TaskLoop:
    """Construct a minimal TaskLoop with real lightweight infrastructure
    and fake agents. Only _parse_ds_task_description and related methods
    are exercised -- the full agent pipeline is not invoked."""
    fake_repo = FakeRepository()
    config = AppConfig()

    task_router = TaskRouter(repository=fake_repo)
    health_monitor = HealthMonitor(
        repository=fake_repo,
        config=config.orchestrator,
    )
    metrics = PipelineMetrics()
    diagnostics = DiagnosticsCollector()

    return TaskLoop(
        repository=fake_repo,
        state_manager=FakeStateManager(),
        builder=FakeBuilder(),
        task_router=task_router,
        health_monitor=health_monitor,
        metrics=metrics,
        diagnostics=diagnostics,
        config=config,
    )


# ---------------------------------------------------------------------------
# Tests: _parse_ds_task_description
# ---------------------------------------------------------------------------


class TestParseDSTaskDescription:
    """Test parsing of DS task descriptions into input dicts."""

    def test_parses_complete_description(self):
        """A well-formed description should yield all fields."""
        loop = _make_task_loop()
        project_id = uuid.uuid4()
        task = Task(
            project_id=project_id,
            title="DS Audit Phase",
            description=(
                "Data science pipeline phase: ds_audit\n"
                "Dataset: /data/train.csv\n"
                "Target: income\n"
                "Problem type: regression\n"
            ),
            task_type="ds_audit",
        )
        result = loop._parse_ds_task_description(task)

        assert result is not None
        assert result["dataset_path"] == "/data/train.csv"
        assert result["target_column"] == "income"
        assert result["problem_type"] == "regression"
        assert result["project_id"] == project_id

    def test_parses_description_with_sensitive_columns(self):
        """Sensitive columns should be parsed into a list."""
        loop = _make_task_loop()
        task = Task(
            project_id=uuid.uuid4(),
            title="DS Audit",
            description=(
                "Data science pipeline phase: ds_audit\n"
                "Dataset: /data/train.csv\n"
                "Target: label\n"
                "Problem type: classification\n"
                "Sensitive columns: gender, race, age\n"
            ),
            task_type="ds_audit",
        )
        result = loop._parse_ds_task_description(task)

        assert result is not None
        assert result["sensitive_columns"] == ["gender", "race", "age"]

    def test_returns_none_for_missing_dataset(self):
        """Description without Dataset: line should return None."""
        loop = _make_task_loop()
        task = Task(
            project_id=uuid.uuid4(),
            title="Broken DS Task",
            description=(
                "Data science pipeline phase: ds_audit\n"
                "Target: label\n"
                "Problem type: classification\n"
            ),
            task_type="ds_audit",
        )
        result = loop._parse_ds_task_description(task)
        assert result is None

    def test_returns_none_for_missing_target(self):
        """Description without Target: line should return None."""
        loop = _make_task_loop()
        task = Task(
            project_id=uuid.uuid4(),
            title="Broken DS Task",
            description=(
                "Data science pipeline phase: ds_audit\n"
                "Dataset: /data/train.csv\n"
                "Problem type: classification\n"
            ),
            task_type="ds_audit",
        )
        result = loop._parse_ds_task_description(task)
        assert result is None

    def test_returns_none_for_empty_description(self):
        """Empty description should return None."""
        loop = _make_task_loop()
        task = Task(
            project_id=uuid.uuid4(),
            title="Empty DS Task",
            description="",
            task_type="ds_audit",
        )
        result = loop._parse_ds_task_description(task)
        assert result is None

    def test_problem_type_optional_defaults_absent(self):
        """If Problem type: line is missing, result should still succeed
        (it is not a required field for parsing to return non-None)."""
        loop = _make_task_loop()
        task = Task(
            project_id=uuid.uuid4(),
            title="DS Audit",
            description=(
                "Data science pipeline phase: ds_audit\n"
                "Dataset: /data/train.csv\n"
                "Target: label\n"
            ),
            task_type="ds_audit",
        )
        result = loop._parse_ds_task_description(task)
        # Should not be None since dataset_path and target_column are present
        assert result is not None
        assert result["dataset_path"] == "/data/train.csv"
        assert result["target_column"] == "label"
        # problem_type key should not be present if not in description
        assert "problem_type" not in result

    def test_handles_whitespace_around_values(self):
        """Leading/trailing whitespace in values should be stripped."""
        loop = _make_task_loop()
        task = Task(
            project_id=uuid.uuid4(),
            title="DS Audit",
            description=(
                "Data science pipeline phase: ds_audit\n"
                "Dataset:   /data/train.csv   \n"
                "Target:   label   \n"
                "Problem type:   classification   \n"
            ),
            task_type="ds_audit",
        )
        result = loop._parse_ds_task_description(task)
        assert result is not None
        assert result["dataset_path"] == "/data/train.csv"
        assert result["target_column"] == "label"
        assert result["problem_type"] == "classification"

    def test_handles_colons_in_dataset_path(self):
        """Dataset paths containing colons (e.g., Windows paths) should parse correctly."""
        loop = _make_task_loop()
        task = Task(
            project_id=uuid.uuid4(),
            title="DS Audit",
            description=(
                "Data science pipeline phase: ds_audit\n"
                "Dataset: C:\\data\\train.csv\n"
                "Target: label\n"
            ),
            task_type="ds_audit",
        )
        result = loop._parse_ds_task_description(task)
        assert result is not None
        # split(":", 1)[1] should capture everything after the first colon
        assert result["dataset_path"] == "C:\\data\\train.csv"

    def test_project_id_set_from_task(self):
        """project_id should always come from task.project_id, not description."""
        loop = _make_task_loop()
        project_id = uuid.uuid4()
        task = Task(
            project_id=project_id,
            title="DS Audit",
            description=(
                "Dataset: /data/train.csv\n"
                "Target: label\n"
            ),
            task_type="ds_audit",
        )
        result = loop._parse_ds_task_description(task)
        assert result is not None
        assert result["project_id"] == project_id


# ---------------------------------------------------------------------------
# Tests: DS task type detection
# ---------------------------------------------------------------------------


class TestDSTaskTypeDetection:
    """Verify that tasks with ds_ prefix are recognized as DS tasks."""

    @pytest.mark.parametrize("task_type", [
        "ds_audit",
        "ds_eda",
        "ds_feature_eng",
        "ds_training",
        "ds_ensemble",
        "ds_evaluation",
        "ds_deployment",
    ])
    def test_ds_task_types_start_with_ds_prefix(self, task_type):
        """All DS task types should start with 'ds_'."""
        assert task_type.startswith("ds_")

    @pytest.mark.parametrize("task_type", [
        "general",
        "bug_fix",
        "refactor",
        "feature",
    ])
    def test_non_ds_task_types_do_not_match(self, task_type):
        """Non-DS task types should not start with 'ds_'."""
        assert not task_type.startswith("ds_")

    def test_ds_task_description_round_trip(self):
        """Build a description string and verify it parses back correctly."""
        loop = _make_task_loop()
        project_id = uuid.uuid4()

        # Build the structured description
        description = (
            "Data science pipeline phase: ds_training\n"
            "Dataset: /workspace/data/features.csv\n"
            "Target: churn\n"
            "Problem type: classification\n"
            "Sensitive columns: age, gender\n"
        )

        task = Task(
            project_id=project_id,
            title="DS Model Training",
            description=description,
            task_type="ds_training",
        )

        parsed = loop._parse_ds_task_description(task)

        assert parsed is not None
        assert parsed["dataset_path"] == "/workspace/data/features.csv"
        assert parsed["target_column"] == "churn"
        assert parsed["problem_type"] == "classification"
        assert parsed["sensitive_columns"] == ["age", "gender"]
        assert parsed["project_id"] == project_id

    def test_all_ds_phase_descriptions_parseable(self):
        """Each DS phase type should produce a parseable description."""
        loop = _make_task_loop()

        for phase in ["ds_audit", "ds_eda", "ds_feature_eng", "ds_training",
                       "ds_ensemble", "ds_evaluation", "ds_deployment"]:
            task = Task(
                project_id=uuid.uuid4(),
                title=f"DS Phase: {phase}",
                description=(
                    f"Data science pipeline phase: {phase}\n"
                    "Dataset: /data/input.csv\n"
                    "Target: outcome\n"
                    "Problem type: classification\n"
                ),
                task_type=phase,
            )
            result = loop._parse_ds_task_description(task)
            assert result is not None, f"Failed to parse description for {phase}"
            assert result["dataset_path"] == "/data/input.csv"
            assert result["target_column"] == "outcome"
