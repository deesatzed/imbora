"""Tests for SOTAppR execution bridge into TaskLoop."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from click.testing import CliRunner

from src.cli import cli
from src.core.models import Project, Task, TaskStatus
from src.sotappr import BuilderRequest, SOTAppRBuilder
from src.sotappr.executor import ExecutionSummary, SOTAppRExecutor
from src.sotappr.observability import SOTAppRObservability


class StubRepository:
    def __init__(self):
        self.projects: list[Project] = []
        self.tasks: list[Task] = []

    def create_project(self, project: Project) -> Project:
        self.projects.append(project)
        return project

    def create_task(self, task: Task) -> Task:
        self.tasks.append(task)
        return task


class StubRunRepository(StubRepository):
    """StubRepository extended with real-shaped in-memory SOTAppR run storage."""

    def __init__(self):
        super().__init__()
        self.runs: dict[UUID, dict] = {}
        self.artifacts: list[dict] = []

    def create_sotappr_run(
        self,
        *,
        project_id,
        mode: str,
        governance_pack: str,
        spec_json: dict,
        report_json: dict,
        repo_path: str,
        report_path: str | None,
        status: str,
    ) -> UUID:
        run_id = uuid.uuid4()
        self.runs[run_id] = {
            "id": run_id,
            "project_id": project_id,
            "mode": mode,
            "governance_pack": governance_pack,
            "spec_json": spec_json,
            "report_json": report_json,
            "repo_path": repo_path,
            "report_path": report_path,
            "status": status,
            "tasks_seeded": 0,
            "tasks_processed": 0,
            "last_error": None,
            "stop_reason": None,
            "estimated_cost_usd": 0.0,
            "elapsed_hours": 0.0,
            "completed": False,
        }
        return run_id

    def update_sotappr_run(
        self,
        *,
        run_id: UUID,
        status: str | None = None,
        tasks_seeded: int | None = None,
        tasks_processed: int | None = None,
        last_error: str | None = None,
        stop_reason: str | None = None,
        estimated_cost_usd: float | None = None,
        elapsed_hours: float | None = None,
        completed: bool = False,
    ) -> None:
        run = self.runs[run_id]
        if status is not None:
            run["status"] = status
        if tasks_seeded is not None:
            run["tasks_seeded"] = tasks_seeded
        if tasks_processed is not None:
            run["tasks_processed"] = tasks_processed
        if last_error is not None:
            run["last_error"] = last_error
        if stop_reason is not None:
            run["stop_reason"] = stop_reason
        if estimated_cost_usd is not None:
            run["estimated_cost_usd"] = estimated_cost_usd
        if elapsed_hours is not None:
            run["elapsed_hours"] = elapsed_hours
        if completed:
            run["completed"] = True

    def get_sotappr_run(self, run_id: UUID) -> dict | None:
        return self.runs.get(run_id)

    def save_sotappr_artifact(
        self, *, run_id: UUID, phase: int, artifact_type: str, payload: dict
    ) -> None:
        self.artifacts.append(
            {
                "run_id": run_id,
                "phase": phase,
                "artifact_type": artifact_type,
                "payload": payload,
            }
        )


class StubTaskLoop:
    def __init__(self, processed: int = 0):
        self.processed = processed
        self.calls: list[tuple[str, int]] = []
        self.run_context_updates: list[UUID | None] = []
        self.trace_context_updates: list[str | None] = []

    def set_run_context(self, run_id: UUID | None, trace_id: str | None = None) -> None:
        self.run_context_updates.append(run_id)
        self.trace_context_updates.append(trace_id)

    def run_loop(self, project_id, max_iterations: int = 100) -> int:
        self.calls.append((str(project_id), max_iterations))
        return self.processed

    def get_last_run_summary(self) -> dict:
        return {
            "processed": self.processed,
            "tokens": 0,
            "estimated_cost_usd": 0.0,
            "elapsed_hours": 0.0,
            "stop_reason": "no_pending_tasks",
        }


class FailingTaskLoop(StubTaskLoop):
    """Task loop that raises a configurable exception from run_loop."""

    def __init__(self, error: Exception):
        super().__init__(processed=0)
        self._error = error

    def run_loop(self, project_id, max_iterations: int = 100) -> int:
        self.calls.append((str(project_id), max_iterations))
        raise self._error


class BudgetPausedTaskLoop(StubTaskLoop):
    """Task loop that reports budget exceeded in summary metadata."""

    def run_loop(self, project_id, max_iterations: int = 100, budget_contract=None) -> int:
        _ = budget_contract
        self.calls.append((str(project_id), max_iterations))
        return self.processed

    def get_last_run_summary(self) -> dict:
        return {
            "processed": self.processed,
            "tokens": 2500,
            "estimated_cost_usd": 0.05,
            "elapsed_hours": 0.25,
            "stop_reason": "budget_cost_exceeded",
        }


def _report():
    builder = SOTAppRBuilder()
    req = BuilderRequest(
        organism_name="Autonomous SWE Reactor",
        stated_problem="Build software safely.",
        root_need="Reliable autonomous delivery.",
        user_confirmed_phase1=True,
    )
    return req, builder.build(req)


class TestSOTAppRExecutor:
    def test_bootstrap_project_seeds_phase3_tasks(self):
        request, report = _report()
        repo = StubRepository()
        loop = StubTaskLoop(processed=0)
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)

        project, tasks = executor.bootstrap_project(
            request=request,
            report=report,
            repo_path="/tmp/repo",
        )

        assert project.name == "Autonomous SWE Reactor"
        assert project.repo_path == "/tmp/repo"
        assert len(tasks) == len(report.phase3.backlog_actions)
        assert len(repo.tasks) == len(report.phase3.backlog_actions)
        assert all(task.project_id == project.id for task in tasks)

    def test_seeded_tasks_are_prioritized_descending(self):
        request, report = _report()
        repo = StubRepository()
        loop = StubTaskLoop(processed=0)
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)

        project, tasks = executor.bootstrap_project(
            request=request,
            report=report,
            repo_path="/tmp/repo",
        )
        _ = project
        priorities = [task.priority for task in tasks]
        assert priorities == sorted(priorities, reverse=True)

    def test_bootstrap_and_execute_runs_task_loop(self):
        request, report = _report()
        repo = StubRepository()
        loop = StubTaskLoop(processed=3)
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)

        summary = executor.bootstrap_and_execute(
            request=request,
            report=report,
            repo_path="/tmp/repo",
            max_iterations=17,
        )

        assert summary.tasks_seeded == len(report.phase3.backlog_actions)
        assert summary.tasks_processed == 3
        assert len(loop.calls) == 1
        assert loop.calls[0][1] == 17

    def test_bootstrap_and_execute_pauses_on_budget_overrun(self):
        request, report = _report()
        repo = StubRunRepository()
        loop = BudgetPausedTaskLoop(processed=2)
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)

        summary = executor.bootstrap_and_execute(
            request=request,
            report=report,
            repo_path="/tmp/repo",
            max_iterations=10,
            budget_contract={"max_cost_usd": 0.03},
        )

        assert summary.status == "paused"
        assert summary.stop_reason == "budget_cost_exceeded"
        assert summary.tasks_processed == 2
        run = repo.runs[UUID(summary.run_id)]
        assert run["status"] == "paused"
        assert "budget_cost_exceeded" in str(run["last_error"])
        assert loop.run_context_updates[0] == UUID(summary.run_id)
        assert loop.run_context_updates[-1] is None
        assert loop.trace_context_updates[0] == summary.run_id
        assert loop.trace_context_updates[-1] is None


class TestSOTAppRExecuteCli:
    def test_cli_execute_runs_with_stubs(self, tmp_path, monkeypatch):
        spec_path = tmp_path / "spec.json"
        report_out = tmp_path / "report.json"
        export_tasks = tmp_path / "tasks.json"
        spec_path.write_text(
            json.dumps(
                {
                    "organism_name": "Autonomous SWE Reactor",
                    "stated_problem": "Build software safely and fast.",
                    "root_need": "Autonomous delivery with hard guarantees.",
                    "user_confirmed_phase1": True,
                }
            ),
            encoding="utf-8",
        )

        @dataclass
        class FakeSotapprCfg:
            governance_pack: str = "balanced"
            report_archive_dir: str = "artifacts/sotappr"

        @dataclass
        class FakeConfig:
            sotappr: FakeSotapprCfg = field(default_factory=FakeSotapprCfg)

        @dataclass
        class FakeBundle:
            config: FakeConfig = field(default_factory=FakeConfig)

        class FakeExecutor:
            def __init__(self):
                self.calls = 0

            def preview_seeded_tasks(self, backlog_actions):
                _ = backlog_actions
                return [{"title": "Task", "description": "Desc", "priority": 1}]

            def bootstrap_and_execute(self, **kwargs):
                self.calls += 1
                _ = kwargs
                return ExecutionSummary(
                    project_id=str(uuid.uuid4()),
                    tasks_seeded=5,
                    tasks_processed=4,
                    run_id=str(uuid.uuid4()),
                    mode="execute",
                )

        fake_executor = FakeExecutor()

        class FakeFactory:
            @staticmethod
            def create(config_dir=None, env=None, api_key=None):
                _ = config_dir, env, api_key
                return FakeBundle()

            @staticmethod
            def close(bundle):
                _ = bundle
                return None

        class FakeSOTAppRExecutor:
            @staticmethod
            def from_bundle(bundle, repo_path, test_command, **kwargs):
                _ = bundle, repo_path, test_command, kwargs
                return fake_executor

        monkeypatch.setattr("src.cli._load_component_factory", lambda: FakeFactory)
        monkeypatch.setattr("src.cli._load_sotappr_executor", lambda: FakeSOTAppRExecutor)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sotappr-execute",
                "--spec",
                str(spec_path),
                "--repo-path",
                str(tmp_path),
                "--report-out",
                str(report_out),
                "--export-tasks",
                str(export_tasks),
                "--max-iterations",
                "9",
            ],
            input="y\n",
        )

        assert result.exit_code == 0, result.output
        assert report_out.exists()
        assert export_tasks.exists()
        assert "Execution summary:" in result.output
        assert fake_executor.calls == 1

    def test_cli_execute_dry_run_uses_pause_mode(self, tmp_path, monkeypatch):
        spec_path = tmp_path / "spec.json"
        report_out = tmp_path / "report.json"
        spec_path.write_text(
            json.dumps(
                {
                    "organism_name": "Autonomous SWE Reactor",
                    "stated_problem": "Build software safely and fast.",
                    "root_need": "Autonomous delivery with hard guarantees.",
                    "user_confirmed_phase1": True,
                }
            ),
            encoding="utf-8",
        )

        @dataclass
        class FakeSotapprCfg:
            governance_pack: str = "strict"
            report_archive_dir: str = "artifacts/sotappr"

        @dataclass
        class FakeConfig:
            sotappr: FakeSotapprCfg = field(default_factory=FakeSotapprCfg)

        @dataclass
        class FakeBundle:
            config: FakeConfig = field(default_factory=FakeConfig)

        class FakeExecutor:
            def __init__(self):
                self.kwargs = None

            def preview_seeded_tasks(self, backlog_actions):
                _ = backlog_actions
                return []

            def bootstrap_and_execute(self, **kwargs):
                self.kwargs = kwargs
                return ExecutionSummary(
                    project_id="dryrun-project-id",
                    tasks_seeded=3,
                    tasks_processed=0,
                    run_id="dryrun-run-id",
                    mode="dry-run",
                )

        fake_executor = FakeExecutor()

        class FakeFactory:
            @staticmethod
            def create(config_dir=None, env=None, api_key=None):
                _ = config_dir, env, api_key
                return FakeBundle()

            @staticmethod
            def close(bundle):
                _ = bundle
                return None

        class FakeSOTAppRExecutor:
            @staticmethod
            def from_bundle(bundle, repo_path, test_command, **kwargs):
                _ = bundle, repo_path, test_command, kwargs
                return fake_executor

        monkeypatch.setattr("src.cli._load_component_factory", lambda: FakeFactory)
        monkeypatch.setattr("src.cli._load_sotappr_executor", lambda: FakeSOTAppRExecutor)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sotappr-execute",
                "--spec",
                str(spec_path),
                "--repo-path",
                str(tmp_path),
                "--report-out",
                str(report_out),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0, result.output
        assert report_out.exists()
        assert "Dry-run mode: TaskLoop execution was skipped." in result.output
        assert "Tasks seeded:" in result.output or "tasks_seeded" in result.output
        assert fake_executor.kwargs is not None
        assert fake_executor.kwargs["execute"] is False
        assert fake_executor.kwargs["mode"] == "dry-run"


class TestSOTAppROperationsCli:
    def test_cli_status_lists_runs(self, monkeypatch):
        class FakeRepo:
            def list_sotappr_runs(self, limit=20, project_id=None):
                _ = limit, project_id
                return [
                    {
                        "id": uuid.uuid4(),
                        "project_id": uuid.uuid4(),
                        "mode": "execute",
                        "status": "completed",
                        "governance_pack": "balanced",
                        "tasks_seeded": 3,
                        "tasks_processed": 3,
                        "created_at": None,
                        "updated_at": None,
                    }
                ]

        class FakeCtx:
            repository = FakeRepo()

            @staticmethod
            def db_engine():
                return None

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(cli, ["sotappr-status", "--limit", "5"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["count"] == 1
        assert payload["runs"][0]["status"] == "completed"

    def test_cli_status_marks_budget_paused_runs(self, monkeypatch):
        class FakeRepo:
            def list_sotappr_runs(self, limit=20, project_id=None):
                _ = limit, project_id
                return [
                    {
                        "id": uuid.uuid4(),
                        "project_id": uuid.uuid4(),
                        "mode": "execute",
                        "status": "paused",
                        "governance_pack": "balanced",
                        "tasks_seeded": 8,
                        "tasks_processed": 4,
                        "stop_reason": "budget_cost_exceeded",
                        "estimated_cost_usd": 0.24,
                        "elapsed_hours": 1.2,
                        "last_error": "Run paused: budget_cost_exceeded",
                        "created_at": None,
                        "updated_at": None,
                    }
                ]

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(cli, ["sotappr-status", "--limit", "5"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["runs"][0]["paused_by_budget"] is True
        assert payload["runs"][0]["stop_reason"] == "budget_cost_exceeded"
        assert payload["runs"][0]["estimated_cost_usd"] == 0.24

    def test_cli_packet_trace_uses_first_class_packet_tables(self, monkeypatch):
        run_id = uuid.uuid4()
        task_id = uuid.uuid4()
        packet_id = uuid.uuid4()

        class FakeRepo:
            @staticmethod
            def list_agent_packets(**kwargs):
                assert str(kwargs["run_id"]) == str(run_id)
                assert str(kwargs["task_id"]) == str(task_id)
                assert str(kwargs["packet_id"]) == str(packet_id)
                return [
                    {
                        "packet_id": packet_id,
                        "task_id": task_id,
                        "run_id": run_id,
                        "attempt_number": 1,
                        "packet_type": "TASK_IR",
                        "channel": "INTER_AGENT",
                        "run_phase": "DISCOVERY",
                        "sender_role": "state_manager",
                        "trace_id": "trace-1",
                        "lifecycle_state": "ARCHIVED",
                        "created_at": None,
                        "packet_json": {"sender": {"role": "state_manager"}},
                        "payload_json": {"ir_header": {"task_id": str(task_id)}},
                    }
                ]

            @staticmethod
            def list_packet_events(**kwargs):
                assert str(kwargs["packet_id"]) == str(packet_id)
                return [
                    {
                        "packet_id": packet_id,
                        "event": "emit",
                        "from_state": "DRAFT",
                        "to_state": "NORMALIZED",
                        "metadata": {},
                        "occurred_at": None,
                    }
                ]

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "packet-trace",
                "--run-id",
                str(run_id),
                "--task-id",
                str(task_id),
                "--packet-id",
                str(packet_id),
                "--limit",
                "10",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["source"] == "agent_packets"
        assert payload["count"] == 1
        assert payload["event_count"] == 1
        assert payload["packets"][0]["packet_type"] == "TASK_IR"

    def test_cli_packet_trace_falls_back_to_artifacts(self, monkeypatch):
        run_uuid = uuid.uuid4()
        packet_id = uuid.uuid4()

        class FakeRepo:
            @staticmethod
            def list_sotappr_artifacts(run_id=None, limit=200, **kwargs):
                _ = kwargs
                assert str(run_id) == str(run_uuid)
                assert limit >= 5
                return [
                    {
                        "id": uuid.uuid4(),
                        "run_id": run_id,
                        "artifact_type": "agent_packet",
                        "payload": {
                            "trace_id": "trace-fallback",
                            "task_id": str(uuid.uuid4()),
                            "attempt": 2,
                            "lifecycle_state": "ARCHIVED",
                            "packet": {
                                "packet_id": str(packet_id),
                                "packet_type": "SIGNAL",
                                "channel": "INTER_AGENT",
                                "run_phase": "DISCOVERY",
                                "sender": {"role": "orchestrator"},
                                "payload": {"signal": "ATTEMPT_STARTED"},
                            },
                        },
                        "created_at": None,
                    }
                ]

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["packet-trace", "--run-id", str(run_uuid), "--limit", "5", "--no-events"],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["source"] == "sotappr_artifacts"
        assert payload["count"] == 1
        assert payload["event_count"] == 0
        assert payload["packets"][0]["packet_type"] == "SIGNAL"

    def test_cli_packet_lineage_summary_uses_lineage_tables(self, monkeypatch):
        expected_run_id = uuid.uuid4()
        expected_project_id = uuid.uuid4()

        class FakeRepo:
            @staticmethod
            def get_packet_lineage_summary(project_id=None, run_id=None, limit=50):
                assert str(project_id) == str(expected_project_id)
                assert str(run_id) == str(expected_run_id)
                assert limit == 10
                return [
                    {
                        "protocol_id": "p-1",
                        "transfer_count": 3,
                        "run_count": 2,
                        "task_count": 2,
                        "cross_use_case_count": 1,
                        "last_seen_at": None,
                    }
                ]

            @staticmethod
            def list_protocol_propagation(protocol_id=None, project_id=None, run_id=None, limit=200):
                _ = protocol_id, project_id, run_id, limit
                return [
                    {
                        "protocol_id": "p-1",
                        "parent_protocol_ids": ["p-0"],
                        "ancestor_swarms": ["swarm-a"],
                        "cross_use_case": True,
                        "transfer_mode": "WINNER_TO_LOSERS",
                        "packet_id": uuid.uuid4(),
                        "packet_type": "TRANSFER",
                        "run_phase": "PROMOTION",
                        "run_id": expected_run_id,
                        "project_id": expected_project_id,
                        "task_id": uuid.uuid4(),
                        "trace_id": "trace-l-1",
                        "created_at": None,
                    }
                ]

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "packet-lineage-summary",
                "--project-id",
                str(expected_project_id),
                "--run-id",
                str(expected_run_id),
                "--limit",
                "10",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["source"] == "packet_lineage"
        assert payload["summary"]["protocol_count"] == 1
        assert payload["summary"]["cross_run_protocols"] == 1
        assert payload["protocols"][0]["protocol_id"] == "p-1"

    def test_cli_packet_lineage_summary_falls_back_to_artifacts(self, monkeypatch):
        expected_run_id = uuid.uuid4()

        class FakeRepo:
            @staticmethod
            def list_sotappr_artifacts(run_id=None, limit=200):
                assert str(run_id) == str(expected_run_id)
                _ = limit
                return [
                    {
                        "id": uuid.uuid4(),
                        "run_id": run_id,
                        "artifact_type": "agent_packet",
                        "payload": {
                            "trace_id": "trace-l-fallback",
                            "task_id": str(uuid.uuid4()),
                            "packet": {
                                "packet_id": str(uuid.uuid4()),
                                "packet_type": "TRANSFER",
                                "run_phase": "PROMOTION",
                                "lineage": {
                                    "protocol_id": "proto-fallback",
                                    "parent_protocol_ids": ["proto-parent"],
                                    "ancestor_swarms": ["swarm-z"],
                                    "cross_use_case": False,
                                    "transfer_mode": "COMMONS_SEED",
                                },
                            },
                        },
                        "created_at": None,
                    }
                ]

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "packet-lineage-summary",
                "--run-id",
                str(expected_run_id),
                "--limit",
                "5",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["source"] == "sotappr_artifacts"
        assert payload["summary"]["protocol_count"] == 1
        assert payload["protocols"][0]["protocol_id"] == "proto-fallback"

    def test_cli_packet_lineage_summary_includes_transfer_arbitration_summary(self, monkeypatch):
        expected_run_id = uuid.uuid4()

        class FakeRepo:
            @staticmethod
            def get_packet_lineage_summary(project_id=None, run_id=None, limit=50):
                _ = project_id, run_id, limit
                return []

            @staticmethod
            def list_protocol_propagation(protocol_id=None, project_id=None, run_id=None, limit=200):
                _ = protocol_id, project_id, run_id, limit
                return []

            @staticmethod
            def list_sotappr_artifacts(run_id=None, limit=200):
                assert str(run_id) == str(expected_run_id)
                _ = limit
                return [
                    {
                        "id": uuid.uuid4(),
                        "run_id": run_id,
                        "artifact_type": "transfer_arbitration_decision",
                        "payload": {
                            "trace_id": "trace-arb-1",
                            "task_id": str(uuid.uuid4()),
                            "attempt": 1,
                            "candidate_count": 4,
                            "validated_count": 4,
                            "selected_count": 2,
                            "blocked_count": 1,
                            "rejected_count": 1,
                            "selected_packet_ids": [str(uuid.uuid4())],
                            "blocked": [{"reason": "protocol_effectiveness_gate_blocked: protocol=p-low"}],
                            "rejected": [{"reason": "policy_reject: unsupported mode"}],
                        },
                        "created_at": None,
                    }
                ]

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "packet-lineage-summary",
                "--run-id",
                str(expected_run_id),
                "--limit",
                "5",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["summary"]["arbitration_decision_count"] == 1
        assert payload["summary"]["arbitration_selected_total"] == 2
        assert payload["transfer_arbitration"]["summary"]["block_reason_counts"]["protocol_effectiveness_gate_blocked"] == 1
        assert payload["transfer_arbitration"]["summary"]["reject_reason_counts"]["policy_reject"] == 1

    def test_cli_packet_replay_uses_timeline_api(self, monkeypatch):
        expected_run_id = uuid.uuid4()
        expected_task_id = uuid.uuid4()
        packet_id = uuid.uuid4()

        class FakeRepo:
            @staticmethod
            def get_packet_timeline(run_id=None, task_id=None, limit=500):
                assert str(run_id) == str(expected_run_id)
                assert str(task_id) == str(expected_task_id)
                assert limit == 50
                return [
                    {
                        "packet_id": packet_id,
                        "packet_type": "SIGNAL",
                        "run_phase": "DISCOVERY",
                        "lifecycle_state": "ARCHIVED",
                        "trace_id": "trace-r-1",
                        "packet_created_at": None,
                        "event": "emit",
                        "from_state": "DRAFT",
                        "to_state": "NORMALIZED",
                        "metadata": {"k": "v"},
                        "occurred_at": None,
                    },
                    {
                        "packet_id": packet_id,
                        "packet_type": "SIGNAL",
                        "run_phase": "DISCOVERY",
                        "lifecycle_state": "ARCHIVED",
                        "trace_id": "trace-r-1",
                        "packet_created_at": None,
                        "event": "schema_pass",
                        "from_state": "NORMALIZED",
                        "to_state": "SCHEMA_VALIDATED",
                        "metadata": {},
                        "occurred_at": None,
                    },
                ]

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "packet-replay",
                "--run-id",
                str(expected_run_id),
                "--task-id",
                str(expected_task_id),
                "--limit",
                "50",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["source"] == "packet_events"
        assert payload["packet_count"] == 1
        assert payload["event_count"] == 2
        assert payload["packets"][0]["packet_type"] == "SIGNAL"

    def test_cli_packet_replay_falls_back_to_artifacts(self, monkeypatch):
        expected_run_id = uuid.uuid4()
        packet_id = uuid.uuid4()

        class FakeRepo:
            @staticmethod
            def list_sotappr_artifacts(run_id=None, limit=200):
                assert str(run_id) == str(expected_run_id)
                assert limit >= 20
                return [
                    {
                        "id": uuid.uuid4(),
                        "run_id": run_id,
                        "artifact_type": "agent_packet",
                        "payload": {
                            "trace_id": "trace-r-fallback",
                            "task_id": str(uuid.uuid4()),
                            "lifecycle_state": "ARCHIVED",
                            "lifecycle_history": [
                                {"event": "emit", "from": "DRAFT", "to": "NORMALIZED"},
                                {"event": "schema_pass", "from": "NORMALIZED", "to": "SCHEMA_VALIDATED"},
                            ],
                            "packet": {
                                "packet_id": str(packet_id),
                                "packet_type": "TASK_IR",
                                "run_phase": "DISCOVERY",
                            },
                        },
                        "created_at": None,
                    }
                ]

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "packet-replay",
                "--run-id",
                str(expected_run_id),
                "--limit",
                "20",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["source"] == "sotappr_artifacts"
        assert payload["packet_count"] == 1
        assert payload["event_count"] == 2

    def test_cli_packet_replay_writes_export_file(self, monkeypatch, tmp_path):
        expected_run_id = uuid.uuid4()

        class FakeRepo:
            @staticmethod
            def get_packet_timeline(run_id=None, task_id=None, limit=500):
                assert str(run_id) == str(expected_run_id)
                _ = task_id, limit
                return []

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        out_path = tmp_path / "replay.json"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "packet-replay",
                "--run-id",
                str(expected_run_id),
                "--out",
                str(out_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert out_path.exists()
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["run_id"] == str(expected_run_id)
        assert payload["packet_count"] == 0

    def test_cli_packet_transfer_arbitrate_selects_top_k(self, tmp_path):
        packet_a = {
            "protocol_version": "apc/1.0",
            "packet_id": str(uuid.uuid4()),
            "packet_type": "TRANSFER",
            "channel": "INTER_AGENT",
            "run_phase": "REFINEMENT",
            "sender": {"agent_id": "a", "role": "arbiter", "swarm_id": "swarm"},
            "recipients": [{"agent_id": "b", "role": "builder", "swarm_id": "swarm"}],
            "trace": {
                "session_id": "s1",
                "run_id": "r1",
                "task_id": "t1",
                "root_packet_id": str(uuid.uuid4()),
                "generation": 15,
                "step": 1,
            },
            "routing": {"delivery_mode": "DIRECT", "priority": 5, "ttl_ms": 60000, "requires_ack": False},
            "confidence": 0.9,
            "lineage": {
                "protocol_id": "p-a",
                "parent_protocol_ids": [],
                "ancestor_swarms": ["swarm-a"],
                "cross_use_case": False,
                "transfer_mode": "SELECTIVE_IMPORT",
            },
            "payload": {
                "protocol_id": "p-a",
                "from_swarm": "swarm-a",
                "to_swarm": "swarm-b",
                "sender_score": 0.9,
                "receiver_score": 0.6,
                "transfer_policy": {"mode": "SELECTIVE_IMPORT", "top_k": 2},
                "accepted": True,
            },
        }
        packet_b = {
            **packet_a,
            "packet_id": str(uuid.uuid4()),
            "payload": {
                "protocol_id": "p-b",
                "from_swarm": "swarm-a",
                "to_swarm": "swarm-c",
                "sender_score": 0.8,
                "receiver_score": 0.65,
                "transfer_policy": {"mode": "SELECTIVE_IMPORT", "top_k": 2},
                "accepted": True,
            },
            "lineage": {
                "protocol_id": "p-b",
                "parent_protocol_ids": [],
                "ancestor_swarms": ["swarm-a"],
                "cross_use_case": False,
                "transfer_mode": "SELECTIVE_IMPORT",
            },
        }
        invalid_packet = {"packet_type": "SIGNAL"}
        in_path = tmp_path / "transfer_packets.json"
        in_path.write_text(
            json.dumps({"packets": [packet_b, packet_a, invalid_packet]}, indent=2),
            encoding="utf-8",
        )
        out_path = tmp_path / "transfer_selection.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "packet-transfer-arbitrate",
                "--packets-in",
                str(in_path),
                "--top-k",
                "1",
                "--out",
                str(out_path),
            ],
        )

        assert result.exit_code == 0, result.output
        json_start = result.output.find("{")
        assert json_start >= 0
        payload = json.loads(result.output[json_start:])
        assert payload["input_count"] == 3
        assert payload["selected_count"] == 1
        assert payload["valid_transfer_count"] == 2
        assert len(payload["rejected"]) >= 1
        assert out_path.exists()
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        assert saved["selected_count"] == 1

    def test_cli_resume_run_uses_stored_repo_path(self, monkeypatch):
        run_uuid = uuid.uuid4()

        @dataclass
        class FakeSotapprCfg:
            governance_pack: str = "balanced"
            report_archive_dir: str = "artifacts/sotappr"

        @dataclass
        class FakeConfig:
            sotappr: FakeSotapprCfg = field(default_factory=FakeSotapprCfg)

        class FakeRepo:
            @staticmethod
            def get_sotappr_run(run_id):
                assert str(run_id) == str(run_uuid)
                return {"repo_path": "/tmp/repo", "project_id": str(uuid.uuid4())}

        @dataclass
        class FakeBundle:
            repository: FakeRepo = field(default_factory=FakeRepo)
            config: FakeConfig = field(default_factory=FakeConfig)

        class FakeFactory:
            @staticmethod
            def create(config_dir=None, env=None, api_key=None):
                _ = config_dir, env, api_key
                return FakeBundle()

            @staticmethod
            def close(bundle):
                _ = bundle

        class FakeExecutor:
            def __init__(self):
                self.resume_called = False

            def resume_run(self, run_id, max_iterations=100):
                self.resume_called = True
                assert str(run_id) == str(run_uuid)
                assert max_iterations == 7
                return ExecutionSummary(
                    project_id=str(uuid.uuid4()),
                    tasks_seeded=9,
                    tasks_processed=4,
                    run_id=str(run_uuid),
                    mode="resume",
                )

        fake_executor = FakeExecutor()

        class FakeSOTAppRExecutor:
            @staticmethod
            def from_bundle(bundle, repo_path, test_command, **kwargs):
                _ = bundle, test_command, kwargs
                assert repo_path == "/tmp/repo"
                return fake_executor

        monkeypatch.setattr("src.cli._load_component_factory", lambda: FakeFactory)
        monkeypatch.setattr("src.cli._load_sotappr_executor", lambda: FakeSOTAppRExecutor)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["sotappr-resume-run", "--run-id", str(run_uuid), "--max-iterations", "7"],
        )

        assert result.exit_code == 0, result.output
        assert "Resume summary:" in result.output
        assert fake_executor.resume_called

    def test_cli_approve_task_transitions_reviewing(self, monkeypatch):
        task_id = str(uuid.uuid4())

        @dataclass
        class FakeTask:
            id: str
            title: str
            status: object

        class FakeStatus:
            value = "DONE"

        class FakeRepo:
            @staticmethod
            def approve_task(task_uuid):
                assert str(task_uuid) == task_id
                return FakeTask(id=task_id, title="review-me", status=FakeStatus())

            @staticmethod
            def get_task(task_uuid):
                _ = task_uuid
                return None

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(cli, ["sotappr-approve-task", "--task-id", task_id])

        assert result.exit_code == 0, result.output
        assert "Approved task" in result.output
        assert "DONE" in result.output

    def test_cli_reset_task_safe_returns_reconciled_counters(self, monkeypatch):
        task_id = str(uuid.uuid4())

        class FakeRepo:
            @staticmethod
            def reset_task_for_retry(task_uuid, status):
                assert str(task_uuid) == task_id
                assert status.value == "PENDING"
                return Task(
                    id=uuid.UUID(task_id),
                    project_id=uuid.uuid4(),
                    title="retry-me",
                    description="d",
                    status=TaskStatus.PENDING,
                    attempt_count=5,
                    council_count=0,
                )

            @staticmethod
            def get_next_hypothesis_attempt(task_uuid):
                assert str(task_uuid) == task_id
                return 6

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(cli, ["sotappr-reset-task-safe", "--task-id", task_id])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["task_id"] == task_id
        assert payload["status"] == "PENDING"
        assert payload["attempt_count"] == 5
        assert payload["next_hypothesis_attempt"] == 6

    def test_cli_dashboard_outputs_operational_payload(self, monkeypatch):
        class FakeRepo:
            @staticmethod
            def list_sotappr_runs(limit=20, project_id=None):
                _ = limit, project_id
                return [
                    {"status": "running", "stop_reason": None, "last_error": None},
                    {
                        "id": uuid.uuid4(),
                        "status": "paused",
                        "stop_reason": "budget_time_exceeded",
                        "last_error": "Run paused: budget_time_exceeded",
                        "estimated_cost_usd": 0.12,
                        "elapsed_hours": 1.1,
                    },
                    {"status": "completed", "stop_reason": None, "last_error": None},
                ]

            @staticmethod
            def get_task_status_summary(project_id=None):
                _ = project_id
                return {"PENDING": 3, "REVIEWING": 2, "DONE": 7}

            @staticmethod
            def list_review_queue(limit=20, project_id=None):
                _ = limit, project_id
                return [
                    Task(
                        project_id=uuid.uuid4(),
                        title="review-1",
                        description="d",
                        priority=9,
                    )
                ]

            @staticmethod
            def get_hypothesis_error_stats(project_id=None):
                _ = project_id
                return [{"error_signature": "timeout", "cnt": 4}]

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(cli, ["sotappr-dashboard"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["run_status_counts"]["completed"] == 1
        assert payload["task_status_counts"]["DONE"] == 7
        assert payload["review_queue_size"] == 1
        assert payload["budget_paused_runs"] == 1
        assert payload["latest_budget_pauses"][0]["stop_reason"] == "budget_time_exceeded"
        assert payload["transfer_arbitration_summary"]["decision_count"] == 0

    def test_cli_benchmark_freeze_writes_snapshot(self, monkeypatch, tmp_path):
        run_a = uuid.uuid4()
        run_b = uuid.uuid4()

        class FakeRepo:
            @staticmethod
            def list_sotappr_runs(limit=20, project_id=None):
                _ = limit, project_id
                return [
                    {
                        "id": run_a,
                        "status": "completed",
                        "tasks_seeded": 4,
                        "tasks_processed": 4,
                        "estimated_cost_usd": 0.1,
                        "elapsed_hours": 1.0,
                    },
                    {
                        "id": run_b,
                        "status": "paused",
                        "tasks_seeded": 4,
                        "tasks_processed": 2,
                        "estimated_cost_usd": 0.2,
                        "elapsed_hours": 1.5,
                    },
                ]

            @staticmethod
            def list_sotappr_artifacts(run_id, phase=None, limit=500):
                _ = run_id, phase, limit
                return [{"artifact_type": "retry_telemetry"}]

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        out_path = tmp_path / "frozen.json"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sotappr-benchmark-freeze",
                "--limit",
                "10",
                "--out",
                str(out_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert out_path.exists()
        payload = json.loads(result.output)
        assert payload["run_count"] == 2
        frozen = json.loads(out_path.read_text(encoding="utf-8"))
        assert frozen["aggregate"]["total_runs"] == 2
        assert "tolerances" in frozen

    def test_cli_benchmark_gate_fails_on_regression(self, monkeypatch, tmp_path):
        run_id = uuid.uuid4()
        baseline = {
            "aggregate": {
                "quality_success_rate": 1.0,
                "avg_elapsed_hours": 1.0,
                "avg_estimated_cost_usd": 0.1,
                "retry_events_per_task": 0.1,
                "rollback_rate": 0.1,
            },
            "status_filter": ["completed"],
            "run_ids": [str(run_id)],
            "tolerances": {
                "quality_drop_abs": 0.01,
                "latency_increase_pct": 0.2,
                "cost_increase_pct": 0.2,
                "retry_increase_pct": 0.2,
                "rollback_increase_pct": 0.2,
            },
        }
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

        class FakeRepo:
            @staticmethod
            def list_sotappr_runs(limit=20, project_id=None):
                _ = limit, project_id
                return [
                    {
                        "id": run_id,
                        "status": "completed",
                        "tasks_seeded": 4,
                        "tasks_processed": 1,
                        "estimated_cost_usd": 0.5,
                        "elapsed_hours": 3.0,
                    }
                ]

            @staticmethod
            def list_sotappr_artifacts(run_id, phase=None, limit=500):
                _ = run_id, phase, limit
                return [{"artifact_type": "retry_telemetry"} for _ in range(3)]

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sotappr-benchmark-gate",
                "--baseline",
                str(baseline_path),
            ],
        )

        assert result.exit_code != 0
        assert "Benchmark regression gate failed" in result.output
        assert '"gate_passed": false' in result.output

    def test_cli_portfolio_outputs_project_rows(self, monkeypatch):
        project_id = uuid.uuid4()

        class FakeRepo:
            @staticmethod
            def get_portfolio_summary(limit=50):
                _ = limit
                return [
                    {
                        "id": project_id,
                        "name": "proj",
                        "repo_path": "/tmp/p",
                        "total_tasks": 10,
                        "pending_tasks": 3,
                        "done_tasks": 6,
                        "stuck_tasks": 1,
                        "last_run_at": None,
                    }
                ]

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(cli, ["sotappr-portfolio", "--limit", "5"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["count"] == 1
        assert payload["projects"][0]["project_id"] == str(project_id)

    def test_cli_replay_search_filters_project(self, monkeypatch):
        wanted_project = uuid.uuid4()
        other_project = uuid.uuid4()
        run_id = uuid.uuid4()

        class FakeRepo:
            @staticmethod
            def search_sotappr_replay(query, limit=20):
                _ = query, limit
                return [
                    {
                        "id": uuid.uuid4(),
                        "run_id": run_id,
                        "project_id": wanted_project,
                        "created_at": None,
                        "payload": {
                            "context": "A",
                            "transferable_insight": "A1",
                        },
                    },
                    {
                        "id": uuid.uuid4(),
                        "run_id": uuid.uuid4(),
                        "project_id": other_project,
                        "created_at": None,
                        "payload": {
                            "context": "B",
                            "transferable_insight": "B1",
                        },
                    },
                ]

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sotappr-replay-search",
                "--query",
                "A",
                "--project-id",
                str(wanted_project),
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["count"] == 1
        assert payload["matches"][0]["project_id"] == str(wanted_project)

    def test_cli_autotune_generates_recommendations(self, monkeypatch):
        class FakeSotapprCfg:
            require_human_review_before_done = True

        class FakeConfig:
            sotappr = FakeSotapprCfg()

        class FakeRepo:
            @staticmethod
            def get_task_status_summary(project_id=None):
                _ = project_id
                return {"PENDING": 10, "DONE": 2, "REVIEWING": 2, "STUCK": 1}

            @staticmethod
            def get_hypothesis_error_stats(project_id=None):
                _ = project_id
                return [{"error_signature": "timeout on test", "cnt": 5}]

            @staticmethod
            def list_review_queue(limit=50, project_id=None):
                _ = limit, project_id
                return [object()] * 10

        class FakeCtx:
            repository = FakeRepo()
            config = FakeConfig()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(cli, ["sotappr-autotune"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["total_tasks"] == 15
        assert len(payload["recommendations"]) >= 2

    def test_cli_drift_check_uses_config_threshold(self, monkeypatch):
        class FakeSentinelCfg:
            drift_threshold = 0.25

        class FakeConfig:
            sentinel = FakeSentinelCfg()

        class FakeRepo:
            @staticmethod
            def get_task_status_summary(project_id=None):
                _ = project_id
                return {"PENDING": 4, "REVIEWING": 2, "STUCK": 2, "DONE": 2}

            @staticmethod
            def get_hypothesis_error_stats(project_id=None):
                _ = project_id
                return [{"error_signature": "assertion", "cnt": 6}]

        class FakeCtx:
            repository = FakeRepo()
            config = FakeConfig()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(cli, ["sotappr-drift-check"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["threshold"] == 0.25
        assert payload["drift_score"] >= 0
        assert "components" in payload

    def test_cli_chaos_drill_extracts_playbooks(self, monkeypatch):
        run_id = uuid.uuid4()

        class FakeRepo:
            @staticmethod
            def list_sotappr_artifacts(run_id, phase=None, limit=200):
                assert phase == 5
                _ = limit
                return [
                    {
                        "artifact_type": "phase5",
                        "payload": {
                            "chaos_playbooks": [
                                {
                                    "scenario": "DB outage",
                                    "detection": "health check fails",
                                    "automated_response": "trip breaker",
                                    "manual_escalation": "page on-call",
                                    "recovery": "restore db",
                                    "post_mortem_hook": "record timeline",
                                }
                            ]
                        },
                    }
                ]

        class FakeCtx:
            repository = FakeRepo()

        monkeypatch.setattr("src.cli._open_repo_context", lambda *args, **kwargs: FakeCtx())
        monkeypatch.setattr("src.cli._close_repo_context", lambda ctx: None)

        runner = CliRunner()
        result = runner.invoke(cli, ["sotappr-chaos-drill", "--run-id", str(run_id)])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["drill_count"] == 1
        assert payload["drills"][0]["scenario"] == "DB outage"


class TestSOTAppRExecutorRunMethods:
    """Tests for run/artifact persistence, observability delegation, paused/failure paths."""

    def _make_executor(self, repo=None, loop=None, observability=None):
        repo = repo or StubRunRepository()
        loop = loop or StubTaskLoop(processed=0)
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)
        executor.observability = observability
        return executor, repo, loop

    def test_bootstrap_and_execute_paused_mode(self, tmp_path):
        request, report = _report()
        obs = SOTAppRObservability(
            jsonl_path=tmp_path / "events.jsonl",
            metrics_path=tmp_path / "metrics.json",
        )
        executor, repo, loop = self._make_executor(
            repo=StubRunRepository(),
            loop=StubTaskLoop(processed=0),
            observability=obs,
        )

        summary = executor.bootstrap_and_execute(
            request=request,
            report=report,
            repo_path="/tmp/repo",
            execute=False,
        )

        assert summary.tasks_processed == 0
        assert summary.mode == "execute"
        # Run should be updated to "paused"
        run = repo.get_sotappr_run(UUID(summary.run_id))
        assert run is not None
        assert run["status"] == "paused"

        # Observability should have emitted sotappr_paused
        lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
        event_types = [json.loads(line)["event_type"] for line in lines]
        assert "sotappr_paused" in event_types

    def test_bootstrap_and_execute_failure_path(self, tmp_path):
        import pytest

        request, report = _report()
        obs = SOTAppRObservability(
            jsonl_path=tmp_path / "events.jsonl",
            metrics_path=tmp_path / "metrics.json",
        )
        executor, repo, _ = self._make_executor(
            repo=StubRunRepository(),
            loop=FailingTaskLoop(RuntimeError("boom")),
            observability=obs,
        )

        with pytest.raises(RuntimeError, match="boom"):
            executor.bootstrap_and_execute(
                request=request,
                report=report,
                repo_path="/tmp/repo",
            )

        # Find the run — there should be exactly one
        assert len(repo.runs) == 1
        run_id = next(iter(repo.runs))
        run = repo.runs[run_id]
        assert run["status"] == "failed"
        assert "boom" in (run["last_error"] or "")

        # Observability should have emitted sotappr_failed
        lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
        event_types = [json.loads(line)["event_type"] for line in lines]
        assert "sotappr_failed" in event_types

        # Metrics should have been flushed
        assert (tmp_path / "metrics.json").exists()

    def test_resume_run_happy_path(self, tmp_path):
        request, report = _report()
        repo = StubRunRepository()
        loop = StubTaskLoop(processed=5)
        obs = SOTAppRObservability(
            jsonl_path=tmp_path / "events.jsonl",
            metrics_path=tmp_path / "metrics.json",
        )
        executor, _, _ = self._make_executor(repo=repo, loop=loop, observability=obs)

        # Manually seed a run to resume
        project_id = uuid.uuid4()
        run_id = repo.create_sotappr_run(
            project_id=project_id,
            mode="execute",
            governance_pack="balanced",
            spec_json={},
            report_json={},
            repo_path="/tmp/repo",
            report_path=None,
            status="paused",
        )
        repo.update_sotappr_run(run_id=run_id, tasks_seeded=10, tasks_processed=3)

        summary = executor.resume_run(run_id=run_id, max_iterations=50)

        assert summary.project_id == str(project_id)
        assert summary.tasks_processed == 8  # 3 existing + 5 new
        assert summary.tasks_seeded == 10
        assert summary.mode == "resume"

        run = repo.get_sotappr_run(run_id)
        assert run["status"] == "completed"
        assert run["completed"] is True

    def test_resume_run_not_found(self):
        import pytest

        repo = StubRunRepository()
        loop = StubTaskLoop(processed=0)
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)

        with pytest.raises(ValueError, match="SOTAppR run not found"):
            executor.resume_run(run_id=uuid.uuid4())

    def test_resume_run_missing_project_id(self):
        import pytest

        repo = StubRunRepository()
        loop = StubTaskLoop(processed=0)
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)

        # Manually insert a run with no project_id
        run_id = uuid.uuid4()
        repo.runs[run_id] = {
            "id": run_id,
            "project_id": None,
            "status": "paused",
            "tasks_processed": 0,
            "tasks_seeded": 0,
        }

        with pytest.raises(ValueError, match="missing project_id"):
            executor.resume_run(run_id=run_id)

    def test_resume_run_failure_path(self, tmp_path):
        import pytest

        repo = StubRunRepository()
        loop = FailingTaskLoop(RuntimeError("resume-boom"))
        obs = SOTAppRObservability(
            jsonl_path=tmp_path / "events.jsonl",
            metrics_path=tmp_path / "metrics.json",
        )
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)
        executor.observability = obs

        project_id = uuid.uuid4()
        run_id = repo.create_sotappr_run(
            project_id=project_id,
            mode="execute",
            governance_pack="balanced",
            spec_json={},
            report_json={},
            repo_path="/tmp/repo",
            report_path=None,
            status="paused",
        )
        repo.update_sotappr_run(run_id=run_id, tasks_seeded=5, tasks_processed=2)

        with pytest.raises(RuntimeError, match="resume-boom"):
            executor.resume_run(run_id=run_id)

        run = repo.get_sotappr_run(run_id)
        assert run["status"] == "failed"
        assert "resume-boom" in (run["last_error"] or "")

    def test_title_from_action_truncates_long_strings(self):
        repo = StubRunRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)

        long_action = "x" * 100
        title = executor._title_from_action(action=long_action, index=1)
        assert title.endswith("...")
        assert len(title) <= len("SOTAppR-1: ") + 72

    def test_title_from_action_preserves_short_strings(self):
        repo = StubRunRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)

        short_action = "Add user auth endpoint"
        title = executor._title_from_action(action=short_action, index=3)
        assert title == f"SOTAppR-3: {short_action}"
        assert "..." not in title

    def test_create_run_stores_and_returns_uuid(self):
        request, report = _report()
        repo = StubRunRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)

        run_id = executor._create_run(
            project_id=uuid.uuid4(),
            mode="execute",
            governance_pack="strict",
            request=request,
            report=report,
            repo_path="/tmp/repo",
            report_path=None,
        )

        assert isinstance(run_id, UUID)
        assert repo.get_sotappr_run(run_id) is not None

    def test_update_run_updates_stored_run(self):
        repo = StubRunRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)

        run_id = repo.create_sotappr_run(
            project_id=uuid.uuid4(),
            mode="execute",
            governance_pack="balanced",
            spec_json={},
            report_json={},
            repo_path="/tmp",
            report_path=None,
            status="planned",
        )

        executor._update_run(run_id=run_id, status="running", tasks_seeded=5)
        run = repo.get_sotappr_run(run_id)
        assert run["status"] == "running"
        assert run["tasks_seeded"] == 5

    def test_get_run_returns_stored_run(self):
        repo = StubRunRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)

        project_id = uuid.uuid4()
        run_id = repo.create_sotappr_run(
            project_id=project_id,
            mode="execute",
            governance_pack="balanced",
            spec_json={},
            report_json={},
            repo_path="/tmp",
            report_path=None,
            status="planned",
        )

        result = executor._get_run(run_id)
        assert result is not None
        assert result["project_id"] == project_id
        assert result["status"] == "planned"

    def test_save_phase_artifacts_saves_all_phases(self):
        _, report = _report()
        repo = StubRunRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)

        run_id = uuid.uuid4()
        repo.runs[run_id] = {}  # Just needs to exist

        executor._save_phase_artifacts(run_id=run_id, report=report)

        # 9 phase artifacts + experience_replay entries
        phase_types = [a["artifact_type"] for a in repo.artifacts]
        assert "alien_goggles" in phase_types
        assert "phase1" in phase_types
        assert "phase8" in phase_types
        assert len([t for t in phase_types if t != "experience_replay"]) == 9

    def test_save_seed_artifact_stores_payload(self):
        repo = StubRunRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)

        run_id = uuid.uuid4()
        repo.runs[run_id] = {}

        payload = {"tasks": [{"title": "Task 1"}, {"title": "Task 2"}]}
        executor._save_seed_artifact(run_id=run_id, payload=payload)

        assert len(repo.artifacts) == 1
        assert repo.artifacts[0]["artifact_type"] == "seeded_tasks"
        assert repo.artifacts[0]["phase"] == 3
        assert repo.artifacts[0]["payload"]["tasks"][0]["title"] == "Task 1"

    def test_emit_event_delegates_to_observability(self, tmp_path):
        obs = SOTAppRObservability(
            jsonl_path=tmp_path / "events.jsonl",
            metrics_path=tmp_path / "metrics.json",
        )
        repo = StubRunRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)
        executor.observability = obs

        executor._emit_event("test_event", {"k": "v"})

        lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event_type"] == "test_event"
        assert record["payload"]["k"] == "v"

    def test_flush_metrics_delegates_to_observability(self, tmp_path):
        obs = SOTAppRObservability(
            jsonl_path=tmp_path / "events.jsonl",
            metrics_path=tmp_path / "metrics.json",
        )
        repo = StubRunRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)
        executor.observability = obs

        executor._emit_event("x", {})
        executor._flush_metrics()

        assert (tmp_path / "metrics.json").exists()
        snapshot = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert snapshot["counters"]["x"] == 1

    def test_emit_event_noop_when_no_observability(self):
        repo = StubRunRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)
        assert executor.observability is None
        # Should not raise
        executor._emit_event("test", {"a": 1})
        executor._flush_metrics()

    def test_get_run_returns_none_without_method(self):
        """Line 429: _get_run returns None when repository has no get_sotappr_run."""
        repo = StubRepository()  # basic repo, no get_sotappr_run
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)

        result = executor._get_run(uuid.uuid4())
        assert result is None

    def test_create_run_returns_none_without_method(self):
        """_create_run returns None when repository has no create_sotappr_run."""
        request, report = _report()
        repo = StubRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)

        result = executor._create_run(
            project_id=uuid.uuid4(),
            mode="execute",
            governance_pack="balanced",
            request=request,
            report=report,
            repo_path="/tmp",
            report_path=None,
        )
        assert result is None

    def test_update_run_noop_without_method(self):
        """_update_run is a no-op when repository has no update_sotappr_run."""
        repo = StubRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)
        # Should not raise
        executor._update_run(run_id=uuid.uuid4(), status="running")

    def test_update_run_noop_when_run_id_none(self):
        """_update_run is a no-op when run_id is None."""
        repo = StubRunRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)
        # Should not raise even with run-capable repo
        executor._update_run(run_id=None, status="running")

    def test_save_phase_artifacts_noop_without_method(self):
        """_save_phase_artifacts is a no-op when repository has no save_sotappr_artifact."""
        _, report = _report()
        repo = StubRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)
        # Should not raise
        executor._save_phase_artifacts(run_id=uuid.uuid4(), report=report)

    def test_save_phase_artifacts_noop_when_run_id_none(self):
        """_save_phase_artifacts is a no-op when run_id is None."""
        _, report = _report()
        repo = StubRunRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)
        executor._save_phase_artifacts(run_id=None, report=report)
        assert len(repo.artifacts) == 0

    def test_save_seed_artifact_noop_without_method(self):
        """_save_seed_artifact is a no-op when repository has no save_sotappr_artifact."""
        repo = StubRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)
        executor._save_seed_artifact(run_id=uuid.uuid4(), payload={"tasks": []})

    def test_save_seed_artifact_noop_when_run_id_none(self):
        """_save_seed_artifact is a no-op when run_id is None."""
        repo = StubRunRepository()
        loop = StubTaskLoop()
        executor = SOTAppRExecutor(repository=repo, task_loop=loop)
        executor._save_seed_artifact(run_id=None, payload={"tasks": []})
        assert len(repo.artifacts) == 0


class TestSOTAppRCliEdgeCases:
    """CLI command edge cases: SOTAppRStop, empty catalog, exception paths."""

    def test_sotappr_build_sotappr_stop(self, tmp_path, monkeypatch):
        from src.sotappr import SOTAppRStop

        spec_path = tmp_path / "spec.json"
        spec_path.write_text(
            json.dumps(
                {
                    "organism_name": "Test",
                    "stated_problem": "Test problem.",
                    "root_need": "Test need.",
                    "user_confirmed_phase1": True,
                }
            ),
            encoding="utf-8",
        )

        # Monkeypatch SOTAppRBuilder.build to raise SOTAppRStop
        monkeypatch.setattr(
            "src.sotappr.SOTAppRBuilder.build",
            lambda self, request: (_ for _ in ()).throw(SOTAppRStop("Phase 1 failed")),
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["sotappr-build", "--spec", str(spec_path)])
        assert result.exit_code != 0
        assert "Phase 1 failed" in result.output

    def test_sotappr_models_empty_result(self, monkeypatch):
        class EmptyCatalog:
            def __init__(self, api_key=None):
                pass

            def list_models(self, **kwargs):
                return []

        monkeypatch.setattr("src.cli._load_openrouter_model_catalog", lambda: EmptyCatalog)

        runner = CliRunner()
        result = runner.invoke(cli, ["sotappr-models"])
        assert result.exit_code != 0
        assert "No models matched" in result.output

    def test_sotappr_models_exception(self, monkeypatch):
        class FailCatalog:
            def __init__(self, api_key=None):
                pass

            def list_models(self, **kwargs):
                raise RuntimeError("network down")

        monkeypatch.setattr("src.cli._load_openrouter_model_catalog", lambda: FailCatalog)

        runner = CliRunner()
        result = runner.invoke(cli, ["sotappr-models"])
        assert result.exit_code != 0
        assert "Failed to query" in result.output

    def test_sotappr_execute_general_exception(self, tmp_path, monkeypatch):
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(
            json.dumps(
                {
                    "organism_name": "Test",
                    "stated_problem": "Test.",
                    "root_need": "Test.",
                    "user_confirmed_phase1": True,
                }
            ),
            encoding="utf-8",
        )

        @dataclass
        class FakeSotapprCfg:
            governance_pack: str = "balanced"
            report_archive_dir: str = "artifacts/sotappr"

        @dataclass
        class FakeConfig:
            sotappr: FakeSotapprCfg = field(default_factory=FakeSotapprCfg)

        @dataclass
        class FakeBundle:
            config: FakeConfig = field(default_factory=FakeConfig)

        class FakeExecutor:
            def preview_seeded_tasks(self, backlog_actions):
                return []

            def bootstrap_and_execute(self, **kwargs):
                raise RuntimeError("unexpected executor crash")

        class FakeFactory:
            @staticmethod
            def create(config_dir=None, env=None, api_key=None):
                return FakeBundle()

            @staticmethod
            def close(bundle):
                pass

        class FakeSOTAppRExecutor:
            @staticmethod
            def from_bundle(bundle, repo_path, test_command, **kwargs):
                return FakeExecutor()

        monkeypatch.setattr("src.cli._load_component_factory", lambda: FakeFactory)
        monkeypatch.setattr("src.cli._load_sotappr_executor", lambda: FakeSOTAppRExecutor)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sotappr-execute",
                "--spec",
                str(spec_path),
                "--repo-path",
                str(tmp_path),
                "--max-iterations",
                "1",
            ],
            input="y\n",
        )
        assert result.exit_code != 0
        assert "SOTAppR execution failed" in result.output
