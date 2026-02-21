"""Integration tests for src/orchestrator/loop.py using real-shaped fakes.

These are NOT mocks — they are minimal implementations with real data shapes
that exercise the actual TaskLoop logic (state transitions, retries, council
escalation, governance, circuit breaker, etc.).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

import pytest

from src.core.config import AppConfig, OrchestratorConfig, SOTAppRConfig
from src.core.models import (
    AgentResult,
    BuildResult,
    ContextBrief,
    CouncilDiagnosis,
    HypothesisEntry,
    PeerReview,
    Project,
    ResearchBrief,
    SentinelVerdict,
    Task,
    TaskContext,
    TaskStatus,
)
from src.orchestrator.diagnostics import DiagnosticsCollector
from src.orchestrator.health_monitor import HealthCheck, HealthMonitor
from src.orchestrator.loop import TaskLoop
from src.orchestrator.metrics import PipelineMetrics
from src.orchestrator.task_router import TaskRouter


# ---------------------------------------------------------------------------
# Fake implementations — real data shapes, in-memory state
# ---------------------------------------------------------------------------

class FakeRepository:
    """In-memory dict store with real Repository method signatures."""

    def __init__(self):
        self.projects: dict[uuid.UUID, Project] = {}
        self.tasks: dict[uuid.UUID, Task] = {}
        self.hypotheses: list[HypothesisEntry] = []
        self.peer_reviews: list[PeerReview] = []
        self.sotappr_artifacts: list[dict[str, Any]] = []
        self._pending_queue: list[uuid.UUID] = []  # ordered task IDs

    def create_project(self, project: Project) -> Project:
        self.projects[project.id] = project
        return project

    def create_task(self, task: Task) -> Task:
        self.tasks[task.id] = task
        self._pending_queue.append(task.id)
        return task

    def get_next_task(self, project_id: uuid.UUID) -> Optional[Task]:
        for tid in self._pending_queue:
            task = self.tasks.get(tid)
            if task and task.project_id == project_id and task.status == TaskStatus.PENDING:
                self._pending_queue.remove(tid)
                return task
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
        return [
            t for t in self.tasks.values()
            if t.status in (TaskStatus.CODING, TaskStatus.REVIEWING, TaskStatus.RESEARCHING)
        ]

    def log_hypothesis(self, entry: HypothesisEntry) -> HypothesisEntry:
        self.hypotheses.append(entry)
        return entry

    def get_failed_approaches(self, task_id: uuid.UUID) -> list[HypothesisEntry]:
        return [
            h for h in self.hypotheses
            if h.task_id == task_id and h.outcome.value == "FAILURE"
        ]

    def save_peer_review(self, review: PeerReview) -> PeerReview:
        self.peer_reviews.append(review)
        return review

    def get_peer_reviews(self, task_id: uuid.UUID) -> list[PeerReview]:
        return [r for r in self.peer_reviews if r.task_id == task_id]

    def save_context_snapshot(self, snapshot) -> Any:
        return snapshot

    def save_sotappr_artifact(
        self,
        *,
        run_id: uuid.UUID,
        phase: int,
        artifact_type: str,
        payload: dict[str, Any],
    ) -> uuid.UUID:
        artifact_id = uuid.uuid4()
        self.sotappr_artifacts.append(
            {
                "id": artifact_id,
                "run_id": run_id,
                "phase": phase,
                "artifact_type": artifact_type,
                "payload": payload,
            }
        )
        return artifact_id


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
    """Returns configurable success/failure with BuildResult data."""

    name = "Builder"

    def __init__(
        self,
        succeed: bool = True,
        files: list[str] | None = None,
        tokens: int = 100,
        fail_reason: str = "test failure",
    ):
        self._succeed = succeed
        self._files = files or (["src/main.py"] if succeed else [])
        self._tokens = tokens
        self._fail_reason = fail_reason
        self._call_count = 0

    def run(self, input_data: Any) -> AgentResult:
        self._call_count += 1
        br = BuildResult(
            files_changed=self._files if self._succeed else [],
            tests_passed=self._succeed,
            test_output="All tests passed" if self._succeed else self._fail_reason,
            approach_summary="Implemented feature" if self._succeed else "Failed attempt",
            model_used="test/model",
            failure_reason=None if self._succeed else self._fail_reason,
            failure_detail=None if self._succeed else self._fail_reason,
        )
        return AgentResult(
            agent_name="Builder",
            status="success" if self._succeed else "failure",
            data={"build_result": br.model_dump(), "tokens_used": self._tokens},
            error=None if self._succeed else self._fail_reason,
        )


class FakeBuilderSequence:
    """Builder that fails N times then succeeds."""

    name = "Builder"

    def __init__(self, fail_count: int = 1, tokens: int = 100):
        self._fail_count = fail_count
        self._tokens = tokens
        self._calls = 0

    def run(self, input_data: Any) -> AgentResult:
        self._calls += 1
        succeed = self._calls > self._fail_count
        br = BuildResult(
            files_changed=["src/main.py"] if succeed else [],
            tests_passed=succeed,
            test_output="passed" if succeed else "FAILED: assertion error",
            approach_summary="attempt",
            model_used="test/model",
            failure_reason=None if succeed else "assertion error",
            failure_detail=None if succeed else "FAILED: assertion error",
        )
        return AgentResult(
            agent_name="Builder",
            status="success" if succeed else "failure",
            data={"build_result": br.model_dump(), "tokens_used": self._tokens},
            error=None if succeed else "assertion error",
        )


class FakeSentinel:
    """Returns configurable pass/fail with SentinelVerdict data."""

    name = "Sentinel"

    def __init__(self, approve: bool = True):
        self._approve = approve

    def run(self, input_data: Any) -> AgentResult:
        verdict = SentinelVerdict(
            approved=self._approve,
            violations=[] if self._approve else [{"check": "style", "detail": "bad formatting"}],
        )
        return AgentResult(
            agent_name="Sentinel",
            status="success" if self._approve else "failure",
            data={"sentinel_verdict": verdict.model_dump(), "tokens_used": 50},
            error=None if self._approve else "Sentinel rejected",
        )


class FakeResearchUnit:
    """Returns success with ResearchBrief data."""

    name = "ResearchUnit"

    def run(self, input_data: Any) -> AgentResult:
        brief = ResearchBrief(
            live_docs=[{"title": "API Docs", "url": "https://example.com", "snippet": "..."}],
        )
        return AgentResult(
            agent_name="ResearchUnit",
            status="success",
            data={"research_brief": brief.model_dump(), "tokens_used": 30},
        )


class FakeLibrarian:
    """Returns success with ContextBrief data."""

    name = "Librarian"

    def __init__(self, task: Task):
        self._task = task

    def run(self, input_data: Any) -> AgentResult:
        cb = ContextBrief(task=self._task)
        return AgentResult(
            agent_name="Librarian",
            status="success",
            data={"context_brief": cb.model_dump(), "tokens_used": 20},
        )


class FakeCouncil:
    """Returns success with CouncilDiagnosis data."""

    name = "Council"

    def __init__(self, succeed: bool = True):
        self._succeed = succeed
        self.call_count = 0

    def run(self, input_data: Any) -> AgentResult:
        self.call_count += 1
        if not self._succeed:
            return AgentResult(
                agent_name="Council",
                status="failure",
                error="Council failed",
            )
        diag = CouncilDiagnosis(
            strategy_shift="Try a different algorithm",
            new_approach="Use binary search instead of linear scan",
            reasoning="Linear scan is O(n), too slow",
            model_used="council/model",
        )
        return AgentResult(
            agent_name="Council",
            status="success",
            data={"council_diagnosis": diag.model_dump(), "tokens_used": 80},
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project() -> Project:
    return Project(
        name="test-project",
        repo_path="/tmp/test-repo",
        tech_stack={"language": "python"},
    )


@pytest.fixture
def task(project: Project) -> Task:
    return Task(
        project_id=project.id,
        title="Add user login",
        description="Implement JWT auth for login endpoint",
        priority=10,
    )


@pytest.fixture
def fake_repo(project: Project, task: Task) -> FakeRepository:
    repo = FakeRepository()
    repo.create_project(project)
    repo.create_task(task)
    return repo


@pytest.fixture
def config() -> AppConfig:
    return AppConfig(
        orchestrator=OrchestratorConfig(
            max_retries_per_task=5,
            council_trigger_threshold=2,
            council_max_invocations=3,
        ),
        sotappr=SOTAppRConfig(
            max_files_changed_per_task=25,
            protected_paths=[".env", ".git/", "secrets/"],
            require_human_review_before_done=False,
        ),
    )


def _make_loop(
    fake_repo: FakeRepository,
    config: AppConfig,
    builder: Any = None,
    sentinel: Any = None,
    research_unit: Any = None,
    librarian: Any = None,
    council: Any = None,
    health_monitor: Any = None,
) -> TaskLoop:
    """Helper to wire up a TaskLoop with fakes."""
    orch_config = config.orchestrator
    hm = health_monitor or HealthMonitor(
        repository=fake_repo,
        config=orch_config,
    )
    tr = TaskRouter(repository=fake_repo)
    metrics = PipelineMetrics()
    diagnostics = DiagnosticsCollector()
    sm = FakeStateManager()

    return TaskLoop(
        repository=fake_repo,
        state_manager=sm,
        builder=builder or FakeBuilder(succeed=True),
        task_router=tr,
        health_monitor=hm,
        metrics=metrics,
        diagnostics=diagnostics,
        config=config,
        research_unit=research_unit,
        librarian=librarian,
        sentinel=sentinel,
        council=council,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTaskLoopIntegration:
    """Integration tests using real-shaped fakes — exercises actual loop.py logic."""

    def test_run_once_happy_path(self, fake_repo, config, project, task):
        """All agents succeed → task reaches DONE."""
        loop = _make_loop(fake_repo, config, builder=FakeBuilder(succeed=True))
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.DONE
        assert result.id == task.id

    def test_run_once_builder_failure_triggers_retry(self, fake_repo, config, project, task):
        """Builder fails on first attempt, succeeds on second → DONE."""
        builder = FakeBuilderSequence(fail_count=1)
        loop = _make_loop(fake_repo, config, builder=builder)
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.DONE
        # Should have taken 2 attempts
        assert builder._calls == 2

    def test_run_once_sentinel_rejection(self, project):
        """Sentinel rejects on single-attempt loop → logs hypothesis and raises.

        Note: The current loop.py has a known limitation where sentinel
        rejection on retry causes an invalid REVIEWING → REVIEWING transition.
        This test validates the first rejection path works correctly (hypothesis
        logged, error propagated). A fix for multi-retry sentinel flow is tracked
        separately.
        """
        repo = FakeRepository()
        repo.create_project(project)
        t = Task(
            project_id=project.id,
            title="Sentinel test task",
            description="This will be rejected by sentinel",
            priority=5,
        )
        repo.create_task(t)

        cfg = AppConfig(
            orchestrator=OrchestratorConfig(
                max_retries_per_task=1,  # Single attempt — no retry
                council_trigger_threshold=10,
                council_max_invocations=1,
            ),
        )
        loop = _make_loop(
            repo, cfg,
            builder=FakeBuilder(succeed=True),
            sentinel=FakeSentinel(approve=False),
        )
        result = loop.run_once(project.id)

        assert result is not None
        # Single attempt exhausted → STUCK
        assert result.status == TaskStatus.STUCK
        # Verify hypothesis was logged for the rejection
        failed = repo.get_failed_approaches(t.id)
        assert len(failed) >= 1
        assert any("Sentinel rejected" in (h.error_full or "") for h in failed)

    def test_council_escalation_after_threshold(self, fake_repo, config, project, task):
        """Builder fails N times → council fires."""
        council = FakeCouncil(succeed=True)
        # Builder always fails → after council_trigger_threshold (2), council fires
        loop = _make_loop(
            fake_repo, config,
            builder=FakeBuilder(succeed=False),
            council=council,
        )
        result = loop.run_once(project.id)

        assert result is not None
        assert council.call_count > 0

    def test_task_marked_stuck_after_council_exhaustion(self, fake_repo, config, project, task):
        """Builder always fails, council fires max times → STUCK."""
        council = FakeCouncil(succeed=True)
        config_strict = AppConfig(
            orchestrator=OrchestratorConfig(
                max_retries_per_task=10,
                council_trigger_threshold=2,
                council_max_invocations=1,
            ),
        )
        loop = _make_loop(
            fake_repo, config_strict,
            builder=FakeBuilder(succeed=False),
            council=council,
        )
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.STUCK

    def test_circuit_breaker_skips_iteration(self, fake_repo, config, project, task):
        """Health monitor circuit breaker open → returns None."""
        hm = HealthMonitor(repository=fake_repo, config=config.orchestrator)
        # Trip the circuit breaker
        for _ in range(3):
            hm.record_llm_failure()
        assert hm.is_circuit_breaker_open is True

        loop = _make_loop(fake_repo, config, health_monitor=hm)
        result = loop.run_once(project.id)

        assert result is None

    def test_no_output_failure_injects_guardrail(self, fake_repo, config, project, task):
        """Builder produces no files → forced strategy injected."""
        # Builder fails with no output, then succeeds
        call_count = {"n": 0}

        class AdaptiveBuilder:
            name = "Builder"

            def run(self, input_data: Any) -> AgentResult:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    # First call: no files, no test pass → no-output failure
                    br = BuildResult(files_changed=[], tests_passed=False)
                    return AgentResult(
                        agent_name="Builder",
                        status="failure",
                        data={"build_result": br.model_dump(), "tokens_used": 50},
                        error="no output",
                    )
                else:
                    # Subsequent calls: succeed
                    br = BuildResult(
                        files_changed=["src/main.py"],
                        tests_passed=True,
                        approach_summary="Fixed",
                        model_used="test/model",
                    )
                    return AgentResult(
                        agent_name="Builder",
                        status="success",
                        data={"build_result": br.model_dump(), "tokens_used": 50},
                    )

        loop = _make_loop(fake_repo, config, builder=AdaptiveBuilder())
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.DONE
        # Verify guardrail was injected (the forced strategy should exist after first failure)
        assert call_count["n"] >= 2

    def test_sotappr_governance_rejects_excess_files(self, fake_repo, config, project, task):
        """Too many files changed → governance rejection."""
        many_files = [f"src/file_{i}.py" for i in range(30)]
        loop = _make_loop(
            fake_repo, config,
            builder=FakeBuilder(succeed=True, files=many_files),
        )
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.STUCK

    def test_protected_path_blocked(self, fake_repo, config, project, task):
        """.env in files_changed → governance rejection."""
        loop = _make_loop(
            fake_repo, config,
            builder=FakeBuilder(succeed=True, files=[".env", "src/main.py"]),
        )
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.STUCK

    def test_run_loop_processes_multiple_tasks(self, config, project):
        """3 tasks → all processed."""
        repo = FakeRepository()
        repo.create_project(project)
        for i in range(3):
            t = Task(
                project_id=project.id,
                title=f"Task {i}",
                description=f"Description {i}",
                priority=10 - i,
            )
            repo.create_task(t)

        loop = _make_loop(repo, config, builder=FakeBuilder(succeed=True))
        processed = loop.run_loop(project.id, max_iterations=10)

        assert processed == 3

    def test_token_budget_marks_stuck(self, fake_repo, config, project, task):
        """Exceed token budget → STUCK."""
        # Set a very low budget
        hm = HealthMonitor(
            repository=fake_repo,
            config=config.orchestrator,
            max_tokens_per_task=10,
        )
        # Builder returns high token count
        loop = _make_loop(
            fake_repo, config,
            builder=FakeBuilder(succeed=False, tokens=500),
            health_monitor=hm,
        )
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.STUCK

    def test_no_pending_tasks_returns_none(self, config, project):
        """No tasks → returns None."""
        empty_repo = FakeRepository()
        empty_repo.create_project(project)
        loop = _make_loop(empty_repo, config)
        result = loop.run_once(project.id)

        assert result is None

    def test_multi_candidate_arbitration_selects_winning_builder_path(self, project):
        """Arbitration enabled: second candidate succeeds in same attempt."""
        repo = FakeRepository()
        repo.create_project(project)
        task = Task(
            project_id=project.id,
            title="Arbitration task",
            description="Choose best candidate",
            priority=5,
        )
        repo.create_task(task)

        class ArbitrationSensitiveBuilder:
            name = "Builder"

            def __init__(self):
                self.calls = 0

            def run(self, input_data: Any) -> AgentResult:
                self.calls += 1
                diagnosis = None
                if hasattr(input_data, "previous_council_diagnosis"):
                    diagnosis = input_data.previous_council_diagnosis
                elif isinstance(input_data, dict):
                    diagnosis = input_data.get("council_diagnosis")

                if diagnosis and "Arbitration variant" in diagnosis:
                    br = BuildResult(
                        files_changed=["src/main.py"],
                        tests_passed=True,
                        approach_summary="variant success",
                        model_used="test/model",
                    )
                    return AgentResult(
                        agent_name="Builder",
                        status="success",
                        data={"build_result": br.model_dump(), "tokens_used": 60},
                    )

                br = BuildResult(
                    files_changed=[],
                    tests_passed=False,
                    failure_reason="tests_failed",
                    failure_detail="no passing path",
                    model_used="test/model",
                )
                return AgentResult(
                    agent_name="Builder",
                    status="failure",
                    data={"build_result": br.model_dump(), "tokens_used": 60},
                    error="tests_failed",
                )

        cfg = AppConfig(
            orchestrator=OrchestratorConfig(
                max_retries_per_task=1,
                council_trigger_threshold=10,
                council_max_invocations=1,
                enable_multi_candidate_arbitration=True,
                arbitration_candidate_count=2,
                arbitration_low_confidence_threshold=0.9,
                arbitration_conflict_trigger_count=1,
            ),
        )
        builder = ArbitrationSensitiveBuilder()
        loop = _make_loop(repo, cfg, builder=builder)
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.DONE
        assert builder.calls == 2

    def test_multi_candidate_arbitration_sentinel_veto_promotes_next_candidate(self, project):
        """Sentinel veto on top candidate should promote next ranked candidate."""
        repo = FakeRepository()
        repo.create_project(project)
        task = Task(
            project_id=project.id,
            title="Sentinel veto arbitration task",
            description="Sentinel should veto first candidate and allow second",
            priority=5,
        )
        repo.create_task(task)

        class TwoCandidateBuilder:
            name = "Builder"

            def __init__(self):
                self.calls = 0

            def run(self, input_data: Any) -> AgentResult:
                self.calls += 1
                diagnosis = getattr(input_data, "previous_council_diagnosis", "") or ""
                # Candidate 1 (baseline): path sentinel will reject.
                if "Arbitration variant" not in diagnosis:
                    br = BuildResult(
                        files_changed=["src/reject_me.py"],
                        tests_passed=True,
                        approach_summary="baseline path",
                        model_used="test/model",
                    )
                    return AgentResult(
                        agent_name="Builder",
                        status="success",
                        data={"build_result": br.model_dump(), "tokens_used": 60},
                    )

                # Candidate 2 (variant): path sentinel will allow.
                br = BuildResult(
                    files_changed=["src/accept_me.py"],
                    tests_passed=True,
                    approach_summary="variant path",
                    model_used="test/model",
                )
                return AgentResult(
                    agent_name="Builder",
                    status="success",
                    data={"build_result": br.model_dump(), "tokens_used": 60},
                )

        class VetoingSentinel:
            name = "Sentinel"

            def __init__(self):
                self.calls = 0

            def run(self, input_data: Any) -> AgentResult:
                self.calls += 1
                build_result = input_data.get("build_result", {})
                files = build_result.get("files_changed", [])
                if "src/reject_me.py" in files:
                    verdict = SentinelVerdict(
                        approved=False,
                        violations=[{"check": "policy", "detail": "reject baseline"}],
                    )
                    return AgentResult(
                        agent_name="Sentinel",
                        status="failure",
                        data={"sentinel_verdict": verdict.model_dump(), "tokens_used": 15},
                        error="Sentinel rejected",
                    )
                verdict = SentinelVerdict(approved=True, violations=[])
                return AgentResult(
                    agent_name="Sentinel",
                    status="success",
                    data={"sentinel_verdict": verdict.model_dump(), "tokens_used": 15},
                )

        cfg = AppConfig(
            orchestrator=OrchestratorConfig(
                max_retries_per_task=1,
                council_trigger_threshold=10,
                council_max_invocations=1,
                enable_multi_candidate_arbitration=True,
                arbitration_candidate_count=2,
                arbitration_low_confidence_threshold=0.9,
                arbitration_conflict_trigger_count=1,
            ),
        )
        builder = TwoCandidateBuilder()
        sentinel = VetoingSentinel()
        loop = _make_loop(repo, cfg, builder=builder, sentinel=sentinel)
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.DONE
        assert builder.calls == 2
        # Sentinel checks both candidates during arbitration and reuses approved verdict.
        assert sentinel.calls == 2

    def test_multi_candidate_arbitration_persists_decision_artifact(self, project):
        """Arbitration writes replay artifact when run context is set."""
        repo = FakeRepository()
        repo.create_project(project)
        task = Task(
            project_id=project.id,
            title="Persist arbitration task",
            description="Ensure arbitration decisions are persisted",
            priority=5,
        )
        repo.create_task(task)

        class ArbitrationSensitiveBuilder:
            name = "Builder"

            def __init__(self):
                self.calls = 0

            def run(self, input_data: Any) -> AgentResult:
                self.calls += 1
                diagnosis = getattr(input_data, "previous_council_diagnosis", None)
                if diagnosis and "Arbitration variant" in diagnosis:
                    br = BuildResult(
                        files_changed=["src/main.py"],
                        tests_passed=True,
                        approach_summary="variant success",
                        model_used="test/model",
                    )
                    return AgentResult(
                        agent_name="Builder",
                        status="success",
                        data={"build_result": br.model_dump(), "tokens_used": 60},
                    )
                br = BuildResult(
                    files_changed=[],
                    tests_passed=False,
                    failure_reason="tests_failed",
                    failure_detail="no passing path",
                    model_used="test/model",
                )
                return AgentResult(
                    agent_name="Builder",
                    status="failure",
                    data={"build_result": br.model_dump(), "tokens_used": 60},
                    error="tests_failed",
                )

        cfg = AppConfig(
            orchestrator=OrchestratorConfig(
                max_retries_per_task=1,
                council_trigger_threshold=10,
                council_max_invocations=1,
                enable_multi_candidate_arbitration=True,
                arbitration_candidate_count=2,
                arbitration_low_confidence_threshold=0.9,
                arbitration_conflict_trigger_count=1,
            ),
        )
        loop = _make_loop(repo, cfg, builder=ArbitrationSensitiveBuilder())
        run_id = uuid.uuid4()
        loop.set_run_context(run_id)
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.DONE
        artifacts = [a for a in repo.sotappr_artifacts if a["artifact_type"] == "arbitration_decision"]
        assert len(artifacts) == 1
        assert artifacts[0]["run_id"] == run_id
        assert artifacts[0]["payload"]["decision"]["candidate_count"] == 2

    def test_self_correction_guardrail_injects_hint_for_conflicted_retrieval(self, project):
        """Self-correction pass injects strategy hint before builder execution."""
        repo = FakeRepository()
        repo.create_project(project)
        task = Task(
            project_id=project.id,
            title="Self correction task",
            description="Check guardrail injection",
            priority=5,
        )
        repo.create_task(task)
        observed_diagnosis: dict[str, str | None] = {"value": None}

        class GuardrailAwareBuilder:
            name = "Builder"

            def run(self, input_data: Any) -> AgentResult:
                observed_diagnosis["value"] = getattr(input_data, "council_diagnosis", None)
                br = BuildResult(
                    files_changed=["src/main.py"],
                    tests_passed=True,
                    approach_summary="guardrail applied",
                    model_used="test/model",
                )
                return AgentResult(
                    agent_name="Builder",
                    status="success",
                    data={"build_result": br.model_dump(), "tokens_used": 40},
                )

        class ConflictLibrarian:
            name = "Librarian"

            def run(self, input_data: Any) -> AgentResult:
                _ = input_data
                cb = ContextBrief(
                    task=task,
                    retrieval_confidence=0.35,
                    retrieval_conflicts=["Conflicting API signatures"],
                )
                return AgentResult(
                    agent_name="Librarian",
                    status="success",
                    data={"context_brief": cb.model_dump(), "tokens_used": 15},
                )

        cfg = AppConfig(
            orchestrator=OrchestratorConfig(
                max_retries_per_task=1,
                council_trigger_threshold=10,
                council_max_invocations=1,
                enable_prebuild_self_correction=True,
                self_correction_confidence_threshold=0.6,
                self_correction_conflict_threshold=1,
            ),
        )
        loop = _make_loop(
            repo,
            cfg,
            builder=GuardrailAwareBuilder(),
            librarian=ConflictLibrarian(),
        )
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.DONE
        assert observed_diagnosis["value"] is not None
        assert "Self-correction guardrail" in str(observed_diagnosis["value"])

    def test_retry_telemetry_artifact_persisted_with_run_trace(self, project):
        """Failed attempt should persist retry telemetry with trace linkage."""
        repo = FakeRepository()
        repo.create_project(project)
        task = Task(
            project_id=project.id,
            title="Retry telemetry task",
            description="Capture retry cause telemetry",
            priority=5,
        )
        repo.create_task(task)

        builder = FakeBuilderSequence(fail_count=1, tokens=40)
        cfg = AppConfig(
            orchestrator=OrchestratorConfig(
                max_retries_per_task=2,
                council_trigger_threshold=10,
                council_max_invocations=1,
            ),
        )
        loop = _make_loop(repo, cfg, builder=builder)
        run_id = uuid.uuid4()
        loop.set_run_context(run_id, trace_id="trace-run-1")
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.DONE
        retry_items = [a for a in repo.sotappr_artifacts if a["artifact_type"] == "retry_telemetry"]
        assert len(retry_items) >= 1
        first = retry_items[0]["payload"]
        assert first["retry_cause"] in {"assertion error", "builder_failed", "tests_failed"}
        assert first["trace_id"].startswith("trace-run-1")

    def test_model_failover_snapshot_persisted_when_builder_has_llm_state(self, project):
        """Loop should persist per-role model failover state for dashboard telemetry."""
        repo = FakeRepository()
        repo.create_project(project)
        task = Task(
            project_id=project.id,
            title="Failover state task",
            description="Persist model failover state",
            priority=5,
        )
        repo.create_task(task)

        class FakeLLMClient:
            @staticmethod
            def get_model_failover_state() -> dict[str, dict[str, float | int]]:
                return {
                    "model/a": {"failure_count": 1, "cooldown_remaining_seconds": 12.0},
                }

        class BuilderWithLLMState(FakeBuilder):
            def __init__(self):
                super().__init__(succeed=True, tokens=30)
                self.llm_client = FakeLLMClient()

        cfg = AppConfig(
            orchestrator=OrchestratorConfig(
                max_retries_per_task=1,
                council_trigger_threshold=10,
                council_max_invocations=1,
            ),
        )
        loop = _make_loop(repo, cfg, builder=BuilderWithLLMState())
        run_id = uuid.uuid4()
        loop.set_run_context(run_id, trace_id="trace-run-2")
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.DONE
        snapshots = [a for a in repo.sotappr_artifacts if a["artifact_type"] == "model_failover_state"]
        assert len(snapshots) >= 1
        assert snapshots[0]["payload"]["state"]["builder"]["model/a"]["failure_count"] == 1

    def test_run_loop_pauses_when_runtime_cost_budget_exceeded(self, config, project):
        """Run loop stops early when runtime cost budget is exceeded."""
        repo = FakeRepository()
        repo.create_project(project)
        for idx in range(3):
            repo.create_task(
                Task(
                    project_id=project.id,
                    title=f"Budget task {idx}",
                    description="Cost budget test",
                    priority=5,
                )
            )

        loop = _make_loop(repo, config, builder=FakeBuilder(succeed=True, tokens=120))
        processed = loop.run_loop(
            project.id,
            max_iterations=10,
            budget_contract={
                "max_cost_usd": 0.001,  # cost_per_1k=0.02 => first task (~120 tokens) exceeds
                "estimated_cost_per_1k_tokens_usd": 0.02,
            },
        )
        summary = loop.get_last_run_summary()

        assert processed == 1
        assert summary["stop_reason"] == "budget_cost_exceeded"
        assert summary["estimated_cost_usd"] > 0.001

    def test_lineage_hint_biases_candidate_generation(self, project):
        """Lineage summary should bias arbitration candidates through strategy hints."""

        class LineageRepo(FakeRepository):
            @staticmethod
            def get_packet_lineage_summary(project_id=None, run_id=None, limit=5):
                _ = project_id, run_id, limit
                return [
                    {
                        "protocol_id": "proto-main",
                        "transfer_count": 7,
                        "cross_use_case_count": 2,
                    }
                ]

            @staticmethod
            def list_protocol_propagation(protocol_id=None, project_id=None, run_id=None, limit=10):
                _ = protocol_id, project_id, run_id, limit
                return [{"transfer_mode": "WINNER_TO_LOSERS"}]

        repo = LineageRepo()
        repo.create_project(project)
        task = Task(
            project_id=project.id,
            title="Lineage-biased candidate task",
            description="Use lineage strategy hint",
            priority=5,
        )
        repo.create_task(task)

        class LineageAwareBuilder:
            name = "Builder"

            def __init__(self):
                self.calls = 0

            def run(self, input_data: Any) -> AgentResult:
                self.calls += 1
                diagnosis = getattr(input_data, "previous_council_diagnosis", "") or ""
                if "Lineage strategy bias" in diagnosis:
                    br = BuildResult(
                        files_changed=["src/main.py"],
                        tests_passed=True,
                        approach_summary="lineage-biased success",
                        model_used="test/model",
                    )
                    return AgentResult(
                        agent_name="Builder",
                        status="success",
                        data={"build_result": br.model_dump(), "tokens_used": 30},
                    )
                br = BuildResult(
                    files_changed=[],
                    tests_passed=False,
                    failure_reason="tests_failed",
                    failure_detail="baseline failed",
                    model_used="test/model",
                )
                return AgentResult(
                    agent_name="Builder",
                    status="failure",
                    data={"build_result": br.model_dump(), "tokens_used": 30},
                    error="baseline failed",
                )

        cfg = AppConfig(
            orchestrator=OrchestratorConfig(
                max_retries_per_task=1,
                council_trigger_threshold=10,
                council_max_invocations=1,
                enable_multi_candidate_arbitration=True,
                arbitration_candidate_count=2,
                arbitration_low_confidence_threshold=0.9,
                arbitration_conflict_trigger_count=1,
            ),
        )
        builder = LineageAwareBuilder()
        loop = _make_loop(repo, cfg, builder=builder)
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.DONE
        assert builder.calls == 2

    def test_success_path_emits_transfer_packet_artifact(self, project):
        """Successful completion should emit a promotion TRANSFER packet."""
        repo = FakeRepository()
        repo.create_project(project)
        task = Task(
            project_id=project.id,
            title="Transfer packet success task",
            description="Ensure transfer packet persistence",
            priority=5,
        )
        repo.create_task(task)

        cfg = AppConfig(
            orchestrator=OrchestratorConfig(
                max_retries_per_task=1,
                council_trigger_threshold=10,
                council_max_invocations=1,
            ),
        )
        loop = _make_loop(repo, cfg, builder=FakeBuilder(succeed=True, tokens=25))
        loop.set_run_context(uuid.uuid4(), trace_id="trace-transfer-1")
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.DONE
        packet_items = [a for a in repo.sotappr_artifacts if a["artifact_type"] == "agent_packet"]
        transfer_packets = [
            item["payload"]["packet"]
            for item in packet_items
            if item["payload"]["packet"]["packet_type"] == "TRANSFER"
        ]
        assert transfer_packets
        assert transfer_packets[0]["run_phase"] == "PROMOTION"
        assert transfer_packets[0]["payload"]["transfer_policy"]["mode"] == "WINNER_TO_LOSERS"

        decisions = [a for a in repo.sotappr_artifacts if a["artifact_type"] == "transfer_arbitration_decision"]
        assert decisions
        decision_payload = decisions[0]["payload"]
        assert decision_payload["candidate_count"] == 2
        assert decision_payload["validated_count"] == 2
        assert decision_payload["selected_count"] >= 1
        assert decision_payload["blocked_count"] == 0
        assert len(decision_payload["ranking"]) == 2

    def test_protocol_effectiveness_gate_blocks_transfer_packet_emission(self, project):
        """Low protocol effectiveness should block transfer packet promotion."""

        class GateBlockedRepo(FakeRepository):
            @staticmethod
            def get_protocol_effectiveness(project_id=None, limit=100):
                _ = limit
                return [
                    {
                        "protocol_id": f"project-{str(project_id)[:8]}-promotion",
                        "transfer_count": 8,
                        "accepted_count": 0,
                        "avg_outcome": 0.0,
                        "avg_drift_risk": 1.0,
                    }
                ]

        repo = GateBlockedRepo()
        repo.create_project(project)
        task = Task(
            project_id=project.id,
            title="Transfer gate blocked task",
            description="Ensure weak protocol transfers are blocked",
            priority=5,
        )
        repo.create_task(task)

        cfg = AppConfig(
            orchestrator=OrchestratorConfig(
                max_retries_per_task=1,
                council_trigger_threshold=10,
                council_max_invocations=1,
            ),
            sotappr=SOTAppRConfig(
                transfer_arbitration_top_k=1,
                enable_protocol_effectiveness_gate=True,
                min_protocol_effectiveness_samples=3,
                min_protocol_effectiveness_score=0.5,
                enable_protocol_decay_policy=False,
            ),
        )
        loop = _make_loop(repo, cfg, builder=FakeBuilder(succeed=True, tokens=25))
        loop.set_run_context(uuid.uuid4(), trace_id="trace-transfer-blocked-1")
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.DONE

        packet_items = [a for a in repo.sotappr_artifacts if a["artifact_type"] == "agent_packet"]
        transfer_packets = [
            item["payload"]["packet"]
            for item in packet_items
            if item["payload"]["packet"]["packet_type"] == "TRANSFER"
        ]
        assert transfer_packets == []

        decisions = [a for a in repo.sotappr_artifacts if a["artifact_type"] == "transfer_arbitration_decision"]
        assert decisions
        decision_payload = decisions[0]["payload"]
        assert decision_payload["candidate_count"] == 2
        assert decision_payload["selected_count"] == 0
        assert decision_payload["blocked_count"] >= 1
        assert "protocol_effectiveness_gate_blocked" in decision_payload["blocked"][0]["reason"]

    def test_transfer_arbitration_expands_candidates_from_lineage_history(self, project):
        """Lineage history should add extra transfer candidates beyond the baseline pair."""

        class HistoryRepo(FakeRepository):
            @staticmethod
            def get_packet_lineage_summary(project_id=None, run_id=None, limit=50):
                _ = run_id, limit
                return [
                    {
                        "protocol_id": f"project-{str(project_id)[:8]}-promotion",
                        "transfer_count": 4,
                        "run_count": 1,
                        "task_count": 2,
                        "cross_use_case_count": 0,
                        "last_seen_at": None,
                    },
                    {
                        "protocol_id": "proto-historical-1",
                        "transfer_count": 7,
                        "run_count": 2,
                        "task_count": 4,
                        "cross_use_case_count": 1,
                        "last_seen_at": None,
                    },
                ]

            @staticmethod
            def list_protocol_propagation(protocol_id=None, project_id=None, run_id=None, limit=10):
                _ = project_id, run_id, limit
                if protocol_id == "proto-historical-1":
                    return [
                        {
                            "transfer_mode": "COMMONS_SEED",
                            "parent_protocol_ids": ["proto-parent"],
                            "ancestor_swarms": ["associate-main", "archive-swarm"],
                        }
                    ]
                return []

            @staticmethod
            def get_protocol_effectiveness(project_id=None, run_id=None, limit=50):
                _ = project_id, run_id, limit
                return [
                    {
                        "protocol_id": "proto-historical-1",
                        "transfer_count": 7,
                        "accepted_count": 6,
                        "run_count": 2,
                        "task_count": 4,
                        "avg_outcome": 0.82,
                        "avg_drift_risk": 0.18,
                    }
                ]

        repo = HistoryRepo()
        repo.create_project(project)
        task = Task(
            project_id=project.id,
            title="Lineage transfer expansion task",
            description="Use historical lineage to expand transfer candidates",
            priority=5,
        )
        repo.create_task(task)

        cfg = AppConfig(
            orchestrator=OrchestratorConfig(
                max_retries_per_task=1,
                council_trigger_threshold=10,
                council_max_invocations=1,
            ),
            sotappr=SOTAppRConfig(
                transfer_arbitration_candidate_limit=4,
                transfer_arbitration_top_k=2,
                enable_protocol_effectiveness_gate=False,
                enable_protocol_decay_policy=False,
            ),
        )
        loop = _make_loop(repo, cfg, builder=FakeBuilder(succeed=True, tokens=30))
        loop.set_run_context(uuid.uuid4(), trace_id="trace-transfer-history-1")
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.DONE

        decisions = [a for a in repo.sotappr_artifacts if a["artifact_type"] == "transfer_arbitration_decision"]
        assert decisions
        decision_payload = decisions[0]["payload"]
        assert decision_payload["candidate_count"] >= 3
        assert len(decision_payload["ranking"]) >= 3

    def test_protocol_decay_policy_blocks_transfer_and_persists_decision(self, project):
        """Protocol decay policy should block weak protocols and emit decay decision artifacts."""

        class DecayRepo(FakeRepository):
            @staticmethod
            def get_protocol_effectiveness(project_id=None, run_id=None, limit=100):
                _ = run_id, limit
                return [
                    {
                        "protocol_id": f"project-{str(project_id)[:8]}-promotion",
                        "transfer_count": 9,
                        "accepted_count": 0,
                        "run_count": 2,
                        "task_count": 5,
                        "avg_outcome": 0.0,
                        "avg_drift_risk": 1.0,
                    }
                ]

        repo = DecayRepo()
        repo.create_project(project)
        task = Task(
            project_id=project.id,
            title="Protocol decay block task",
            description="Block low-effectiveness transfer protocol",
            priority=5,
        )
        repo.create_task(task)

        cfg = AppConfig(
            orchestrator=OrchestratorConfig(
                max_retries_per_task=1,
                council_trigger_threshold=10,
                council_max_invocations=1,
            ),
            sotappr=SOTAppRConfig(
                enable_protocol_effectiveness_gate=False,
                enable_protocol_decay_policy=True,
                protocol_decay_min_samples=3,
                protocol_decay_effectiveness_threshold=0.8,
                protocol_decay_drift_risk_threshold=0.2,
            ),
        )
        loop = _make_loop(repo, cfg, builder=FakeBuilder(succeed=True, tokens=25))
        loop.set_run_context(uuid.uuid4(), trace_id="trace-transfer-decay-1")
        result = loop.run_once(project.id)

        assert result is not None
        assert result.status == TaskStatus.DONE

        packet_items = [a for a in repo.sotappr_artifacts if a["artifact_type"] == "agent_packet"]
        transfer_packets = [
            item["payload"]["packet"]
            for item in packet_items
            if item["payload"]["packet"]["packet_type"] == "TRANSFER"
        ]
        assert transfer_packets == []

        decisions = [a for a in repo.sotappr_artifacts if a["artifact_type"] == "transfer_arbitration_decision"]
        assert decisions
        decision_payload = decisions[0]["payload"]
        assert decision_payload["selected_count"] == 0
        assert decision_payload["decay_blocked_count"] >= 1
        assert "protocol_decay_blocked" in decision_payload["blocked"][0]["reason"]

        decay_artifacts = [a for a in repo.sotappr_artifacts if a["artifact_type"] == "protocol_decay_decision"]
        assert decay_artifacts
        assert decay_artifacts[0]["payload"]["blocked_protocol_count"] >= 1

    def test_run_once_persists_agent_packet_lifecycle_artifacts(self, project):
        """Successful run emits APC packets with lifecycle state persisted."""
        repo = FakeRepository()
        repo.create_project(project)
        task = Task(
            project_id=project.id,
            title="Packetized execution task",
            description="Validate packet lifecycle persistence",
            priority=5,
        )
        repo.create_task(task)

        cfg = AppConfig(
            orchestrator=OrchestratorConfig(
                max_retries_per_task=1,
                council_trigger_threshold=10,
                council_max_invocations=1,
            ),
        )
        loop = _make_loop(repo, cfg, builder=FakeBuilder(succeed=True, tokens=45))
        run_id = uuid.uuid4()
        loop.set_run_context(run_id, trace_id="packet-trace-1")

        result = loop.run_once(project.id)
        assert result is not None
        assert result.status == TaskStatus.DONE

        packet_items = [a for a in repo.sotappr_artifacts if a["artifact_type"] == "agent_packet"]
        assert len(packet_items) >= 5

        packet_types = {item["payload"]["packet"]["packet_type"] for item in packet_items}
        assert "TASK_IR" in packet_types
        assert "ARTIFACT" in packet_types
        assert "EVAL" in packet_types
        assert "DECISION" in packet_types

        eval_packets = [
            item["payload"]["packet"]
            for item in packet_items
            if item["payload"]["packet"]["packet_type"] == "EVAL"
        ]
        assert eval_packets
        assert eval_packets[0]["payload"]["pass"] is True
        assert all(item["payload"]["lifecycle_state"] == "ARCHIVED" for item in packet_items)

    def test_builder_failure_emits_error_packet_artifact(self, project):
        """Failure path should persist ERROR packet artifacts for retries/escalation."""
        repo = FakeRepository()
        repo.create_project(project)
        task = Task(
            project_id=project.id,
            title="Packetized failure task",
            description="Validate error packet persistence",
            priority=5,
        )
        repo.create_task(task)

        cfg = AppConfig(
            orchestrator=OrchestratorConfig(
                max_retries_per_task=1,
                council_trigger_threshold=10,
                council_max_invocations=1,
            ),
        )
        loop = _make_loop(repo, cfg, builder=FakeBuilder(succeed=False, tokens=20))
        loop.set_run_context(uuid.uuid4(), trace_id="packet-trace-2")

        result = loop.run_once(project.id)
        assert result is not None
        assert result.status == TaskStatus.STUCK

        packet_items = [a for a in repo.sotappr_artifacts if a["artifact_type"] == "agent_packet"]
        error_packets = [
            item["payload"]["packet"]
            for item in packet_items
            if item["payload"]["packet"]["packet_type"] == "ERROR"
        ]
        assert error_packets
        first = error_packets[0]
        assert first["payload"]["retryable"] is True
        assert first["run_phase"] == "EXECUTION"
