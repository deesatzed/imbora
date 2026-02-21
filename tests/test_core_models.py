"""Tests for src/core/models.py — Pydantic data models."""

import uuid
from datetime import datetime

from src.core.models import (
    AgentResult,
    BuildResult,
    ContextBrief,
    ContextSnapshot,
    CouncilDiagnosis,
    HypothesisEntry,
    HypothesisOutcome,
    Methodology,
    PeerReview,
    Project,
    ResearchBrief,
    SentinelVerdict,
    Task,
    TaskContext,
    TaskStatus,
)


class TestEnums:
    def test_task_status_values(self):
        assert TaskStatus.PENDING == "PENDING"
        assert TaskStatus.RESEARCHING == "RESEARCHING"
        assert TaskStatus.CODING == "CODING"
        assert TaskStatus.REVIEWING == "REVIEWING"
        assert TaskStatus.STUCK == "STUCK"
        assert TaskStatus.DONE == "DONE"

    def test_hypothesis_outcome_values(self):
        assert HypothesisOutcome.SUCCESS == "SUCCESS"
        assert HypothesisOutcome.FAILURE == "FAILURE"

    def test_task_status_from_string(self):
        assert TaskStatus("PENDING") == TaskStatus.PENDING
        assert TaskStatus("DONE") == TaskStatus.DONE


class TestProject:
    def test_defaults(self):
        p = Project(name="test", repo_path="/tmp/repo")
        assert isinstance(p.id, uuid.UUID)
        assert p.tech_stack == {}
        assert p.banned_dependencies == []
        assert p.project_rules is None
        assert isinstance(p.created_at, datetime)

    def test_full_init(self):
        p = Project(
            name="my-project",
            repo_path="/home/user/project",
            tech_stack={"lang": "python"},
            project_rules="no wildcards",
            banned_dependencies=["flask"],
        )
        assert p.name == "my-project"
        assert p.banned_dependencies == ["flask"]
        assert p.tech_stack["lang"] == "python"


class TestTask:
    def test_defaults(self):
        pid = uuid.uuid4()
        t = Task(project_id=pid, title="Fix bug", description="Fix the auth bug")
        assert t.status == TaskStatus.PENDING
        assert t.priority == 0
        assert t.attempt_count == 0
        assert t.council_count == 0
        assert t.completed_at is None

    def test_full_init(self):
        pid = uuid.uuid4()
        t = Task(
            project_id=pid,
            title="Add feature",
            description="Add dark mode",
            status=TaskStatus.CODING,
            priority=5,
            attempt_count=2,
        )
        assert t.status == TaskStatus.CODING
        assert t.priority == 5
        assert t.attempt_count == 2


class TestHypothesisEntry:
    def test_defaults(self):
        tid = uuid.uuid4()
        h = HypothesisEntry(
            task_id=tid,
            attempt_number=1,
            approach_summary="Used regex parsing",
        )
        assert h.outcome == HypothesisOutcome.FAILURE
        assert h.error_signature is None
        assert h.files_changed == []

    def test_with_error(self):
        tid = uuid.uuid4()
        h = HypothesisEntry(
            task_id=tid,
            attempt_number=2,
            approach_summary="Used AST parsing",
            outcome=HypothesisOutcome.FAILURE,
            error_signature="TypeError: expected str",
            error_full="Traceback: ...",
            files_changed=["parser.py", "utils.py"],
            duration_seconds=45.2,
            model_used="anthropic/claude-sonnet-4",
        )
        assert h.error_signature == "TypeError: expected str"
        assert len(h.files_changed) == 2
        assert h.duration_seconds == 45.2


class TestMethodology:
    def test_defaults(self):
        m = Methodology(
            problem_description="Parse JSON from LLM",
            solution_code="import json\njson.loads(text)",
        )
        assert m.problem_embedding is None
        assert m.tags == []
        assert m.language is None

    def test_with_embedding(self):
        embedding = [0.1] * 384
        m = Methodology(
            problem_description="Test",
            problem_embedding=embedding,
            solution_code="pass",
            tags=["python", "parsing"],
            language="python",
        )
        assert len(m.problem_embedding) == 384
        assert m.tags == ["python", "parsing"]


class TestPeerReview:
    def test_creation(self):
        tid = uuid.uuid4()
        pr = PeerReview(
            task_id=tid,
            model_used="google/gemini-2.5-flash",
            diagnosis="Builder is stuck in a regex loop",
            recommended_approach="Use a proper parser library",
            reasoning="Regex can't handle nested structures",
        )
        assert pr.model_used == "google/gemini-2.5-flash"
        assert "regex" in pr.diagnosis


class TestContextSnapshot:
    def test_creation(self):
        tid = uuid.uuid4()
        cs = ContextSnapshot(
            task_id=tid,
            attempt_number=1,
            git_ref="stash@{0}",
            file_manifest={"src/main.py": "abc123"},
        )
        assert cs.git_ref == "stash@{0}"
        assert cs.file_manifest["src/main.py"] == "abc123"


class TestInterAgentModels:
    def test_agent_result(self):
        ar = AgentResult(
            agent_name="Builder",
            status="success",
            data={"files_changed": ["main.py"]},
        )
        assert ar.error is None
        assert ar.duration_seconds == 0.0

    def test_agent_result_failure(self):
        ar = AgentResult(
            agent_name="Sentinel",
            status="failure",
            error="Dependency jail violation",
        )
        assert ar.status == "failure"
        assert "jail" in ar.error

    def test_task_context(self):
        pid = uuid.uuid4()
        task = Task(project_id=pid, title="Test", description="Test task")
        tc = TaskContext(
            task=task,
            forbidden_approaches=["Used regex", "Used string split"],
            checkpoint_ref="stash@{0}",
        )
        assert len(tc.forbidden_approaches) == 2
        assert tc.previous_council_diagnosis is None

    def test_research_brief(self):
        rb = ResearchBrief(
            live_docs=[{"title": "FastAPI Docs", "url": "https://fastapi.tiangolo.com", "snippet": "..."}],
            api_signatures=["def get(path: str)"],
        )
        assert len(rb.live_docs) == 1
        assert rb.version_warnings == []

    def test_context_brief(self):
        pid = uuid.uuid4()
        task = Task(project_id=pid, title="Test", description="Test")
        cb = ContextBrief(
            task=task,
            forbidden_approaches=["regex approach"],
            project_rules="no wildcards",
        )
        assert cb.past_solutions == []
        assert cb.council_diagnosis is None
        assert cb.retrieval_confidence == 0.0
        assert cb.retrieval_conflicts == []
        assert cb.retrieval_strategy_hint is None

    def test_build_result(self):
        br = BuildResult(
            files_changed=["auth.py"],
            test_output="OK",
            tests_passed=True,
            diff="+ def login():",
            approach_summary="Added JWT auth",
        )
        assert br.tests_passed is True
        assert len(br.files_changed) == 1
        assert br.failure_reason is None
        assert br.failure_detail is None

    def test_sentinel_verdict_approved(self):
        sv = SentinelVerdict(approved=True)
        assert sv.violations == []

    def test_sentinel_verdict_rejected(self):
        sv = SentinelVerdict(
            approved=False,
            violations=[{"check": "dependency_jail", "detail": "flask is banned"}],
            recommendations=["Use fastapi instead"],
        )
        assert not sv.approved
        assert len(sv.violations) == 1

    def test_council_diagnosis(self):
        cd = CouncilDiagnosis(
            strategy_shift="Switch from regex to AST",
            new_approach="Use ast.parse() for code analysis",
            reasoning="Regex fails on nested structures",
            model_used="google/gemini-2.5-flash",
        )
        assert "AST" in cd.strategy_shift
