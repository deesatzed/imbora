"""Tests for src/db/repository.py — data access layer.

Requires a running PostgreSQL instance. Skipped if unavailable.
All tests use real database operations — no mocks.
"""

import uuid

import pytest

from src.core.models import (
    ContextSnapshot,
    HypothesisEntry,
    HypothesisOutcome,
    Methodology,
    PeerReview,
    Project,
    Task,
    TaskStatus,
)
from src.db.repository import Repository

from tests.conftest import requires_postgres


@requires_postgres
class TestProjectCRUD:
    def test_create_and_get_project(self, repository, sample_project):
        created = repository.create_project(sample_project)
        assert created.id == sample_project.id

        fetched = repository.get_project(sample_project.id)
        assert fetched is not None
        assert fetched.name == "test-project"
        assert fetched.repo_path == "/tmp/test-repo"
        assert fetched.tech_stack["language"] == "python"
        assert "flask" in fetched.banned_dependencies

    def test_get_project_not_found(self, repository):
        result = repository.get_project(uuid.uuid4())
        assert result is None


@requires_postgres
class TestTaskCRUD:
    def test_create_and_get_task(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        created = repository.create_task(sample_task)
        assert created.status == TaskStatus.PENDING

        fetched = repository.get_task(sample_task.id)
        assert fetched is not None
        assert fetched.title == "Implement user auth"
        assert fetched.priority == 10

    def test_get_next_task_by_priority(self, repository, sample_project):
        repository.create_project(sample_project)

        low = Task(
            project_id=sample_project.id,
            title="Low priority",
            description="Low",
            priority=1,
        )
        high = Task(
            project_id=sample_project.id,
            title="High priority",
            description="High",
            priority=100,
        )
        repository.create_task(low)
        repository.create_task(high)

        next_task = repository.get_next_task(sample_project.id)
        assert next_task is not None
        assert next_task.title == "High priority"
        assert next_task.priority == 100

    def test_update_task_status(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        repository.update_task_status(sample_task.id, TaskStatus.CODING)
        task = repository.get_task(sample_task.id)
        assert task.status == TaskStatus.CODING

    def test_update_task_status_done_sets_completed_at(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        repository.update_task_status(sample_task.id, TaskStatus.DONE)
        task = repository.get_task(sample_task.id)
        assert task.status == TaskStatus.DONE
        assert task.completed_at is not None

    def test_increment_attempt(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        repository.increment_task_attempt(sample_task.id)
        repository.increment_task_attempt(sample_task.id)
        task = repository.get_task(sample_task.id)
        assert task.attempt_count == 2

    def test_increment_council_count(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        repository.increment_task_council_count(sample_task.id)
        task = repository.get_task(sample_task.id)
        assert task.council_count == 1

    def test_get_tasks_by_status(self, repository, sample_project):
        repository.create_project(sample_project)
        for i in range(3):
            repository.create_task(Task(
                project_id=sample_project.id,
                title=f"Task {i}",
                description=f"Desc {i}",
                priority=i,
            ))

        pending = repository.get_tasks_by_status(sample_project.id, TaskStatus.PENDING)
        assert len(pending) >= 3

    def test_get_in_progress_tasks(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)
        repository.update_task_status(sample_task.id, TaskStatus.CODING)

        in_progress = repository.get_in_progress_tasks()
        coding_ids = [t.id for t in in_progress]
        assert sample_task.id in coding_ids

    def test_get_next_task_none_when_all_done(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)
        repository.update_task_status(sample_task.id, TaskStatus.DONE)

        next_task = repository.get_next_task(sample_project.id)
        assert next_task is None

    def test_get_next_hypothesis_attempt_defaults_to_one(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        next_attempt = repository.get_next_hypothesis_attempt(sample_task.id)
        assert next_attempt == 1

    def test_reset_task_for_retry_reconciles_attempt_counter(
        self, repository, sample_project, sample_task
    ):
        repository.create_project(sample_project)
        repository.create_task(sample_task)
        repository.update_task_status(sample_task.id, TaskStatus.STUCK)

        repository.increment_task_attempt(sample_task.id)
        repository.increment_task_attempt(sample_task.id)
        repository.increment_task_council_count(sample_task.id)

        for i in range(5):
            repository.log_hypothesis(
                HypothesisEntry(
                    task_id=sample_task.id,
                    attempt_number=i + 1,
                    approach_summary=f"attempt-{i + 1}",
                )
            )

        reset = repository.reset_task_for_retry(sample_task.id, status=TaskStatus.PENDING)
        assert reset is not None
        assert reset.status == TaskStatus.PENDING
        assert reset.council_count == 0
        assert reset.attempt_count == 5
        assert repository.get_next_hypothesis_attempt(sample_task.id) == 6


@requires_postgres
class TestHypothesisLog:
    def test_log_and_retrieve(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        entry = HypothesisEntry(
            task_id=sample_task.id,
            attempt_number=1,
            approach_summary="Used regex for parsing",
            outcome=HypothesisOutcome.FAILURE,
            error_signature="TypeError: expected str",
            error_full="Full traceback...",
            files_changed=["parser.py"],
            duration_seconds=30.5,
            model_used="anthropic/claude-sonnet-4",
        )
        repository.log_hypothesis(entry)

        failed = repository.get_failed_approaches(sample_task.id)
        assert len(failed) >= 1
        assert failed[0].approach_summary == "Used regex for parsing"
        assert failed[0].error_signature == "TypeError: expected str"

    def test_hypothesis_count(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        for i in range(3):
            repository.log_hypothesis(HypothesisEntry(
                task_id=sample_task.id,
                attempt_number=i + 1,
                approach_summary=f"Attempt {i + 1}",
            ))

        count = repository.get_hypothesis_count(sample_task.id)
        assert count == 3

    def test_has_duplicate_error(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        repository.log_hypothesis(HypothesisEntry(
            task_id=sample_task.id,
            attempt_number=1,
            approach_summary="First try",
            error_signature="TypeError: expected str",
        ))

        assert repository.has_duplicate_error(sample_task.id, "TypeError: expected str")
        assert not repository.has_duplicate_error(sample_task.id, "ValueError: invalid")

    def test_no_failed_approaches_for_new_task(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        failed = repository.get_failed_approaches(sample_task.id)
        assert failed == []


@requires_postgres
class TestMethodologies:
    def test_save_without_embedding(self, repository):
        meth = Methodology(
            problem_description="Parse JSON from LLM output",
            solution_code="import json\nresult = json.loads(text)",
            methodology_notes="Strip markdown fences first",
            tags=["python", "json"],
            language="python",
        )
        repository.save_methodology(meth)

    def test_save_with_embedding(self, repository):
        embedding = [0.1] * 384
        meth = Methodology(
            problem_description="Handle rate limiting",
            problem_embedding=embedding,
            solution_code="time.sleep(backoff)",
            tags=["api"],
        )
        repository.save_methodology(meth)

    def test_find_similar_methodologies(self, repository):
        # Store a methodology with a known embedding
        embedding = [0.5] * 384
        meth = Methodology(
            problem_description="Parse API response",
            problem_embedding=embedding,
            solution_code="json.loads(resp.text)",
            tags=["api"],
        )
        repository.save_methodology(meth)

        # Search with similar embedding
        query_embedding = [0.5] * 384
        results = repository.find_similar_methodologies(query_embedding, limit=5)
        assert len(results) >= 1
        found_meth, similarity = results[0]
        assert similarity > 0.9  # Same vector should be very similar

    def test_search_methodologies_text(self, repository):
        meth = Methodology(
            problem_description="JWT authentication token validation",
            solution_code="import jwt\njwt.decode(token, key)",
            tags=["auth"],
        )
        repository.save_methodology(meth)

        results = repository.search_methodologies_text("JWT authentication")
        assert len(results) >= 1
        assert "JWT" in results[0].problem_description


@requires_postgres
class TestPeerReviews:
    def test_save_and_retrieve(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        review = PeerReview(
            task_id=sample_task.id,
            model_used="google/gemini-2.5-flash",
            diagnosis="Builder is stuck in regex pattern",
            recommended_approach="Use AST parsing",
            reasoning="Regex cannot handle nested structures",
        )
        repository.save_peer_review(review)

        reviews = repository.get_peer_reviews(sample_task.id)
        assert len(reviews) >= 1
        assert reviews[0].diagnosis == "Builder is stuck in regex pattern"
        assert reviews[0].model_used == "google/gemini-2.5-flash"


@requires_postgres
class TestContextSnapshots:
    def test_save_and_get_latest(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        snap1 = ContextSnapshot(
            task_id=sample_task.id,
            attempt_number=1,
            git_ref="stash@{0}",
            file_manifest={"src/main.py": "abc123"},
        )
        snap2 = ContextSnapshot(
            task_id=sample_task.id,
            attempt_number=2,
            git_ref="stash@{1}",
            file_manifest={"src/main.py": "def456"},
        )
        repository.save_context_snapshot(snap1)
        repository.save_context_snapshot(snap2)

        latest = repository.get_latest_snapshot(sample_task.id)
        assert latest is not None
        assert latest.attempt_number == 2
        assert latest.git_ref == "stash@{1}"

    def test_no_snapshot_returns_none(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        latest = repository.get_latest_snapshot(sample_task.id)
        assert latest is None


@requires_postgres
class TestSOTAppRRepository:
    def test_approve_task_requires_reviewing(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)

        assert repository.approve_task(sample_task.id) is None

        repository.update_task_status(sample_task.id, TaskStatus.REVIEWING)
        approved = repository.approve_task(sample_task.id)
        assert approved is not None
        assert approved.status == TaskStatus.DONE

    def test_task_status_summary_and_review_queue(self, repository, sample_project):
        repository.create_project(sample_project)

        pending = Task(project_id=sample_project.id, title="pending", description="pending")
        reviewing = Task(project_id=sample_project.id, title="review", description="review")
        done = Task(project_id=sample_project.id, title="done", description="done")

        repository.create_task(pending)
        repository.create_task(reviewing)
        repository.create_task(done)
        repository.update_task_status(reviewing.id, TaskStatus.REVIEWING)
        repository.update_task_status(done.id, TaskStatus.REVIEWING)
        repository.update_task_status(done.id, TaskStatus.DONE)

        counts = repository.get_task_status_summary(project_id=sample_project.id)
        assert counts["PENDING"] >= 1
        assert counts["REVIEWING"] >= 1
        assert counts["DONE"] >= 1

        queue = repository.list_review_queue(project_id=sample_project.id, limit=10)
        queue_ids = {task.id for task in queue}
        assert reviewing.id in queue_ids
        assert done.id not in queue_ids

    def test_sotappr_run_artifacts_and_replay_search(self, repository, sample_project):
        repository.create_project(sample_project)
        run_id = repository.create_sotappr_run(
            project_id=sample_project.id,
            mode="execute",
            governance_pack="balanced",
            spec_json={"spec": "v1"},
            report_json={"phase8": {"health_card": {"sota_confidence": 8}}},
            repo_path=sample_project.repo_path,
            status="planned",
        )
        repository.update_sotappr_run(
            run_id,
            status="running",
            tasks_seeded=2,
            tasks_processed=1,
            stop_reason="budget_cost_exceeded",
            estimated_cost_usd=0.11,
            elapsed_hours=0.5,
        )

        repository.save_sotappr_artifact(
            run_id=run_id,
            phase=8,
            artifact_type="experience_replay",
            payload={
                "context": "Import failure in CI",
                "options_considered": "retry, pin dep",
                "reasoning": "pinning was safer",
                "what_we_learned": "lockfile drift",
                "transferable_insight": "always pin",
            },
        )

        run = repository.get_sotappr_run(run_id)
        assert run is not None
        assert run["status"] == "running"
        assert int(run["tasks_seeded"]) == 2
        assert run["stop_reason"] == "budget_cost_exceeded"
        assert float(run["estimated_cost_usd"]) == 0.11
        assert float(run["elapsed_hours"]) == 0.5

        artifacts = repository.list_sotappr_artifacts(run_id=run_id)
        assert len(artifacts) >= 1

        replay = repository.search_sotappr_replay(query="pin", limit=10)
        assert len(replay) >= 1
        assert str(replay[0]["run_id"]) == str(run_id)

    def test_save_and_list_agent_packets_with_events(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)
        run_id = repository.create_sotappr_run(
            project_id=sample_project.id,
            mode="execute",
            governance_pack="balanced",
            spec_json={"spec": "v1"},
            report_json={"phase8": {}},
            repo_path=sample_project.repo_path,
            status="running",
        )

        packet_id = uuid.uuid4()
        packet = {
            "protocol_version": "apc/1.0",
            "packet_id": str(packet_id),
            "packet_type": "SIGNAL",
            "channel": "INTER_AGENT",
            "run_phase": "DISCOVERY",
            "sender": {"agent_id": "orchestrator-1", "role": "orchestrator", "swarm_id": "assoc"},
            "recipients": [{"agent_id": "builder-1", "role": "builder", "swarm_id": "assoc"}],
            "trace": {
                "session_id": "sess-1",
                "run_id": str(run_id),
                "task_id": str(sample_task.id),
                "root_packet_id": str(packet_id),
                "generation": 1,
                "step": 1,
            },
            "routing": {"delivery_mode": "DIRECT", "priority": 5, "ttl_ms": 60000, "requires_ack": False},
            "confidence": 0.8,
            "symbolic_keys": ["attempt"],
            "payload": {"signal": "ATTEMPT_STARTED", "reason": "test"},
        }

        packet_row_id = repository.save_agent_packet(
            run_id=run_id,
            task_id=sample_task.id,
            attempt_number=1,
            trace_id="trace-1",
            packet=packet,
            packet_hash="hash-123",
            lifecycle_state="ARCHIVED",
            lifecycle_history=[
                {"event": "emit", "from": "DRAFT", "to": "NORMALIZED", "at": "2026-02-19T00:00:00+00:00"},
                {"event": "schema_pass", "from": "NORMALIZED", "to": "SCHEMA_VALIDATED"},
            ],
        )
        assert packet_row_id is not None

        rows = repository.list_agent_packets(run_id=run_id, packet_type="SIGNAL", limit=10)
        assert len(rows) == 1
        assert str(rows[0]["packet_id"]) == str(packet_id)
        assert rows[0]["lifecycle_state"] == "ARCHIVED"

        events = repository.list_packet_events(run_id=run_id, packet_id=packet_id, limit=20)
        assert len(events) >= 2
        names = {row["event"] for row in events}
        assert "emit" in names
        assert "schema_pass" in names

    def test_get_packet_timeline(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)
        run_id = repository.create_sotappr_run(
            project_id=sample_project.id,
            mode="execute",
            governance_pack="balanced",
            spec_json={"spec": "v1"},
            report_json={"phase8": {}},
            repo_path=sample_project.repo_path,
            status="running",
        )
        packet_id = uuid.uuid4()
        packet = {
            "protocol_version": "apc/1.0",
            "packet_id": str(packet_id),
            "packet_type": "TASK_IR",
            "channel": "INTER_AGENT",
            "run_phase": "DISCOVERY",
            "sender": {"agent_id": "state-manager-1", "role": "state_manager", "swarm_id": "assoc"},
            "recipients": [{"agent_id": "builder-1", "role": "builder", "swarm_id": "assoc"}],
            "trace": {
                "session_id": "sess-2",
                "run_id": str(run_id),
                "task_id": str(sample_task.id),
                "root_packet_id": str(packet_id),
                "generation": 1,
                "step": 2,
            },
            "routing": {"delivery_mode": "DIRECT", "priority": 5, "ttl_ms": 60000, "requires_ack": False},
            "confidence": 0.9,
            "symbolic_keys": ["task_ir"],
            "payload": {"ir_header": {"task_id": str(sample_task.id)}, "objects": []},
        }
        repository.save_agent_packet(
            run_id=run_id,
            task_id=sample_task.id,
            attempt_number=1,
            trace_id="trace-2",
            packet=packet,
            lifecycle_state="ARCHIVED",
            lifecycle_history=[{"event": "emit", "from": "DRAFT", "to": "NORMALIZED"}],
        )

        timeline = repository.get_packet_timeline(run_id=run_id, task_id=sample_task.id, limit=20)
        assert len(timeline) >= 1
        assert str(timeline[0]["packet_id"]) == str(packet_id)

    def test_packet_lineage_summary_and_propagation(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)
        run_id = repository.create_sotappr_run(
            project_id=sample_project.id,
            mode="execute",
            governance_pack="balanced",
            spec_json={"spec": "v1"},
            report_json={"phase8": {}},
            repo_path=sample_project.repo_path,
            status="running",
        )

        packet_id = uuid.uuid4()
        packet = {
            "protocol_version": "apc/1.0",
            "packet_id": str(packet_id),
            "packet_type": "TRANSFER",
            "channel": "INTER_AGENT",
            "run_phase": "PROMOTION",
            "sender": {"agent_id": "arbiter-1", "role": "arbiter", "swarm_id": "assoc"},
            "recipients": [{"agent_id": "builder-1", "role": "builder", "swarm_id": "assoc"}],
            "trace": {
                "session_id": "sess-3",
                "run_id": str(run_id),
                "task_id": str(sample_task.id),
                "root_packet_id": str(packet_id),
                "generation": 25,
                "step": 3,
            },
            "routing": {"delivery_mode": "DIRECT", "priority": 5, "ttl_ms": 60000, "requires_ack": False},
            "confidence": 0.95,
            "symbolic_keys": ["transfer"],
            "lineage": {
                "protocol_id": "proto-1",
                "parent_protocol_ids": ["proto-0"],
                "ancestor_swarms": ["swarm-a"],
                "cross_use_case": True,
                "transfer_mode": "WINNER_TO_LOSERS",
            },
            "payload": {
                "protocol_id": "proto-1",
                "from_swarm": "swarm-a",
                "to_swarm": "swarm-b",
                "sender_score": 1.5,
                "receiver_score": 1.0,
                "transfer_policy": {"mode": "WINNER_TO_LOSERS", "top_k": 2},
                "accepted": True,
            },
        }

        repository.save_agent_packet(
            run_id=run_id,
            task_id=sample_task.id,
            attempt_number=1,
            trace_id="trace-l-1",
            packet=packet,
            lifecycle_state="ARCHIVED",
            lifecycle_history=[{"event": "emit", "from": "DRAFT", "to": "NORMALIZED"}],
        )

        summary = repository.get_packet_lineage_summary(
            project_id=sample_project.id,
            run_id=run_id,
            limit=10,
        )
        assert len(summary) >= 1
        assert summary[0]["protocol_id"] == "proto-1"
        assert int(summary[0]["transfer_count"]) >= 1

        propagation = repository.list_protocol_propagation(
            protocol_id="proto-1",
            project_id=sample_project.id,
            run_id=run_id,
            limit=10,
        )
        assert len(propagation) >= 1
        first = propagation[0]
        assert first["protocol_id"] == "proto-1"
        assert bool(first["cross_use_case"]) is True

    def test_transfer_quality_rollup_and_eval_correlation(self, repository, sample_project, sample_task):
        repository.create_project(sample_project)
        repository.create_task(sample_task)
        task_b = Task(
            project_id=sample_project.id,
            title="task-b",
            description="d",
            priority=1,
        )
        repository.create_task(task_b)
        run_id = repository.create_sotappr_run(
            project_id=sample_project.id,
            mode="execute",
            governance_pack="balanced",
            spec_json={"spec": "v1"},
            report_json={"phase8": {}},
            repo_path=sample_project.repo_path,
            status="running",
        )

        transfer_a = uuid.uuid4()
        repository.save_agent_packet(
            run_id=run_id,
            task_id=sample_task.id,
            attempt_number=1,
            trace_id="trace-q-1",
            packet={
                "protocol_version": "apc/1.0",
                "packet_id": str(transfer_a),
                "packet_type": "TRANSFER",
                "channel": "INTER_AGENT",
                "run_phase": "PROMOTION",
                "sender": {"agent_id": "arbiter-1", "role": "arbiter", "swarm_id": "assoc"},
                "recipients": [{"agent_id": "builder-1"}],
                "trace": {
                    "session_id": "sess-q",
                    "run_id": str(run_id),
                    "task_id": str(sample_task.id),
                    "root_packet_id": str(transfer_a),
                    "generation": 25,
                    "step": 1,
                },
                "routing": {"delivery_mode": "DIRECT", "priority": 5, "ttl_ms": 60000, "requires_ack": False},
                "confidence": 0.9,
                "lineage": {
                    "protocol_id": "proto-q",
                    "parent_protocol_ids": [],
                    "ancestor_swarms": ["swarm-a"],
                    "cross_use_case": True,
                    "transfer_mode": "WINNER_TO_LOSERS",
                },
                "payload": {
                    "protocol_id": "proto-q",
                    "from_swarm": "swarm-a",
                    "to_swarm": "swarm-b",
                    "sender_score": 0.9,
                    "receiver_score": 0.7,
                    "transfer_policy": {"mode": "WINNER_TO_LOSERS", "top_k": 2},
                    "accepted": True,
                },
            },
            lifecycle_state="ARCHIVED",
            lifecycle_history=[{"event": "emit", "from": "DRAFT", "to": "NORMALIZED"}],
        )

        transfer_b = uuid.uuid4()
        repository.save_agent_packet(
            run_id=run_id,
            task_id=task_b.id,
            attempt_number=1,
            trace_id="trace-q-2",
            packet={
                "protocol_version": "apc/1.0",
                "packet_id": str(transfer_b),
                "packet_type": "TRANSFER",
                "channel": "INTER_AGENT",
                "run_phase": "PROMOTION",
                "sender": {"agent_id": "arbiter-1", "role": "arbiter", "swarm_id": "assoc"},
                "recipients": [{"agent_id": "builder-2"}],
                "trace": {
                    "session_id": "sess-q",
                    "run_id": str(run_id),
                    "task_id": str(task_b.id),
                    "root_packet_id": str(transfer_b),
                    "generation": 25,
                    "step": 2,
                },
                "routing": {"delivery_mode": "DIRECT", "priority": 5, "ttl_ms": 60000, "requires_ack": False},
                "confidence": 0.8,
                "lineage": {
                    "protocol_id": "proto-q",
                    "parent_protocol_ids": [],
                    "ancestor_swarms": ["swarm-a"],
                    "cross_use_case": False,
                    "transfer_mode": "COMMONS_SEED",
                },
                "payload": {
                    "protocol_id": "proto-q",
                    "from_swarm": "swarm-a",
                    "to_swarm": "swarm-c",
                    "sender_score": 0.85,
                    "receiver_score": 0.7,
                    "transfer_policy": {"mode": "COMMONS_SEED", "top_k": 2},
                    "accepted": False,
                },
            },
            lifecycle_state="ARCHIVED",
            lifecycle_history=[{"event": "emit", "from": "DRAFT", "to": "NORMALIZED"}],
        )

        for task_id, trace_id, outcome, drift in [
            (sample_task.id, "trace-q-1", 0.82, 0.11),
            (task_b.id, "trace-q-2", 0.61, 0.26),
        ]:
            eval_id = uuid.uuid4()
            repository.save_agent_packet(
                run_id=run_id,
                task_id=task_id,
                attempt_number=1,
                trace_id=trace_id,
                packet={
                    "protocol_version": "apc/1.0",
                    "packet_id": str(eval_id),
                    "packet_type": "EVAL",
                    "channel": "INTER_AGENT",
                    "run_phase": "VALIDATION",
                    "sender": {"agent_id": "sentinel-1", "role": "sentinel", "swarm_id": "assoc"},
                    "recipients": [{"agent_id": "builder-1"}],
                    "trace": {
                        "session_id": "sess-q",
                        "run_id": str(run_id),
                        "task_id": str(task_id),
                        "root_packet_id": str(eval_id),
                        "generation": 25,
                        "step": 9,
                    },
                    "routing": {"delivery_mode": "DIRECT", "priority": 5, "ttl_ms": 60000, "requires_ack": False},
                    "confidence": 0.95,
                    "proof_bundle": {"evidence_refs": [], "falsifiers": [], "gate_scores": {}, "signatures": []},
                    "payload": {
                        "metric_set": {
                            "rtf": 1.0,
                            "stability": 1.0,
                            "audit": 1.0,
                            "outcome": outcome,
                            "learnability": 1.0,
                            "generalization_ratio": 1.0,
                            "drift_risk": drift,
                        },
                        "thresholds": {
                            "rtf": 1.0,
                            "stability": 0.9,
                            "audit": 1.0,
                            "outcome": 0.5,
                            "learnability": 0.7,
                            "generalization_ratio": 0.8,
                        },
                        "pass": True,
                        "notes": [],
                    },
                },
                lifecycle_state="ARCHIVED",
                lifecycle_history=[{"event": "emit", "from": "DRAFT", "to": "NORMALIZED"}],
            )

        quality = repository.get_transfer_quality_rollup(
            project_id=sample_project.id,
            run_id=run_id,
        )
        assert len(quality) >= 2
        quality_modes = {row["transfer_mode"] for row in quality}
        assert "WINNER_TO_LOSERS" in quality_modes
        assert "COMMONS_SEED" in quality_modes

        corr = repository.get_transfer_eval_correlation(
            project_id=sample_project.id,
            run_id=run_id,
        )
        assert len(corr) >= 1
        groups = {bool(row["has_cross_use_case_transfer"]) for row in corr}
        assert True in groups

        effectiveness = repository.get_protocol_effectiveness(
            project_id=sample_project.id,
            run_id=run_id,
            limit=10,
        )
        assert len(effectiveness) >= 1
        assert effectiveness[0]["protocol_id"] == "proto-q"
        assert int(effectiveness[0]["transfer_count"]) >= 1
