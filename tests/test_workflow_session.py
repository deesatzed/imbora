"""Tests for guided workflow session artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from src.workflow.session import (
    AlignmentSpec,
    ExecutionCharter,
    IntakeCard,
    SessionPlan,
    build_runtime_manifest,
    estimate_budget,
    materialize_knowledge_items,
    write_session_artifacts,
)


class TestEstimateBudget:
    def test_estimate_budget_increases_with_scope(self):
        small = estimate_budget(task="fix bug", feature_count=1, knowledge_count=0)
        large = estimate_budget(
            task="build a new full-stack app with realtime collaboration and auth",
            feature_count=5,
            knowledge_count=4,
        )
        assert large.estimated_hours > small.estimated_hours
        assert large.estimated_cost_usd >= small.estimated_cost_usd
        assert large.max_hours == large.estimated_hours
        assert large.max_cost_usd == large.estimated_cost_usd


class TestMaterializeKnowledge:
    def test_materialize_knowledge_items_copies_local_and_urls(self, tmp_path):
        source_file = tmp_path / "source.txt"
        source_file.write_text("hello", encoding="utf-8")
        out_dir = tmp_path / "knowledge"

        manifest = materialize_knowledge_items(
            knowledge_dir=out_dir,
            local_paths=[source_file],
            urls=["https://example.com/docs"],
            query_results={"jwt auth": [{"title": "JWT", "url": "https://x", "snippet": "s"}]},
        )

        assert len(manifest.items) == 3
        saved_local = [item for item in manifest.items if item.source_type == "local"][0]
        assert saved_local.saved_path is not None
        assert Path(saved_local.saved_path).exists()

        url_file = out_dir / "external_urls.json"
        assert url_file.exists()
        query_file = out_dir / "queries" / "jwt_auth.json"
        assert query_file.exists()


class TestWriteSessionArtifacts:
    def test_write_session_artifacts_writes_all_gate_files(self, tmp_path):
        budget = estimate_budget("task", feature_count=1, knowledge_count=0)
        budget.approved = True
        plan = SessionPlan(
            session_id="session_test",
            intake=IntakeCard(
                session_id="session_test",
                task="Implement auth",
                target_path="/tmp/project",
                project_mode="new",
            ),
            alignment=AlignmentSpec(
                ux_expectations=["clear UX"],
                features=["auth"],
                methodology="test-first",
                acceptance_criteria=["tests pass"],
                approved=True,
            ),
            budget=budget,
            knowledge=materialize_knowledge_items(
                knowledge_dir=tmp_path / "knowledge",
                local_paths=[],
                urls=[],
                query_results={},
            ),
            charter=ExecutionCharter(
                anticipated_end_goal="Ship auth",
                execution_summary="Build and verify",
                approved=True,
            ),
            runtime=build_runtime_manifest(
                model_roles={"builder": "model/a"},
                prompts_dir=tmp_path,
                knowledge_manifest=materialize_knowledge_items(
                    knowledge_dir=tmp_path / "k2",
                    local_paths=[],
                    urls=[],
                    query_results={},
                ),
            ),
            status="approved",
        )

        outputs = write_session_artifacts(plan=plan, session_root=tmp_path / "session")
        assert outputs["session"].exists()
        payload = json.loads(outputs["session"].read_text(encoding="utf-8"))
        assert payload["status"] == "approved"
        assert payload["alignment"]["approved"] is True
