"""Tests for src/orchestrator/arbitration.py."""

from __future__ import annotations

from src.core.models import AgentResult, BuildResult
from src.orchestrator.arbitration import BuildArbiter


def _candidate(
    *,
    status: str,
    tests_passed: bool,
    files_changed: list[str] | None = None,
    failure_reason: str | None = None,
) -> tuple[AgentResult, BuildResult]:
    build = BuildResult(
        files_changed=files_changed or [],
        tests_passed=tests_passed,
        failure_reason=failure_reason,
    )
    result = AgentResult(
        agent_name="Builder",
        status=status,
        data={"build_result": build.model_dump()},
        error=failure_reason,
    )
    return result, build


class TestBuildArbiter:
    def test_choose_prefers_passing_candidate(self):
        arbiter = BuildArbiter()
        c1 = _candidate(status="failure", tests_passed=False, files_changed=[], failure_reason="tests_failed")
        c2 = _candidate(status="success", tests_passed=True, files_changed=["src/main.py"])

        decision = arbiter.choose([c1, c2], retrieval_confidence=0.4)
        assert decision.selected_index == 2
        assert decision.selected_build_result.tests_passed is True

    def test_choose_penalizes_no_artifacts(self):
        arbiter = BuildArbiter()
        c1 = _candidate(status="success", tests_passed=False, files_changed=[])
        c2 = _candidate(status="success", tests_passed=False, files_changed=["src/a.py"])

        decision = arbiter.choose([c1, c2], retrieval_confidence=0.4)
        assert decision.selected_index == 2

    def test_to_dict_includes_score_breakdown(self):
        arbiter = BuildArbiter()
        c1 = _candidate(status="success", tests_passed=True, files_changed=["src/main.py"])
        c2 = _candidate(status="success", tests_passed=False, files_changed=["src/alt.py"])
        decision = arbiter.choose([c1, c2], retrieval_confidence=0.8)
        data = decision.to_dict()

        assert data["selected_index"] == 1
        assert len(data["scores"]) == 2
        assert "notes" in data["scores"][0]
        assert "ranked_indices" in data
        assert "vetoed_candidates" in data

    def test_choose_honors_vetoes(self):
        arbiter = BuildArbiter()
        c1 = _candidate(status="success", tests_passed=True, files_changed=["src/main.py"])
        c2 = _candidate(status="success", tests_passed=True, files_changed=["src/alt.py"])

        decision = arbiter.choose(
            [c1, c2],
            retrieval_confidence=0.8,
            vetoes={1: ["sentinel_veto:policy_violation"]},
        )
        assert decision.selected_index == 2
        assert decision.vetoed_candidates[0]["candidate_index"] == 1

    def test_tie_break_reason_when_scores_match(self):
        arbiter = BuildArbiter()
        c1 = _candidate(status="success", tests_passed=True, files_changed=["src/a.py"])
        c2 = _candidate(status="success", tests_passed=True, files_changed=["src/b.py"])

        decision = arbiter.choose([c1, c2], retrieval_confidence=0.8)
        assert decision.tie_break_reason == "stable_candidate_order_preference"
