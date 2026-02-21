"""Tests for src/orchestrator/wrapup.py — WrapUp salvage workflow."""

import uuid
from types import SimpleNamespace

from src.core.models import AgentResult, BuildResult, Task
from src.orchestrator.wrapup import WrapUpWorkflow


def _make_task(title: str = "Test task", description: str = "A test") -> Task:
    """Create a real Task for testing."""
    return Task(
        project_id=uuid.uuid4(),
        title=title,
        description=description,
    )


def _make_build_result(
    files: list[str] | None = None,
    approach: str = "Tried approach X",
    test_output: str = "FAILED: 2 tests",
) -> BuildResult:
    """Create a real BuildResult for testing."""
    return BuildResult(
        files_changed=files or [],
        test_output=test_output,
        tests_passed=False,
        approach_summary=approach,
    )


# ---------------------------------------------------------------------------
# No partial files -> not salvaged
# ---------------------------------------------------------------------------

def test_no_files_returns_not_salvaged():
    wf = WrapUpWorkflow()
    task = _make_task()
    build = _make_build_result(files=[])

    result = wf.attempt_salvage(task, build)
    assert result["salvaged"] is False
    assert result["files"] == []
    assert result["quality_score"] == 0.0
    assert result["reason"] == "no_partial_output"


def test_empty_files_list_returns_not_salvaged():
    wf = WrapUpWorkflow()
    task = _make_task()
    build = BuildResult()  # files_changed defaults to []

    result = wf.attempt_salvage(task, build)
    assert result["salvaged"] is False
    assert result["reason"] == "no_partial_output"


# ---------------------------------------------------------------------------
# Files exist, no sentinel -> accepts partial output
# ---------------------------------------------------------------------------

def test_files_exist_no_sentinel_accepts_partial():
    wf = WrapUpWorkflow(council=None, sentinel=None)
    task = _make_task()
    build = _make_build_result(files=["src/auth.py", "tests/test_auth.py"])

    result = wf.attempt_salvage(task, build)
    assert result["salvaged"] is True
    assert result["files"] == ["src/auth.py", "tests/test_auth.py"]
    assert result["reason"] == "no_sentinel_available_accepting_partial"


def test_files_exist_no_sentinel_no_council():
    wf = WrapUpWorkflow()
    task = _make_task()
    build = _make_build_result(files=["main.py"])

    result = wf.attempt_salvage(task, build)
    assert result["salvaged"] is True
    assert result["files"] == ["main.py"]
    assert result["reason"] == "no_sentinel_available_accepting_partial"


# ---------------------------------------------------------------------------
# Council recommends discard
# ---------------------------------------------------------------------------

def test_council_recommends_discard():
    council_result = AgentResult(
        agent_name="Council",
        status="success",
        data={
            "council_diagnosis": {
                "new_approach": "Discard everything and start fresh",
                "strategy_shift": "complete restart",
                "reasoning": "code is unsalvageable",
            }
        },
    )

    class FakeCouncil:
        def run(self, input_data):
            return council_result

    wf = WrapUpWorkflow(council=FakeCouncil(), sentinel=None)
    task = _make_task()
    build = _make_build_result(files=["broken.py"])

    result = wf.attempt_salvage(task, build)
    assert result["salvaged"] is False
    assert result["reason"] == "council_recommends_discard"
    assert result["files"] == ["broken.py"]


def test_council_does_not_recommend_discard():
    council_result = AgentResult(
        agent_name="Council",
        status="success",
        data={
            "council_diagnosis": {
                "new_approach": "Keep partial work and refine",
                "strategy_shift": "incremental fix",
                "reasoning": "some code is usable",
            }
        },
    )

    class FakeCouncil:
        def run(self, input_data):
            return council_result

    # No sentinel, so it should accept partial output after council says keep
    wf = WrapUpWorkflow(council=FakeCouncil(), sentinel=None)
    task = _make_task()
    build = _make_build_result(files=["partial.py"])

    result = wf.attempt_salvage(task, build)
    assert result["salvaged"] is True
    assert result["reason"] == "no_sentinel_available_accepting_partial"


def test_council_failure_continues_to_sentinel():
    class FailingCouncil:
        def run(self, input_data):
            raise RuntimeError("Council model unavailable")

    sentinel_result = AgentResult(
        agent_name="Sentinel",
        status="success",
        data={"sentinel_verdict": {"quality_score": 0.60}},
    )

    class FakeSentinel:
        def run(self, input_data):
            return sentinel_result

    wf = WrapUpWorkflow(council=FailingCouncil(), sentinel=FakeSentinel())
    task = _make_task()
    build = _make_build_result(files=["partial.py"])

    result = wf.attempt_salvage(task, build)
    assert result["salvaged"] is True
    assert result["quality_score"] == 0.60


def test_council_returns_failure_status_continues():
    council_result = AgentResult(
        agent_name="Council",
        status="failure",
        error="Model timeout",
    )

    class FakeCouncil:
        def run(self, input_data):
            return council_result

    wf = WrapUpWorkflow(council=FakeCouncil(), sentinel=None)
    task = _make_task()
    build = _make_build_result(files=["file.py"])

    result = wf.attempt_salvage(task, build)
    # Council status="failure" does not trigger discard path; continues
    assert result["salvaged"] is True
    assert result["reason"] == "no_sentinel_available_accepting_partial"


# ---------------------------------------------------------------------------
# Sentinel scoring — above threshold
# ---------------------------------------------------------------------------

def test_sentinel_above_threshold_salvages():
    sentinel_result = AgentResult(
        agent_name="Sentinel",
        status="success",
        data={"sentinel_verdict": {"quality_score": 0.55}},
    )

    class FakeSentinel:
        def run(self, input_data):
            return sentinel_result

    wf = WrapUpWorkflow(sentinel=FakeSentinel(), relaxed_quality_threshold=0.40)
    task = _make_task()
    build = _make_build_result(files=["auth.py"])

    result = wf.attempt_salvage(task, build)
    assert result["salvaged"] is True
    assert result["quality_score"] == 0.55
    assert "acceptable" in result["reason"]
    assert "0.55" in result["reason"]


def test_sentinel_exactly_at_threshold_salvages():
    sentinel_result = AgentResult(
        agent_name="Sentinel",
        status="success",
        data={"sentinel_verdict": {"quality_score": 0.40}},
    )

    class FakeSentinel:
        def run(self, input_data):
            return sentinel_result

    wf = WrapUpWorkflow(sentinel=FakeSentinel(), relaxed_quality_threshold=0.40)
    task = _make_task()
    build = _make_build_result(files=["auth.py"])

    result = wf.attempt_salvage(task, build)
    assert result["salvaged"] is True


# ---------------------------------------------------------------------------
# Sentinel scoring — below threshold
# ---------------------------------------------------------------------------

def test_sentinel_below_threshold_rejects():
    sentinel_result = AgentResult(
        agent_name="Sentinel",
        status="success",
        data={"sentinel_verdict": {"quality_score": 0.20}},
    )

    class FakeSentinel:
        def run(self, input_data):
            return sentinel_result

    wf = WrapUpWorkflow(sentinel=FakeSentinel(), relaxed_quality_threshold=0.40)
    task = _make_task()
    build = _make_build_result(files=["auth.py"])

    result = wf.attempt_salvage(task, build)
    assert result["salvaged"] is False
    assert result["quality_score"] == 0.20
    assert "below_threshold" in result["reason"]
    assert "0.20" in result["reason"]
    assert "0.40" in result["reason"]


def test_sentinel_zero_score_rejects():
    sentinel_result = AgentResult(
        agent_name="Sentinel",
        status="success",
        data={"sentinel_verdict": {"quality_score": 0.0}},
    )

    class FakeSentinel:
        def run(self, input_data):
            return sentinel_result

    wf = WrapUpWorkflow(sentinel=FakeSentinel(), relaxed_quality_threshold=0.40)
    task = _make_task()
    build = _make_build_result(files=["bad.py"])

    result = wf.attempt_salvage(task, build)
    assert result["salvaged"] is False
    assert result["quality_score"] == 0.0


# ---------------------------------------------------------------------------
# Sentinel failure handling
# ---------------------------------------------------------------------------

def test_sentinel_exception_recorded_in_reason():
    class FailingSentinel:
        def run(self, input_data):
            raise ConnectionError("Sentinel service down")

    wf = WrapUpWorkflow(sentinel=FailingSentinel())
    task = _make_task()
    build = _make_build_result(files=["file.py"])

    result = wf.attempt_salvage(task, build)
    assert result["salvaged"] is False
    assert "sentinel_check_failed" in result["reason"]
    assert "Sentinel service down" in result["reason"]


def test_sentinel_missing_data_field_uses_zero_score():
    sentinel_result = AgentResult(
        agent_name="Sentinel",
        status="success",
        data={},  # no sentinel_verdict key
    )

    class FakeSentinel:
        def run(self, input_data):
            return sentinel_result

    wf = WrapUpWorkflow(sentinel=FakeSentinel(), relaxed_quality_threshold=0.40)
    task = _make_task()
    build = _make_build_result(files=["file.py"])

    result = wf.attempt_salvage(task, build)
    assert result["quality_score"] == 0.0
    assert result["salvaged"] is False


def test_sentinel_returns_result_with_none_data_via_simplenamespace():
    """Exercise the `if sentinel_result.data` guard using a non-Pydantic object.

    AgentResult enforces data as dict, so we use SimpleNamespace to simulate
    a sentinel that returns an object where data is None (e.g., a broken
    third-party adapter).
    """
    fake_result = SimpleNamespace(data=None)

    class FakeSentinel:
        def run(self, input_data):
            return fake_result

    wf = WrapUpWorkflow(sentinel=FakeSentinel(), relaxed_quality_threshold=0.40)
    task = _make_task()
    build = _make_build_result(files=["file.py"])

    result = wf.attempt_salvage(task, build)
    # data is None so verdict_data = {}, quality = 0.0 => below threshold
    assert result["salvaged"] is False
    assert result["quality_score"] == 0.0


# ---------------------------------------------------------------------------
# Custom relaxed threshold
# ---------------------------------------------------------------------------

def test_custom_high_threshold():
    sentinel_result = AgentResult(
        agent_name="Sentinel",
        status="success",
        data={"sentinel_verdict": {"quality_score": 0.70}},
    )

    class FakeSentinel:
        def run(self, input_data):
            return sentinel_result

    wf = WrapUpWorkflow(sentinel=FakeSentinel(), relaxed_quality_threshold=0.80)
    task = _make_task()
    build = _make_build_result(files=["auth.py"])

    result = wf.attempt_salvage(task, build)
    assert result["salvaged"] is False
    assert "below_threshold" in result["reason"]


def test_custom_low_threshold():
    sentinel_result = AgentResult(
        agent_name="Sentinel",
        status="success",
        data={"sentinel_verdict": {"quality_score": 0.15}},
    )

    class FakeSentinel:
        def run(self, input_data):
            return sentinel_result

    wf = WrapUpWorkflow(sentinel=FakeSentinel(), relaxed_quality_threshold=0.10)
    task = _make_task()
    build = _make_build_result(files=["auth.py"])

    result = wf.attempt_salvage(task, build)
    assert result["salvaged"] is True


# ---------------------------------------------------------------------------
# Full pipeline: council approves, sentinel scores above threshold
# ---------------------------------------------------------------------------

def test_full_pipeline_council_approves_sentinel_passes():
    council_result = AgentResult(
        agent_name="Council",
        status="success",
        data={
            "council_diagnosis": {
                "new_approach": "Keep and polish the partial work",
                "strategy_shift": "minor fixes",
                "reasoning": "Core logic is sound",
            }
        },
    )

    sentinel_result = AgentResult(
        agent_name="Sentinel",
        status="success",
        data={"sentinel_verdict": {"quality_score": 0.65}},
    )

    class FakeCouncil:
        def run(self, input_data):
            return council_result

    class FakeSentinel:
        def run(self, input_data):
            return sentinel_result

    wf = WrapUpWorkflow(
        council=FakeCouncil(),
        sentinel=FakeSentinel(),
        relaxed_quality_threshold=0.40,
    )
    task = _make_task("Implement auth", "Add JWT auth to API")
    build = _make_build_result(
        files=["src/auth.py", "tests/test_auth.py"],
        approach="Used PyJWT for token generation",
        test_output="FAILED: 1 test (token expiry edge case)",
    )

    result = wf.attempt_salvage(task, build)
    assert result["salvaged"] is True
    assert result["files"] == ["src/auth.py", "tests/test_auth.py"]
    assert result["quality_score"] == 0.65
    assert "acceptable" in result["reason"]


def test_result_dict_always_has_required_keys():
    wf = WrapUpWorkflow()
    task = _make_task()
    build = _make_build_result(files=[])

    result = wf.attempt_salvage(task, build)
    assert "salvaged" in result
    assert "files" in result
    assert "quality_score" in result
    assert "reason" in result
