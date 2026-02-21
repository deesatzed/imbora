"""Focused tests for TaskLoop failure diagnostics and guardrails."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

from src.core.models import AgentResult, BuildResult, Task, TaskContext, TaskStatus
from src.orchestrator.loop import TaskLoop


def _make_task() -> Task:
    return Task(
        project_id=uuid.uuid4(),
        title="Guardrail task",
        description="Verify failure handling behavior",
    )


def test_derive_failure_text_prefers_specific_sources() -> None:
    loop = TaskLoop.__new__(TaskLoop)
    build_result = BuildResult(
        failure_reason="no_parseable_file_blocks",
        failure_detail="No parseable file blocks",
    )

    assert (
        loop._derive_failure_text(
            build_result=build_result,
            error_override="override",
            agent_error="agent error",
        )
        == "override"
    )
    assert (
        loop._derive_failure_text(
            build_result=build_result,
            agent_error="agent error",
        )
        == "agent error"
    )
    assert (
        loop._derive_failure_text(build_result=build_result)
        == "No parseable file blocks"
    )


def test_derive_failure_text_no_output_fallback_message() -> None:
    loop = TaskLoop.__new__(TaskLoop)
    build_result = BuildResult(files_changed=[], tests_passed=False)

    message = loop._derive_failure_text(build_result=build_result)
    assert "without writing files" in message
    assert "no_file_changes" in message


def test_handle_failure_uses_agent_error_in_hypothesis_log() -> None:
    loop = TaskLoop.__new__(TaskLoop)
    task = _make_task()
    captured: dict[str, str | None] = {}

    loop.state_manager = SimpleNamespace(rewind_to_checkpoint=lambda _task, _ref: True)

    def _capture_log(_task, _attempt, **kwargs) -> None:
        captured["error"] = kwargs.get("error")

    loop._log_hypothesis = _capture_log

    loop._handle_failure(
        task=task,
        build_data={},
        attempt=1,
        checkpoint_ref=None,
        agent_error="LLM output contained no parseable file blocks - no code was written",
    )

    assert (
        captured["error"]
        == "LLM output contained no parseable file blocks - no code was written"
    )


def test_apply_forced_strategy_injects_guidance_into_task_context() -> None:
    loop = TaskLoop.__new__(TaskLoop)
    task = _make_task()
    loop._forced_builder_strategy = {task.id: "Execution guardrail guidance"}

    context = TaskContext(task=task, forbidden_approaches=[])
    updated = loop._apply_forced_strategy(task, context)

    assert isinstance(updated, TaskContext)
    assert updated.previous_council_diagnosis is not None
    assert "Execution guardrail guidance" in updated.previous_council_diagnosis


def test_process_task_uses_global_attempt_counter_for_hypothesis_numbering() -> None:
    task = _make_task()
    task.attempt_count = 5

    loop = TaskLoop.__new__(TaskLoop)
    loop._progress_callback = None
    loop.config = SimpleNamespace(
        orchestrator=SimpleNamespace(
            max_retries_per_task=1,
            council_trigger_threshold=2,
            council_max_invocations=3,
        )
    )
    loop.state_manager = SimpleNamespace(
        run=lambda _task: AgentResult(
            agent_name="StateManager",
            status="failure",
            error="simulated state-manager failure",
        )
    )
    loop._total_tokens_per_task = {}
    loop._council_diagnosis_cache = {}
    loop._forced_builder_strategy = {}
    loop._no_output_attempts = {}
    loop._active_trace_id = None
    loop._active_run_id = None
    loop.diagnostics = SimpleNamespace(
        start_run=lambda **_kw: None,
        complete_run=lambda **_kw: None,
        record_event=lambda **_kw: None,
    )
    loop.repository = SimpleNamespace()

    captured: dict[str, int] = {}

    def _capture_hypothesis(_task, attempt, *_args, **_kwargs) -> None:
        captured["attempt"] = attempt

    loop._log_hypothesis = _capture_hypothesis

    class _Router:
        @staticmethod
        def transition(task_obj, new_status, reason=None):
            _ = reason
            task_obj.status = new_status
            return task_obj

        @staticmethod
        def increment_attempt(task_obj):
            task_obj.attempt_count += 1
            return task_obj

        @staticmethod
        def mark_stuck(task_obj, reason):
            _ = reason
            task_obj.status = TaskStatus.STUCK
            return task_obj

    loop.task_router = _Router()

    result = loop._process_task(task)

    assert result.status == TaskStatus.STUCK
    assert captured["attempt"] == 6


def test_llm_provider_failure_classifier() -> None:
    loop = TaskLoop.__new__(TaskLoop)
    assert loop._looks_like_llm_provider_failure("Request failed after 3 attempts: timeout")
    assert loop._looks_like_llm_provider_failure("OpenRouter network error")
    assert not loop._looks_like_llm_provider_failure("Tests failed (rc=1)")
