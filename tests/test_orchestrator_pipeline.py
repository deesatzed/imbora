"""Tests for src/orchestrator/pipeline.py — sequential agent chaining."""

import uuid

import pytest

from src.agents.base_agent import BaseAgent
from src.core.models import AgentResult, Task, TaskStatus
from src.orchestrator.diagnostics import DiagnosticsCollector
from src.orchestrator.metrics import PipelineMetrics
from src.orchestrator.pipeline import Pipeline, PipelineResult


class SuccessAgent(BaseAgent):
    """Agent that always succeeds — for testing pipeline flow."""

    def __init__(self, name: str = "SuccessAgent", tokens: int = 0):
        super().__init__(name=name, role="test")
        self._tokens = tokens

    def process(self, input_data):
        return AgentResult(
            agent_name=self.name,
            status="success",
            data={"output": f"{self.name} processed", "tokens_used": self._tokens},
        )


class FailureAgent(BaseAgent):
    """Agent that always fails — for testing pipeline failure handling."""

    def __init__(self, name: str = "FailureAgent", error_msg: str = "intentional failure"):
        super().__init__(name=name, role="test")
        self._error_msg = error_msg

    def process(self, input_data):
        return AgentResult(
            agent_name=self.name,
            status="failure",
            error=self._error_msg,
        )


class ExceptionAgent(BaseAgent):
    """Agent that raises an exception — for testing pipeline error handling."""

    def __init__(self):
        super().__init__(name="ExceptionAgent", role="test")

    def process(self, input_data):
        raise RuntimeError("unexpected error")


@pytest.fixture
def metrics():
    return PipelineMetrics()


@pytest.fixture
def diagnostics():
    return DiagnosticsCollector()


@pytest.fixture
def task():
    return Task(
        project_id=uuid.uuid4(),
        title="Test task",
        description="Testing pipeline",
    )


class TestPipeline:
    def test_single_stage_success(self, metrics, diagnostics, task):
        pipeline = Pipeline(metrics=metrics, diagnostics=diagnostics)
        pipeline.add_stage(SuccessAgent("Step1"))

        result = pipeline.execute(task, initial_input="test input")
        assert result.success is True
        assert len(result.stage_results) == 1
        assert result.final_result.status == "success"

    def test_multi_stage_success(self, metrics, diagnostics, task):
        pipeline = Pipeline(metrics=metrics, diagnostics=diagnostics)
        pipeline.add_stage(SuccessAgent("Step1"))
        pipeline.add_stage(SuccessAgent("Step2"))
        pipeline.add_stage(SuccessAgent("Step3"))

        result = pipeline.execute(task, initial_input="test input")
        assert result.success is True
        assert len(result.stage_results) == 3

    def test_failure_stops_pipeline(self, metrics, diagnostics, task):
        pipeline = Pipeline(metrics=metrics, diagnostics=diagnostics)
        pipeline.add_stage(SuccessAgent("Step1"))
        pipeline.add_stage(FailureAgent("Step2"))
        pipeline.add_stage(SuccessAgent("Step3"))

        result = pipeline.execute(task, initial_input="test input")
        assert result.success is False
        assert len(result.stage_results) == 2  # Step3 never ran
        assert result.failed_stage == "Step2"

    def test_exception_handled_as_failure(self, metrics, diagnostics, task):
        pipeline = Pipeline(metrics=metrics, diagnostics=diagnostics)
        pipeline.add_stage(ExceptionAgent())

        result = pipeline.execute(task, initial_input="test input")
        assert result.success is False
        assert result.failed_stage == "ExceptionAgent"

    def test_metrics_collected(self, metrics, diagnostics, task):
        pipeline = Pipeline(metrics=metrics, diagnostics=diagnostics)
        pipeline.add_stage(SuccessAgent("Step1", tokens=100))
        pipeline.add_stage(SuccessAgent("Step2", tokens=200))

        pipeline.execute(task, initial_input="test input")

        runs = metrics.get_runs()
        assert len(runs) == 1
        assert runs[0].total_tokens == 300

    def test_diagnostics_collected(self, metrics, diagnostics, task):
        pipeline = Pipeline(metrics=metrics, diagnostics=diagnostics)
        pipeline.add_stage(SuccessAgent("Step1"))

        pipeline.execute(task, initial_input="test input")

        history = diagnostics.get_history()
        assert len(history) == 1
        assert len(history[0].entries) == 1
        assert history[0].entries[0].agent_name == "Step1"

    def test_pipeline_result_get_stage(self, metrics, diagnostics, task):
        pipeline = Pipeline(metrics=metrics, diagnostics=diagnostics)
        pipeline.add_stage(SuccessAgent("Alpha"))
        pipeline.add_stage(SuccessAgent("Beta"))

        result = pipeline.execute(task, initial_input="test")
        alpha = result.get_stage_result("Alpha")
        assert alpha is not None
        assert alpha.agent_name == "Alpha"

        missing = result.get_stage_result("Gamma")
        assert missing is None

    def test_pipeline_result_total_tokens(self, metrics, diagnostics, task):
        pipeline = Pipeline(metrics=metrics, diagnostics=diagnostics)
        pipeline.add_stage(SuccessAgent("A", tokens=50))
        pipeline.add_stage(SuccessAgent("B", tokens=75))

        result = pipeline.execute(task, initial_input="test")
        assert result.total_tokens == 125

    def test_method_chaining(self, metrics, diagnostics):
        pipeline = Pipeline(metrics=metrics, diagnostics=diagnostics)
        result = pipeline.add_stage(SuccessAgent("A")).add_stage(SuccessAgent("B"))
        assert result is pipeline

    def test_clear_stages(self, metrics, diagnostics):
        pipeline = Pipeline(metrics=metrics, diagnostics=diagnostics)
        pipeline.add_stage(SuccessAgent("A"))
        assert len(pipeline._stages) == 1
        pipeline.clear()
        assert len(pipeline._stages) == 0

    def test_custom_transform(self, metrics, diagnostics, task):
        """Test that a custom transform function passes data between stages."""
        pipeline = Pipeline(metrics=metrics, diagnostics=diagnostics)

        def transform_fn(result, original):
            return {"transformed": True, "from": result.agent_name}

        pipeline.add_stage(SuccessAgent("Step1"), transform=transform_fn)
        pipeline.add_stage(SuccessAgent("Step2"))

        result = pipeline.execute(task, initial_input="raw input")
        assert result.success is True

    def test_run_number_passed(self, metrics, diagnostics, task):
        pipeline = Pipeline(metrics=metrics, diagnostics=diagnostics)
        pipeline.add_stage(SuccessAgent("A"))

        pipeline.execute(task, initial_input="test", run_number=5)

        runs = metrics.get_runs()
        assert runs[0].run_number == 5
