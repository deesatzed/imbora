"""Tests for src/agents/base_agent.py — abstract base agent lifecycle."""

from typing import Any

import pytest

from src.agents.base_agent import BaseAgent, _summarize_input
from src.core.models import AgentResult, Task, TaskContext, TaskStatus
import uuid


class ConcreteAgent(BaseAgent):
    """Test implementation of BaseAgent."""

    def __init__(self, should_fail: bool = False):
        super().__init__(name="TestAgent", role="builder")
        self.should_fail = should_fail
        self.received_input = None

    def process(self, input_data: Any) -> AgentResult:
        self.received_input = input_data
        if self.should_fail:
            raise RuntimeError("Intentional test failure")
        return AgentResult(
            agent_name=self.name,
            status="success",
            data={"processed": True},
        )


class TestBaseAgentInit:
    def test_name_and_role(self):
        agent = ConcreteAgent()
        assert agent.name == "TestAgent"
        assert agent.role == "builder"

    def test_initial_metrics(self):
        agent = ConcreteAgent()
        metrics = agent.get_metrics()
        assert metrics["total_processed"] == 0
        assert metrics["total_errors"] == 0
        assert metrics["last_duration_seconds"] == 0.0


class TestBaseAgentRun:
    def test_successful_run(self):
        agent = ConcreteAgent()
        result = agent.run("test input")
        assert result.status == "success"
        assert result.data["processed"] is True
        assert result.duration_seconds > 0
        assert agent.received_input == "test input"

    def test_successful_run_updates_metrics(self):
        agent = ConcreteAgent()
        agent.run("input1")
        agent.run("input2")
        metrics = agent.get_metrics()
        assert metrics["total_processed"] == 2
        assert metrics["total_errors"] == 0

    def test_failed_run_returns_failure_result(self):
        agent = ConcreteAgent(should_fail=True)
        result = agent.run("test input")
        assert result.status == "failure"
        assert "Intentional test failure" in result.error
        assert result.duration_seconds > 0

    def test_failed_run_updates_error_metrics(self):
        agent = ConcreteAgent(should_fail=True)
        agent.run("input")
        metrics = agent.get_metrics()
        assert metrics["total_errors"] == 1
        assert metrics["total_processed"] == 0

    def test_mixed_runs(self):
        agent = ConcreteAgent()
        agent.run("ok")
        agent.should_fail = True
        agent.run("fail")
        agent.should_fail = False
        agent.run("ok again")
        metrics = agent.get_metrics()
        assert metrics["total_processed"] == 2
        assert metrics["total_errors"] == 1

    def test_result_has_agent_name(self):
        agent = ConcreteAgent()
        result = agent.run("test")
        assert result.agent_name == "TestAgent"


class TestBaseAgentAbstract:
    def test_cannot_instantiate_base_directly(self):
        with pytest.raises(TypeError):
            BaseAgent(name="Bad", role="none")


class TestSummarizeInput:
    def test_task_context(self):
        pid = uuid.uuid4()
        task = Task(project_id=pid, title="Fix auth", description="Fix JWT")
        ctx = TaskContext(task=task)
        assert "Fix auth" in _summarize_input(ctx)

    def test_task_directly(self):
        pid = uuid.uuid4()
        task = Task(project_id=pid, title="Build UI", description="React")
        assert "Build UI" in _summarize_input(task)

    def test_plain_object(self):
        result = _summarize_input({"key": "value"})
        assert "dict" in result

    def test_string_input(self):
        result = _summarize_input("hello")
        assert "str" in result
