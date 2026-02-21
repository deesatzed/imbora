"""Tests for BaseDataScienceAgent.

Uses real-shaped fakes (in-memory implementations matching real interfaces)
for LLM client, model router, and repository — consistent with the project's
testing approach in test_orchestrator_loop_integration.py. No mocks.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from src.agents.base_agent import BaseAgent
from src.core.config import DataScienceConfig
from src.core.models import AgentResult
from src.datasci.agents.base_ds_agent import BaseDataScienceAgent


# ---------------------------------------------------------------------------
# Real-shaped fakes (in-memory implementations, NOT mocks)
# ---------------------------------------------------------------------------

class FakeLLMResponse:
    """Matches the interface returned by OpenRouterClient.complete_with_fallback()."""
    def __init__(self, content: str):
        self.content = content


class FakeLLMClient:
    """In-memory LLM client that returns deterministic responses."""
    def __init__(self, response_text: str = "LLM analysis complete."):
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def complete_with_fallback(self, **kwargs) -> FakeLLMResponse:
        self.calls.append(kwargs)
        return FakeLLMResponse(self.response_text)


class FakeModelRouter:
    """In-memory model router matching ModelRouter interface."""
    def __init__(self):
        self.models = {
            "ds_analyst": "google/gemini-3-flash-preview",
            "ds_evaluator": "google/gemini-3-flash-preview",
        }
        self.fallbacks = {
            "ds_analyst": ["anthropic/claude-sonnet-4"],
            "ds_evaluator": ["anthropic/claude-sonnet-4"],
        }

    def get_model(self, role: str) -> str:
        return self.models.get(role, "fallback/model")

    def get_fallback_models(self, role: str) -> list[str]:
        return self.fallbacks.get(role, [])

    def get_model_chain(self, role: str) -> list[str]:
        primary = self.get_model(role)
        fallbacks = self.get_fallback_models(role)
        chain: list[str] = []
        for model in [primary, *fallbacks]:
            if model and model not in chain:
                chain.append(model)
        return chain


class FakeRepository:
    """In-memory repository for DS experiment CRUD."""
    def __init__(self):
        self.experiments: dict[uuid.UUID, dict[str, Any]] = {}
        self.artifacts: list[dict[str, Any]] = []

    def create_ds_experiment(
        self,
        project_id: uuid.UUID,
        experiment_phase: str,
        experiment_config: dict | None = None,
        task_id: uuid.UUID | None = None,
        run_id: uuid.UUID | None = None,
        dataset_fingerprint: str | None = None,
        parent_experiment_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        exp_id = uuid.uuid4()
        self.experiments[exp_id] = {
            "id": exp_id,
            "project_id": project_id,
            "experiment_phase": experiment_phase,
            "experiment_config": experiment_config or {},
            "task_id": task_id,
            "run_id": run_id,
            "dataset_fingerprint": dataset_fingerprint,
            "parent_experiment_id": parent_experiment_id,
            "status": "RUNNING",
            "metrics": {},
            "artifacts_manifest": {},
        }
        return exp_id

    def update_ds_experiment(
        self,
        experiment_id: uuid.UUID,
        status: str = "COMPLETED",
        metrics: dict | None = None,
        artifacts_manifest: dict | None = None,
    ) -> None:
        if experiment_id in self.experiments:
            self.experiments[experiment_id]["status"] = status
            if metrics:
                self.experiments[experiment_id]["metrics"] = metrics
            if artifacts_manifest:
                self.experiments[experiment_id]["artifacts_manifest"] = artifacts_manifest


# ---------------------------------------------------------------------------
# Concrete subclass for testing (BaseDataScienceAgent is abstract)
# ---------------------------------------------------------------------------

class ConcreteTestAgent(BaseDataScienceAgent):
    """Minimal concrete subclass for testing base class functionality."""

    def __init__(self, llm_client, model_router, repository, ds_config):
        super().__init__(
            name="TestDSAgent",
            role="ds_analyst",
            llm_client=llm_client,
            model_router=model_router,
            repository=repository,
            ds_config=ds_config,
        )
        self._should_fail = False

    def process(self, input_data: Any) -> AgentResult:
        if self._should_fail:
            raise RuntimeError("Intentional test failure")
        return AgentResult(
            agent_name=self.name,
            status="success",
            data={"processed": True, "input_type": type(input_data).__name__},
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ds_config():
    return DataScienceConfig()


@pytest.fixture
def fake_llm():
    return FakeLLMClient(response_text="Analysis: dataset looks clean.")


@pytest.fixture
def fake_router():
    return FakeModelRouter()


@pytest.fixture
def fake_repo():
    return FakeRepository()


@pytest.fixture
def agent(fake_llm, fake_router, fake_repo, ds_config):
    return ConcreteTestAgent(
        llm_client=fake_llm,
        model_router=fake_router,
        repository=fake_repo,
        ds_config=ds_config,
    )


# ---------------------------------------------------------------------------
# Tests: Inheritance and initialization
# ---------------------------------------------------------------------------

class TestBaseDataScienceAgentInit:
    def test_extends_base_agent(self, agent):
        assert isinstance(agent, BaseAgent)
        assert isinstance(agent, BaseDataScienceAgent)

    def test_name_and_role(self, agent):
        assert agent.name == "TestDSAgent"
        assert agent.role == "ds_analyst"

    def test_injected_dependencies(self, agent, fake_llm, fake_router, fake_repo, ds_config):
        assert agent.llm_client is fake_llm
        assert agent.model_router is fake_router
        assert agent.repository is fake_repo
        assert agent.ds_config is ds_config

    def test_metrics_initialized(self, agent):
        metrics = agent.get_metrics()
        assert metrics["total_processed"] == 0
        assert metrics["total_errors"] == 0
        assert metrics["last_duration_seconds"] == 0.0

    def test_logger_exists(self, agent):
        assert agent.logger is not None


# ---------------------------------------------------------------------------
# Tests: run() lifecycle (inherited from BaseAgent)
# ---------------------------------------------------------------------------

class TestBaseAgentLifecycle:
    def test_run_success(self, agent):
        result = agent.run({"dataset": "test.csv"})
        assert result.status == "success"
        assert result.agent_name == "TestDSAgent"
        assert result.data["processed"] is True
        assert result.duration_seconds > 0

    def test_run_updates_metrics(self, agent):
        agent.run({"data": "x"})
        metrics = agent.get_metrics()
        assert metrics["total_processed"] == 1
        assert metrics["total_errors"] == 0

    def test_run_failure_captured(self, agent):
        agent._should_fail = True
        result = agent.run({"data": "x"})
        assert result.status == "failure"
        assert "Intentional test failure" in result.error
        assert result.duration_seconds > 0

    def test_run_failure_updates_metrics(self, agent):
        agent._should_fail = True
        agent.run({"data": "x"})
        metrics = agent.get_metrics()
        assert metrics["total_processed"] == 0
        assert metrics["total_errors"] == 1


# ---------------------------------------------------------------------------
# Tests: _call_ds_llm()
# ---------------------------------------------------------------------------

class TestCallDSLLM:
    def test_basic_call(self, agent, fake_llm):
        response = agent._call_ds_llm("Analyze this dataset")
        assert response == "Analysis: dataset looks clean."
        assert len(fake_llm.calls) == 1

    def test_default_role(self, agent, fake_llm):
        agent._call_ds_llm("prompt")
        call = fake_llm.calls[0]
        assert "google/gemini-3-flash-preview" in call["models"]

    def test_evaluator_role(self, agent, fake_llm):
        agent._call_ds_llm("evaluate", role="ds_evaluator")
        call = fake_llm.calls[0]
        assert "google/gemini-3-flash-preview" in call["models"]

    def test_system_prompt_included(self, agent, fake_llm):
        agent._call_ds_llm("prompt", system_prompt="You are a data scientist.")
        call = fake_llm.calls[0]
        assert len(call["messages"]) == 2
        assert call["messages"][0].role == "system"
        assert call["messages"][0].content == "You are a data scientist."
        assert call["messages"][1].role == "user"

    def test_no_system_prompt(self, agent, fake_llm):
        agent._call_ds_llm("prompt")
        call = fake_llm.calls[0]
        assert len(call["messages"]) == 1
        assert call["messages"][0].role == "user"

    def test_custom_parameters(self, agent, fake_llm):
        agent._call_ds_llm("prompt", max_tokens=8192, temperature=0.7)
        call = fake_llm.calls[0]
        assert call["max_tokens"] == 8192
        assert call["temperature"] == 0.7

    def test_fallback_models_passed(self, agent, fake_llm):
        agent._call_ds_llm("prompt", role="ds_analyst")
        call = fake_llm.calls[0]
        assert "anthropic/claude-sonnet-4" in call["models"]

    def test_multiple_calls_tracked(self, agent, fake_llm):
        agent._call_ds_llm("call1")
        agent._call_ds_llm("call2")
        agent._call_ds_llm("call3")
        assert len(fake_llm.calls) == 3


# ---------------------------------------------------------------------------
# Tests: _save_experiment()
# ---------------------------------------------------------------------------

class TestSaveExperiment:
    def test_basic_save(self, agent, fake_repo):
        pid = uuid.uuid4()
        exp_id = agent._save_experiment(project_id=pid, phase="data_audit")
        assert isinstance(exp_id, uuid.UUID)
        assert exp_id in fake_repo.experiments
        assert fake_repo.experiments[exp_id]["experiment_phase"] == "data_audit"
        assert fake_repo.experiments[exp_id]["project_id"] == pid
        assert fake_repo.experiments[exp_id]["status"] == "RUNNING"

    def test_save_with_config(self, agent, fake_repo):
        pid = uuid.uuid4()
        config = {"cv_folds": 5, "models": ["xgb", "tabpfn"]}
        exp_id = agent._save_experiment(project_id=pid, phase="model_training", config=config)
        assert fake_repo.experiments[exp_id]["experiment_config"] == config

    def test_save_with_all_optional_fields(self, agent, fake_repo):
        pid = uuid.uuid4()
        tid = uuid.uuid4()
        rid = uuid.uuid4()
        parent_id = uuid.uuid4()

        exp_id = agent._save_experiment(
            project_id=pid,
            phase="eda",
            config={"detail_level": "full"},
            task_id=tid,
            run_id=rid,
            dataset_fingerprint="sha256:abc123def456",
            parent_experiment_id=parent_id,
        )
        exp = fake_repo.experiments[exp_id]
        assert exp["task_id"] == tid
        assert exp["run_id"] == rid
        assert exp["dataset_fingerprint"] == "sha256:abc123def456"
        assert exp["parent_experiment_id"] == parent_id

    def test_multiple_experiments(self, agent, fake_repo):
        pid = uuid.uuid4()
        ids = []
        for phase in ["data_audit", "eda", "feature_eng", "model_training"]:
            ids.append(agent._save_experiment(project_id=pid, phase=phase))
        assert len(fake_repo.experiments) == 4
        assert len(set(ids)) == 4  # All unique IDs


# ---------------------------------------------------------------------------
# Tests: _complete_experiment()
# ---------------------------------------------------------------------------

class TestCompleteExperiment:
    def test_basic_complete(self, agent, fake_repo):
        pid = uuid.uuid4()
        exp_id = agent._save_experiment(project_id=pid, phase="eda")
        assert fake_repo.experiments[exp_id]["status"] == "RUNNING"

        agent._complete_experiment(exp_id)
        assert fake_repo.experiments[exp_id]["status"] == "COMPLETED"

    def test_complete_with_metrics(self, agent, fake_repo):
        pid = uuid.uuid4()
        exp_id = agent._save_experiment(project_id=pid, phase="model_training")

        metrics = {"accuracy": 0.93, "f1": 0.91, "training_time": 12.5}
        agent._complete_experiment(exp_id, metrics=metrics)
        assert fake_repo.experiments[exp_id]["metrics"] == metrics

    def test_complete_with_artifacts(self, agent, fake_repo):
        pid = uuid.uuid4()
        exp_id = agent._save_experiment(project_id=pid, phase="deployment")

        artifacts = {"model_path": "/artifacts/model.joblib", "api_scaffold": True}
        agent._complete_experiment(exp_id, artifacts_manifest=artifacts)
        assert fake_repo.experiments[exp_id]["artifacts_manifest"] == artifacts

    def test_complete_with_failed_status(self, agent, fake_repo):
        pid = uuid.uuid4()
        exp_id = agent._save_experiment(project_id=pid, phase="ensemble")

        agent._complete_experiment(exp_id, status="FAILED")
        assert fake_repo.experiments[exp_id]["status"] == "FAILED"

    def test_full_lifecycle(self, agent, fake_repo):
        """Complete experiment lifecycle: create -> complete with metrics + artifacts."""
        pid = uuid.uuid4()
        exp_id = agent._save_experiment(
            project_id=pid,
            phase="evaluation",
            config={"rubric": "comprehensive"},
        )
        assert fake_repo.experiments[exp_id]["status"] == "RUNNING"

        agent._complete_experiment(
            exp_id,
            metrics={"overall_grade": "A-", "accuracy": 0.94},
            artifacts_manifest={"report_path": "/reports/eval.json"},
            status="COMPLETED",
        )
        exp = fake_repo.experiments[exp_id]
        assert exp["status"] == "COMPLETED"
        assert exp["metrics"]["overall_grade"] == "A-"
        assert exp["artifacts_manifest"]["report_path"] == "/reports/eval.json"


# ---------------------------------------------------------------------------
# Tests: DS config access
# ---------------------------------------------------------------------------

class TestDSConfigAccess:
    def test_default_config_values(self, agent):
        assert agent.ds_config.enabled is False
        assert agent.ds_config.cv_folds == 5
        assert agent.ds_config.llm_fe_generations == 5
        assert agent.ds_config.conformal_alpha == 0.10
        assert agent.ds_config.enable_tabpfn is True
        assert agent.ds_config.enable_tabicl is True
        assert agent.ds_config.enable_nam is True
        assert agent.ds_config.cleanlab_enabled is True

    def test_custom_config(self, fake_llm, fake_router, fake_repo):
        custom_config = DataScienceConfig(
            enabled=True,
            cv_folds=10,
            llm_fe_generations=10,
            conformal_alpha=0.05,
            enable_tabpfn=False,
        )
        agent = ConcreteTestAgent(
            llm_client=fake_llm,
            model_router=fake_router,
            repository=fake_repo,
            ds_config=custom_config,
        )
        assert agent.ds_config.enabled is True
        assert agent.ds_config.cv_folds == 10
        assert agent.ds_config.enable_tabpfn is False
