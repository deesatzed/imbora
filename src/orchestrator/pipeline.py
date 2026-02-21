"""Sequential agent pipeline for The Associate.

Chains agents in order, passing output from one to the next. Each agent
execution is wrapped with metrics collection and diagnostic recording.

Phase 2 pipeline: State Manager → Builder
Phase 3 will add: Research Unit → Librarian → Sentinel
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.models import AgentResult, Task
from src.orchestrator.diagnostics import DiagnosticsCollector
from src.orchestrator.metrics import PipelineMetrics

logger = logging.getLogger("associate.orchestrator.pipeline")


class Pipeline:
    """Sequential agent execution with metrics and diagnostics.

    Agents are executed in order. Each agent's output is available for
    the next agent through a transform function. If any agent fails,
    the pipeline stops and returns the failure.
    """

    def __init__(
        self,
        metrics: PipelineMetrics,
        diagnostics: DiagnosticsCollector,
    ):
        self.metrics = metrics
        self.diagnostics = diagnostics
        self._stages: list[PipelineStage] = []

    def add_stage(
        self,
        agent: BaseAgent,
        transform: Any = None,
    ) -> "Pipeline":
        """Add an agent stage to the pipeline.

        Args:
            agent: The agent to execute at this stage.
            transform: Optional callable(AgentResult, original_input) -> next_input
                      that transforms the output for the next stage.
                      If None, passes the AgentResult.data directly.

        Returns:
            self for method chaining.
        """
        self._stages.append(PipelineStage(agent=agent, transform=transform))
        return self

    def clear(self) -> None:
        """Remove all stages from the pipeline."""
        self._stages.clear()

    def execute(self, task: Task, initial_input: Any, run_number: int = 1) -> PipelineResult:
        """Execute all pipeline stages sequentially.

        Args:
            task: The task being processed (for metrics/diagnostics).
            initial_input: Input to the first agent.
            run_number: Attempt number for this pipeline run.

        Returns:
            PipelineResult with the final agent result and all stage results.
        """
        # Start metrics and diagnostics tracking
        self.metrics.start_run(task.id, run_number)
        self.diagnostics.start_run(task.id, run_number)

        stage_results: list[AgentResult] = []
        current_input = initial_input

        for i, stage in enumerate(self._stages):
            agent = stage.agent
            logger.info(
                "Pipeline stage %d/%d: %s",
                i + 1, len(self._stages), agent.name,
            )

            # Track metrics
            metric = self.metrics.start_agent(task.id, agent.name)

            # Execute agent
            result = agent.run(current_input)

            # Extract tokens from result data if available
            tokens = result.data.get("tokens_used", 0) if result.data else 0

            # Complete metrics
            self.metrics.complete_agent(
                metric,
                status=result.status,
                tokens_used=tokens,
                error=result.error,
            )

            # Record diagnostics
            input_summary = _summarize_for_diagnostics(current_input)
            self.diagnostics.record_agent(
                task_id=task.id,
                agent_name=agent.name,
                result=result,
                tokens_used=tokens,
                input_summary=input_summary,
            )

            stage_results.append(result)

            # Check for failure — stop pipeline
            if result.status != "success":
                logger.warning(
                    "Pipeline stopped at stage %d (%s): %s",
                    i + 1, agent.name, result.error,
                )
                self.metrics.complete_run("failure")
                self.diagnostics.complete_run("failure", error_summary=result.error)
                return PipelineResult(
                    success=False,
                    final_result=result,
                    stage_results=stage_results,
                    failed_stage=agent.name,
                )

            # Transform output for next stage
            if stage.transform and i < len(self._stages) - 1:
                current_input = stage.transform(result, initial_input)
            elif i < len(self._stages) - 1:
                # Default: pass result data as next input
                current_input = result

        # All stages passed
        self.metrics.complete_run("success")
        self.diagnostics.complete_run("success")

        return PipelineResult(
            success=True,
            final_result=stage_results[-1] if stage_results else None,
            stage_results=stage_results,
        )


class PipelineStage:
    """A single stage in the pipeline."""

    def __init__(self, agent: BaseAgent, transform: Any = None):
        self.agent = agent
        self.transform = transform


class PipelineResult:
    """Result of a pipeline execution."""

    def __init__(
        self,
        success: bool,
        final_result: AgentResult | None = None,
        stage_results: list[AgentResult] | None = None,
        failed_stage: str | None = None,
    ):
        self.success = success
        self.final_result = final_result
        self.stage_results = stage_results or []
        self.failed_stage = failed_stage

    @property
    def total_tokens(self) -> int:
        total = 0
        for r in self.stage_results:
            total += r.data.get("tokens_used", 0) if r.data else 0
        return total

    def get_stage_result(self, agent_name: str) -> AgentResult | None:
        """Get a specific agent's result by name."""
        for r in self.stage_results:
            if r.agent_name == agent_name:
                return r
        return None


def _summarize_for_diagnostics(input_data: Any) -> str:
    """Create a concise summary of pipeline stage input."""
    if hasattr(input_data, "task") and hasattr(input_data.task, "title"):
        return f"task='{input_data.task.title}'"
    if hasattr(input_data, "title"):
        return f"task='{input_data.title}'"
    if isinstance(input_data, dict):
        keys = list(input_data.keys())[:5]
        return f"dict(keys={keys})"
    return type(input_data).__name__
