"""Abstract base agent for The Associate.

Adapted from cbarch's BaseAgent pattern (ABC with metrics and structured logging).
Adds database/LLM injection and failure tracking for hypothesis log integration.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from src.core.models import AgentResult


class BaseAgent(ABC):
    """Base class for all Associate agents.

    Every agent follows the same lifecycle:
    1. Receive input (task context, brief, etc.)
    2. Process (LLM calls, DB queries, tool execution)
    3. Return an AgentResult with structured data
    4. Log metrics and errors throughout

    Subclasses must implement `process()`. They receive dependencies
    via __init__ injection — no global state, no singletons.
    """

    def __init__(self, name: str, role: str):
        """Initialize agent with name and role.

        Args:
            name: Human-readable agent name (e.g., "StateManager").
            role: Model router role key (e.g., "builder", "sentinel").
        """
        self.name = name
        self.role = role
        self.logger = logging.getLogger(f"associate.agent.{name.lower()}")
        self._metrics: dict[str, Any] = {
            "total_processed": 0,
            "total_errors": 0,
            "last_duration_seconds": 0.0,
        }

    @abstractmethod
    def process(self, input_data: Any) -> AgentResult:
        """Process input and return a structured result.

        Each subclass defines the input type (TaskContext, ContextBrief, etc.)
        and populates AgentResult.data with its specific output model.

        Args:
            input_data: Agent-specific input payload.

        Returns:
            AgentResult with status, data, and optional error.
        """

    def run(self, input_data: Any) -> AgentResult:
        """Execute the agent with lifecycle logging and metrics.

        This wraps process() with start/complete/error tracking.
        Agents should override process(), not run().

        Args:
            input_data: Agent-specific input payload.

        Returns:
            AgentResult from process(), or a failure result on exception.
        """
        self._log_start(input_data)
        start = time.monotonic()

        try:
            result = self.process(input_data)
            duration = time.monotonic() - start
            result.duration_seconds = duration
            self._metrics["total_processed"] += 1
            self._metrics["last_duration_seconds"] = duration
            self._log_complete(duration, result)
            return result
        except Exception as e:
            duration = time.monotonic() - start
            self._metrics["total_errors"] += 1
            self._metrics["last_duration_seconds"] = duration
            self._log_error(e)
            return AgentResult(
                agent_name=self.name,
                status="failure",
                error=str(e),
                duration_seconds=duration,
            )

    def _log_start(self, input_data: Any) -> None:
        context = _summarize_input(input_data)
        self.logger.info("[%s] Starting: %s", self.name, context)

    def _log_complete(self, duration: float, result: AgentResult) -> None:
        self.logger.info(
            "[%s] Complete: status=%s (%.2fs)",
            self.name, result.status, duration,
        )

    def _log_error(self, error: Exception) -> None:
        self.logger.error(
            "[%s] Error: %s", self.name, error, exc_info=True,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Return a copy of the agent's runtime metrics."""
        return self._metrics.copy()


def _summarize_input(input_data: Any) -> str:
    """Create a short log-safe summary of agent input."""
    if hasattr(input_data, "task") and hasattr(input_data.task, "title"):
        return f"task='{input_data.task.title}'"
    if hasattr(input_data, "title"):
        return f"task='{input_data.title}'"
    return type(input_data).__name__
