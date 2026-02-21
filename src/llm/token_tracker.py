"""Token-level cost tracking for LLM calls.

Adapted from ClawWork's TrackedProvider + EconomicTracker pattern.
Intercepts every LLM call and records (input_tokens, output_tokens, cost)
per call, attributed to the active task/run. Supports both PostgreSQL
persistence and JSONL shadow-logging (Item 10).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from src.core.models import TokenCostRecord

logger = logging.getLogger("associate.llm.token_tracker")


class TokenTracker:
    """Accumulates per-call token costs and persists to DB + JSONL.

    Usage:
        tracker = TokenTracker(repository=repo)
        tracker.set_context(task_id=..., run_id=..., agent_role="builder")
        # After each LLM call:
        tracker.record(model="...", input_tokens=100, output_tokens=200, cost_usd=0.01)
    """

    def __init__(
        self,
        repository: Optional[Any] = None,
        jsonl_path: Optional[str] = None,
        cost_per_1k_input: float = 0.01,
        cost_per_1k_output: float = 0.03,
    ):
        self.repository = repository
        self.jsonl_path = jsonl_path
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        self._task_id: Optional[uuid.UUID] = None
        self._run_id: Optional[uuid.UUID] = None
        self._agent_role: str = ""
        self._session_totals: dict[str, Any] = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "call_count": 0,
        }
        self._task_totals: dict[uuid.UUID, dict[str, Any]] = {}

    def set_context(
        self,
        task_id: Optional[uuid.UUID] = None,
        run_id: Optional[uuid.UUID] = None,
        agent_role: str = "",
    ) -> None:
        """Set the attribution context for subsequent records."""
        self._task_id = task_id
        self._run_id = run_id
        self._agent_role = agent_role

    def record(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: Optional[float] = None,
    ) -> TokenCostRecord:
        """Record a single LLM call's token usage and cost.

        If cost_usd is not provided, it is estimated from token counts
        using the configured per-1k rates.
        """
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        if cost_usd is None:
            cost_usd = (
                (input_tokens / 1000.0) * self.cost_per_1k_input
                + (output_tokens / 1000.0) * self.cost_per_1k_output
            )

        record = TokenCostRecord(
            task_id=self._task_id,
            run_id=self._run_id,
            agent_role=self._agent_role,
            model_used=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
        )

        # Update session totals
        self._session_totals["total_input_tokens"] += input_tokens
        self._session_totals["total_output_tokens"] += output_tokens
        self._session_totals["total_tokens"] += total_tokens
        self._session_totals["total_cost_usd"] += cost_usd
        self._session_totals["call_count"] += 1

        # Update per-task totals
        if self._task_id is not None:
            if self._task_id not in self._task_totals:
                self._task_totals[self._task_id] = {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "call_count": 0,
                }
            tt = self._task_totals[self._task_id]
            tt["total_input_tokens"] += input_tokens
            tt["total_output_tokens"] += output_tokens
            tt["total_tokens"] += total_tokens
            tt["total_cost_usd"] += cost_usd
            tt["call_count"] += 1

        # Persist to database
        self._persist_to_db(record)

        # Shadow-log to JSONL (Item 10)
        self._persist_to_jsonl(record)

        logger.debug(
            "Token cost recorded: model=%s in=%d out=%d cost=$%.6f role=%s",
            model, input_tokens, output_tokens, cost_usd, self._agent_role,
        )
        return record

    def get_session_totals(self) -> dict[str, Any]:
        """Return accumulated session-level totals."""
        return dict(self._session_totals)

    def get_task_totals(self, task_id: uuid.UUID) -> dict[str, Any]:
        """Return accumulated totals for a specific task."""
        return dict(self._task_totals.get(task_id, {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "call_count": 0,
        }))

    def _persist_to_db(self, record: TokenCostRecord) -> None:
        """Write record to the token_costs table."""
        if self.repository is None:
            return
        if not hasattr(self.repository, "save_token_cost"):
            return
        try:
            self.repository.save_token_cost(record)
        except Exception as e:
            logger.warning("Failed to persist token cost to DB: %s", e)

    def _persist_to_jsonl(self, record: TokenCostRecord) -> None:
        """Append record to JSONL shadow log (Item 10)."""
        if not self.jsonl_path:
            return
        try:
            path = Path(self.jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "id": str(record.id),
                "task_id": str(record.task_id) if record.task_id else None,
                "run_id": str(record.run_id) if record.run_id else None,
                "agent_role": record.agent_role,
                "model_used": record.model_used,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "total_tokens": record.total_tokens,
                "cost_usd": record.cost_usd,
                "created_at": record.created_at.isoformat(),
            }
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning("Failed to write token cost to JSONL: %s", e)
