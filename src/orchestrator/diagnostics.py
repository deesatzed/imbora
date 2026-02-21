"""Diagnostic snapshots for The Associate — Dirac Pattern 6.

Structured diagnostic snapshots at each pipeline stage. Maintains a rolling
in-memory record for the current run. On failure/STUCK, the full diagnostic
can be dumped for debugging.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from src.core.models import AgentResult

logger = logging.getLogger("associate.orchestrator.diagnostics")

MAX_HISTORY = 50  # Rolling window of diagnostic entries


@dataclass
class DiagnosticEntry:
    """Single diagnostic record from an agent execution."""
    timestamp: datetime
    task_id: uuid.UUID
    agent_name: str
    status: str
    duration_seconds: float
    tokens_used: int = 0
    input_summary: str = ""
    output_summary: str = ""
    error: Optional[str] = None
    trace_id: str | None = None
    run_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunDiagnostics:
    """Complete diagnostic snapshot for a single task run."""
    task_id: uuid.UUID
    run_number: int
    run_id: str | None = None
    trace_id: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    entries: list[DiagnosticEntry] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    outcome: str = "in_progress"
    error_summary: Optional[str] = None

    def add_entry(self, entry: DiagnosticEntry) -> None:
        self.entries.append(entry)

    @property
    def total_duration(self) -> float:
        return sum(e.duration_seconds for e in self.entries)

    @property
    def total_tokens(self) -> int:
        return sum(e.tokens_used for e in self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "run_number": self.run_number,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "started_at": self.started_at.isoformat(),
            "metadata": dict(self.metadata),
            "outcome": self.outcome,
            "error_summary": self.error_summary,
            "total_duration_seconds": round(self.total_duration, 2),
            "total_tokens": self.total_tokens,
            "entries": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "agent_name": e.agent_name,
                    "status": e.status,
                    "duration_seconds": round(e.duration_seconds, 2),
                    "tokens_used": e.tokens_used,
                    "input_summary": e.input_summary,
                    "output_summary": e.output_summary,
                    "error": e.error,
                    "trace_id": e.trace_id,
                    "run_id": e.run_id,
                    "extra": dict(e.extra),
                }
                for e in self.entries
            ],
            "events": list(self.events),
        }


class DiagnosticsCollector:
    """Collects structured diagnostic snapshots during pipeline execution.

    Maintains a rolling deque of diagnostic snapshots. On failure/STUCK,
    the full diagnostic can be dumped to a file for debugging.
    """

    def __init__(self, max_history: int = MAX_HISTORY):
        self._history: deque[RunDiagnostics] = deque(maxlen=max_history)
        self._current: Optional[RunDiagnostics] = None

    def start_run(
        self,
        task_id: uuid.UUID,
        run_number: int,
        run_id: str | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunDiagnostics:
        """Begin a new diagnostic run."""
        diag = RunDiagnostics(
            task_id=task_id,
            run_number=run_number,
            run_id=run_id,
            trace_id=trace_id,
            started_at=datetime.now(UTC),
            metadata=dict(metadata or {}),
        )
        self._current = diag
        self._history.append(diag)
        return diag

    def record_agent(
        self,
        task_id: uuid.UUID,
        agent_name: str,
        result: AgentResult,
        tokens_used: int = 0,
        input_summary: str = "",
        trace_id: str | None = None,
        run_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> DiagnosticEntry:
        """Record a diagnostic entry from an agent execution."""
        current_trace_id = trace_id or (self._current.trace_id if self._current else None)
        current_run_id = run_id or (self._current.run_id if self._current else None)
        entry = DiagnosticEntry(
            timestamp=datetime.now(UTC),
            task_id=task_id,
            agent_name=agent_name,
            status=result.status,
            duration_seconds=result.duration_seconds,
            tokens_used=tokens_used,
            input_summary=input_summary,
            output_summary=_summarize_data(result.data),
            error=result.error,
            trace_id=current_trace_id,
            run_id=current_run_id,
            extra=dict(extra or {}),
        )

        if self._current and self._current.task_id == task_id:
            self._current.add_entry(entry)

        return entry

    def record_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        trace_id: str | None = None,
        run_id: str | None = None,
        task_id: uuid.UUID | None = None,
    ) -> None:
        """Record a structured run-level event in the active diagnostics stream."""
        if self._current is None:
            return
        if task_id is not None and self._current.task_id != task_id:
            return
        self._current.events.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "event_type": event_type,
                "trace_id": trace_id or self._current.trace_id,
                "run_id": run_id or self._current.run_id,
                "payload": dict(payload),
            }
        )

    def complete_run(self, outcome: str, error_summary: Optional[str] = None) -> Optional[RunDiagnostics]:
        """Mark the current run as complete."""
        if self._current is None:
            return None

        self._current.outcome = outcome
        self._current.error_summary = error_summary
        completed = self._current
        self._current = None
        return completed

    def dump_to_file(self, output_path: str | Path, task_id: Optional[uuid.UUID] = None) -> Path:
        """Dump diagnostics to a JSON file for debugging.

        Args:
            output_path: Directory or file path to write to.
            task_id: Optional filter — only dump diagnostics for this task.

        Returns:
            Path to the written file.
        """
        path = Path(output_path)

        # Filter diagnostics
        runs = list(self._history)
        if task_id:
            runs = [r for r in runs if r.task_id == task_id]

        data = {
            "generated_at": datetime.now(UTC).isoformat(),
            "total_runs": len(runs),
            "runs": [r.to_dict() for r in runs],
        }

        if path.is_dir():
            filename = f"diagnostics_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
            path = path / filename

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Diagnostics dumped to %s (%d runs)", path, len(runs))
        return path

    def get_current(self) -> Optional[RunDiagnostics]:
        """Get the current in-progress diagnostic run."""
        return self._current

    def get_history(self, limit: int = 10) -> list[RunDiagnostics]:
        """Get recent diagnostic runs."""
        runs = list(self._history)
        return runs[-limit:] if len(runs) > limit else runs


def _summarize_data(data: dict[str, Any], max_len: int = 200) -> str:
    """Create a concise summary of agent output data."""
    if not data:
        return ""
    keys = list(data.keys())
    summary = f"keys={keys}"
    if len(summary) > max_len:
        summary = summary[:max_len] + "..."
    return summary
