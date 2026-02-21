"""Minimal observability exporters for SOTAppR workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class SOTAppRObservability:
    """Writes JSONL events and aggregate counters."""

    jsonl_path: Path
    metrics_path: Path
    counters: dict[str, int] = field(default_factory=dict)

    def emit_event(self, event_type: str, payload: dict) -> None:
        self.counters[event_type] = self.counters.get(event_type, 0) + 1
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def flush_metrics(self) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "timestamp": datetime.now(UTC).isoformat(),
            "counters": dict(sorted(self.counters.items())),
        }
        self.metrics_path.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
