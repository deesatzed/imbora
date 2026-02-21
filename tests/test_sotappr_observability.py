"""Tests for SOTAppR observability exporters (JSONL events + metrics flush)."""

from __future__ import annotations

import json

from src.sotappr.observability import SOTAppRObservability


class TestEmitEvent:
    def test_emit_event_writes_jsonl(self, tmp_path):
        jsonl = tmp_path / "events.jsonl"
        metrics = tmp_path / "metrics.json"
        obs = SOTAppRObservability(jsonl_path=jsonl, metrics_path=metrics)

        obs.emit_event("build_start", {"task": "A"})
        obs.emit_event("build_end", {"task": "A", "ok": True})

        lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

        for line in lines:
            record = json.loads(line)
            assert "event_type" in record
            assert "timestamp" in record
            assert "payload" in record

        assert json.loads(lines[0])["event_type"] == "build_start"
        assert json.loads(lines[1])["event_type"] == "build_end"

    def test_emit_event_increments_counters(self, tmp_path):
        jsonl = tmp_path / "events.jsonl"
        metrics = tmp_path / "metrics.json"
        obs = SOTAppRObservability(jsonl_path=jsonl, metrics_path=metrics)

        obs.emit_event("task_done", {})
        obs.emit_event("task_done", {})
        obs.emit_event("task_fail", {})

        assert obs.counters["task_done"] == 2
        assert obs.counters["task_fail"] == 1

    def test_emit_event_creates_parent_dirs(self, tmp_path):
        jsonl = tmp_path / "deep" / "nested" / "events.jsonl"
        metrics = tmp_path / "metrics.json"
        obs = SOTAppRObservability(jsonl_path=jsonl, metrics_path=metrics)

        obs.emit_event("init", {"v": 1})

        assert jsonl.exists()
        record = json.loads(jsonl.read_text(encoding="utf-8").strip())
        assert record["event_type"] == "init"


class TestFlushMetrics:
    def test_flush_metrics_writes_json(self, tmp_path):
        jsonl = tmp_path / "events.jsonl"
        metrics = tmp_path / "metrics.json"
        obs = SOTAppRObservability(jsonl_path=jsonl, metrics_path=metrics)

        obs.emit_event("a", {})
        obs.emit_event("a", {})
        obs.emit_event("b", {})
        obs.flush_metrics()

        snapshot = json.loads(metrics.read_text(encoding="utf-8"))
        assert "timestamp" in snapshot
        assert "counters" in snapshot
        assert snapshot["counters"]["a"] == 2
        assert snapshot["counters"]["b"] == 1

    def test_flush_metrics_creates_parent_dirs(self, tmp_path):
        jsonl = tmp_path / "events.jsonl"
        metrics = tmp_path / "a" / "b" / "metrics.json"
        obs = SOTAppRObservability(jsonl_path=jsonl, metrics_path=metrics)

        obs.emit_event("x", {})
        obs.flush_metrics()

        assert metrics.exists()
        snapshot = json.loads(metrics.read_text(encoding="utf-8"))
        assert snapshot["counters"]["x"] == 1
