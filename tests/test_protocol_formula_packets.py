"""Tests for APC/1.0 formula packet validation and evaluation flow."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from src.cli import cli
from src.core.models import AgentPacket
from src.protocol.formula_evaluator import evaluate_formula_packet
from src.protocol.validation import validate_agent_packet


def _idea_formula_packet() -> dict:
    root_id = str(uuid4())
    return {
        "protocol_version": "apc/1.0",
        "packet_id": root_id,
        "packet_type": "IDEA",
        "channel": "INTER_AGENT",
        "run_phase": "DISCOVERY",
        "created_at": "2026-02-19T08:00:00Z",
        "sender": {
            "agent_id": "builder-1",
            "role": "builder",
            "swarm_id": "swarm-a",
        },
        "recipients": [{"agent_id": "validator-1"}],
        "trace": {
            "session_id": "sess-1",
            "run_id": "run-1",
            "task_id": "task-1",
            "root_packet_id": root_id,
            "generation": 4,
            "step": 10,
        },
        "routing": {
            "delivery_mode": "DIRECT",
            "priority": 5,
            "ttl_ms": 60000,
            "requires_ack": True,
        },
        "confidence": 0.75,
        "symbolic_keys": ["formula", "mc"],
        "payload": {
            "method_id": "method-1",
            "intent": "Stress-test a formula bundle with Monte Carlo simulation.",
            "assumptions": ["time-invariant parameters"],
            "required_capabilities": ["ode-solver", "stability-analysis"],
            "expected_impact": {
                "quality_gain": 0.2,
                "cost_change": 0.1,
                "risk_change": -0.1,
            },
            "failure_modes": [
                {
                    "id": "fm-1",
                    "description": "nonlinear blowup",
                    "mitigation": "boundedness check",
                }
            ],
            "formula_bundle": {
                "representation_version": "formula_bundle/1.0",
                "latex_full": r"\\frac{dx}{dt} = a x - b x y - e x^2",
                "sympy_str": "Eq(Derivative(x,t), a*x - b*x*y - e*x**2)",
                "ast_json": {"type": "ODESystem", "vars": ["x", "y"]},
                "domains": ["data-science", "dynamics"],
                "var_ontology": {"x": "state", "y": "volatility"},
                "functor_map": "ds -> ode",
                "ode_system": {
                    "state_variables": ["x", "y"],
                    "equations": [
                        "a*x - b*x*y - e*(x**2)",
                        "c*x*y - d*y",
                    ],
                    "parameters": {"a": 1.2, "b": 0.7, "c": 0.6, "d": 0.5, "e": 0.02},
                    "initial_state": {"x": 1.0, "y": 1.0},
                    "time_start": 0.0,
                    "time_end": 20.0,
                    "num_points": 120,
                },
            },
        },
    }


def test_agent_packet_accepts_formula_bundle_payload():
    packet = AgentPacket.model_validate(_idea_formula_packet())
    assert packet.packet_type.value == "IDEA"
    assert packet.payload["formula_bundle"]["ode_system"]["state_variables"] == ["x", "y"]


def test_agent_packet_requires_proof_bundle_for_eval():
    packet = _idea_formula_packet()
    packet["packet_type"] = "EVAL"
    packet["run_phase"] = "VALIDATION"
    packet["payload"] = {
        "metric_set": {
            "rtf": 1.0,
            "stability": 1.0,
            "audit": 1.0,
            "outcome": 0.8,
            "learnability": 0.8,
            "generalization_ratio": 0.9,
        },
        "thresholds": {
            "rtf": 1.0,
            "stability": 0.9,
            "audit": 1.0,
            "outcome": 0.5,
            "learnability": 0.7,
            "generalization_ratio": 0.8,
        },
        "pass": True,
    }
    with pytest.raises(ValidationError):
        AgentPacket.model_validate(packet)


def test_formula_evaluator_returns_metric_bundle():
    packet = validate_agent_packet(_idea_formula_packet())
    result = evaluate_formula_packet(packet, mc_runs=30, seed=7)
    assert "outcome" in result.metrics
    assert 0.0 <= result.metrics["outcome"] <= 1.0
    assert result.artifact.mc_runs == 30
    assert "x" in result.artifact.mean_final
    assert "y" in result.artifact.mean_final


def test_cli_packet_evaluate_formula_writes_followup_packets(tmp_path: Path):
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_idea_formula_packet(), indent=2), encoding="utf-8")
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "packet-evaluate-formula",
            "--packet-in",
            str(packet_path),
            "--out-dir",
            str(out_dir),
            "--mc-runs",
            "25",
            "--seed",
            "9",
        ],
    )

    assert result.exit_code == 0, result.output
    files = list(out_dir.glob("formula_eval_packets_*.json"))
    assert files, "Expected emitted packet bundle file"
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["artifact_packet"]["packet_type"] == "ARTIFACT"
    assert payload["eval_packet"]["packet_type"] == "EVAL"

