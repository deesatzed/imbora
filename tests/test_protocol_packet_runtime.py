"""Tests for APC/1.0 runtime policy + lifecycle state transitions."""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.protocol.packet_runtime import PacketPolicyError, PacketRuntime
from src.protocol.validation import validate_agent_packet


def _base_packet(packet_type: str = "SIGNAL", run_phase: str = "DISCOVERY") -> dict:
    root_id = str(uuid4())
    return {
        "protocol_version": "apc/1.0",
        "packet_id": str(uuid4()),
        "packet_type": packet_type,
        "channel": "INTER_AGENT",
        "run_phase": run_phase,
        "sender": {
            "agent_id": "orchestrator-1",
            "role": "orchestrator",
            "swarm_id": "associate-main",
        },
        "recipients": [{"agent_id": "builder-1", "role": "builder", "swarm_id": "associate-main"}],
        "trace": {
            "session_id": "sess-1",
            "run_id": "run-1",
            "task_id": "task-1",
            "root_packet_id": root_id,
            "generation": 1,
            "step": 1,
        },
        "routing": {
            "delivery_mode": "DIRECT",
            "priority": 5,
            "ttl_ms": 60_000,
            "requires_ack": False,
        },
        "confidence": 0.8,
        "symbolic_keys": ["k1"],
        "payload": {
            "signal": "READY",
            "reason": "test",
        },
    }


def _transfer_packet(
    *,
    run_phase: str,
    generation: int,
    sender_score: float = 0.9,
    receiver_score: float = 0.6,
    top_k: int = 2,
    mode: str = "SELECTIVE_IMPORT",
    cross_use_case: bool = False,
) -> dict:
    base = _base_packet(packet_type="TRANSFER", run_phase=run_phase)
    base["trace"]["generation"] = generation
    base["payload"] = {
        "protocol_id": "p-1",
        "from_swarm": "swarm-a",
        "to_swarm": "swarm-b",
        "sender_score": sender_score,
        "receiver_score": receiver_score,
        "transfer_policy": {
            "mode": mode,
            "top_k": top_k,
        },
        "accepted": True,
    }
    base["lineage"] = {
        "protocol_id": "p-1",
        "parent_protocol_ids": ["p-0"],
        "ancestor_swarms": ["swarm-a"],
        "cross_use_case": cross_use_case,
        "transfer_mode": mode,
    }
    return base


def test_packet_runtime_processes_signal_to_archived():
    runtime = PacketRuntime()
    packet = validate_agent_packet(_base_packet())
    state, history = runtime.process_packet(packet, promote=False, verification_pass=True)

    assert state == "ARCHIVED"
    assert history[0]["to"] == "DRAFT"
    assert history[-1]["to"] == "ARCHIVED"


def test_packet_runtime_rejects_phase_policy_violation():
    runtime = PacketRuntime()
    bad = _base_packet(packet_type="TRANSFER", run_phase="DISCOVERY")
    bad["payload"] = {
        "protocol_id": "p-1",
        "from_swarm": "a",
        "to_swarm": "b",
        "sender_score": 0.9,
        "receiver_score": 0.6,
        "transfer_policy": {"mode": "selective"},
    }
    bad["lineage"] = {
        "protocol_id": "p-1",
        "ancestor_swarms": ["a"],
        "cross_use_case": False,
    }
    packet = validate_agent_packet(bad)
    with pytest.raises(PacketPolicyError):
        runtime.process_packet(packet)


def test_eval_is_promotable_uses_ci_thresholds():
    runtime = PacketRuntime()
    assert runtime.eval_is_promotable(
        {
            "rtf": 1.0,
            "stability": 1.0,
            "audit": 1.0,
            "outcome": 0.8,
            "learnability": 0.9,
            "generalization_ratio": 0.9,
            "drift_risk": 0.1,
        }
    )
    assert not runtime.eval_is_promotable(
        {
            "rtf": 1.0,
            "stability": 0.2,
            "audit": 1.0,
            "outcome": 0.8,
            "learnability": 0.9,
            "generalization_ratio": 0.9,
            "drift_risk": 0.1,
        }
    )


def test_transfer_rejected_in_independent_discovery_window():
    runtime = PacketRuntime()
    packet = validate_agent_packet(
        _transfer_packet(
            run_phase="REFINEMENT",
            generation=8,
        )
    )
    with pytest.raises(PacketPolicyError, match="TRANSFER disabled for generation"):
        runtime.process_packet(packet)


def test_transfer_rejected_when_top_k_exceeds_limit():
    runtime = PacketRuntime()
    packet = validate_agent_packet(
        _transfer_packet(
            run_phase="CONVERGENCE",
            generation=15,
            top_k=3,
        )
    )
    with pytest.raises(PacketPolicyError, match="top_k"):
        runtime.process_packet(packet)


def test_transfer_rejected_when_promotion_mode_invalid():
    runtime = PacketRuntime()
    packet = validate_agent_packet(
        _transfer_packet(
            run_phase="PROMOTION",
            generation=25,
            mode="SELECTIVE_IMPORT",
        )
    )
    with pytest.raises(PacketPolicyError, match="mode must be WINNER_TO_LOSERS or COMMONS_SEED"):
        runtime.process_packet(packet)


def test_transfer_passes_for_valid_promotion_path():
    runtime = PacketRuntime()
    packet = validate_agent_packet(
        _transfer_packet(
            run_phase="PROMOTION",
            generation=25,
            mode="WINNER_TO_LOSERS",
            top_k=2,
            sender_score=0.9,
            receiver_score=0.7,
            cross_use_case=True,
        )
    )
    state, _history = runtime.process_packet(packet)
    assert state == "ARCHIVED"


def test_rank_transfer_candidates_orders_by_selection_score():
    runtime = PacketRuntime()
    p1 = validate_agent_packet(
        _transfer_packet(
            run_phase="REFINEMENT",
            generation=15,
            sender_score=0.9,
            receiver_score=0.5,
            top_k=2,
            mode="SELECTIVE_IMPORT",
        )
    )
    p2 = validate_agent_packet(
        _transfer_packet(
            run_phase="REFINEMENT",
            generation=15,
            sender_score=0.75,
            receiver_score=0.6,
            top_k=2,
            mode="SELECTIVE_IMPORT",
        )
    )

    ranked = runtime.rank_transfer_candidates([p2, p1])
    assert len(ranked) == 2
    assert ranked[0]["packet_id"] == str(p1.packet_id)
    assert ranked[0]["selection_score"] >= ranked[1]["selection_score"]
    assert isinstance(ranked[0]["recommended_accept"], bool)
