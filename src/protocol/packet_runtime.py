"""Runtime policy and lifecycle engine for APC/1.0 packets."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.models import AgentPacket, PacketType


def _default_state_machine_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "contracts" / "packet_state_machine_v1.json"


class PacketPolicyError(ValueError):
    """Raised when packet phase/type policy is violated."""


class PacketStateTransitionError(ValueError):
    """Raised when a lifecycle transition event is invalid for current state."""


class PacketRuntime:
    """Applies phase policy and packet lifecycle transitions from the JSON contract."""

    def __init__(self, spec_path: Path | None = None):
        path = spec_path or _default_state_machine_path()
        payload = json.loads(path.read_text(encoding="utf-8"))

        self._phase_policy: dict[str, dict[str, Any]] = dict(payload.get("phase_policy", {}))
        lifecycle = payload.get("state_machines", {}).get("packet_lifecycle", {})
        self._initial_state: str = str(lifecycle.get("initial_state", "DRAFT"))
        self._terminal_states: set[str] = set(lifecycle.get("terminal_states", []))
        self._ci_thresholds: dict[str, float] = {
            str(k): float(v) for k, v in dict(payload.get("ci_thresholds", {})).items()
        }
        self._transfer_throttling: dict[str, dict[str, Any]] = dict(payload.get("transfer_throttling", {}))

        self._transitions: dict[tuple[str, str], str] = {}
        for edge in lifecycle.get("transitions", []):
            from_state = str(edge.get("from", ""))
            event = str(edge.get("event", ""))
            to_state = str(edge.get("to", ""))
            if from_state and event and to_state:
                self._transitions[(from_state, event)] = to_state

        self._state_by_packet: dict[uuid.UUID, str] = {}
        self._history_by_packet: dict[uuid.UUID, list[dict[str, Any]]] = {}

    def validate_phase_policy(self, packet: AgentPacket) -> None:
        phase = packet.run_phase.value
        phase_cfg = self._phase_policy.get(phase)
        if phase_cfg is None:
            raise PacketPolicyError(f"Unknown run phase '{phase}'")

        allowed = set(phase_cfg.get("allowed_packet_types", []))
        if packet.packet_type.value not in allowed:
            raise PacketPolicyError(
                f"Packet type {packet.packet_type.value} not allowed in run phase {phase}"
            )

        if packet.packet_type == PacketType.TRANSFER:
            if not bool(phase_cfg.get("transfer_allowed", False)):
                raise PacketPolicyError(f"TRANSFER is not allowed in run phase {phase}")
            self._validate_transfer_policy(packet=packet, phase_cfg=phase_cfg)

    def _validate_transfer_policy(self, *, packet: AgentPacket, phase_cfg: dict[str, Any]) -> None:
        payload = packet.payload if isinstance(packet.payload, dict) else {}
        transfer_policy = payload.get("transfer_policy")
        transfer_policy = transfer_policy if isinstance(transfer_policy, dict) else {}

        sender_score = float(payload.get("sender_score", 0.0) or 0.0)
        receiver_score = float(payload.get("receiver_score", 0.0) or 0.0)
        top_k_raw = transfer_policy.get("top_k")
        top_k: int | None = None
        if isinstance(top_k_raw, (int, float, str)) and str(top_k_raw).strip():
            try:
                top_k = int(top_k_raw)
            except (TypeError, ValueError):
                top_k = None
        mode = (
            str(transfer_policy.get("mode") or "")
            or str(transfer_policy.get("transfer_mode") or "")
            or str((packet.lineage.transfer_mode if packet.lineage else "") or "")
        )
        generation = int(packet.trace.generation)

        window_name, window = self._generation_window(generation)
        if window is not None:
            if not bool(window.get("transfer_enabled", False)):
                raise PacketPolicyError(
                    f"TRANSFER disabled for generation {generation} ({window_name})"
                )
            multiplier = float(window.get("min_score_multiplier", 0.0) or 0.0)
            if multiplier > 0.0 and not (sender_score > receiver_score * multiplier):
                raise PacketPolicyError(
                    "TRANSFER rejected by throttling: sender_score must exceed "
                    f"receiver_score * {multiplier:.2f}"
                )
            top_k_limit = window.get("top_k")
            if top_k_limit is not None:
                if top_k is None:
                    raise PacketPolicyError(
                        f"TRANSFER requires transfer_policy.top_k for generation window {window_name}"
                    )
                if top_k > int(top_k_limit):
                    raise PacketPolicyError(
                        f"TRANSFER rejected: top_k={top_k} exceeds limit {int(top_k_limit)} in {window_name}"
                    )
            cross_allowed = bool(window.get("cross_use_case_allowed", False))
            if packet.lineage and packet.lineage.cross_use_case and not cross_allowed:
                raise PacketPolicyError(
                    f"TRANSFER rejected: cross_use_case not allowed in generation window {window_name}"
                )

        guard = str(phase_cfg.get("transfer_guard") or "").strip()
        if not guard:
            return
        if guard == "sender_score > receiver_score * 1.2":
            if not (sender_score > receiver_score * 1.2):
                raise PacketPolicyError(
                    "TRANSFER rejected by phase guard: sender_score > receiver_score * 1.2"
                )
            return
        if guard == "top_k <= 2":
            if top_k is None or top_k > 2:
                raise PacketPolicyError("TRANSFER rejected by phase guard: top_k <= 2")
            return
        if guard == "mode in [WINNER_TO_LOSERS, COMMONS_SEED]":
            allowed_modes = {"WINNER_TO_LOSERS", "COMMONS_SEED"}
            if mode not in allowed_modes:
                raise PacketPolicyError(
                    "TRANSFER rejected by phase guard: mode must be WINNER_TO_LOSERS or COMMONS_SEED"
                )
            return

    def _generation_window(self, generation: int) -> tuple[str, dict[str, Any] | None]:
        if generation <= 10:
            return "generation_0_10", self._transfer_throttling.get("generation_0_10")
        if generation <= 20:
            return "generation_11_20", self._transfer_throttling.get("generation_11_20")
        return "generation_21_plus", self._transfer_throttling.get("generation_21_plus")

    def _transfer_selection_metadata(self, packet: AgentPacket) -> dict[str, Any]:
        payload = packet.payload if isinstance(packet.payload, dict) else {}
        sender_score = float(payload.get("sender_score", 0.0) or 0.0)
        receiver_score = float(payload.get("receiver_score", 0.0) or 0.0)
        delta = sender_score - receiver_score
        cross = bool(packet.lineage.cross_use_case) if packet.lineage else False
        score = max(0.0, delta) + (0.25 * float(packet.confidence)) + (0.05 if cross else 0.0)
        _window_name, window = self._generation_window(int(packet.trace.generation))
        selector = None
        if window is not None:
            selector = window.get("selector")
        if selector is None:
            selector = "phase_guard_only"
        return {
            "selector": selector,
            "selection_score": round(float(score), 6),
            "sender_score": sender_score,
            "receiver_score": receiver_score,
            "score_delta": round(float(delta), 6),
            "recommended_accept": bool(score > 0.0 and delta > 0.0),
        }

    def rank_transfer_candidates(self, packets: list[AgentPacket]) -> list[dict[str, Any]]:
        """Score and rank validated TRANSFER packets for selector-based routing."""
        ranked: list[dict[str, Any]] = []
        for packet in packets:
            if packet.packet_type != PacketType.TRANSFER:
                continue
            self.validate_phase_policy(packet)
            metadata = self._transfer_selection_metadata(packet)
            ranked.append(
                {
                    "packet_id": str(packet.packet_id),
                    "to_swarm": str((packet.payload or {}).get("to_swarm", "")),
                    **metadata,
                }
            )
        ranked.sort(key=lambda row: float(row.get("selection_score") or 0.0), reverse=True)
        return ranked

    def start_packet(self, packet_id: uuid.UUID) -> str:
        self._state_by_packet[packet_id] = self._initial_state
        self._history_by_packet[packet_id] = [
            {
                "at": datetime.now(UTC).isoformat(),
                "event": "init",
                "from": None,
                "to": self._initial_state,
            }
        ]
        return self._initial_state

    def transition(
        self,
        packet_id: uuid.UUID,
        event: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if packet_id not in self._state_by_packet:
            self.start_packet(packet_id)

        current = self._state_by_packet[packet_id]
        next_state = self._transitions.get((current, event))
        if next_state is None:
            next_state = self._transitions.get(("*", event))
        if next_state is None:
            raise PacketStateTransitionError(
                f"Invalid packet transition: state={current}, event={event}"
            )

        self._state_by_packet[packet_id] = next_state
        event_record: dict[str, Any] = {
            "at": datetime.now(UTC).isoformat(),
            "event": event,
            "from": current,
            "to": next_state,
        }
        if metadata:
            event_record["metadata"] = dict(metadata)
        self._history_by_packet.setdefault(packet_id, []).append(event_record)
        return next_state

    def get_state(self, packet_id: uuid.UUID) -> str:
        return self._state_by_packet.get(packet_id, self._initial_state)

    def get_history(self, packet_id: uuid.UUID) -> list[dict[str, Any]]:
        return list(self._history_by_packet.get(packet_id, []))

    def is_terminal(self, state: str) -> bool:
        return state in self._terminal_states

    def eval_is_promotable(self, metrics: dict[str, float]) -> bool:
        if not self._ci_thresholds:
            return False
        checks = [
            float(metrics.get("rtf", -1.0)) >= self._ci_thresholds.get("rtf", 1.0),
            float(metrics.get("stability", -1.0)) >= self._ci_thresholds.get("stability", 0.9),
            float(metrics.get("audit", -1.0)) >= self._ci_thresholds.get("audit", 1.0),
            float(metrics.get("outcome", -1.0)) >= self._ci_thresholds.get("outcome", 0.5),
            float(metrics.get("learnability", -1.0)) >= self._ci_thresholds.get("learnability", 0.7),
            float(metrics.get("generalization_ratio", -1.0))
            >= self._ci_thresholds.get("generalization_ratio", 0.8),
            float(metrics.get("drift_risk", 1.0)) <= self._ci_thresholds.get("max_drift_risk", 0.3),
        ]
        return all(checks)

    def process_packet(
        self,
        packet: AgentPacket,
        *,
        promote: bool = False,
        verification_pass: bool = True,
    ) -> tuple[str, list[dict[str, Any]]]:
        packet_id = packet.packet_id
        self.start_packet(packet_id)
        self.transition(packet_id, "emit")
        self.transition(packet_id, "schema_pass")

        try:
            self.validate_phase_policy(packet)
        except PacketPolicyError:
            self.transition(packet_id, "policy_fail")
            raise

        transfer_metadata = None
        if packet.packet_type == PacketType.TRANSFER:
            transfer_metadata = self._transfer_selection_metadata(packet)
        self.transition(packet_id, "policy_pass", metadata=transfer_metadata)
        self.transition(packet_id, "route_selected")
        self.transition(packet_id, "delivery_ack")
        self.transition(packet_id, "consumer_apply")

        if not verification_pass:
            self.transition(packet_id, "verification_fail")
            return self.get_state(packet_id), self.get_history(packet_id)

        self.transition(packet_id, "verification_pass")
        if promote:
            self.transition(packet_id, "promotion_eligible")
            self.transition(packet_id, "persisted")
        else:
            self.transition(packet_id, "non_promotable_complete")

        return self.get_state(packet_id), self.get_history(packet_id)
