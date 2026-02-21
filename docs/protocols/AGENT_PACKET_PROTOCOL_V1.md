# Agent Packet Protocol v1 (APC/1.0)

This document defines the machine-native communication layer for `the-associateDS` using xplurx concepts, with zero dependency on human-readable interpretation during execution.

## Goal

Enable inter-agent and intra-agent coordination via typed packets that exchange:
- Ideas (methods, assumptions, constraints, proofs), not full code by default
- Artifact references (hash-addressed outputs), not inline blobs
- Verifiable claims and evaluation evidence (CI-style gate scores)
- Transfer lineage and throttled protocol propagation (to avoid homogenization)

## xplurx Concepts Used

1. IR-first semantics: derived from `xplurx/evo/semantics/ir.py`
2. Patch-style communication: CLAIM/ARTIFACT/SIGNAL from `xplurx/docs/build_specs/08_SUBSTRATE_BUILD.md`
3. Protocol lineage and transfer events: `xplurx/evo/tournament/protocol_exchange.py`
4. Transfer throttling by generation window: `xplurx/PARAMETER_TUNING_GUIDE.md`
5. Fair differentiation and research briefs: `xplurx/evo/tournament/strategy_profiles.py`
6. CI gate invariants: RTF/STAB/AUD/OUT/LEARN from `xplurx/evo/ci/gates.py`
7. Anti-drift policy gates: `xplurx/docs/validation/11_ANTI_DRIFT_CHECKLIST.md`

## Artifacts Added

1. Schema: `config/contracts/agent_packet_v1.schema.json`
2. State machines: `config/contracts/packet_state_machine_v1.json`
3. Runtime policy engine: `src/protocol/packet_runtime.py`
4. Loop integration: `src/orchestrator/loop.py` (`_emit_agent_packet`)
5. First-class DB persistence: `src/db/schema.sql` + `src/db/repository.py`
6. Trace CLI: `associate packet-trace`

## Exact Envelope Contract

Every packet MUST validate against `apc/1.0` and include:
- `protocol_version`, `packet_id`, `packet_type`, `channel`, `run_phase`
- `sender`, `recipients`, `trace`, `routing`
- `confidence`, `symbolic_keys`, `payload`

Optional but strongly recommended:
- `proof_bundle` for CLAIM and EVAL
- `lineage` for TRANSFER

## Packet Types

Core packet types:
- `TASK_IR`: canonical task/problem IR bundle
- `IDEA`: method proposal packet (assumptions, failure modes, expected impact)
- `CLAIM`: hypothesis assertion with confidence
- `ARTIFACT`: hash-addressed artifact reference
- `SIGNAL`: control-plane coordination (START, PAUSE, VETO, etc.)
- `EVAL`: gate scores + thresholds + pass/fail
- `TRANSFER`: protocol sharing metadata with throttling policy
- `ARBITRATION`: candidate comparison and winner selection
- `DECISION`: execution decision and next actions
- `RETRACTION`: immutable correction packet (no mutation in place)
- `ACK`/`NACK`/`ERROR`/`HEARTBEAT`

## No-Human-Interpretation Design

The protocol enforces machine-native coordination by:
- Requiring typed payloads with schema validation
- Requiring falsifier/evidence structures for critical claims
- Passing references and metrics, not prose-only discussion
- Encoding transfer rules and phase constraints as explicit machine guards

## Inter-Agent vs Intra-Agent

- `channel=INTER_AGENT`: cross-agent communication (can be direct or substrate-backed)
- `channel=INTRA_AGENT`: internal module coordination within one agent (`recipients.maxItems=1`)

## Transfer Throttling Rules

Defined in `packet_state_machine_v1.json`:
- Generation 0-10: no transfer (independent discovery)
- Generation 11-20: selective transfer only if `sender_score > receiver_score * 1.2`, top-k capped
- Generation 21+: controlled convergence with top-k cap and optional cross-use-case transfer

This is directly adapted from xplurx anti-homogenization policy and is now enforced at runtime in
`src/protocol/packet_runtime.py` for `TRANSFER` packets.

## Lineage Analytics

First-class lineage analytics are available through:
- repository APIs: `get_packet_lineage_summary`, `list_protocol_propagation`
- CLI: `associate packet-lineage-summary`

The CLI reports per-protocol transfer counts, cross-use-case counts, transfer mode distribution,
and sample propagation rows across runs.

Transfer quality and drift correlation rollups are included when repository methods are available:
- `get_transfer_quality_rollup`
- `get_transfer_eval_correlation`
- `get_protocol_effectiveness`

Replay tooling:
- CLI: `associate packet-replay --run-id <uuid> [--task-id <uuid>]`
- Reconstructs ordered packet transitions and event stream from packet tables (or artifact fallback).
- Optional export: `associate packet-replay --run-id <uuid> --out artifacts/packets/replay.json`

Transfer arbitration tooling:
- CLI: `associate packet-transfer-arbitrate --packets-in <path>.json --top-k 2`
- Consumes multiple candidate TRANSFER packets, applies runtime throttling/policy checks,
  ranks by selection score, and outputs selected/rejected candidates.
- Runtime: `TaskLoop` promotion path uses the same arbitration pipeline and persists
  `transfer_arbitration_decision` artifacts for cross-run analytics.
- Runtime policy: low-effectiveness protocols can be decay-blocked before transfer emission,
  with `protocol_decay_decision` artifacts persisted for auditability.

## Validation Gates (Promotion Criteria)

A packet path can reach promotion only if:
- `rtf == 1.0`
- `stability >= 0.9`
- `audit == 1.0`
- `outcome >= 0.5`
- `learnability >= 0.7`
- `generalization_ratio >= 0.8`
- `drift_risk <= 0.3`

## State Machines

Three state machines are defined:
1. Packet lifecycle
2. Inter-agent coordination lifecycle
3. Intra-agent coordination lifecycle

### Packet Lifecycle Summary

`DRAFT -> NORMALIZED -> SCHEMA_VALIDATED -> POLICY_VALIDATED -> ROUTED -> DELIVERED -> APPLIED -> VERIFIED -> PROMOTED -> ARCHIVED`

Failure/side states:
`REJECTED_SCHEMA`, `REJECTED_POLICY`, `EXPIRED`, `SUPERSEDED`, `RETRACTED`

### Inter-Agent Lifecycle Summary

`SESSION_OPEN -> DIFFERENTIATED_RESEARCH -> INDEPENDENT_DISCOVERY -> SELECTIVE_TRANSFER -> CONTROLLED_CONVERGENCE -> ARBITRATION -> EXECUTION -> VALIDATION -> PROMOTION -> CLOSED`

Global halt path:
`* -> HALTED` on stop signal or budget exhaustion.

### Intra-Agent Lifecycle Summary

`LOCAL_IDLE -> CONTEXT_SYNC -> IDEA_SYNTHESIS -> SELF_CRITIQUE -> LOCAL_BUILD -> LOCAL_VERIFY -> PUBLISH -> WAIT_FOR_ACK -> LOCAL_IDLE`

Global halt path:
`* -> LOCAL_HALT` on cancel/error.

## Example IDEA Packet (Minimal)

```json
{
  "protocol_version": "apc/1.0",
  "packet_id": "7f0c2f5d-0f76-4a4c-93d0-f0b2e2891730",
  "packet_type": "IDEA",
  "channel": "INTER_AGENT",
  "run_phase": "DISCOVERY",
  "created_at": "2026-02-19T07:50:00Z",
  "sender": {
    "agent_id": "builder-1",
    "role": "builder",
    "swarm_id": "swarm-hybrid"
  },
  "recipients": [
    {
      "agent_id": "sentinel-1"
    }
  ],
  "trace": {
    "session_id": "sess-01",
    "run_id": "run-01",
    "task_id": "task-01",
    "root_packet_id": "7f0c2f5d-0f76-4a4c-93d0-f0b2e2891730",
    "generation": 4,
    "step": 12
  },
  "routing": {
    "delivery_mode": "DIRECT",
    "priority": 5,
    "ttl_ms": 60000,
    "requires_ack": true
  },
  "confidence": 0.74,
  "symbolic_keys": ["predictive", "feature-selection", "calibration"],
  "payload": {
    "method_id": "method-gbdt-calibrated-v2",
    "intent": "Use calibrated gradient boosting on time-split features.",
    "assumptions": [
      "Label delay <= 7 days",
      "No future leakage in feature windows"
    ],
    "required_capabilities": [
      "time_split_validation",
      "calibration",
      "drift_monitoring"
    ],
    "expected_impact": {
      "quality_gain": 0.12,
      "cost_change": 0.08,
      "risk_change": -0.10
    },
    "failure_modes": [
      {
        "id": "fm-1",
        "description": "Temporal leakage from joined aggregates",
        "mitigation": "Feature window materialization and leakage test"
      }
    ]
  }
}
```

## Next Wiring Steps in `the-associateDS`

1. Add packet bus abstraction for async routing (currently in-loop emit)
2. Feed packet replay outputs into adaptive strategy priors for future attempts
3. Add protocol decay/retirement policy for low-effectiveness lineage clusters
4. Extend dashboard surfacing for transfer arbitration decisions across runs

## Implementation Constraints

- Keep packet writes immutable; corrections only via `RETRACTION`
- Reject packets that fail phase policy, drift checks, or gate requirements
- Avoid free-form text as control input when typed payload exists
