# The Associate SOTA Plan (Allow-Code) Checklist

This is the execution checklist for upgrading `the-associate` toward OpenClaw/Moltbot-class reliability and capability.

## Phase 0 - Baseline and Control Metrics
- [x] Define benchmark suite and frozen eval inputs (quality, latency, cost, rollback rate).
- [x] Add run-level IDs and trace stitching across orchestrator, agents, and LLM calls.
- [x] Track confidence/conflict/retry-cause in diagnostics payloads.
- [x] Add dashboards or CLI summary for per-model failover and cooldown state.
- [x] Establish weekly regression gate on benchmark deltas.

## Phase 1 - Reliability Substrate (OpenClaw-style)
- [x] Add model failover cooldown policy in LLM client.
- [x] Add per-model failure counters and cooldown state introspection.
- [x] Add provider-failure classifier and feed LLM health failures into circuit breaker logic.
- [ ] Add lane-aware queueing for concurrent task classes (fast-path vs heavy tasks).
- [ ] Add supervisor no-output timeout and cancellation scopes for long-running tool/test commands.
- [ ] Add auth/profile rotation policy for upstream provider credentials.

## Phase 2 - Arbitration and Multi-Candidate Reasoning
- [x] Define candidate schema (strategy, evidence, risk, expected test impact, confidence).
- [x] Generate at least 2 builder candidates when confidence is low or conflict is high.
- [x] Implement council-aware arbitration scoring (tests, risk, policy, novelty penalties).
- [x] Add explicit tie-break and veto paths (Sentinel/Council veto authority).
- [x] Persist arbitration decisions for replay and tuning.

## Phase 3 - Active Memory Dynamics (VAMS-inspired)
- [x] Add retrieval confidence/conflict signals to hybrid memory results.
- [x] Aggregate memory signals into Librarian context.
- [x] Inject memory strategy hints into Builder prompt context.
- [ ] Add energy/convergence memory state and retrieval gating.
- [ ] Add contradiction detector for recalled methodologies before prompt composition.
- [ ] Add memory conflict telemetry into run diagnostics and retry strategy.

## Phase 4 - Clarification and Decision Intelligence (ASL/SOC-inspired)
- [ ] Add ambiguity detector before Builder execution.
- [ ] Add clarification question policy when spec ambiguity crosses threshold.
- [ ] Add unit/contract consistency checks before coding.
- [ ] Add decision-store with similarity retrieval for recurring failure modes.
- [ ] Feed prior decision outcomes into strategy selection.

## Phase 5 - OpenClaw/Moltbot Interop
- [ ] Add adapter layer for OpenClaw session-key and route semantics.
- [ ] Add memory plugin compatibility boundary for external stores.
- [ ] Align run lifecycle semantics with embedded-run queue/abort/wait behavior.
- [ ] Add portability tests running same task in native and OpenClaw-backed execution modes.

## Phase 6 - Hardening and SOTA Validation
- [ ] Run controlled A/B against current baseline across benchmark corpus.
- [ ] Tune thresholds: fallback, cooldown, conflict cutoffs, council trigger points.
- [ ] Expand fault-injection suite: provider outage, malformed outputs, no-file generations.
- [ ] Produce release candidate with migration notes and rollback plan.
- [ ] Publish architecture + benchmark report with reproducible artifacts.
