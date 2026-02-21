# SOTAppR Incident Report and Mitigation Plan

Date: 2026-02-15  
System: The Associate (`ralfed/the-associate`)  
Primary run: `c885b294-7339-4ed7-ac40-c847964e7280`  
Project: `28c66f92-8a34-48a3-a5e4-f80173422f50`  
Primary affected task: `d97b8f93-a483-47a3-9d2b-ecd8ec773b36` (`SOTAppR-2`)

## 1. Executive Summary

The run completed overall, but one core task became repeatedly failing before recovery:

- Run status: `completed`
- Tasks seeded: `5`
- Tasks processed: `6`
- Current task states:
  - `DONE`: 4 tasks
  - `REVIEWING`: 1 task (`SOTAppR-2`)
  - `STUCK`: 0 tasks

Main finding: this was **not purely model failure** and **not purely code failure**. It was a combined issue:

- Model behavior: exploratory loops with no code output on some attempts.
- Code/system behavior: low-quality error capture (`Unknown test failure`) and state/governance edge cases amplified retries.
- Ops behavior: manual recovery reset introduced attempt-number collision risk in hypothesis logging.

## 2. Evidence Snapshot

From `hypothesis_log` for task `d97b8f93-a483-47a3-9d2b-ecd8ec773b36`:

- Attempts 1-4: `FAILURE` with `error_signature = Unknown test failure`
- Attempt 5: `FAILURE` with `error_signature = SOTAppR governance rejected: changed 18 files (limit=15)`
- Model used on those failed attempts: `minimax/minimax-m2.5`

From `peer_reviews`:

- Council diagnosis explicitly states attempts were stuck in exploration/tool-use loops and did not produce implementation.
- Council recommended a test-first, contract-extraction approach.

Failure frequency for this project:

- `Unknown test failure`: 6
- `SOTAppR governance rejected: changed 18 files (limit=15)`: 1
- Success entries with empty error signature: 4

## 3. Root Causes (5)

## RC1. Error Signal Collapse (`Unknown test failure`)
Type: Code/observability defect  
Severity: High  
Owner: Orchestrator + Builder

### Why it happened
- Failure logging path defaulted to generic text when structured failure detail was missing.
- This masked whether the failure was no-code output, tool policy block, parser mismatch, or test failure.

### Impact
- Reduced diagnosability.
- Slower recovery and unnecessary retries.

### Mitigations (3)
1. Propagate `builder_result.error` into hypothesis logging path in orchestrator failure handling.
2. Add explicit failure reason codes in `BuildResult` (`no_parseable_file_blocks`, `policy_blocked_path`, `tests_failed_rc`).
3. Add tests asserting no generic `Unknown test failure` when known reason exists.

## RC2. Builder Attempt Strategy Drift (Explore vs Implement)
Type: Model + prompt/strategy mismatch  
Severity: High  
Owner: Builder prompts + model routing

### Why it happened
- Builder attempts entered repo/tool exploration loops and did not consistently produce code patches.
- Council confirmed non-implementation behavior.

### Impact
- Multiple non-productive retries.
- Triggered escalation and council invocation without new artifacts.

### Mitigations (3)
1. Harden builder prompt contract: require at least one concrete file block or explicit blocking reason.
2. Add orchestrator guardrail: if no files changed on attempt N, force strategy switch (test-contract-first template).
3. Prefer stronger builder model for complex synthesis tasks (already improved by switching builder role to `anthropic/claude-opus-4.6` during recovery).

## RC3. Governance Threshold Too Tight for Task Scope
Type: Config/policy tuning issue  
Severity: Medium  
Owner: SOTAppR governance config

### Why it happened
- Task required wide edits; governance limit was `15` files.
- Attempt touching 18 files was blocked.

### Impact
- Productive patch rejected even after iterative failures.

### Mitigations (3)
1. Keep production `max_files_changed_per_task` at `40` (already set in `config/production.yaml`).
2. Introduce task-class-based thresholds (feature task vs refactor task).
3. Add preflight chunking: split large patch plans into staged subtasks before builder execution.

## RC4. Orchestrator State Machine Edge Cases
Type: Code logic defect  
Severity: Medium  
Owner: Orchestrator

### Why it happened
- Missing legal transition (`CODING -> RESEARCHING`) for retry/fallback cycles.
- Duplicate transition attempt (`REVIEWING -> REVIEWING`) could raise invalid transition failures.

### Impact
- Increased chance of non-semantic task failure during retry paths.

### Mitigations (3)
1. Keep and verify transition fix allowing `CODING -> RESEARCHING`.
2. Keep and verify duplicate transition guard in success handler.
3. Add state-transition fuzz/property tests for retry/escalation paths.

## RC5. Recovery Workflow Can Violate Hypothesis Attempt Uniqueness
Type: Operations + data model risk  
Severity: Medium  
Owner: CLI/admin recovery tooling

### Why it happened
- Manual attempt counter reset can conflict with unique index `(task_id, attempt_number)` in `hypothesis_log`.
- This produced duplicate key warnings during resumed execution.

### Impact
- Lossy telemetry and noisy recovery runs.

### Mitigations (3)
1. Never reset attempts to 0 blindly; resume from `max(hypothesis_log.attempt_number)+1`.
2. Add safe admin command for task recovery that reindexes counters correctly.
3. Add duplicate-safe hypothesis write mode in orchestrator (`upsert` or retry with increment).

## 4. What Is Model-Based vs Code-Based vs Both

- Model-dominant:
  - Exploration-loop behavior without implementation output (RC2).
- Code/config-dominant:
  - Error signal collapse (RC1).
  - Transition edge cases (RC4).
  - Governance threshold mismatch (RC3; policy design).
  - Recovery counter collision risk (RC5).
- Combined:
  - RC2 + RC1 created a high-friction loop: weak model behavior plus weak error visibility.

## 5. Step-by-Step To-Do List

## Phase A: Stabilize diagnostics (do first)
1. Update orchestrator failure handling to log builder explicit error text.
2. Extend `BuildResult` with `failure_reason` enum and optional `failure_detail`.
3. Add unit tests covering each builder failure branch.
4. Run targeted tests for orchestrator + builder + logging paths.

Definition of done:
- New failures show actionable signatures instead of `Unknown test failure`.

## Phase B: Enforce productive attempt behavior
1. Tighten builder system prompt requirements around file block output.
2. Add no-output retry strategy switch in orchestrator.
3. Add max consecutive "no file changes" guard before forced council.
4. Add tests for no-file-output scenario and strategy switch.

Definition of done:
- No repeated attempts with zero code output without strategy shift.

## Phase C: Governance tuning and chunking
1. Keep production cap at `40` and document rationale in runbook.
2. Add configurable per-task-type edit limits.
3. Add preflight patch-scope estimator and automatic task split if above threshold.
4. Add integration test for large patch split behavior.

Definition of done:
- Large valid work is split, not rejected late.

## Phase D: State machine hardening
1. Keep current transition fixes and add explicit transition audit logs.
2. Add property tests for valid transitions under retries/escalations.
3. Add regression tests for duplicate `REVIEWING -> REVIEWING` cases.

Definition of done:
- No transition-related failures in stress test suite.

## Phase E: Recovery safety tooling
1. Add `sotappr-reset-task-safe` command with counter reconciliation logic.
2. Add DB-side helper query for next attempt number derivation.
3. Add duplicate-attempt conflict test.
4. Update runbook recovery section with safe procedure.

Definition of done:
- Resume/recovery produces no uniqueness warnings.

## 6. Immediate Operator Actions (Current Run)

1. Approve the remaining reviewing task:
   - `associate sotappr-approve-task --task-id d97b8f93-a483-47a3-9d2b-ecd8ec773b36`
2. Re-check run health:
   - `associate sotappr-status --run-id c885b294-7339-4ed7-ac40-c847964e7280`
3. Confirm review queue empty and all tasks `DONE`.

## 7. Implementation Priority

- P0: RC1 diagnostics fix, RC2 attempt guardrails.
- P1: RC4 state-machine property tests, RC5 safe recovery command.
- P2: RC3 dynamic governance/chunking improvements.

## 8. Success Metrics

Track over next 10 runs:

- `Unknown test failure` count: target `< 5%` of failures.
- Average attempts/task: target `< 2.0`.
- Governance late-rejection rate: target `< 2%`.
- Transition error count: target `0`.
- Recovery duplicate-key warnings: target `0`.

## 9. Progress Update (Implemented)

Implemented in codebase:

- P0 diagnostics:
  - Builder now emits explicit `failure_reason` and `failure_detail`.
  - Orchestrator failure logging now prioritizes agent/build-specific error details.
  - No-output failures now produce explicit signatures instead of generic unknowns.
- P0 guardrails:
  - Forced strategy injection now activates after no-output attempts to push test-contract-first implementation and require concrete file output.
  - Builder prompt and system prompt now require at least one file block when not blocked.
- P1 recovery safety:
  - Hypothesis attempt numbering now uses persisted global `task.attempt_count`, preventing duplicate `(task_id, attempt_number)` collisions across resumed retries.
  - Added safe repository reset primitive: `reset_task_for_retry(...)` with attempt reconciliation from hypothesis history.
  - Added CLI command: `sotappr-reset-task-safe`.
