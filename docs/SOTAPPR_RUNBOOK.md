# SOTAppR Runbook

This runbook defines the standard UX workflow for running and operating the SOTAppR autonomous SWE builder in The Associate.

## 1. Prepare

- Ensure PostgreSQL is reachable (schema auto-initializes on CLI command startup).
- Ensure `OPENROUTER_API_KEY` is set for real execution.
- Choose config profile:
  - default: `config/default.yaml`
  - production-like: `config/production.yaml` (stricter governance + human gate)

## 2. Build and Execute

Run build + seed + execute:

```bash
associate sotappr-execute \
  --spec spec.json \
  --repo-path /path/to/repo \
  --report-out artifacts/sotappr_report.json \
  --export-tasks artifacts/seeded_tasks.json
```

Run build + seed only (no loop execution):

```bash
associate sotappr-execute \
  --spec spec.json \
  --repo-path /path/to/repo \
  --dry-run
```

## 3. Observe and Control

List latest runs:

```bash
associate sotappr-status --limit 20
```

Inspect a specific run and review queue:

```bash
associate sotappr-status --run-id <RUN_UUID>
```

Resume a paused/failed run:

```bash
associate sotappr-resume-run --run-id <RUN_UUID> --max-iterations 100
```

Approve a task in human-review gate mode:

```bash
associate sotappr-approve-task --task-id <TASK_UUID>
```

Reset a task safely for retry (reconciles attempt counters to avoid hypothesis collisions):

```bash
associate sotappr-reset-task-safe --task-id <TASK_UUID> --status PENDING
```

## 4. Operational Telemetry

Dashboard:

```bash
associate sotappr-dashboard
```

Portfolio summary:

```bash
associate sotappr-portfolio
```

Search experience replay knowledge:

```bash
associate sotappr-replay-search --query "timeout"
```

Autotune recommendations:

```bash
associate sotappr-autotune
```

Drift signal:

```bash
associate sotappr-drift-check
```

Chaos drill checklist from Phase 5:

```bash
associate sotappr-chaos-drill --run-id <RUN_UUID>
```

## 5. Live Model Selection

List the latest OpenRouter models ranked by newness:

```bash
associate sotappr-models --criteria newness --limit 20
```

Rank by cost (or only paid models):

```bash
associate sotappr-models --criteria cost --only-paid --limit 20
```

Rank by latency (probes recent models for `latency_last_30m`):

```bash
associate sotappr-models --criteria latency --limit 20 --latency-probe-limit 80
```

Interactively select and apply a model to a role in `config/models.yaml`:

```bash
associate sotappr-models --criteria newness --choose --role builder
```

## 6. Mission Workflow

For guided, gate-by-gate execution with explicit user alignment:

```bash
associate mission-start \
  --task "Build a REST API for user management" \
  --target-path /path/to/repo
```

This runs an intake sequence: alignment spec → budget contract → knowledge manifest → execution charter. Add `--execute` to proceed into TaskLoop after charter approval.

## 7. Benchmark Regression Gate

Freeze a baseline snapshot from completed runs:

```bash
associate sotappr-benchmark-freeze \
  --project-id <PROJECT_UUID> \
  --limit 20 \
  --statuses completed,paused \
  --out artifacts/sotappr/frozen_benchmark.json
```

Run regression gate against the frozen baseline:

```bash
associate sotappr-benchmark-gate \
  --baseline artifacts/sotappr/frozen_benchmark.json \
  --project-id <PROJECT_UUID>
```

Gate checks: quality success rate, average elapsed hours, average estimated cost, retry events per task, rollback rate.

## 8. Governance Controls

Governance controls are configured in `sotappr` config:

- `governance_pack`
- `require_human_review_before_done`
- `max_files_changed_per_task`
- `protected_paths`
- `max_estimated_cost_per_task_usd`
- `estimated_cost_per_1k_tokens_usd`

When limits are exceeded, tasks are prevented from auto-completion and execution is constrained.

## 9. Recommended Operator Cadence

- Every run: check `sotappr-status` and `sotappr-dashboard`.
- Daily: review `sotappr-portfolio` and `sotappr-drift-check`.
- Weekly: run `sotappr-chaos-drill` and review `sotappr-replay-search` insights.
