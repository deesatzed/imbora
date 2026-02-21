# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

The Associate is an autonomous software engineering agent orchestrator. It takes a problem spec, runs a 9-phase architecture design process (SOTAppR, phases 0-8), seeds executable tasks, then processes each task through a 5-agent pipeline that writes real code, runs real tests, and commits to a target git repository.

**Stack:** Python 3.12+, Click (CLI), Pydantic v2, PostgreSQL+pgvector, OpenRouter (LLM proxy), Tavily (web search), sentence-transformers (embeddings), httpx, psycopg3.

## Build & Development Commands

```bash
# Infrastructure
docker-compose up -d                    # Start PostgreSQL 17 + pgvector (port 5433)

# Install
pip install -e ".[dev]"                 # Editable install with dev deps

# Run tests
pytest                                  # All tests
pytest tests/test_agents_builder.py     # Single file
pytest -k "test_sentinel_drift"         # Pattern match
pytest --cov=src                        # With coverage
pytest --cov=src --cov-report=html      # HTML coverage report

# Lint & format
ruff check src/                         # Lint (line-length 100, Python 3.12 target, rules: E,F,I,N,W)
ruff format src/                        # Auto-format

# CLI — core commands (after install)
associate sotappr-build --spec spec.json --out report.json
associate sotappr-execute --spec spec.json --repo-path /path/to/repo --report-out report.json
associate sotappr-execute --spec spec.json --dry-run    # Seed tasks only, no execution
associate sotappr-models --criteria newness --limit 20  # Query live OpenRouter catalog
associate sotappr-status --limit 20
associate sotappr-dashboard
associate sotappr-portfolio              # List all projects with aggregate stats

# CLI — run management
associate sotappr-resume-run --run-id <id>       # Resume a paused run
associate sotappr-approve-task --task-id <id>    # Human approval gate
associate sotappr-reset-task-safe --task-id <id> # Reset stuck task to PENDING

# CLI — quality & observability
associate sotappr-benchmark-freeze       # Freeze current metrics as baseline
associate sotappr-benchmark-gate         # Compare metrics against frozen baseline
associate sotappr-drift-check            # Embedding drift against threshold
associate sotappr-chaos-drill            # Inject chaos for adversarial validation
associate sotappr-autotune               # Auto-tune thresholds from run history
associate sotappr-replay-search --query "..." # Search past run artifacts

# CLI — APC/1.0 packet commands
associate packet-evaluate-formula        # Evaluate xplurx formula packet
associate packet-transfer-arbitrate      # Rank/select TRANSFER packet candidates
associate packet-trace --run-id <id>     # Show packet event stream
associate packet-lineage-summary         # Transfer lineage and protocol effectiveness
associate packet-replay --run-id <id>    # Replay packet stream from DB to file
```

Use `--verbose` on any command for debug output.

### Required Environment Variables

```bash
OPENROUTER_API_KEY=sk-or-...           # LLM calls via OpenRouter
DATABASE_URL=postgresql://associate:associate@localhost:5433/the_associate
TAVILY_API_KEY=...                     # Web search for Research Unit
```

**Port note:** Docker exposes PostgreSQL on external port `5433` (mapped to internal 5432). The `DATABASE_URL` must use port `5433`. The `config/default.yaml` defaults to `port: 5432` which is the internal port — the `DATABASE_URL` env var override takes precedence.

### Test Configuration

Tests requiring external services are auto-skipped via markers in `conftest.py`:
- `requires_postgres` — skipped when PostgreSQL is unreachable
- `requires_openrouter` — skipped when `OPENROUTER_API_KEY` is not set

`conftest.py` also provides shared fixtures: `config_dir`, `app_config`, `model_registry`, `db_config`, `db_engine` (bootstraps real schema via `initialize_schema()`), `repository`, `sample_project`, `sample_task`. `.env` is loaded at import time via `python-dotenv`.

`asyncio_mode = "auto"` is set in `pyproject.toml` — `async def test_*` functions work without `@pytest.mark.asyncio`.

Integration tests in `test_orchestrator_loop_integration.py` use **real-shaped fakes** (FakeRepository, FakeBuilder, FakeStateManager) — these are in-memory implementations matching real interfaces, not mocks. They exercise actual TaskLoop logic.

## Architecture

### 5-Agent Pipeline

Each task flows through agents in this order:

```
State Manager → Research Unit → Librarian → Builder → Sentinel
                (optional)      (optional)
```

- **State Manager** (`src/agents/state_manager.py`): Creates git checkpoints, loads forbidden approaches from hypothesis log, builds `TaskContext`
- **Research Unit** (`src/agents/research_unit.py`): Live Tavily web search for current API docs. Returns `ResearchBrief`
- **Librarian** (`src/agents/librarian.py`): RAG context aggregation — merges past solutions, forbidden approaches, project rules, live docs into `ContextBrief` with retrieval strategy hints based on confidence/conflict signals. Also injects `skill_procedure` (from configurable `skills_dir`) and `complexity_tier`
- **Builder** (`src/agents/builder.py`): LLM code generation. Parses `--- FILE: path ---` blocks from output. Must produce structured sections: `## Approach`, `## Edge Cases`, `## Self-Audit`, `## Files`. Returns `BuildResult`. Receives budget-aware prompt hints from `budget_hints.py`
- **Sentinel** (`src/agents/sentinel.py`): 6 deterministic audit checks + optional LLM deep-check: dependency jail, style match, chaos check, placeholder scan, drift alignment (embedding similarity), claim validation. Supports task-type-specific rubrics via `sentinel_rubrics.py` (loads per-type JSON rubric files from configurable `rubrics_dir`)

On failure, the escalation ladder is: retry → **Council** trigger (at attempt 2) → STUCK (after 3 council invocations). Council (`src/agents/council.py`) uses a different LLM model family for diagnosis. When all retries exhaust, the **WrapUp** workflow (`src/orchestrator/wrapup.py`) asks Council to score partial artifacts at a relaxed quality threshold.

All agents extend `BaseAgent` (`src/agents/base_agent.py`) which provides lifecycle timing, metrics, and structured `AgentResult` returns.

### Inter-Agent Contracts (Pydantic Models)

All data flowing between agents is typed via models in `src/core/models.py`:

```
TaskContext → ResearchBrief → ContextBrief → BuildResult → SentinelVerdict
                                                        → CouncilDiagnosis (on failure)
```

Additional pipeline state: `ExecutionState` (typed shared state with task_id, run_id, trace_id, current_phase, attempt_number, token_budget_remaining, tokens_used, complexity_tier, quality_score).

Key enums: `TaskStatus` (6 states: `PENDING → RESEARCHING → CODING → REVIEWING → DONE | STUCK`), `ComplexityTier` (TRIVIAL/LOW/MEDIUM/HIGH/VERY_HIGH), `MethodologyType` (BUG_FIX/PATTERN/DECISION/GOTCHA).

APC/1.0 models: `PacketType`, `PacketChannel`, `RunPhase`, `DeliveryMode` enums; `PacketActor`, `PacketRecipient`, `PacketTrace`, `PacketRouting`, `AgentPacket` structs.

Legal task transitions enforced by `TaskRouter` (`src/orchestrator/task_router.py`).

### Agent Packet Protocol (APC/1.0)

`src/protocol/` implements the APC/1.0 inter-agent communication protocol:

- **`packet_runtime.py`** — `PacketRuntime`: state-machine-driven lifecycle engine. Loads policy from `config/contracts/packet_state_machine_v1.json`. Enforces phase policy (which packet types are legal per run phase), validates TRANSFER packet throttling rules, manages per-packet state transitions
- **`formula_evaluator.py`** — Evaluates xplurx-derived formula packets
- **`validation.py`** — Packet validation against `config/contracts/agent_packet_v1.schema.json`

The `config/contracts/` directory contains the two JSON contract files that define all protocol policy. `PacketRuntime` uses the state machine JSON as its sole source of truth.

The protocol specification is documented in `docs/protocols/AGENT_PACKET_PROTOCOL_V1.md`.

### Dependency Injection

`ComponentFactory.create()` in `src/core/factory.py` builds the entire dependency graph:

```
AppConfig + ModelRegistry → DatabaseEngine → Repository → EmbeddingEngine →
OpenRouterClient → ModelRouter → SecurityPolicy → HybridSearch →
MethodologyStore → HypothesisTracker → TavilyClient
```

All bundled in `ComponentBundle` dataclass and injected into agents via `__init__`. No global state.

### Config Cascade

Config loaded by `load_config()` in `src/core/config.py`:

```
config/default.yaml → config/{env}.yaml (deep merge) → DATABASE_URL env var override
```

Key sections in `default.yaml`: `database`, `llm` (timeouts, retries, cooldown), `search`, `embeddings` (model: `all-MiniLM-L6-v2`, dimension: 384), `orchestrator` (retry limits, council thresholds, token budget, arbitration settings, complexity scoring, prebuild self-correction), `sentinel`, `security` (autonomy level, command allowlist, forbidden paths), `sotappr` (governance pack, file/cost caps, wrapup workflow toggle, protocol effectiveness gates, protocol decay policy), `token_tracking` (enabled flag, JSONL path, cost rates), `logging`.

`config/production.yaml` overlays strict governance: human review gate, tighter file/cost caps.

### LLM Model Selection

`config/models.yaml` maps roles (builder, sentinel, librarian, council, research) to OpenRouter model IDs. **Model IDs are user-managed and must never be hardcoded or assumed.** The `sotappr-models` CLI command queries the live OpenRouter catalog and can update `models.yaml`. Fallback chains are defined per role.

`OpenRouterClient` (`src/llm/client.py`) handles retry with exponential backoff (2s base, 60s cap), per-model failure counting with cooldown (threshold: 2 failures, 90s cooldown), and fallback chains via `complete_with_fallback()`.

`TokenTracker` (`src/llm/token_tracker.py`) intercepts every LLM call, records input/output token counts and cost, persists to both PostgreSQL (`token_costs` table) and a JSONL shadow file.

### Memory System (Cross-Task Learning)

The anti-fragile core: every failure is persisted in `hypothesis_log` with a normalized `error_signature`. Before each Builder attempt, the Librarian injects forbidden approaches mined across **all** project tasks — not just the current task's history.

**Two-key hybrid search** (`src/memory/hybrid_search.py`):
- pgvector cosine similarity (60% weight) + PostgreSQL tsvector FTS (40% weight)
- Score agreement between backends produces confidence/conflict signals
- Signals propagate through Librarian into Builder prompt as `retrieval_strategy_hint`

`MethodologyStore` (`src/memory/methodology_store.py`) wraps Repository + EmbeddingEngine + HybridSearch for save/retrieve operations.

### SOTAppR Engine

`SOTAppRBuilder.build()` in `src/sotappr/engine.py` executes 9 phases with hard-stop quality gates:

| Phase | Name | Output |
|-------|------|--------|
| 0 | Alien Goggles | Root-need analysis, reframe |
| 1 | Contract Spec | QuadrupleContract (must/must-not/validate/refuse) |
| 2 | Divergent Architecture | Scored architecture sketches |
| 3 | Growth Layers | Backlog actions (seeded as TaskLoop tasks) |
| 4 | Vibe Autopsy | Findings + remediation |
| 5 | Chaos Playbook | Adversarial validation |
| 6 | Evolution Score | Future-proofing, migration paths |
| 7 | Ethics/Governance | ADR entries |
| 8 | Delivery | Final assembly |

`SOTAppRExecutor` (`src/sotappr/executor.py`) bridges SOTAppR reports into TaskLoop by creating Project records and seeding tasks from Phase 3 backlog_actions. `--dry-run` seeds without executing.

### Database

PostgreSQL + pgvector. Schema in `src/db/schema.sql` with 12 tables:

**Core tables:** `projects`, `tasks`, `hypothesis_log`, `methodologies`, `peer_reviews`, `context_snapshots`, `sotappr_runs`, `sotappr_artifacts`, `token_costs`.

**APC/1.0 tables:** `agent_packets` (6 indexes), `packet_events` (4 indexes), `packet_lineage` (2 indexes).

Repository (`src/db/repository.py`) provides the data access layer. `hypothesis_log` is indexed on `error_signature` for cross-task pattern mining. `methodologies` has a `search_vector tsvector` generated column with GIN index, plus `scope`, `methodology_type`, and `files_affected` columns.

**No migration framework** — schema is applied via raw SQL bootstrap in `DatabaseEngine.initialize_schema()` using `CREATE TABLE IF NOT EXISTS` and `ALTER TABLE IF NOT EXISTS` patterns.

### Security

`SecurityPolicy` (`src/security/policy.py`) enforces:
- Command allowlist (git, npm, cargo, pytest, python3, etc.)
- Forbidden paths (/etc, /root, /tmp, /var, etc.)
- Workspace sandboxing (all writes within `workspace_dir`)
- Path traversal prevention (symlink detection)
- Rate limiting (200 actions/hour)
- Three autonomy levels: `READ_ONLY`, `SUPERVISED`, `FULL`

### Orchestrator Subsystems

- **TaskLoop** (`src/orchestrator/loop.py`): Main loop — health check → fetch task → retry loop (State Manager → Research → Librarian → Builder → Sentinel) with escalation, token budget, SOTAppR governance
- **Pipeline** (`src/orchestrator/pipeline.py`): Agent pipeline coordinator — sequences the 5-agent chain for a single task attempt
- **BuildArbiter** (`src/orchestrator/arbitration.py`): Multi-candidate scoring when confidence is low (tests_signal 0.70, output_signal 0.25, reliability_signal 0.10, risk penalties)
- **HealthMonitor** (`src/orchestrator/health_monitor.py`): Stuck task detection, LLM circuit breaker (opens after 3 consecutive failures, cooldown = min(30s * count, 300s))
- **TaskRouter** (`src/orchestrator/task_router.py`): State machine enforcing legal task status transitions
- **BudgetHints** (`src/orchestrator/budget_hints.py`): Generates token-budget guidance strings injected into Builder system prompt based on remaining budget and attempt number
- **ComplexityScoring** (`src/orchestrator/complexity.py`): Keyword-heuristic classifier producing `ComplexityTier` (TRIVIAL through VERY_HIGH). Feeds model routing and Builder context
- **WrapUp** (`src/orchestrator/wrapup.py`): Partial-output salvage when Builder exhausts all retries. Asks Council to score partial artifacts at a relaxed quality threshold

### Workflow Module

`src/workflow/session.py` provides pre-execution planning: `SessionPlan`, `BudgetContract`, `IntakeCard`, `AlignmentSpec`, `ExecutionCharter`. Writes gate artifacts to a `knowledge/` directory.

## Entry Points

- **Installed CLI:** `associate` command → `the_associate/cli.py` (stable launcher that fixes `sys.path`) → `src/cli.py`
- **Direct dev:** `python3 -m src.cli`

## Prompt Templates

`config/prompts/` contains system prompts loaded by `PromptLoader` in `src/core/config.py`:
- `builder_system.txt` — Builder agent system prompt
- `council_system.txt` — Council agent system prompt
- `sentinel_deep_check.txt` — Sentinel deep-check template

## Vendored Projects

Two standalone projects are vendored in the repository. They are **not** part of the `associate` package and have their own dependencies/entry points:

- **`ClawWork/`** — Live economic benchmark framework (from HKUDS, MIT license). Transforms AI assistants into "AI coworkers" completing 220 GDPVal professional tasks across 44 occupations. Several `src/` modules cite ClawWork as their pattern source (wrapup, budget_hints, complexity, token_tracker, sentinel_rubrics). Has its own FastAPI + React dashboard (`./start_dashboard.sh`), E2B sandboxed execution, and Tavily/Jina search integration.
- **`xplurx/`** — Evolutionary language system for self-evident symbolic protocols. Conceptual foundation for the APC/1.0 Agent Packet Protocol. `src/protocol/formula_evaluator.py` evaluates xplurx-derived formulas.
- **`demo_app/`** — Scaffolded demonstration application generated by a prior SOTAppR execution run. Not integrated into the main package.

## Current Development State

**Coverage:** 94% at last committed baseline (with Docker/Postgres running); 83% without Docker (71 postgres-gated tests skip). Gap from 100% requires action plan per project policy.

**Lint status:** 91 active ruff errors (71 E501 line-length, 12 F401 unused imports, plus minor N/W/I issues).

**Upgrade roadmap:** `docs/OPENCLAW_MOLTBOT_SOTA_ALLOW_CODE_CHECKLIST.md` tracks a 6-phase plan toward OpenClaw/Moltbot-class reliability. Phase 0 complete (all 5 items); Phase 2 complete (all 5 items); Phase 1 (3/6 items); Phase 3 (3/6 items); Phases 4, 5, 6 untouched.

**Operational docs:**
- `docs/SOTAPPR_RUNBOOK.md` — operator workflow guide
- `docs/SOTAPPR_INCIDENT_REPORT_2026-02-15.md` — postmortem with RC1-RC5 mitigations
- `docs/BENCHMARKING.md` — benchmark suite documentation
- `docs/protocols/AGENT_PACKET_PROTOCOL_V1.md` — APC/1.0 specification
- `docs/ADAPTATION_REPORT_2026-02-18.md` — multi-AI adaptation session report

## Known Infrastructure Notes

- **No CI/CD pipeline** — `.github/workflows/` does not exist
- **No git at workspace root** — the `the-associateDS` directory is not itself a git repository; the inner project repo is at a different path
- **No migration framework** — schema changes are idempotent SQL with `IF NOT EXISTS` guards, no Alembic
