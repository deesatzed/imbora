# The Associate

An autonomous SWE agent orchestrator with cross-task failure learning, multi-layered code auditing, and architecture-first planning.

## What Makes This Different

**Cross-task failure pattern mining.** Every failed attempt is recorded with a normalized error signature and persisted to PostgreSQL. Before each build attempt, the system queries for recurring failure patterns across all tasks in the project — not just the current task — and injects them as forbidden approaches. If Task 1 failed with a missing import, Task 5 won't try the same import.

**6+1 Sentinel audit gate.** After the Builder generates code and tests pass, the Sentinel runs 6 deterministic checks on the diff: dependency jail, style match, chaos check (happy-path bias counter), placeholder scan, drift alignment (embedding-based semantic similarity between task intent and output), and claim validation (detects "production ready" or "tests pass" claims that contradict actual evidence). An optional 7th check calls a different LLM to review the diff.

**SOTAppR planning (phases 0-8).** Before code generation, projects go through a 9-phase architecture-first planning pipeline with typed artifacts, hard-stop quality gates, adversarial validation phases, and ethics governance.

For a technical deep-dive, see [Cross-Task Learning in Autonomous SWE Agents](docs/blog/cross-task-learning-swe-agents.md).

## Architecture

```
Task Queue ──► State Manager ──► Research Unit ──► Librarian ──► Builder ──► Sentinel
(PostgreSQL)       │                   │              │             │            │
                   │ TaskContext   ResearchBrief  ContextBrief  BuildResult  SentinelVerdict
                   │                                                           │
                   │                                              ┌── PASS ──► git commit
                   │                                              └── FAIL ──► retry
                   │                                                    │
                   └──────── Hypothesis Tracker ◄───────────────────────┘
                                    │
                               N failures?
                                    ▼
                              Council Agent
                              (2nd-opinion LLM)
```

**5 Agents:**

| Agent | Role |
|-------|------|
| **State Manager** | Creates git checkpoints, loads task context, fetches forbidden approaches |
| **Research Unit** | Live web search via Tavily for current API docs and version info |
| **Librarian** | Retrieves past solutions from methodology store (pgvector + full-text hybrid search) |
| **Builder** | Generates code via LLM, writes files, runs tests |
| **Sentinel** | 6 deterministic checks + optional LLM deep review on every diff |

Plus **Council** for escalation: when the Builder fails N times, a different LLM model provides a second opinion on what to try next.

**Data contracts:** All inter-agent communication uses typed Pydantic models (`TaskContext`, `ResearchBrief`, `ContextBrief`, `BuildResult`, `SentinelVerdict`, `CouncilDiagnosis`).

**Persistence:** PostgreSQL with pgvector. 8 tables covering projects, tasks, hypothesis log, context snapshots, methodologies (with 384-dim embeddings), peer reviews, and SOTAppR artifacts.

## Quickstart

### Prerequisites

- Python 3.12+
- Docker (for PostgreSQL with pgvector)
- An [OpenRouter](https://openrouter.ai) API key

### Setup

```bash
# Clone
git clone https://github.com/deesatzed/ralfed.git
cd ralfed/the-associate

# Start PostgreSQL with pgvector
docker compose up -d

# Configure environment
cp .env.example .env
# Edit .env: set OPENROUTER_API_KEY, optionally TAVILY_API_KEY

# Install
pip install -e .
```

### Run

```bash
# Execute SOTAppR planning + task execution against a target repo
associate sotappr-execute \
    --spec spec.json \
    --repo-path /path/to/target/repo \
    --report-out artifacts/sotappr_report.json \
    --test-command "pytest tests/ -v"
```

This will:
1. Run SOTAppR phases 0-8 to plan the architecture
2. Seed tasks from the growth layers
3. Execute each task through the full agent pipeline
4. Commit passing code to the target repo's git history

### Configuration

`.env` variables:

```
OPENROUTER_API_KEY=sk-or-v1-...    # Required — all LLM calls route through OpenRouter
DATABASE_URL=postgresql://...       # Default: associate:associate@localhost:5433/the_associate
TAVILY_API_KEY=...                  # Optional — enables Research Unit web search
```

Model selection is configured via the model router. All model choices are made by the user, not hardcoded.

## Verified Run Results

Tested against a fresh repository with 5 SOTAppR-seeded tasks:

| Metric | Value |
|--------|-------|
| Tasks seeded | 5 |
| Tasks completed | 5/5 |
| First-attempt success rate | 100% |
| Files generated | 21 |
| Lines of code | 2,019 |
| Test exit code | 0 |
| Council escalations | 0 |

## Test Coverage

669 tests passing, 71 skipped (Postgres-dependent tests skip when Docker is not running). Coverage at 83% without Docker; 94% at last committed baseline with Docker running.

**Known gaps:** `db/repository.py` and `db/engine.py` require live PostgreSQL. `memory/hypothesis_tracker.py` and `core/factory.py` need additional unit tests. Coverage action plan tracked in `docs/OPENCLAW_MOLTBOT_SOTA_ALLOW_CODE_CHECKLIST.md`.

## Project Structure

```
the-associate/
├── src/
│   ├── cli.py                # Click CLI — all user-facing commands (entry point)
│   ├── agents/               # 5 agents + base class
│   │   ├── base_agent.py     # Abstract base with lifecycle/metrics
│   │   ├── state_manager.py
│   │   ├── research_unit.py
│   │   ├── librarian.py      # RAG context aggregation + retrieval strategy hints
│   │   ├── builder.py        # LLM code generation + file parsing
│   │   ├── sentinel.py       # 6 deterministic + 1 LLM audit checks
│   │   └── council.py        # Escalation diagnosis (different model family)
│   ├── core/
│   │   ├── models.py         # All Pydantic data contracts
│   │   ├── config.py         # YAML cascade loader
│   │   ├── exceptions.py
│   │   └── factory.py        # ComponentFactory — DI for all infrastructure
│   ├── db/
│   │   ├── schema.sql        # PostgreSQL + pgvector schema (8 tables)
│   │   ├── engine.py         # psycopg3 connection management
│   │   ├── repository.py     # 37 CRUD methods + vector search
│   │   └── embeddings.py     # sentence-transformers
│   ├── llm/
│   │   ├── client.py         # OpenRouter HTTP client with retry/fallback/cooldown
│   │   ├── router.py         # Model selection by role
│   │   ├── model_catalog.py  # Live OpenRouter catalog queries
│   │   └── response_parser.py
│   ├── memory/
│   │   ├── hypothesis_tracker.py   # Cross-task failure patterns
│   │   ├── hybrid_search.py        # pgvector cosine (60%) + FTS BM25 (40%) merge
│   │   └── methodology_store.py    # Solution pattern storage
│   ├── orchestrator/
│   │   ├── loop.py           # TaskLoop — main retry loop with 5-agent pipeline
│   │   ├── arbitration.py    # BuildArbiter — multi-candidate scoring
│   │   ├── task_router.py    # Task state machine (PENDING→CODING→DONE/STUCK)
│   │   ├── pipeline.py       # Sequential agent execution
│   │   ├── health_monitor.py # Stuck detection, LLM circuit breaker
│   │   ├── diagnostics.py    # Run diagnostics + trace events
│   │   └── metrics.py
│   ├── security/
│   │   └── policy.py         # Workspace sandboxing, command allowlist
│   ├── sotappr/
│   │   ├── engine.py         # 9-phase planning pipeline (phases 0-8)
│   │   ├── executor.py       # SOTAppR → task execution bridge
│   │   ├── benchmark.py      # Baseline freeze + regression gate
│   │   ├── models.py         # Pydantic models for all phase artifacts
│   │   └── observability.py  # JSONL event + metrics emission
│   ├── workflow/
│   │   └── session.py        # Pre-execution planning (SessionPlan, BudgetContract)
│   └── tools/
│       ├── file_ops.py       # Sandboxed file I/O
│       ├── git_ops.py        # Git operations
│       ├── shell.py          # Command execution
│       └── search.py         # Web search (Tavily)
├── tests/                    # 46 test files
├── config/
│   ├── default.yaml          # Primary runtime config
│   ├── production.yaml       # Strict governance overlay
│   ├── models.yaml           # Role→model ID mapping (user-managed)
│   └── prompts/              # System prompts for builder, council, sentinel
├── docs/
│   ├── blog/                 # Technical deep-dive articles
│   ├── SOTAPPR_RUNBOOK.md
│   ├── BENCHMARKING.md
│   └── OPENCLAW_MOLTBOT_SOTA_ALLOW_CODE_CHECKLIST.md
├── docker-compose.yml        # PostgreSQL + pgvector
└── .env.example
```

## Showpiece: Charity Trivia Night

To see The Associate in action, we provide a ready-to-run showpiece project: a **Charity Trivia Night** platform built entirely by The Associate from a single JSON spec.

The spec describes a FastAPI + SQLite platform with multi-charity tenants, a trivia question bank (6 genres, 4 difficulty tiers, 20+ real questions), configurable events with purpose/seriousness levels, live game sessions with scoring, leaderboards, and a CLI admin interface.

```bash
# From the trivia-night repo (see ../trivia-night/)
associate --verbose sotappr-execute \
  --spec trivia_spec.json \
  --repo-path . \
  --test-command "python3 -m pytest tests/ -v" \
  --max-iterations 30
```

What you'll observe:
- **SOTAppR 8-phase planning** generates architecture contracts, adversarial validation, and ethics governance before a line of code is written
- **Sentinel rejections** when the Builder emits TODOs, bare excepts, or drifts from the task intent
- **Cross-task failure learning** when import errors from Task 1 are injected as warnings into Task 3
- **Council escalation** when a different LLM model diagnoses persistent failures

After completion, start the server (`uvicorn app.main:app`) and hit the interactive API docs at `/docs`.

See [`../trivia-night/SHOWPIECE.md`](../trivia-night/SHOWPIECE.md) for the full execution guide and demo script.

## CLI Commands

| Command | Purpose |
|---------|---------|
| `associate sotappr-build` | Run SOTAppR phases 0-8, produce report JSON |
| `associate sotappr-execute` | Build plan, seed tasks, execute TaskLoop |
| `associate mission-start` | Guided workflow: alignment → budget → knowledge → charter → execute |
| `associate sotappr-status` | Show task/run status summary |
| `associate sotappr-resume-run` | Resume a paused/failed run |
| `associate sotappr-approve-task` | Approve a task in human-review gate mode |
| `associate sotappr-reset-task-safe` | Reset task with attempt counter reconciliation |
| `associate sotappr-dashboard` | Dashboard view |
| `associate sotappr-models` | Query live OpenRouter catalog, optionally update models.yaml |
| `associate sotappr-benchmark-freeze` | Freeze baseline metrics snapshot |
| `associate sotappr-benchmark-gate` | Run regression gate against frozen baseline |
| `associate sotappr-portfolio` | Portfolio summary across projects |
| `associate sotappr-replay-search` | Search experience replay knowledge |
| `associate sotappr-autotune` | Threshold tuning recommendations |
| `associate sotappr-drift-check` | Drift signal analysis |
| `associate sotappr-chaos-drill` | Chaos drill checklist from Phase 5 |

## What's Not Here Yet

- No SWE-bench results (not benchmarked against standardized datasets)
- No interactive debugging (retries use error context, not breakpoints)
- No multi-repo support (single git repo per project)
- SOTAppR doesn't adapt phases to task size (all 8 phases run every time)
- No CI/CD pipeline

## Related Reading

- [Cross-Task Learning in Autonomous SWE Agents](docs/blog/cross-task-learning-swe-agents.md) — technical deep-dive on the three novel contributions
- [SOTAppR Runbook](docs/SOTAPPR_RUNBOOK.md) — operator guide for running SOTAppR
- [Incident Report](docs/SOTAPPR_INCIDENT_REPORT_2026-02-15.md) — root cause analysis from initial test runs
- [Benchmarking Guide](docs/BENCHMARKING.md) — SWE-bench evaluation + built-in regression gate
- [SOTA Upgrade Checklist](docs/OPENCLAW_MOLTBOT_SOTA_ALLOW_CODE_CHECKLIST.md) — roadmap toward OpenClaw/Moltbot-class reliability

## License

See repository root for license information.
