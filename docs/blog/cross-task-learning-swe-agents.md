# Cross-Task Learning in Autonomous SWE Agents: Lessons from Building The Associate

**Date**: 2026-02-15
**Authors**: The Associate Contributors
**Tags**: SWE agents, autonomous coding, failure analysis, code auditing, software architecture

---

## Abstract

Most autonomous SWE agents treat each task as isolated: the agent receives a description, generates code, runs tests, and either succeeds or retries with the same blank slate. Failure information dies with the task.

The Associate takes a different approach. It persists every failed attempt — normalized, categorized, and cross-referenced across all tasks in a project — then feeds those patterns back into future attempts as explicit "forbidden approaches." Combined with a multi-layered audit gate and an architecture-first planning phase, this creates a system that genuinely learns within a project scope.

This post describes three design decisions in The Associate that we believe are novel contributions to the SWE agent field, with code from the actual implementation.

---

## The Problem: Why SWE Agents Repeat Themselves

Consider what happens when an SWE agent encounters the same class of error across different tasks:

- Task 1: "Add user authentication" — fails because the agent tries `import bcrypt` but the project uses `passlib`
- Task 3: "Add password reset" — fails with the same `import bcrypt` error
- Task 5: "Add session management" — tries `import bcrypt` again

The agent has no memory. Each task starts fresh. The same import error wastes three attempts across three different tasks before a human intervenes.

SWE-Agent, OpenHands, Aider, and Cline all exhibit this pattern. They optimize within a single task (retrying with a modified prompt), but they don't transfer failure knowledge between tasks. This is not a critique of those tools — they solve different problems — but it identifies a gap.

---

## Contribution 1: Cross-Task Failure Pattern Mining

The Associate's `HypothesisTracker` records every build attempt with a normalized error signature:

```python
# src/memory/hypothesis_tracker.py

class HypothesisTracker:
    def record_attempt(
        self,
        task_id: uuid.UUID,
        attempt_number: int,
        approach_summary: str,
        outcome: HypothesisOutcome,
        error_signature: Optional[str] = None,
        error_full: Optional[str] = None,
        files_changed: Optional[list[str]] = None,
        duration_seconds: Optional[float] = None,
        model_used: Optional[str] = None,
    ) -> HypothesisEntry:
        entry = HypothesisEntry(
            task_id=task_id,
            attempt_number=attempt_number,
            approach_summary=approach_summary,
            outcome=outcome,
            error_signature=error_signature,
            error_full=error_full,
            files_changed=files_changed or [],
            duration_seconds=duration_seconds,
            model_used=model_used,
        )
        return self.repository.log_hypothesis(entry)
```

Each entry is persisted to PostgreSQL, not ephemeral memory. The `error_signature` field is the key to cross-task deduplication.

### Error Normalization

Raw error messages contain paths, line numbers, UUIDs, and timestamps that differ between occurrences of the same logical error. The normalizer strips these:

```python
# src/memory/hypothesis_tracker.py

def normalize_error_for_dedup(error_text: str) -> str:
    sig = error_text.strip()

    # Remove UUIDs
    sig = re.sub(
        r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',
        '<UUID>', sig,
    )

    # Remove quoted strings
    sig = re.sub(r"'[^']*'", "'<STR>'", sig)
    sig = re.sub(r'"[^"]*"', '"<STR>"', sig)

    # Apply base normalization (timestamps, paths, line numbers, addresses)
    sig = normalize_error_signature(sig)

    # Remove remaining standalone numbers
    sig = re.sub(r'\b\d+\b', '<NUM>', sig)

    # Collapse repeated normalized tokens
    sig = re.sub(r'(<\w+>)\s*\1+', r'\1', sig)

    return sig
```

After normalization, `ModuleNotFoundError: No module named 'bcrypt'` and `ModuleNotFoundError: No module named 'passlib'` both become `ModuleNotFoundError: No module named '<STR>'` — the same signature.

### Cross-Task Pattern Detection

The `get_common_failure_patterns()` method groups normalized errors across all tasks in a project:

```python
# src/memory/hypothesis_tracker.py

def get_common_failure_patterns(
    self, project_id: uuid.UUID, min_count: int = 2,
) -> list[FailurePattern]:
    # Get all tasks for this project
    all_tasks = []
    for status in all_statuses:
        all_tasks.extend(self.repository.get_tasks_by_status(project_id, status))

    # Collect all failures across all tasks
    error_groups: dict[str, list[tuple[uuid.UUID, HypothesisEntry]]] = defaultdict(list)
    success_map: dict[str, list[str]] = defaultdict(list)

    for task in all_tasks:
        failed = self.repository.get_failed_approaches(task.id)
        for entry in failed:
            if entry.error_signature:
                error_groups[entry.error_signature].append((task.id, entry))

        # Check for successful resolutions of the same errors
        if task.status == TaskStatus.DONE:
            for entry in failed:
                if entry.error_signature:
                    success_map[entry.error_signature].append(
                        f"Task '{task.title}' resolved this error"
                    )

    # Build failure patterns with urgency
    patterns = []
    for error_sig, occurrences in error_groups.items():
        if len(occurrences) < min_count:
            continue

        task_ids = {task_id for task_id, _ in occurrences}
        urgency = _calculate_urgency(len(occurrences), len(task_ids))
        resolution = success_map.get(error_sig, [None])[0]

        patterns.append(FailurePattern(
            error_signature=error_sig,
            count=len(occurrences),
            task_ids=task_ids,
            category=_categorize_error(error_sig),
            urgency=urgency,
            example_approaches=[e.approach_summary for _, e in occurrences],
            successful_resolution=resolution,
        ))
    return patterns
```

Urgency is calculated based on frequency and spread:

```python
def _calculate_urgency(count: int, num_tasks: int) -> str:
    if count >= 5 or num_tasks >= 3:
        return "critical"
    if count >= 3 or num_tasks >= 2:
        return "high"
    if count >= 2:
        return "medium"
    return "low"
```

### How It Feeds Back

Before each build attempt, the orchestrator calls `get_enriched_forbidden_approaches()` which combines task-local failures with project-wide patterns:

```python
def get_enriched_forbidden_approaches(
    self, task_id: uuid.UUID, project_id: uuid.UUID
) -> list[str]:
    # Task-local forbidden approaches
    local = self.get_forbidden_approaches(task_id)

    # Project-wide recurring patterns
    patterns = self.get_common_failure_patterns(project_id, min_count=2)

    project_warnings = []
    for pattern in patterns:
        if pattern.urgency in ("critical", "high"):
            warning = (
                f"[PROJECT PATTERN - {pattern.urgency.upper()}] "
                f"Error '{pattern.error_signature[:100]}' has occurred "
                f"{pattern.count} times across {len(pattern.task_ids)} tasks"
            )
            if pattern.successful_resolution:
                warning += f". Successful resolution: {pattern.successful_resolution}"
            project_warnings.append(warning)

    return local + project_warnings
```

These forbidden approaches are injected directly into the Builder's prompt. The Builder LLM sees:

```
DO NOT USE these approaches — they already failed:
- Attempt #1: Used bcrypt for password hashing (error: ModuleNotFoundError...)
- [PROJECT PATTERN - HIGH] Error 'ModuleNotFoundError: No module named <STR>'
  has occurred 3 times across 2 tasks. Successful resolution: Task 'Add user
  authentication' resolved this error
```

This is structurally different from retry-with-modified-prompt. The Builder doesn't just know its own failures — it knows the project's failure landscape, including which errors were eventually resolved and how.

### What Existing Tools Do Instead

| System | Within-task retry | Cross-task memory | Error normalization |
|--------|:-:|:-:|:-:|
| SWE-Agent | Trajectory replay | None | None |
| OpenHands | Retry with context | None | None |
| Aider | Edit/retry cycle | Git history (manual) | None |
| Devin | Internal replanning | Unknown (closed) | Unknown |
| Claude Code | Tool retry | Conversation context | None |
| **The Associate** | **Hypothesis log** | **Cross-task patterns** | **Multi-pass normalization** |

The closest academic work is SWE-Search (ICLR 2025), which uses Monte Carlo Tree Search over solution trajectories. But SWE-Search operates within a single issue — it doesn't transfer learned failure patterns across issues.

---

## Contribution 2: Multi-Layered Sentinel Gate (6+1 Checks)

Most SWE agents validate output by running tests. If the tests pass, the code ships. The Associate adds a second validation layer: the Sentinel.

The Sentinel runs 6 deterministic checks on every diff before it can be committed, plus an optional 7th LLM-based deep review:

```python
# src/agents/sentinel.py

# Run all 6 checks
checks = [
    ("dependency_jail",   self._check_dependency_jail,   (diff, banned)),
    ("style_match",       self._check_style_match,       (diff,)),
    ("chaos_check",       self._check_chaos,             (diff,)),
    ("placeholder_scan",  self._check_placeholders,      (diff,)),
    ("drift_alignment",   self._check_drift_alignment,   (task_desc, approach_summary)),
    ("claim_validation",  self._check_claims,
                          (approach_summary, build_result, self_audit)),
]

for check_name, check_fn, args in checks:
    violations, recommendations = check_fn(*args)
    all_violations.extend(violations)
    all_recommendations.extend(recommendations)

# Optional 7th check: LLM deep review
# Only runs when enabled AND all 6 rule-based checks passed
if self.llm_deep_check and len(all_violations) == 0:
    deep_violations, deep_recommendations, tokens_used = self._check_llm_deep(
        task_desc, approach_summary, diff
    )
```

### The 6 Checks in Detail

**1. Dependency Jail** — Scans the diff for import statements referencing banned packages, destructive function calls (`delete()`, `drop()`, `kill()`), and writes to protected paths (`.git/`, `.env`, `.ssh/`, credential files).

**2. Style Match** — Detects mixed tabs/spaces, lines over 200 characters, and wildcard imports (`from X import *`). These are the kinds of issues that tests don't catch but code review does.

**3. Chaos Check** — Counters the "Happy Path Bias" common in LLM-generated code. Flags bare `except:` clauses, `eval()`/`exec()` calls, hardcoded credentials, and string method access without null checks.

**4. Placeholder Scan** — Rejects `TODO`, `FIXME`, `HACK`, `NotImplementedError`, ellipsis-as-implementation, and `# stub`/`# placeholder` comments. LLMs frequently emit these as fill-in-later markers that never get filled in.

**5. Drift Alignment** — Uses sentence-transformer embeddings to compute cosine similarity between the original task description and the Builder's approach summary. If similarity falls below 0.40, the build is rejected for task drift:

```python
def _check_drift_alignment(self, task_description, approach_summary):
    similarity = self._compute_alignment(task_description, approach_summary)

    if similarity < self.drift_threshold:
        severity = self._drift_severity(similarity)
        # < 0.1 = CRITICAL, < 0.2 = HIGH, < 0.3 = MEDIUM, else LOW
        violations.append({
            "check": "drift_alignment",
            "detail": f"Task drift detected (severity: {severity}). "
                      f"Alignment score: {similarity:.3f}",
        })
```

This catches the case where an LLM "helpfully" implements something adjacent to but different from the task.

**6. Claim Validation** — Scans the Builder's approach summary for assertions like "production ready", "tested", "fixed", "secure", and validates them against actual evidence from the build result. An LLM saying "all tests pass" when tests failed triggers a violation. The check also cross-references the Builder's self-audit checklist against actual findings from other checks:

```python
def _cross_check_self_audit(self, self_audit, build_result, violations, recommendations):
    # Builder claims "no placeholders" but placeholder scan found violations
    placeholder_violations = [v for v in violations if v.get("check") == "placeholder_scan"]
    if placeholder_violations and ("yes" in audit_lower) and ("placeholder" in audit_lower):
        violations.append({
            "check": "claim_validation",
            "detail": "Self-audit contradiction: Builder claimed no placeholders "
                      "in self-audit, but placeholder scan found violations.",
        })
```

### The Optional 7th Check

When all 6 rule-based checks pass, the Sentinel can optionally call a different LLM model (via the model router) to review the diff against the original task. This uses a separate model family from the Builder to get genuine diversity of perspective. The 7th check only fires after the deterministic checks pass, avoiding wasted LLM calls on obviously-bad diffs.

### Why This Matters

The key insight is that **tests are necessary but not sufficient**. Tests verify functional correctness. The Sentinel verifies structural correctness, intent alignment, security posture, and claim integrity. These are the things that human reviewers catch in code review — the Sentinel automates a substantial fraction of that review.

No other open-source SWE agent we're aware of implements this multi-check pattern. Most rely entirely on test outcomes.

---

## Contribution 3: Architecture-First Planning (SOTAppR)

Before any code is generated, The Associate can optionally run a project through SOTAppR (State-of-the-Art Prompt Recipe): an 8-phase planning framework that produces typed artifacts at each phase.

| Phase | Name | Purpose |
|-------|------|---------|
| 0 | Alien Goggles | First-principles reframing — question the need itself |
| 1 | Contract Specification | Organism declaration + quadruple contracts (goal, constraints, output, failure conditions) |
| 2 | Divergent Architecture | Generate 3+ competing architecture sketches, score on 5 dimensions, select with tribunal |
| 3 | Growth Layers | 6-layer incremental build plan (skeleton, nervous, muscle, skin, immune, sensory) |
| 4 | Vibe Autopsy | Technical depression index — detect burnout markers and remediate |
| 5 | Adversarial Validation | Chaos playbooks, edge case inventories, critical questions |
| 6 | Future Proofing | Blind spot analysis, evolution scores per component, migration paths |
| 7 | Ethics Governance | Privacy, bias, transparency, and misuse audits with data governance |
| 8 | Delivery | Health card, checklist validation, experience replay extraction |

Each phase has hard-stop quality gates. Phase 2, for example, requires at least 3 architecture sketches, a total complexity budget under 100, and minimum scores of 2 on each of 5 dimensions (Feasibility, Novelty, Resilience, Evolvability, Simplicity). Phase 7 blocks delivery if any "High" ethical risk is unmitigated.

The output of SOTAppR is a structured artifact (Pydantic models, stored as JSON) that seeds the task list. The Associate's orchestrator then works through these seeded tasks with the full agent pipeline.

### How It Differs from Other Planning Approaches

Most SWE agents skip planning entirely or do a single "think step by step" prompt. SOTAppR is structurally different:

- **Typed artifacts**: Each phase produces a validated Pydantic model, not free-text
- **Hard-stop gates**: The system won't proceed past a phase with insufficient quality
- **Adversarial phases**: Phases 4, 5, and 6 exist specifically to attack the plan
- **Ethics as a gate**: Phase 7 is not optional and can block delivery

This is closer to how architecture review boards operate in large engineering organizations than to how AI coding assistants plan.

---

## Architecture Overview

The full pipeline chains 5 agents with the orchestrator:

```
                   ┌──────────────────────────────────────────┐
                   │           Orchestrator Loop               │
                   │                                          │
  Task Queue ──────│─► State Manager ──► Research Unit ──┐    │
  (PostgreSQL)     │        │                            │    │
                   │        │ TaskContext    ResearchBrief│    │
                   │        ▼                            ▼    │
                   │     Librarian ◄─── Methodology Store     │
                   │        │            (pgvector + FTS)      │
                   │        │ ContextBrief                     │
                   │        ▼                                  │
                   │     Builder ──────► File Writes + Tests   │
                   │        │                                  │
                   │        │ BuildResult                      │
                   │        ▼                                  │
                   │     Sentinel ─── 6+1 Checks              │
                   │        │                                  │
                   │        │ SentinelVerdict                  │
                   │        ▼                                  │
                   │   ┌─ PASS ──► git commit ──► next task   │
                   │   │                                      │
                   │   └─ FAIL ──► Hypothesis Log ──► retry   │
                   │                   │                      │
                   │              N failures?                  │
                   │                   ▼                      │
                   │              Council Agent                │
                   │              (2nd-opinion LLM)            │
                   └──────────────────────────────────────────┘
```

All data flows through typed Pydantic models (`TaskContext`, `ResearchBrief`, `ContextBrief`, `BuildResult`, `SentinelVerdict`, `CouncilDiagnosis`). There is no untyped dictionary passing between agents.

### Persistence Layer

PostgreSQL with pgvector handles 8 tables:

- `projects` — project metadata, tech stack, banned dependencies
- `tasks` — task queue with status machine (PENDING → RESEARCHING → CODING → REVIEWING → DONE/STUCK)
- `hypothesis_log` — every attempt with normalized error signatures
- `context_snapshots` — git refs for checkpoint/rewind
- `methodologies` — solution patterns with 384-dim embeddings for vector search
- `peer_reviews` — council diagnoses
- `sotappr_runs` / `sotappr_artifacts` — SOTAppR phase artifacts

The `methodologies` table supports the hybrid search pattern (pgvector cosine similarity + tsvector full-text search), weighted 0.6 vector / 0.4 text:

```python
# src/memory/hybrid_search.py

class HybridSearch:
    def search(self, query, limit=5, language=None, tags=None):
        # 1. Vector search via pgvector cosine distance
        vector_results = self._vector_search(query, limit * 3)

        # 2. Full-text search via tsvector + ts_rank
        text_results = self._text_search(query, limit * 3)

        # 3. Merge and deduplicate by methodology ID
        merged = self._merge_results(vector_results, text_results)

        # 4. Calculate combined score
        for result in merged.values():
            result.combined_score = (
                self.vector_weight * result.vector_score
                + self.text_weight * result.text_score
            )

        # 5. Sort, filter, return
        merged.sort(key=lambda r: r.combined_score, reverse=True)
        return [r for r in merged if r.combined_score >= self.min_score][:limit]
```

---

## Verified Results

The Associate was tested against a fresh repository with 5 seeded tasks (from SOTAppR Phase 3 growth layers). Results from a single run:

| Metric | Value |
|--------|-------|
| Tasks seeded | 5 |
| Tasks completed (DONE) | 5/5 |
| First-attempt success | 5/5 |
| Total files generated | 21 |
| Lines of code written | 2,019 |
| Test exit code | 0 |
| Council escalations | 0 |

All 5 tasks completed on the first attempt with no retries and no council escalations. LLM calls were routed through OpenRouter.

### Test Coverage

The Associate's own test suite:

| Module | Coverage |
|--------|----------|
| agents/ | 93% |
| core/ | 96% |
| db/ | 89% |
| llm/ | 78% |
| memory/ | 91% |
| orchestrator/ | 72% |
| tools/ | 88% |
| **Overall** | **81%** |

Coverage gaps are concentrated in `orchestrator/pipeline.py` (end-to-end flow) and `llm/client.py` (HTTP calls to OpenRouter). These require live API integration tests and are tracked for improvement.

---

## What's Missing (Honest Assessment)

The Associate is a working system, not a finished product. Here's what it doesn't do yet:

**No benchmark results.** We have not run The Associate against SWE-bench, SWE-bench Lite, or any standardized benchmark. The 5/5 run described above was against a controlled demo repository, not a real-world issue corpus. Without benchmark numbers, any performance comparison to other tools would be speculative.

**Single-file bias.** The Builder currently generates code using `--- FILE: path ---` block markers in a single LLM response. This works for tasks that touch 1-5 files but may struggle with large-scale refactoring across dozens of files.

**No interactive debugging.** When tests fail, the Builder retries with the error context in its prompt. It cannot set breakpoints, inspect variables, or step through execution. SWE-Agent's AgentOps and OpenHands' browser tool both provide richer debugging affordances.

**No multi-repo support.** The Associate operates on a single git repository per project. Monorepo support and cross-repo coordination are not implemented.

**Coverage gaps in orchestrator tests.** The pipeline orchestration layer (the part that chains agents together) has the lowest test coverage. This is the most critical path and needs dedicated integration testing.

**SOTAppR is opinionated.** The 8-phase framework works well for greenfield projects but may be excessive for bug fixes or small features. Adaptive phase selection (skip phases that don't apply) is not yet implemented.

---

## Comparison with the Field

Rather than a feature matrix, here is where The Associate sits architecturally:

**SWE-Agent / OpenHands** operate at the tool-use level: they give an LLM access to a terminal, file editor, and browser, then let it navigate to a solution. The Associate operates at the orchestration level: it decomposes work into typed phases with distinct agents, each with a specific responsibility and validation gate. These are complementary approaches — The Associate could use SWE-Agent as its Builder engine.

**Devin** is the closest commercial comparison in ambition (full autonomous coding), but its architecture is not public. The Associate is fully open-source.

**Aider** and **Claude Code** are conversation-oriented tools designed for human-in-the-loop pairing. The Associate is designed for autonomous batch execution — seed a task list, walk away, come back to committed code.

**Academic work** (SWE-Search, SERA, ReasoningBank) tends to focus on single-issue resolution with novel search or reasoning strategies. The Associate's contribution is at the project level: learning across tasks, not just within them.

---

## Showpiece: Charity Trivia Night

To see all three contributions working together in a single autonomous run, we provide a ready-to-run showpiece: a **Charity Trivia Night** platform built entirely by The Associate from a JSON spec.

The spec describes a FastAPI + SQLite application with 5 features: multi-charity tenants, a trivia question bank with 6 genres and 4 difficulty tiers (20+ real questions seeded on startup), configurable events with purpose/seriousness levels, live game sessions with scoring, leaderboards, and a Click-based CLI. The spec is designed so that:

- **Cross-module dependencies** (models, database, routes, CLI) make early import errors likely, activating the HypothesisTracker across tasks
- **Real trivia questions** (no placeholders) stress the Sentinel's placeholder scan
- **Genre/difficulty filtering** triggers drift alignment when the Builder generates generic CRUD instead of domain-specific filtering
- **The SOTAppR report** produces 5 quadruple contracts, architecture scoring, chaos playbooks, and ethics governance — all visible in the report JSON

```bash
# Run the full showpiece
associate --verbose sotappr-execute \
  --spec trivia_spec.json \
  --repo-path /path/to/trivia-night \
  --test-command "python3 -m pytest tests/ -v"
```

After completion, start the server and hit the interactive Swagger docs at `http://localhost:8000/docs` — create charities, run trivia games, check leaderboards. See the [showpiece execution guide](../../trivia-night/SHOWPIECE.md) for the full demo script.

---

## Running The Associate

```bash
# Start PostgreSQL with pgvector
docker compose up -d

# Configure
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY

# Install
pip install -e .

# Run SOTAppR planning + execution against a target repo
python -m src.sotappr.cli execute /path/to/target/repo \
    --mode execute \
    --test-command "pytest tests/ -v"
```

The full source is at [github.com/deesatzed/ralfed](https://github.com/deesatzed/ralfed) in the `the-associate/` directory.

---

## Conclusion

The Associate's three contributions — cross-task failure mining, multi-layered audit gates, and architecture-first planning — address gaps we observed while building autonomous coding systems. None of these ideas are individually revolutionary. Error normalization is a solved problem. Static analysis is decades old. Architecture review boards predate software.

What's novel is combining them into an agent orchestration layer with SQL-backed persistence, typed contracts between agents, and feedback loops that span across tasks. The system remembers what failed, validates what was built, and plans before it codes.

The code is open-source. The test results are real. The coverage gaps are documented. We welcome scrutiny and contributions.

---

*The Associate is part of the [ralfed](https://github.com/deesatzed/ralfed) project.*
