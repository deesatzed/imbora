# Adaptation Report: ClawWork + SecondBrain.md

**Generated:** 2026-02-18
**Sources:** `ClawWork/` directory, `SecondBrain.md/` directory
**Target:** The Associate autonomous SWE orchestrator

---

## Tier 1 — High-Value, Directly Portable

| # | Feature | Source | Where It Lands in The Associate |
|---|---------|--------|---------------------------------|
| 1 | **Token-level cost tracking** | ClawWork `TrackedProvider` + `EconomicTracker` | Wrap `src/llm/client.py` with a transparent interceptor recording `(input_tokens, output_tokens, cost)` per call, tied to active task/run. Store in a new `token_costs` table or extend `sotappr_runs`. Surface in `associate sotappr-status`. |
| 2 | **Global/cross-project methodology scope** | SecondBrain | Add `scope TEXT NOT NULL DEFAULT 'project'` column to `methodologies` table + `scope: str = "project"` field on `Methodology` model. Librarian falls back to `scope='global'` search when local results are low-confidence. |
| 3 | **Typed methodology taxonomy** | SecondBrain | Add `methodology_type: str` (BUG_FIX / PATTERN / DECISION / GOTCHA) to `Methodology`. Enables type-filtered retrieval and type-weighted ranking. SOTAppR Phase 7 ADR outputs should auto-tag as DECISION. |
| 4 | **`files_affected` on methodologies** | SecondBrain | Store `BuildResult.files_changed` into methodology at save time. Add file-path filter to `HybridSearch._apply_filters()` — gives Librarian a third retrieval signal. |
| 5 | **Quality filter before saving** | SecondBrain + ClawWork (0.6 cliff) | Only save methodologies when `task.attempt_count >= 1` (non-trivial task) or `len(solution_code) > threshold`. Sentinel verdicts gain a numeric `quality_score` (0.0-1.0) checked against configurable threshold. |

## Tier 2 — High-Value, Moderate Integration Effort

| # | Feature | Source | Where It Lands |
|---|---------|--------|----------------|
| 6 | **Occupation-specific Sentinel rubrics** | ClawWork `eval/meta_prompts/*.json` | Generate task-type-specific rubrics (DB migrations vs API integrations vs UI components). Cache per task classification, inject into Sentinel's quality gate. |
| 7 | **Task complexity scoring** | ClawWork `estimate_task_hours.py` | LLM estimates complexity from task description. Feeds into task prioritization, model routing (complex tasks to stronger models), hypothesis tracker analysis. |
| 8 | **Budget-aware prompt adaptation** | ClawWork `live_agent_prompt.py` | Inject tier-specific guidance into Builder's system prompt based on remaining token budget. Include iteration deadline formula: `submit_by = max(max_steps - 3, int(max_steps * 0.7))`. |
| 9 | **WrapUpWorkflow — iteration limit handling** | ClawWork `wrapup_workflow.py` | On Builder timeout, collect partial outputs, ask Council which partial artifact is best, score against Sentinel at lower threshold, record the attempt. |

## Tier 3 — Useful Enhancements, Lower Priority

| # | Feature | Source | Where It Lands |
|---|---------|--------|----------------|
| 10 | **JSONL shadow-logging** | ClawWork | Append-only JSONL alongside PostgreSQL for token costs. Human-readable, greppable, survives DB outages. |
| 11 | **Typed execution state dataclass** | ClawWork `ClawWorkState` | Formalize shared execution state (task_id, run_id, token_budget, current_phase) as typed dataclass injected into all agents. |
| 12 | **Dual API key for evaluation** | ClawWork `_inject_evaluation_credentials()` | Separate credentials for Sentinel evaluation vs Builder execution. |
| 13 | **Skills as procedure files** | SecondBrain | `skills/` directory with per-domain procedure markdown. Librarian routes by task type, injects as `skill_procedure` on `ContextBrief`. |

## Not Applicable

| Feature | Source | Why Not |
|---------|--------|---------|
| GDPVal dataset / BLS wages | ClawWork | SWE domain, not professional job displacement |
| Nanobot framework classes | ClawWork | Not portable without adopting nanobot |
| LangGraph dependency | ClawWork | Concept portable; library not |
| Frontend React dashboard | ClawWork | No web frontend planned |
| E2B cloud sandbox | ClawWork | Associate uses local repo sandbox |
| Diary entries | SecondBrain | Headless system; hypothesis_log covers temporal audit |
| File-based storage | SecondBrain | PostgreSQL + pgvector is superior |
| Visual layer | SecondBrain | Out of scope for autonomous core |
| Inline breadcrumb comments | SecondBrain | files_affected field achieves same linkage |
