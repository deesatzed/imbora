# The Associate (DS Pipeline) — Handoff Packet
**Generated:** 2026-02-20T19:30:00-05:00
**Branch:** main @ 1acedf5
**Last Commit:** 2026-02-20 — Exploit benchmark findings: 5 pipeline improvements from root cause analysis

---

## Quick Resume Checklist
- [ ] Clone/pull: `git clone https://github.com/deesatzed/imbora.git && cd the-associateDS`
- [ ] Start DB: `docker-compose up -d` (PostgreSQL 17 + pgvector on port 5433)
- [ ] Install: `pip install -e ".[dev,datasci]"`
- [ ] Set env: Copy `.env` with `OPENROUTER_API_KEY`, `DATABASE_URL`, `TAVILY_API_KEY`
- [ ] Verify: `pytest tests/test_datasci_*.py -x -q` — expect **713 passed, 0 failed**
- [ ] Review "Next Steps" section below

---

## What This Project Does
The Associate is an autonomous software engineering agent orchestrator with a 9-phase data science pipeline that takes raw CSV datasets and produces deployment-ready ML models. The DS pipeline runs fully autonomously: data audit, EDA, AI-based feature engineering assessment, feature engineering (LLM-evolutionary + domain heuristics), data augmentation (SMOTE/ADASYN/LLM-synth), model training (XGBoost/CatBoost/LightGBM/Self-Paced Ensemble), stacking ensemble, evaluation (SHAP + conformal calibration), and deployment artifact generation.

**Tech Stack:** Python 3.13, scikit-learn, XGBoost, CatBoost, LightGBM, imblearn, SHAP, OpenRouter (LLM), PostgreSQL+pgvector, Click CLI, Pydantic v2
**Architecture Pattern:** CLI orchestrator with 9-phase agent pipeline + benchmark evaluation framework

---

## Project Structure
```
the-associateDS/
  src/
    datasci/                    # DS pipeline (17,468 LOC across 47 files)
      agents/                   # Phase agents (data_audit, eda, ensemble, evaluation, etc.)
        base_ds_agent.py        # Base class for all DS agents
        data_audit.py           # Phase 1: Data profiling + LLM semantic analysis
        data_augmentation.py    # Phase 3.5: SMOTE/ADASYN/LLM-synth
        deployment.py           # Phase 7: Model artifacts + API scaffold
        eda.py                  # Phase 2: Statistical EDA
        ensemble.py             # Phase 5: Stacking ensemble + Pareto optimization
        evaluation.py           # Phase 6: SHAP + conformal + robustness
        feature_engineering.py  # Phase 3: Per-column treatments
        model_training.py       # Phase 4: Multi-candidate CV training
      benchmark/                # Benchmark evaluation framework
        catalog.py              # 25 curated datasets across 4 imbalance tiers
        loader.py               # Dataset fetcher with CSV caching
        runner.py               # Pipeline executor across datasets
        report.py               # Result aggregation + SOTA gap computation
      pipeline.py               # DSPipelineOrchestrator — phases 1-7
      models.py                 # Pydantic models for all pipeline state
      llm_feature_engineer.py   # LLM-evolutionary feature generation
      column_ai_enricher.py     # AI semantic column analysis
      synth_generator.py        # Synthetic data generation
      fe_assessment.py          # Phase 2.5: AI FE assessment
      pareto.py                 # Multi-objective Pareto optimization
      uncertainty.py            # Conformal prediction + routing
    core/                       # Config, factory, models
    llm/                        # OpenRouter client + token tracking
    agents/                     # Core SWE agents (state_manager, builder, sentinel, etc.)
    orchestrator/               # Task loop, pipeline, arbitration
    db/                         # PostgreSQL repository + schema
    protocol/                   # APC/1.0 inter-agent protocol
    sotappr/                    # SOTAppR engine (9-phase architecture design)
  tests/                        # 1,620 tests (12,796 LOC across 24 DS test files)
  config/                       # YAML config, prompts, rubrics, contracts
  docs/                         # Development journals, runbooks, protocols
```

**Entry Points:**
- `the_associate/cli.py` -> `src/cli.py` — CLI entry (`associate` command)
- `src/datasci/pipeline.py` — DS pipeline orchestrator
- `src/datasci/benchmark/runner.py` — Benchmark runner

**Key Modules:**
| Module | Path | Purpose | Status |
|--------|------|---------|--------|
| DS Pipeline | `src/datasci/pipeline.py` | 9-phase orchestrator | ✅ |
| Benchmark Runner | `src/datasci/benchmark/runner.py` | Multi-dataset evaluation | ✅ |
| Benchmark Catalog | `src/datasci/benchmark/catalog.py` | 25 curated datasets | ✅ |
| Ensemble Agent | `src/datasci/agents/ensemble.py` | Stacking + degradation fallback | ✅ |
| Evaluation Agent | `src/datasci/agents/evaluation.py` | SHAP + conformal | ✅ |
| LLM Feature Engineer | `src/datasci/llm_feature_engineer.py` | Evolutionary FE | ⚠️ 0% lift |
| Data Augmentation | `src/datasci/agents/data_augmentation.py` | SMOTE/ADASYN/synth | ✅ |
| Model Training | `src/datasci/agents/model_training.py` | Multi-candidate CV | ✅ |
| Core Orchestrator | `src/orchestrator/loop.py` | SWE task loop | ⚠️ 17 test failures |
| APC/1.0 Protocol | `src/protocol/packet_runtime.py` | Inter-agent protocol | ⚠️ test failures |

---

## How to Run

### Local Development
```bash
# Setup (one-time)
docker-compose up -d
pip install -e ".[dev,datasci]"
# Set .env with OPENROUTER_API_KEY, DATABASE_URL=postgresql://associate:associate@localhost:5433/the_associate

# Run DS pipeline on a single dataset
associate datasci-run --dataset /path/to/data.csv --target-column target

# Run full benchmark suite
associate datasci-benchmark --tier all --out benchmark_results.json

# Run single dataset benchmark
associate datasci-benchmark --max-datasets 1 --tier mild
```

### Tests
```bash
# DS tests only (fast, no external deps)
pytest tests/test_datasci_*.py -x -q

# Full suite (requires PostgreSQL)
pytest tests/ -q
```
**DS Tests:** 713 passed, 0 failed, 0 skipped
**Full Suite:** 1,603 passed, 17 failed, 0 skipped
**Known Failures:** 17 failures in core orchestrator/protocol tests (pre-existing, unrelated to DS pipeline)

---

## Current State Assessment

### What's Working ✅
- **Full DS pipeline** (9 phases) — runs end-to-end on any classification CSV
- **Benchmark framework** — 25 curated datasets, automated SOTA comparison
- **Benchmark results** — 21/25 datasets succeeded, avg AUC gap +0.016 vs SOTA
- **Ensemble degradation fallback** — auto-detects when stacking hurts, falls back to best base
- **Quality gates** — LLM-assessed pass/fail for each phase output
- **Augmentation path fix** — correct data flows to augmentation agent
- **713 DS tests** — all passing, real-shaped fakes (no mocks)
- **Model deployment artifacts** — joblib model, API scaffold, monitoring config, data contract

### What's Incomplete ⚠️
- **LLM-FE evolutionary loop** — generates features but achieves 0% lift across all 21 datasets. Logging now upgraded (WARNING level) but the LLM-generated transforms consistently fail to beat domain-agnostic features. Root cause unclear — may need prompt engineering, better examples, or different LLM models.
- **Post-improvement benchmark** — 5 improvements committed but full re-benchmark not yet run
- **Academic paper** — analysis complete, paper structure not yet created
- **4 skipped datasets** — imblearn fetch_datasets failures (E. coli, Glass 7, Solar Flare) and OpenML timeout (Webpage)

### What's Broken ❌
- **17 core orchestrator tests** — pre-existing failures in `test_orchestrator_loop_integration.py`, `test_protocol_packet_runtime.py`, `test_e2e_sotappr_execute.py`. These are in the SWE agent orchestrator, not the DS pipeline.

### Current Blockers
- None for DS pipeline work. Pipeline, benchmarks, and tests all functional.

---

## Recent Changes

| Date | SHA | Change | Why |
|------|-----|--------|-----|
| 2026-02-20 | `1acedf5` | 5 pipeline improvements from root cause analysis | Exploit benchmark findings: ensemble fallback, LLM-FE logging, augmentation path fix, eval ensemble logging, SOTA corrections |
| 2026-02-20 | `20c78e0` | Add benchmark results (+0.016 AUC gap) | Full 25-dataset benchmark completed |
| 2026-02-19 | `fbe3c11` | Fix DS pipeline LLM integration | LLMMessage protocol, get_model_chain, models= API |
| 2026-02-19 | `bbcdcd9` | Multi-dataset benchmark system | catalog, loader, runner, report, CLI, 30+ tests |
| 2026-02-19 | `2717d69` | Phase 2.5: AI-based FE Assessment | LLM assesses which FE categories apply before expensive loop |
| 2026-02-18 | `2b36d4d` | Fix 5 root causes (accuracy 0.699->0.829) | Grade C->B on pilot dataset |
| 2026-02-18 | `ef1afea` | Initial commit: 12-phase DS pipeline | Full pipeline from audit through deployment |

**Uncommitted Changes:** `docs/DS_PIPELINE_BENCHMARK_JOURNAL.md` (new file, development journal)
**Stashed Work:** none

---

## Configuration & Secrets

### Environment Variables
| Variable | Purpose | Where to Get |
|----------|---------|--------------|
| `OPENROUTER_API_KEY` | LLM calls (all agents) | https://openrouter.ai/keys |
| `DATABASE_URL` | PostgreSQL connection | `postgresql://associate:associate@localhost:5433/the_associate` |
| `TAVILY_API_KEY` | Web search (research unit) | https://tavily.com |

### External Dependencies
| Service | Purpose | Local Alternative |
|---------|---------|-------------------|
| PostgreSQL 17 + pgvector | Task/hypothesis persistence, embeddings | `docker-compose up -d` |
| OpenRouter API | LLM calls for all agents | Required, no local alternative |
| OpenML | Benchmark dataset downloads | Cached after first download |
| imblearn | Curated imbalanced datasets | Cached after first download |

---

## Benchmark Results Summary (Pre-Improvement Baseline)

**Run Date:** 2026-02-20
**Datasets:** 25 attempted, 21 succeeded, 4 skipped
**Average AUC Gap:** +0.016 (pipeline beats SOTA on average)
**Datasets Beating SOTA:** 8 / 18 (with SOTA values)

### By Tier
| Tier | Datasets | Succeeded | Avg AUC Gap |
|------|----------|-----------|-------------|
| Mild | 5 | 5 | -0.017 |
| Moderate | 5 | 5 | +0.066* |
| Severe | 8 | 8 | +0.017 |
| Extreme | 3 | 3 | -0.024 |

*Moderate tier inflated by stale SOTA values (Wine Quality, Vehicle Insurance) — corrected in 1acedf5.

### Top/Bottom Performers
| Dataset | AUC | SOTA | Gap | Notes |
|---------|-----|------|-----|-------|
| Wine Quality (Red) | 1.000 | 0.950* | +0.050 | Binary recast, Self-Paced Ensemble |
| Satimage | 0.990 | 0.930 | +0.060 | Genuine outperformance |
| PC1 Software Defects | 0.856 | 0.810 | +0.046 | Self-Paced Ensemble on 14:1 data |
| Credit Card Fraud | 0.979 | 0.980 | -0.001 | Within noise margin |
| Haberman Survival | 0.685 | 0.720 | -0.035 | Ensemble degradation (now fixed) |
| COIL 2000 Insurance | 0.705 | 0.780 | -0.075 | Hardest dataset, 577:1 IR |

*SOTA corrected from 0.82 to 0.95 in latest commit.

---

## Known Issues & Tech Debt
- [ ] **LLM-FE 0% lift** — LLM-generated feature transforms never improve CV score across any dataset. May need prompt engineering, domain-specific examples, or model selection tuning. Logging now at WARNING level for diagnosis.
- [ ] **37 ruff lint errors in DS module** — mostly E501 (line length), I001 (import sorting), 1 F841 (unused variable). Pre-existing.
- [ ] **4 skipped benchmark datasets** — imblearn fetch_datasets unreliable for E. coli, Glass 7, Solar Flare. Webpage has OpenML timeout.
- [ ] **17 core orchestrator test failures** — unrelated to DS pipeline, in APC/1.0 protocol tests
- [ ] **Conformal coverage gap** — Coverage consistently ~0.97 with gap ~0.07, suggesting alpha=0.10 may be too conservative

---

## Next Steps (Priority Order)

1. **Re-run full benchmark with 5 improvements** — Measure delta from ensemble fallback + augmentation path fix. "Done" = new benchmark_results.json with updated gaps, especially Haberman and German Credit.

2. **Prepare academic paper** — Structure: Introduction (autonomous ML pipelines), Related Work (AutoML, imbalanced classification), Method (9-phase pipeline architecture), Experiments (25-dataset benchmark), Results (SOTA comparison), Discussion (ensemble degradation, LLM-FE limitations), Conclusion. "Done" = LaTeX draft with tables, figures, and statistical significance tests.

3. **Diagnose LLM-FE 0% lift** — Read WARNING logs from benchmark run to identify transform failure patterns. Experiment with different prompts, models, or providing dataset-specific examples. "Done" = at least 1 dataset shows measurable FE improvement.

4. **Fix 4 skipped datasets** — Add fallback loading strategies (direct CSV download, sklearn fetch) for imblearn failures. "Done" = 25/25 datasets succeed.

5. **Statistical significance testing** — Paired Wilcoxon signed-rank test on AUC gaps across all datasets. Bootstrap confidence intervals. "Done" = p-values and CIs for paper.

---

## Key Files Reference
| File | Purpose | When to Modify |
|------|---------|----------------|
| `src/datasci/pipeline.py` | Phase orchestration | Adding phases, changing phase order, modifying quality gates |
| `src/datasci/agents/ensemble.py` | Stacking + fallback | Changing ensemble strategy, degradation threshold |
| `src/datasci/benchmark/catalog.py` | Dataset registry | Adding datasets, updating SOTA values |
| `src/datasci/benchmark/runner.py` | Benchmark execution | Changing metrics extraction, adding per-dataset config |
| `src/datasci/llm_feature_engineer.py` | LLM-FE loop | Prompt engineering, transform evaluation |
| `src/datasci/models.py` | Pydantic state models | Adding fields to pipeline state |
| `config/default.yaml` | Pipeline config | Changing thresholds, model selection, augmentation config |
| `config/models.yaml` | LLM model mapping | Updating model IDs (user-managed) |
| `tests/test_datasci_ensemble.py` | Ensemble tests | After any ensemble.py changes |
| `docs/DS_PIPELINE_BENCHMARK_JOURNAL.md` | Dev journal | After each development session |

---

## Open Questions / Decisions Needed
- **LLM model for FE:** Current model used for feature engineering proposals — is it the right one? User manages model selection via `config/models.yaml` and OpenRouter.
- **Paper venue:** Target journal/conference for the benchmark paper?
- **Binary recasting validity:** Some "moderate" datasets are multi-class problems recast to binary. Should we run them as multi-class instead?
- **Augmentation strategy for extreme imbalance:** COIL 2000 (577:1) — current SMOTE/ADASYN may not be sufficient. Should we try LLM-synth or deep generative models?

---

## Appendix: Machine-Readable Summary
```json
{
  "project": "the-associate-ds-pipeline",
  "generated": "2026-02-20T19:30:00-05:00",
  "repo": {
    "url": "https://github.com/deesatzed/imbora.git",
    "branch": "main",
    "commit": "1acedf5ccfedaa104ce85947bf0272d5a6b2efbd",
    "commit_date": "2026-02-20T19:07:21-05:00",
    "uncommitted_changes": true,
    "stashed_work": 0
  },
  "stack": {
    "language": "python",
    "language_version": "3.13.9",
    "framework": "scikit-learn + xgboost + catboost + lightgbm + imblearn",
    "cli": "click",
    "models": "pydantic v2",
    "database": "postgresql 17 + pgvector"
  },
  "health": {
    "ds_tests_passing": 713,
    "ds_tests_failing": 0,
    "ds_tests_skipped": 0,
    "full_tests_passing": 1603,
    "full_tests_failing": 17,
    "full_tests_skipped": 0,
    "lint_errors": 37,
    "lint_clean": false
  },
  "benchmark": {
    "datasets_attempted": 25,
    "datasets_succeeded": 21,
    "datasets_failed": 0,
    "datasets_skipped": 4,
    "avg_auc_gap": 0.016,
    "datasets_beating_sota": 8,
    "datasets_with_sota": 18
  },
  "status": {
    "working": [
      "ds_pipeline_9_phases",
      "benchmark_framework",
      "ensemble_degradation_fallback",
      "augmentation_path_fix",
      "quality_gates",
      "model_deployment_artifacts"
    ],
    "incomplete": [
      "llm_feature_engineering_0pct_lift",
      "post_improvement_benchmark_rerun",
      "academic_paper",
      "4_skipped_datasets"
    ],
    "broken": [
      "17_core_orchestrator_tests_preexisting"
    ],
    "blockers": []
  },
  "next_steps": [
    {"task": "Re-run full benchmark with 5 improvements", "priority": 1, "scope": "medium"},
    {"task": "Prepare academic paper structure", "priority": 2, "scope": "large"},
    {"task": "Diagnose LLM-FE 0% lift", "priority": 3, "scope": "medium"},
    {"task": "Fix 4 skipped datasets", "priority": 4, "scope": "small"},
    {"task": "Statistical significance testing", "priority": 5, "scope": "small"}
  ]
}
```
