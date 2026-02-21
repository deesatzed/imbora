# SWE-bench Benchmarking Guide

This document describes how to evaluate The Associate against the SWE-bench dataset — the industry-standard benchmark for autonomous software engineering agents.

## What is SWE-bench?

SWE-bench is a benchmark consisting of real-world GitHub issues and their corresponding pull requests from popular Python repositories. Each instance contains:

- A **problem statement** (GitHub issue text)
- A **codebase snapshot** (git checkout at a specific commit)
- A **gold patch** (the human-authored fix)
- **Test cases** that validate the fix

Agents are scored on how many instances they can resolve — i.e., produce a patch that makes all relevant tests pass.

### Variants

| Variant | Instances | Difficulty | Recommended |
|---------|-----------|------------|-------------|
| SWE-bench Lite | 300 | Moderate | Yes (start here) |
| SWE-bench Full | 2,294 | Mixed | After Lite validation |
| SWE-bench Verified | 500 | Verified solvable | For publication |

## Prerequisites

1. **Docker** — The evaluation harness runs each instance in an isolated container
2. **Python 3.10+** — For the harness scripts
3. **OPENROUTER_API_KEY** — The Associate uses OpenRouter for LLM calls
4. **PostgreSQL** — The Associate's task and hypothesis persistence
5. **Disk space** — Each instance pulls a Docker image; plan for ~50GB

## Setup

### 1. Clone the SWE-bench evaluation harness

```bash
git clone https://github.com/princeton-nlp/SWE-bench.git
cd SWE-bench
pip install -e .
```

### 2. Download the dataset

```bash
# SWE-bench Lite (recommended starting point)
python -c "
from datasets import load_dataset
ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
ds.to_json('swe_bench_lite.json')
print(f'Loaded {len(ds)} instances')
"
```

### 3. Ensure The Associate is installed

```bash
cd /path/to/ralfzero/ralfed/the-associate
pip install -e ".[dev]"

# Verify CLI works
associate --help
```

## Wiring The Associate as the SWE-bench Agent

The SWE-bench harness expects an agent that:

1. Receives a problem statement and codebase
2. Produces a git diff (patch)

### Agent Wrapper Script

Create a wrapper script that adapts The Associate's SOTAppR pipeline to the SWE-bench interface:

```python
#!/usr/bin/env python3
"""swe_bench_agent.py — Adapter between SWE-bench harness and The Associate."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from src.core.factory import ComponentFactory
from src.sotappr.engine import SOTAppRBuilder
from src.sotappr.executor import SOTAppRExecutor
from src.sotappr.models import BuilderRequest, FeatureInput


def run_instance(instance: dict, repo_path: str) -> str:
    """Process a single SWE-bench instance and return the patch."""
    config_dir = Path(__file__).parent / "config"
    bundle = ComponentFactory.create(config_dir=config_dir)

    try:
        request = BuilderRequest(
            organism_name=f"SWE-bench-{instance['instance_id']}",
            stated_problem=instance["problem_statement"],
            root_need="Fix the reported issue and make all tests pass.",
            user_confirmed_phase1=True,
            features=[
                FeatureInput(
                    name="fix",
                    description=instance["problem_statement"][:500],
                ),
            ],
        )

        builder = SOTAppRBuilder()
        report = builder.build(request)

        # Determine test command from instance hints
        test_cmd = instance.get("test_cmd", "python -m pytest tests/ -x -v")

        executor = SOTAppRExecutor.from_bundle(
            bundle=bundle,
            repo_path=repo_path,
            test_command=test_cmd,
        )

        executor.bootstrap_and_execute(
            request=request,
            report=report,
            repo_path=repo_path,
            max_iterations=15,
            mode="execute",
            governance_pack="balanced",
        )

        # Capture the diff
        result = subprocess.run(
            ["git", "diff", "HEAD~1"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        return result.stdout

    finally:
        ComponentFactory.close(bundle)


if __name__ == "__main__":
    instance_file = sys.argv[1]
    repo_path = sys.argv[2]

    with open(instance_file) as f:
        instance = json.load(f)

    patch = run_instance(instance, repo_path)
    print(patch)
```

### Invocation

```bash
# Run a single instance
python swe_bench_agent.py instance.json /path/to/cloned/repo

# Run the full evaluation harness
python -m swebench.harness.run_evaluation \
    --predictions_path predictions.json \
    --swe_bench_tasks swe_bench_lite.json \
    --log_dir logs/ \
    --testbed /tmp/swe-bench-testbed \
    --timeout 600
```

## Interpreting Results

The harness outputs a JSON report with:

- **resolved**: Number of instances where all tests pass after the agent's patch
- **total**: Total instances attempted
- **resolution_rate**: `resolved / total`

### Current state-of-the-art context

Top agents on SWE-bench Lite resolve 50-70%+ of instances as of early 2026. The Associate's architecture (SOTAppR planning + retry loop + council escalation) is designed for reliability over speed.

### Key metrics to track

1. **Resolution rate** — Primary metric (% of instances resolved)
2. **Cost per instance** — Token usage * OpenRouter pricing
3. **Retries per instance** — How many attempts before success/stuck
4. **Council invocations** — How often the escalation path fires

## Known Limitations

1. **Single-file bias** — SWE-bench instances often require multi-file changes; The Associate's builder handles this but may need prompt tuning per repository
2. **No interactive debugging** — The Associate cannot interactively explore test failures; it relies on test output parsing
3. **Repository-specific knowledge** — Large repositories (Django, scikit-learn) benefit from domain-specific context that the Librarian agent may not have
4. **Cost** — Each instance may consume 5-50k tokens depending on complexity and retry count
5. **Test command discovery** — The Associate relies on the configured test command; SWE-bench instances may need repo-specific test commands

## Running a Quick Validation

Before a full benchmark run, validate with a single known-solvable instance:

```bash
# Pick a simple instance from the dataset
python -c "
import json
with open('swe_bench_lite.json') as f:
    instances = [json.loads(line) for line in f]

# Find a simple one (short problem statement, few files changed)
simple = sorted(instances, key=lambda x: len(x['problem_statement']))[:5]
for s in simple:
    print(f\"{s['instance_id']}: {s['problem_statement'][:80]}...\")
"

# Run The Associate on that instance
# (clone the repo at the correct commit first)
```

## Future Work

- **Automated benchmark runner** — Script to iterate over all instances, collect patches, and run the evaluation harness
- **Per-repository prompt tuning** — Custom builder prompts for high-value repositories
- **Result dashboard** — Visualization of resolution rates, costs, and retry patterns
- **Regression tracking** — Compare resolution rates across Associate versions

## Baseline Freeze + Regression Gate (Built-in CLI)

The Associate now ships a lightweight benchmark gate over real SOTAppR runs.

Freeze a baseline snapshot:

```bash
associate sotappr-benchmark-freeze \
  --project-id <project-uuid> \
  --limit 20 \
  --statuses completed,paused \
  --out artifacts/sotappr/frozen_benchmark.json
```

Run regression gate against the frozen baseline:

```bash
associate sotappr-benchmark-gate \
  --baseline artifacts/sotappr/frozen_benchmark.json \
  --project-id <project-uuid>
```

Gate checks compare baseline vs current on:
- quality success rate
- average elapsed hours
- average estimated cost
- retry events per processed task
- rollback rate (runs with retries)
