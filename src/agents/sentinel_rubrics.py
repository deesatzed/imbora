"""Task-type-specific Sentinel evaluation rubrics (Item 6).

Adapted from ClawWork's eval/meta_prompts pattern. Loads JSON rubric files
per task type (db_migration, api_integration, ui_component, etc.) and
injects them into the Sentinel's quality gate alongside the generic rubric.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("associate.agent.sentinel_rubrics")

# Default rubric dimensions when no task-type-specific rubric exists
DEFAULT_RUBRIC = {
    "task_type": "general",
    "dimensions": [
        {"name": "completeness", "weight": 0.35, "description": "All task requirements addressed"},
        {"name": "correctness", "weight": 0.30, "description": "Logic correct, no bugs"},
        {"name": "quality", "weight": 0.20, "description": "Clean code, proper error handling"},
        {"name": "test_coverage", "weight": 0.15, "description": "Tests cover critical paths"},
    ],
}

# Keyword mappings for task type classification
TASK_TYPE_KEYWORDS: dict[str, list[str]] = {
    "db_migration": ["database", "migration", "schema", "table", "column", "sql", "postgres"],
    "api_integration": ["api", "endpoint", "rest", "http", "request", "response", "route"],
    "ui_component": ["component", "ui", "frontend", "render", "display", "template", "view"],
    "testing": ["test", "coverage", "fixture", "mock", "assert", "pytest", "unittest"],
    "refactoring": ["refactor", "restructure", "reorganize", "cleanup", "extract", "simplify"],
    "configuration": ["config", "settings", "environment", "yaml", "toml", "env"],
    "security": ["auth", "security", "permission", "token", "encrypt", "sanitize", "validate"],
    "ds_audit": ["data audit", "column profile", "label quality", "cleanlab", "dataset fingerprint"],
    "ds_eda": ["eda", "exploratory", "correlation", "outlier", "distribution"],
    "ds_feature_eng": ["feature engineering", "feature selection", "transform", "encoding", "llm-fe"],
    "ds_training": ["model training", "cross-validation", "hyperparameter", "tabpfn", "xgboost"],
    "ds_ensemble": ["ensemble", "stacking", "meta-learner", "uncertainty", "pareto"],
    "ds_evaluation": ["evaluation", "robustness", "fairness", "calibration", "shap"],
    "ds_deployment": ["deployment", "artifact", "fastapi", "monitoring", "data contract"],
}


def classify_task_type(title: str, description: str) -> str:
    """Classify a task into a type based on keywords."""
    combined = f"{title} {description}".lower()
    scores: dict[str, int] = {}
    for task_type, keywords in TASK_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > 0:
            scores[task_type] = score

    if not scores:
        return "general"

    return max(scores, key=scores.get)  # type: ignore[arg-type]


def load_rubric(rubrics_dir: Optional[str], task_type: str) -> dict[str, Any]:
    """Load a task-type-specific rubric from the rubrics directory.

    Falls back to DEFAULT_RUBRIC if no file exists for this type.
    """
    if rubrics_dir is None:
        return DEFAULT_RUBRIC

    rubric_path = Path(rubrics_dir) / f"{task_type}.json"
    if not rubric_path.exists():
        logger.debug("No rubric file for type '%s', using default", task_type)
        return DEFAULT_RUBRIC

    try:
        with open(rubric_path) as f:
            rubric = json.load(f)
        logger.info("Loaded rubric for task type '%s' from %s", task_type, rubric_path)
        return rubric
    except Exception as e:
        logger.warning("Failed to load rubric %s: %s", rubric_path, e)
        return DEFAULT_RUBRIC


def format_rubric_for_prompt(rubric: dict[str, Any]) -> str:
    """Format a rubric into text for inclusion in the Sentinel prompt."""
    lines = [f"Evaluation rubric for task type: {rubric.get('task_type', 'general')}"]
    lines.append("Score each dimension 0.0-1.0:")
    for dim in rubric.get("dimensions", []):
        name = dim.get("name", "unknown")
        weight = dim.get("weight", 0.25)
        desc = dim.get("description", "")
        lines.append(f"  - {name} (weight {weight:.0%}): {desc}")
    return "\n".join(lines)
