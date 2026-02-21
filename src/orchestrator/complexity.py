"""Task complexity scoring (Item 7).

Adapted from ClawWork's estimate_task_hours.py pattern. Estimates task
complexity from the task description using keyword heuristics.
Does NOT use LLM calls — this is a fast, deterministic classifier.

The complexity tier feeds into:
- Task prioritization in the orchestrator
- Model routing decisions (complex tasks to stronger models)
- Builder prompt context
"""

from __future__ import annotations

import logging

from src.core.models import ComplexityTier, Task

logger = logging.getLogger("associate.orchestrator.complexity")

# Complexity signal keywords
HIGH_COMPLEXITY_KEYWORDS = [
    "migration", "refactor", "architecture", "redesign", "system",
    "integration", "concurrent", "async", "distributed", "multi-",
    "security", "authentication", "authorization", "encryption",
    "performance", "optimization", "caching", "scaling",
]

MEDIUM_COMPLEXITY_KEYWORDS = [
    "api", "endpoint", "database", "query", "model", "schema",
    "validation", "error handling", "logging", "configuration",
    "testing", "coverage", "fixture", "middleware",
]

LOW_COMPLEXITY_KEYWORDS = [
    "add", "update", "fix", "typo", "rename", "move", "comment",
    "documentation", "readme", "style", "format", "lint",
]


def score_task_complexity(task: Task) -> ComplexityTier:
    """Score task complexity from title and description.

    Returns a ComplexityTier based on keyword analysis of the task text.
    """
    combined = f"{task.title} {task.description}".lower()
    words = combined.split()
    word_count = len(words)

    # Count keyword matches
    high_hits = sum(1 for kw in HIGH_COMPLEXITY_KEYWORDS if kw in combined)
    medium_hits = sum(1 for kw in MEDIUM_COMPLEXITY_KEYWORDS if kw in combined)
    low_hits = sum(1 for kw in LOW_COMPLEXITY_KEYWORDS if kw in combined)

    # Description length is a signal
    desc_length_score = 0
    if word_count > 100:
        desc_length_score = 2
    elif word_count > 50:
        desc_length_score = 1

    # Composite score
    score = (high_hits * 3) + (medium_hits * 1) + desc_length_score - (low_hits * 1)

    if score >= 6:
        tier = ComplexityTier.VERY_HIGH
    elif score >= 4:
        tier = ComplexityTier.HIGH
    elif score >= 2:
        tier = ComplexityTier.MEDIUM
    elif score >= 1:
        tier = ComplexityTier.LOW
    else:
        tier = ComplexityTier.TRIVIAL

    logger.debug(
        "Task '%s' complexity: %s (score=%d, high=%d, med=%d, low=%d, words=%d)",
        task.title[:40], tier.value, score, high_hits, medium_hits, low_hits, word_count,
    )
    return tier
