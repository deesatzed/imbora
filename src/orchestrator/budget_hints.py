"""Budget-aware prompt adaptation (Item 8).

Adapted from ClawWork's live_agent_prompt.py tier-specific guidance pattern.
Generates budget hints for the Builder system prompt based on remaining
token budget and iteration position.
"""

from __future__ import annotations


def generate_budget_hint(
    tokens_used: int,
    token_budget: int,
    attempt: int,
    max_retries: int,
) -> str | None:
    """Generate a budget guidance hint for the Builder prompt.

    Returns None if budget is healthy and no guidance is needed.
    """
    remaining = token_budget - tokens_used
    budget_ratio = remaining / max(token_budget, 1)
    iteration_ratio = attempt / max(max_retries, 1)

    # Calculate the deadline for submitting code (Item 8 formula)
    submit_by = max(max_retries - 2, int(max_retries * 0.7))

    parts: list[str] = []

    # Iteration deadline
    if attempt >= submit_by:
        parts.append(
            f"DEADLINE: You are on attempt {attempt}/{max_retries}. "
            f"Ensure the ## Files section is complete with working code NOW. "
            "No more exploratory approaches — produce a minimal viable implementation."
        )
    elif attempt > 1:
        parts.append(
            f"Iteration {attempt}/{max_retries} (submit by attempt {submit_by})."
        )

    # Token budget tiers
    if budget_ratio <= 0.0:
        parts.append(
            "TOKEN BUDGET EXHAUSTED. Produce the most minimal implementation possible."
        )
    elif budget_ratio < 0.20:
        parts.append(
            f"Token budget is critically low ({remaining:,} remaining). "
            "Produce a minimal viable implementation. No optimization passes."
        )
    elif budget_ratio < 0.50:
        parts.append(
            f"Token budget is limited ({remaining:,} remaining). "
            "Focus on correctness first, minimize response length."
        )

    if not parts:
        return None

    return "\n".join(parts)
