"""Tests for src/orchestrator/budget_hints.py — Budget-aware prompt adaptation."""

from src.orchestrator.budget_hints import generate_budget_hint


# ---------------------------------------------------------------------------
# Returns None when budget is healthy and early attempt
# ---------------------------------------------------------------------------

def test_returns_none_when_budget_healthy_and_first_attempt():
    result = generate_budget_hint(
        tokens_used=0, token_budget=100_000, attempt=1, max_retries=5
    )
    assert result is None


def test_returns_none_when_budget_above_50_percent_and_first_attempt():
    result = generate_budget_hint(
        tokens_used=40_000, token_budget=100_000, attempt=1, max_retries=5
    )
    assert result is None


# ---------------------------------------------------------------------------
# Token budget exhausted (ratio <= 0)
# ---------------------------------------------------------------------------

def test_budget_exhausted():
    result = generate_budget_hint(
        tokens_used=100_000, token_budget=100_000, attempt=1, max_retries=5
    )
    assert result is not None
    assert "TOKEN BUDGET EXHAUSTED" in result


def test_budget_exceeded():
    result = generate_budget_hint(
        tokens_used=120_000, token_budget=100_000, attempt=1, max_retries=5
    )
    assert result is not None
    assert "TOKEN BUDGET EXHAUSTED" in result


# ---------------------------------------------------------------------------
# Token budget critically low (< 20%)
# ---------------------------------------------------------------------------

def test_budget_critically_low():
    result = generate_budget_hint(
        tokens_used=85_000, token_budget=100_000, attempt=1, max_retries=5
    )
    assert result is not None
    assert "critically low" in result
    assert "15,000 remaining" in result


def test_budget_at_19_percent():
    result = generate_budget_hint(
        tokens_used=81_000, token_budget=100_000, attempt=1, max_retries=5
    )
    assert result is not None
    assert "critically low" in result


# ---------------------------------------------------------------------------
# Token budget limited (< 50%)
# ---------------------------------------------------------------------------

def test_budget_limited():
    result = generate_budget_hint(
        tokens_used=60_000, token_budget=100_000, attempt=1, max_retries=5
    )
    # 40% remaining < 50%
    assert result is not None
    assert "limited" in result
    assert "40,000 remaining" in result


def test_budget_at_30_percent():
    # 30% remaining — still "limited" tier (20-50%)
    result = generate_budget_hint(
        tokens_used=70_000, token_budget=100_000, attempt=1, max_retries=5
    )
    assert result is not None
    assert "limited" in result


# ---------------------------------------------------------------------------
# Iteration deadline (attempt >= submit_by)
# ---------------------------------------------------------------------------

def test_deadline_when_at_submit_by():
    # max_retries=5, submit_by = max(5-2, int(5*0.7)) = max(3, 3) = 3
    result = generate_budget_hint(
        tokens_used=0, token_budget=100_000, attempt=3, max_retries=5
    )
    assert result is not None
    assert "DEADLINE" in result
    assert "attempt 3/5" in result


def test_deadline_when_past_submit_by():
    # max_retries=5, submit_by = 3, attempt=4
    result = generate_budget_hint(
        tokens_used=0, token_budget=100_000, attempt=4, max_retries=5
    )
    assert result is not None
    assert "DEADLINE" in result


def test_deadline_at_max_retries():
    result = generate_budget_hint(
        tokens_used=0, token_budget=100_000, attempt=5, max_retries=5
    )
    assert result is not None
    assert "DEADLINE" in result
    assert "attempt 5/5" in result


# ---------------------------------------------------------------------------
# Iteration progress (attempt > 1 but before deadline)
# ---------------------------------------------------------------------------

def test_iteration_progress_shown_on_attempt_2():
    # max_retries=5, submit_by=3, attempt=2
    result = generate_budget_hint(
        tokens_used=0, token_budget=100_000, attempt=2, max_retries=5
    )
    assert result is not None
    assert "Iteration 2/5" in result
    assert "submit by attempt 3" in result


def test_no_iteration_progress_on_attempt_1():
    result = generate_budget_hint(
        tokens_used=0, token_budget=100_000, attempt=1, max_retries=5
    )
    # Budget healthy and attempt 1 => None
    assert result is None


# ---------------------------------------------------------------------------
# Combined: deadline + budget warning
# ---------------------------------------------------------------------------

def test_deadline_and_budget_exhausted_combined():
    result = generate_budget_hint(
        tokens_used=100_000, token_budget=100_000, attempt=5, max_retries=5
    )
    assert result is not None
    assert "DEADLINE" in result
    assert "TOKEN BUDGET EXHAUSTED" in result


def test_deadline_and_budget_limited_combined():
    result = generate_budget_hint(
        tokens_used=60_000, token_budget=100_000, attempt=4, max_retries=5
    )
    assert result is not None
    assert "DEADLINE" in result
    assert "limited" in result


def test_iteration_progress_and_budget_low_combined():
    result = generate_budget_hint(
        tokens_used=85_000, token_budget=100_000, attempt=2, max_retries=5
    )
    assert result is not None
    assert "Iteration 2/5" in result
    assert "critically low" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_zero_budget_returns_exhausted():
    result = generate_budget_hint(
        tokens_used=0, token_budget=0, attempt=1, max_retries=5
    )
    # remaining = 0, budget_ratio = 0/max(0,1) = 0.0 => exhausted
    assert result is not None
    assert "TOKEN BUDGET EXHAUSTED" in result


def test_zero_max_retries():
    result = generate_budget_hint(
        tokens_used=0, token_budget=100_000, attempt=0, max_retries=0
    )
    # budget_ratio = 1.0 (healthy), iteration_ratio = 0/max(0,1) = 0
    # submit_by = max(0-2, int(0*0.7)) = max(-2, 0) = 0
    # attempt=0 >= submit_by=0 => DEADLINE
    assert result is not None
    assert "DEADLINE" in result


def test_large_budget_healthy():
    result = generate_budget_hint(
        tokens_used=100, token_budget=1_000_000, attempt=1, max_retries=10
    )
    assert result is None


def test_submit_by_formula_for_large_max_retries():
    # max_retries=10: submit_by = max(10-2, int(10*0.7)) = max(8, 7) = 8
    result = generate_budget_hint(
        tokens_used=0, token_budget=100_000, attempt=8, max_retries=10
    )
    assert result is not None
    assert "DEADLINE" in result
    assert "attempt 8/10" in result


def test_submit_by_formula_attempt_7_of_10_no_deadline():
    # max_retries=10: submit_by = 8, attempt=7 < 8 => no deadline
    result = generate_budget_hint(
        tokens_used=0, token_budget=100_000, attempt=7, max_retries=10
    )
    assert result is not None
    # Should show iteration progress, not deadline
    assert "Iteration 7/10" in result
    assert "DEADLINE" not in result
