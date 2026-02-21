"""Tests for multi-objective Pareto optimization."""

from __future__ import annotations

import pytest

from src.datasci.pareto import (
    ParetoOptimizer,
    compute_pareto_front,
    dominates,
    select_knee_point,
)

import numpy as np


class TestDominates:
    def test_strict_domination(self):
        assert dominates(np.array([2.0, 3.0]), np.array([1.0, 2.0])) is True

    def test_equal_not_dominating(self):
        assert dominates(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is False

    def test_partial_domination(self):
        # a >= b in all, but not strictly > in all
        assert dominates(np.array([2.0, 2.0]), np.array([1.0, 2.0])) is True

    def test_no_domination(self):
        assert dominates(np.array([1.0, 3.0]), np.array([2.0, 2.0])) is False

    def test_reverse(self):
        assert dominates(np.array([1.0, 2.0]), np.array([2.0, 3.0])) is False


class TestComputeParetoFront:
    def test_empty(self):
        assert compute_pareto_front([], ["accuracy", "speed"]) == []

    def test_single_solution(self):
        solutions = [{"accuracy": 0.9, "speed": 0.8}]
        front = compute_pareto_front(solutions, ["accuracy", "speed"])
        assert front == [0]

    def test_one_dominates(self):
        solutions = [
            {"accuracy": 0.9, "speed": 0.8},  # Dominates
            {"accuracy": 0.7, "speed": 0.6},  # Dominated
        ]
        front = compute_pareto_front(solutions, ["accuracy", "speed"])
        assert front == [0]

    def test_two_on_front(self):
        solutions = [
            {"accuracy": 0.95, "speed": 0.5},  # High accuracy, low speed
            {"accuracy": 0.80, "speed": 0.9},  # Lower accuracy, high speed
        ]
        front = compute_pareto_front(solutions, ["accuracy", "speed"])
        assert sorted(front) == [0, 1]

    def test_classic_front(self):
        solutions = [
            {"accuracy": 0.95, "speed": 0.3},  # On front
            {"accuracy": 0.90, "speed": 0.7},  # On front
            {"accuracy": 0.80, "speed": 0.9},  # On front
            {"accuracy": 0.85, "speed": 0.5},  # Dominated by solution 1
        ]
        front = compute_pareto_front(solutions, ["accuracy", "speed"])
        assert sorted(front) == [0, 1, 2]

    def test_three_objectives(self):
        solutions = [
            {"accuracy": 0.95, "speed": 0.3, "interpretability": 0.2},
            {"accuracy": 0.80, "speed": 0.9, "interpretability": 0.5},
            {"accuracy": 0.70, "speed": 0.5, "interpretability": 0.95},
            {"accuracy": 0.60, "speed": 0.4, "interpretability": 0.3},  # Dominated
        ]
        objectives = ["accuracy", "speed", "interpretability"]
        front = compute_pareto_front(solutions, objectives)
        assert sorted(front) == [0, 1, 2]

    def test_all_equal(self):
        solutions = [
            {"accuracy": 0.9, "speed": 0.8},
            {"accuracy": 0.9, "speed": 0.8},
        ]
        front = compute_pareto_front(solutions, ["accuracy", "speed"])
        # Neither dominates the other, both on front
        assert sorted(front) == [0, 1]


class TestSelectKneePoint:
    def test_empty(self):
        assert select_knee_point([], [], ["accuracy"]) is None

    def test_single_point(self):
        solutions = [{"accuracy": 0.9, "speed": 0.8}]
        result = select_knee_point(solutions, [0], ["accuracy", "speed"])
        assert result == 0

    def test_two_points(self):
        solutions = [
            {"accuracy": 0.95, "speed": 0.3},
            {"accuracy": 0.80, "speed": 0.9},
        ]
        result = select_knee_point(solutions, [0, 1], ["accuracy", "speed"])
        assert result in [0, 1]  # One of the two endpoints

    def test_knee_in_middle(self):
        # Create L-shaped front where the knee is in the middle
        solutions = [
            {"accuracy": 0.99, "speed": 0.1},   # Extreme: high acc, low speed
            {"accuracy": 0.90, "speed": 0.85},  # Knee: balanced
            {"accuracy": 0.50, "speed": 0.95},  # Extreme: low acc, high speed
        ]
        result = select_knee_point(solutions, [0, 1, 2], ["accuracy", "speed"])
        assert result == 1  # The balanced middle point

    def test_returns_valid_index(self):
        solutions = [
            {"accuracy": 0.95, "speed": 0.3},
            {"accuracy": 0.85, "speed": 0.7},
            {"accuracy": 0.75, "speed": 0.9},
        ]
        pareto = compute_pareto_front(solutions, ["accuracy", "speed"])
        knee = select_knee_point(solutions, pareto, ["accuracy", "speed"])
        assert knee in pareto


class TestParetoOptimizer:
    def test_empty(self):
        opt = ParetoOptimizer(["accuracy", "speed"])
        assert opt.select_best() is None

    def test_add_and_select(self):
        opt = ParetoOptimizer(["accuracy", "speed"])
        opt.add_solution("xgb", {"accuracy": 0.92, "speed": 0.7})
        opt.add_solution("tabpfn", {"accuracy": 0.95, "speed": 0.3})
        opt.add_solution("lgbm", {"accuracy": 0.88, "speed": 0.9})

        best = opt.select_best()
        assert best is not None
        assert "name" in best
        assert "scores" in best

    def test_compute_front(self):
        opt = ParetoOptimizer(["accuracy", "interpretability"])
        opt.add_solution("xgb", {"accuracy": 0.92, "interpretability": 0.3})
        opt.add_solution("nam", {"accuracy": 0.85, "interpretability": 0.95})
        opt.add_solution("weak", {"accuracy": 0.60, "interpretability": 0.2})  # Dominated

        front = opt.compute_front()
        names = [s["name"] for s in front]
        assert "xgb" in names
        assert "nam" in names
        assert "weak" not in names

    def test_get_summary(self):
        opt = ParetoOptimizer(["accuracy", "speed"])
        opt.add_solution("m1", {"accuracy": 0.9, "speed": 0.8})
        opt.add_solution("m2", {"accuracy": 0.7, "speed": 0.6})

        summary = opt.get_summary()
        assert summary["total_solutions"] == 2
        assert summary["pareto_front_size"] >= 1
        assert summary["selected"] is not None
        assert "name" in summary["selected"]

    def test_metadata_preserved(self):
        opt = ParetoOptimizer(["accuracy"])
        opt.add_solution(
            "xgb",
            {"accuracy": 0.92},
            metadata={"hyperparams": {"n_estimators": 100}},
        )
        front = opt.compute_front()
        assert front[0]["metadata"]["hyperparams"]["n_estimators"] == 100

    def test_select_best_dominated_excluded(self):
        opt = ParetoOptimizer(["accuracy", "speed", "interpretability"])
        opt.add_solution("best", {"accuracy": 0.95, "speed": 0.9, "interpretability": 0.8})
        opt.add_solution("worst", {"accuracy": 0.50, "speed": 0.3, "interpretability": 0.2})

        best = opt.select_best()
        assert best["name"] == "best"
