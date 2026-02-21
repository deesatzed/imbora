"""Multi-objective Pareto optimization.

Computes Pareto fronts and selects optimal configurations balancing
competing objectives (accuracy, interpretability, speed).
Pure computation — no LLM calls.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("associate.datasci.pareto")


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if solution a dominates solution b (all objectives maximized).

    a dominates b if a is >= b in all objectives and strictly > in at least one.
    """
    return bool(np.all(a >= b) and np.any(a > b))


def compute_pareto_front(
    solutions: list[dict[str, float]],
    objectives: list[str],
) -> list[int]:
    """Compute the Pareto-optimal front indices.

    All objectives are maximized. To minimize an objective, negate its values
    before passing.

    Args:
        solutions: List of dicts mapping objective_name → score.
        objectives: List of objective names to consider.

    Returns:
        List of indices into solutions that are Pareto-optimal.
    """
    if not solutions:
        return []

    n = len(solutions)
    # Build matrix: rows = solutions, cols = objectives
    matrix = np.array([[s.get(obj, 0.0) for obj in objectives] for s in solutions])

    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if dominates(matrix[j], matrix[i]):
                is_pareto[i] = False
                break

    return [i for i in range(n) if is_pareto[i]]


def select_knee_point(
    solutions: list[dict[str, float]],
    pareto_indices: list[int],
    objectives: list[str],
) -> int | None:
    """Select the knee point of the Pareto front.

    The knee point is the solution with maximum distance from the line
    connecting the extreme points of the Pareto front. This represents
    the best balanced trade-off.

    Args:
        solutions: All solutions (same list used for compute_pareto_front).
        pareto_indices: Pareto-optimal indices.
        objectives: Objective names.

    Returns:
        Index of the knee point in the original solutions list, or None if empty.
    """
    if not pareto_indices:
        return None
    if len(pareto_indices) == 1:
        return pareto_indices[0]

    # Extract Pareto front as matrix
    matrix = np.array([
        [solutions[i].get(obj, 0.0) for obj in objectives]
        for i in pareto_indices
    ])

    # Normalize to [0, 1] per objective for fair distance computation
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0  # Avoid division by zero
    normed = (matrix - mins) / ranges

    # Knee = point with max distance from the line connecting the two extremes
    # Use the first and last points sorted by first objective
    sort_idx = np.argsort(normed[:, 0])
    p1 = normed[sort_idx[0]]
    p2 = normed[sort_idx[-1]]

    # Distance from line p1→p2 for each point
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-10:
        # All points are essentially the same; return first
        return pareto_indices[0]

    distances = np.zeros(len(pareto_indices))
    for idx, point in enumerate(normed):
        # Point-to-line distance in n-D
        t = np.dot(point - p1, line_vec) / (line_len**2)
        projection = p1 + t * line_vec
        distances[idx] = np.linalg.norm(point - projection)

    best_local_idx = int(np.argmax(distances))
    return pareto_indices[best_local_idx]


class ParetoOptimizer:
    """Optimizer for multi-objective model selection.

    Maintains a population of model configurations scored on multiple
    objectives and selects the best balanced configuration.
    """

    def __init__(self, objectives: list[str]):
        """Initialize with objective names.

        Args:
            objectives: List of objective names (all maximized).
        """
        self.objectives = objectives
        self.solutions: list[dict[str, Any]] = []

    def add_solution(
        self,
        name: str,
        scores: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a model configuration to the population.

        Args:
            name: Model/configuration name.
            scores: Dict of objective_name → score.
            metadata: Optional extra metadata.
        """
        self.solutions.append({
            "name": name,
            "scores": scores,
            "metadata": metadata or {},
        })

    def compute_front(self) -> list[dict[str, Any]]:
        """Compute Pareto front from current population.

        Returns:
            List of Pareto-optimal solutions with their scores.
        """
        score_dicts = [s["scores"] for s in self.solutions]
        front_indices = compute_pareto_front(score_dicts, self.objectives)
        return [self.solutions[i] for i in front_indices]

    def select_best(self) -> dict[str, Any] | None:
        """Select the best balanced configuration (knee point).

        Returns:
            The knee-point solution, or None if no solutions.
        """
        if not self.solutions:
            return None

        score_dicts = [s["scores"] for s in self.solutions]
        front_indices = compute_pareto_front(score_dicts, self.objectives)
        knee_idx = select_knee_point(score_dicts, front_indices, self.objectives)

        if knee_idx is not None:
            return self.solutions[knee_idx]
        return None

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the optimization results.

        Returns:
            Dict with front size, selected configuration, and all front solutions.
        """
        front = self.compute_front()
        selected = self.select_best()
        return {
            "total_solutions": len(self.solutions),
            "pareto_front_size": len(front),
            "pareto_front": [
                {"name": s["name"], "scores": s["scores"]}
                for s in front
            ],
            "selected": {
                "name": selected["name"],
                "scores": selected["scores"],
            } if selected else None,
        }
