"""Formula packet evaluator for APC/1.0 IDEA/TASK_IR payloads."""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.core.models import AgentPacket, FormulaBundle, FormulaODESystem, PacketType

_SAFE_NUMPY_FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "tanh": np.tanh,
    "minimum": np.minimum,
    "maximum": np.maximum,
}

_CI_THRESHOLDS = {
    "rtf": 1.0,
    "stability": 0.9,
    "audit": 1.0,
    "outcome": 0.5,
    "learnability": 0.7,
    "generalization_ratio": 0.8,
}


class FormulaEvaluationArtifact(BaseModel):
    packet_id: str
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    mc_runs: int
    mean_final: dict[str, float]
    std_final: dict[str, float]
    equilibrium_estimate: dict[str, float]
    bounded_ratio: float
    convergence_ratio: float
    stability_eigenvalues_real: list[float]
    sample_trajectories: list[list[dict[str, float]]] = Field(default_factory=list)


class FormulaEvaluationResult(BaseModel):
    metrics: dict[str, float]
    thresholds: dict[str, float]
    passed: bool
    notes: list[str] = Field(default_factory=list)
    artifact: FormulaEvaluationArtifact


def _hash_payload(payload: dict[str, Any]) -> str:
    serialized = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _extract_formula_bundle(packet: AgentPacket) -> FormulaBundle:
    if packet.packet_type not in {PacketType.IDEA, PacketType.TASK_IR, PacketType.ARTIFACT}:
        raise ValueError(f"Packet type {packet.packet_type.value} is not formula-evaluable")

    payload = packet.payload
    if "formula_bundle" not in payload:
        raise ValueError("Packet payload missing formula_bundle")
    return FormulaBundle.model_validate(payload["formula_bundle"])


def _build_derivative(
    ode: FormulaODESystem,
    params: dict[str, float],
) -> Any:
    equations = list(ode.equations)
    variables = list(ode.state_variables)

    def derivative(state: np.ndarray, t_value: float) -> np.ndarray:
        env: dict[str, Any] = {
            **_SAFE_NUMPY_FUNCS,
            "np": np,
            "t": float(t_value),
        }
        env.update(params)
        for idx, name in enumerate(variables):
            env[name] = float(state[idx])

        result = []
        for expr in equations:
            value = eval(expr, {"__builtins__": {}}, env)
            result.append(float(value))
        return np.asarray(result, dtype=float)

    return derivative


def _rk4(
    derivative: Any,
    init_state: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    dim = init_state.shape[0]
    steps = t_grid.shape[0]
    out = np.zeros((steps, dim), dtype=float)
    out[0] = init_state
    for i in range(1, steps):
        dt = float(t_grid[i] - t_grid[i - 1])
        prev = out[i - 1]
        t_prev = float(t_grid[i - 1])
        k1 = derivative(prev, t_prev)
        k2 = derivative(prev + 0.5 * dt * k1, t_prev + 0.5 * dt)
        k3 = derivative(prev + 0.5 * dt * k2, t_prev + 0.5 * dt)
        k4 = derivative(prev + dt * k3, t_prev + dt)
        out[i] = prev + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def _jacobian(
    derivative: Any,
    eq_point: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    dim = eq_point.shape[0]
    base = derivative(eq_point, 0.0)
    jac = np.zeros((dim, dim), dtype=float)
    for j in range(dim):
        shifted = eq_point.copy()
        shifted[j] += eps
        diff = derivative(shifted, 0.0) - base
        jac[:, j] = diff / eps
    return jac


def _metric_learnability(bundle: FormulaBundle) -> float:
    score = 0.0
    if bundle.latex_full:
        score += 0.2
    if bundle.sympy_str:
        score += 0.2
    if bundle.ast_json:
        score += 0.2
    if bundle.var_ontology:
        score += 0.2
    if bundle.functor_map:
        score += 0.2
    return min(1.0, score)


def _metric_symbol_coverage(bundle: FormulaBundle) -> float:
    score = 0.0
    if bundle.sympy_str:
        score += 0.4
    if bundle.ast_json:
        score += 0.3
    if bundle.latex_full:
        score += 0.3
    return min(1.0, score)


def _metric_rtf(bundle: FormulaBundle) -> float:
    if bundle.sympy_str and bundle.sympy_canonical:
        return 1.0
    if bundle.sympy_str and bundle.ast_json:
        return 0.95
    if bundle.ode_system:
        return 0.9
    return 0.8


def _metric_audit(packet: AgentPacket) -> float:
    if packet.proof_bundle and packet.proof_bundle.falsifiers:
        return 1.0
    return 0.5


def _bounded_ratio(trajectories: list[np.ndarray], bound: float = 1e6) -> float:
    if not trajectories:
        return 0.0
    good = 0
    for traj in trajectories:
        if np.all(np.isfinite(traj)) and np.max(np.abs(traj)) < bound:
            good += 1
    return good / len(trajectories)


def evaluate_formula_packet(
    packet: AgentPacket,
    *,
    mc_runs: int = 200,
    seed: int = 42,
    param_noise_std: float = 0.12,
    init_jitter_ratio: float = 0.10,
) -> FormulaEvaluationResult:
    """Evaluate a packet's formula bundle and emit CI-like metrics."""
    bundle = _extract_formula_bundle(packet)
    if bundle.ode_system is None:
        raise ValueError("formula_bundle.ode_system is required for simulation")

    ode = bundle.ode_system
    vars_order = list(ode.state_variables)
    base_init = np.asarray([ode.initial_state[v] for v in vars_order], dtype=float)
    base_params = dict(ode.parameters)

    t_grid = np.linspace(ode.time_start, ode.time_end, ode.num_points)
    rng = np.random.default_rng(seed)

    trajectories: list[np.ndarray] = []
    finals: list[np.ndarray] = []
    for _ in range(mc_runs):
        run_params: dict[str, float] = {}
        for k, v in base_params.items():
            multiplier = float(rng.normal(loc=1.0, scale=param_noise_std))
            run_params[k] = float(v) * multiplier

        jitter = rng.uniform(
            low=1.0 - init_jitter_ratio,
            high=1.0 + init_jitter_ratio,
            size=base_init.shape[0],
        )
        init = base_init * jitter
        derivative = _build_derivative(ode=ode, params=run_params)
        traj = _rk4(derivative=derivative, init_state=init, t_grid=t_grid)
        trajectories.append(traj)
        finals.append(traj[-1])

    final_arr = np.asarray(finals, dtype=float)
    eq_point = np.mean(final_arr, axis=0)
    std_final = np.std(final_arr, axis=0)

    convergence_tol = max(1e-3, float(np.mean(std_final)))
    convergence_ratio = float(
        np.mean(np.linalg.norm(final_arr - eq_point, axis=1) <= convergence_tol)
    )
    bounded = _bounded_ratio(trajectories)

    derivative_base = _build_derivative(ode=ode, params=base_params)
    jac = _jacobian(derivative=derivative_base, eq_point=eq_point)
    eigvals = np.linalg.eigvals(jac)
    eig_real = [float(np.real(v)) for v in eigvals]
    max_real = max(eig_real) if eig_real else 0.0

    if max_real <= -0.05:
        stability = 1.0
    elif max_real <= 0:
        stability = 0.9
    else:
        stability = max(0.0, 0.9 - min(0.9, max_real))

    outcome = min(1.0, (0.6 * bounded) + (0.4 * convergence_ratio))
    learnability = _metric_learnability(bundle)
    symbol_coverage = _metric_symbol_coverage(bundle)
    rtf = _metric_rtf(bundle)
    audit = _metric_audit(packet)

    split = max(1, int(len(final_arr) * 0.8))
    train_std = float(np.mean(np.std(final_arr[:split], axis=0)))
    test_std = float(np.mean(np.std(final_arr[split:], axis=0)))
    if train_std <= 1e-9:
        generalization_ratio = 1.0
    else:
        generalization_ratio = max(0.0, min(2.0, 1.0 - ((test_std - train_std) / train_std)))

    drift_risk = min(1.0, max(0.0, 1.0 - ((outcome + stability) / 2.0)))

    metrics = {
        "rtf": float(rtf),
        "stability": float(stability),
        "audit": float(audit),
        "outcome": float(outcome),
        "learnability": float(learnability),
        "symbol_coverage": float(symbol_coverage),
        "generalization_ratio": float(generalization_ratio),
        "drift_risk": float(drift_risk),
    }

    passed = (
        metrics["rtf"] >= _CI_THRESHOLDS["rtf"]
        and metrics["stability"] >= _CI_THRESHOLDS["stability"]
        and metrics["audit"] >= _CI_THRESHOLDS["audit"]
        and metrics["outcome"] >= _CI_THRESHOLDS["outcome"]
        and metrics["learnability"] >= _CI_THRESHOLDS["learnability"]
        and metrics["generalization_ratio"] >= _CI_THRESHOLDS["generalization_ratio"]
    )

    notes: list[str] = []
    if metrics["audit"] < _CI_THRESHOLDS["audit"]:
        notes.append("Audit gate below threshold: provide falsifiers in proof_bundle.")
    if metrics["stability"] < _CI_THRESHOLDS["stability"]:
        notes.append("Stability gate below threshold: system has weak or unstable equilibrium.")
    if metrics["learnability"] < _CI_THRESHOLDS["learnability"]:
        notes.append("Learnability gate below threshold: increase multi-representation completeness.")

    sample_trajectories: list[list[dict[str, float]]] = []
    for traj in trajectories[:3]:
        points = []
        for row in traj[:: max(1, len(traj) // 20)]:
            points.append({vars_order[i]: float(row[i]) for i in range(len(vars_order))})
        sample_trajectories.append(points)

    artifact = FormulaEvaluationArtifact(
        packet_id=str(packet.packet_id),
        mc_runs=mc_runs,
        mean_final={vars_order[i]: float(eq_point[i]) for i in range(len(vars_order))},
        std_final={vars_order[i]: float(std_final[i]) for i in range(len(vars_order))},
        equilibrium_estimate={vars_order[i]: float(eq_point[i]) for i in range(len(vars_order))},
        bounded_ratio=float(bounded),
        convergence_ratio=float(convergence_ratio),
        stability_eigenvalues_real=eig_real,
        sample_trajectories=sample_trajectories,
    )

    return FormulaEvaluationResult(
        metrics=metrics,
        thresholds=dict(_CI_THRESHOLDS),
        passed=passed,
        notes=notes,
        artifact=artifact,
    )


def evaluation_artifact_hash(result: FormulaEvaluationResult) -> str:
    payload = {
        "metrics": result.metrics,
        "passed": result.passed,
        "artifact": result.artifact.model_dump(mode="json"),
    }
    return _hash_payload(payload)

