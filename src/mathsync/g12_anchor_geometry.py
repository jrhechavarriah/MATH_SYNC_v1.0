from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from mathsync.legacy_v1_synthetic import (
    SENSORS,
    build_design_matrix,
    evaluate_methods_on_identical_samples,
    inverse_clock_parameters_from_solution,
)


SPAN_WINDOWS = {
    0.25: (45.75, 74.25),
    0.50: (31.50, 88.50),
    1.00: (3.00, 117.00),
}


@dataclass(frozen=True)
class HuberDiagnosticResult:
    z: np.ndarray
    weights: np.ndarray
    converged: bool
    iterations: int
    status: str
    final_relative_parameter_change: float


def make_anchor_times_geometry(
    rng: np.random.Generator,
    n_anchors: int,
    span_fraction: float,
    irregular_perturbation_s: float = 0.8,
) -> np.ndarray:
    if span_fraction not in SPAN_WINDOWS:
        raise ValueError(f"Unsupported span_fraction={span_fraction}")
    if n_anchors < 1:
        raise ValueError("n_anchors must be positive")

    start, end = SPAN_WINDOWS[span_fraction]
    base = np.linspace(start, end, n_anchors)
    perturbation = rng.uniform(
        -irregular_perturbation_s,
        irregular_perturbation_s,
        n_anchors,
    )
    return np.sort(np.clip(base + perturbation, start, end))


def simulate_geometry_observations(
    *,
    seed: int,
    n_anchors: int,
    span_fraction: float,
    drift_ppm: float,
    offset_s: float,
    jitter_ms: float,
    missing_prob: float,
    outlier_frac: float,
    outlier_ms: float,
):
    """
    Geometry-specific simulator.

    Important difference from legacy S1:
    no forced minimum number of retained non-reference anchors is applied.
    This is intentional so that sparse/missing designs can become genuinely
    rank deficient and be retained as G12-B outcomes.
    """
    rng = np.random.default_rng(seed)
    tau = make_anchor_times_geometry(
        rng,
        n_anchors=n_anchors,
        span_fraction=span_fraction,
    )

    rows = []
    true_clock: Dict[str, Tuple[float, float]] = {}

    for sensor_idx, (sensor_name, _) in enumerate(SENSORS):
        if sensor_idx == 0:
            a_i, b_i = 1.0, 0.0
        else:
            a_i = 1.0 + rng.uniform(-drift_ppm, drift_ppm) * 1e-6
            b_i = rng.uniform(-offset_s, offset_s)

        true_clock[sensor_name] = (a_i, b_i)

        observed = (
            a_i * tau
            + b_i
            + rng.normal(0.0, jitter_ms / 1000.0, n_anchors)
        )

        if sensor_idx == 0:
            keep = np.ones(n_anchors, dtype=bool)
            is_outlier = np.zeros(n_anchors, dtype=bool)
        else:
            keep = rng.random(n_anchors) >= missing_prob
            is_outlier = np.zeros(n_anchors, dtype=bool)
            candidates = np.where(keep)[0]
            n_outliers = int(round(outlier_frac * len(candidates)))

            if n_outliers > 0:
                idx = rng.choice(candidates, n_outliers, replace=False)
                is_outlier[idx] = True
                signs = rng.choice([-1.0, 1.0], size=n_outliers)
                magnitudes = rng.uniform(
                    0.5 * outlier_ms,
                    1.5 * outlier_ms,
                    size=n_outliers,
                ) / 1000.0
                observed[idx] += signs * magnitudes

        for event_idx in np.where(keep)[0]:
            rows.append(
                (
                    sensor_idx,
                    sensor_name,
                    int(event_idx),
                    float(tau[event_idx]),
                    float(observed[event_idx]),
                    bool(is_outlier[event_idx]),
                )
            )

    observations = pd.DataFrame(
        rows,
        columns=[
            "sensor_idx",
            "sensor",
            "event",
            "tau_true",
            "t_obs",
            "is_outlier",
        ],
    )
    return tau, observations, true_clock


def solve_joint_huber_with_diagnostics(
    observations: pd.DataFrame,
    n_events: int,
    n_sensors: int,
    *,
    max_iter: int = 30,
    huber_c: float = 1.345,
    tolerance: float = 1e-10,
    minimum_scale: float = 1e-6,
) -> HuberDiagnosticResult:
    """
    Diagnostic mirror of the frozen legacy Huber IRLS implementation.

    The numerical update rule matches legacy_v1_synthetic.solve_joint_huber.
    Extra outputs expose final weights, iteration count, and convergence status.
    """
    A, b = build_design_matrix(observations, n_events, n_sensors)
    p = A.shape[1]

    if A.shape[0] < p or np.linalg.matrix_rank(A) < p:
        return HuberDiagnosticResult(
            z=np.full(p, np.nan, dtype=float),
            weights=np.full(A.shape[0], np.nan, dtype=float),
            converged=False,
            iterations=0,
            status="rank_deficient_design",
            final_relative_parameter_change=np.nan,
        )

    z, *_ = np.linalg.lstsq(A, b, rcond=None)
    final_rel = np.nan
    final_weights = np.ones(A.shape[0], dtype=float)

    for iteration in range(1, max_iter + 1):
        residual = A @ z - b
        median_r = np.median(residual)
        mad = np.median(np.abs(residual - median_r))
        scale = max(1.4826 * mad, minimum_scale)

        standardized = np.abs(residual) / (huber_c * scale)
        weights = np.ones_like(standardized)
        mask = standardized > 1.0
        weights[mask] = 1.0 / standardized[mask]

        sqrt_w = np.sqrt(weights)
        weighted_A = A * sqrt_w[:, None]
        weighted_b = b * sqrt_w

        if np.linalg.matrix_rank(weighted_A) < p:
            return HuberDiagnosticResult(
                z=np.asarray(z, dtype=float),
                weights=np.asarray(weights, dtype=float),
                converged=False,
                iterations=iteration,
                status="weighted_design_rank_deficient",
                final_relative_parameter_change=np.nan,
            )

        z_new, *_ = np.linalg.lstsq(weighted_A, weighted_b, rcond=None)
        denom = 1.0 + np.linalg.norm(z)
        final_rel = float(np.linalg.norm(z_new - z) / denom)
        final_weights = weights

        if np.linalg.norm(z_new - z) <= tolerance * denom:
            return HuberDiagnosticResult(
                z=np.asarray(z_new, dtype=float),
                weights=np.asarray(final_weights, dtype=float),
                converged=True,
                iterations=iteration,
                status="ok",
                final_relative_parameter_change=final_rel,
            )

        z = z_new

    return HuberDiagnosticResult(
        z=np.asarray(z, dtype=float),
        weights=np.asarray(final_weights, dtype=float),
        converged=False,
        iterations=max_iter,
        status="max_iterations_reached",
        final_relative_parameter_change=final_rel,
    )


def structural_diagnostics(A: np.ndarray, weights: np.ndarray | None = None) -> dict:
    A = np.asarray(A, dtype=float)
    m, p = A.shape
    rank_a = int(np.linalg.matrix_rank(A))
    identifiable = bool(m >= p and rank_a == p)

    if weights is None or not np.all(np.isfinite(weights)):
        weighted_A = A
    else:
        weights = np.asarray(weights, dtype=float)
        weighted_A = A * np.sqrt(weights)[:, None]

    s = np.linalg.svd(weighted_A, compute_uv=False)
    sigma_min = float(s[-1]) if len(s) else np.nan
    sigma_max = float(s[0]) if len(s) else np.nan

    if not identifiable or not np.isfinite(sigma_min) or sigma_min <= 0.0:
        condition = np.inf
    else:
        condition = float(sigma_max / sigma_min)

    return {
        "n_observations": int(m),
        "n_parameters": int(p),
        "rank_A": rank_a,
        "identifiable": identifiable,
        "sigma_min_weighted_A": sigma_min,
        "condition_number_weighted_A": condition,
    }


def evaluate_geometry_trial(
    *,
    seed: int,
    scenario: str,
    config: dict,
    n_anchors: int,
    span_fraction: float,
    n_eval_samples: int,
) -> dict:
    tau, observations, true_clock = simulate_geometry_observations(
        seed=seed,
        n_anchors=n_anchors,
        span_fraction=span_fraction,
        drift_ppm=config["drift_ppm"],
        offset_s=config["offset_s"],
        jitter_ms=config["jitter_ms"],
        missing_prob=config["missing_prob"],
        outlier_frac=config["outlier_frac"],
        outlier_ms=config["outlier_ms"],
    )

    n_events = len(tau)
    n_sensors = len(SENSORS)
    A, _ = build_design_matrix(observations, n_events, n_sensors)

    initial = structural_diagnostics(A, weights=None)
    huber = solve_joint_huber_with_diagnostics(
        observations,
        n_events,
        n_sensors,
    )

    diag = structural_diagnostics(
        A,
        weights=huber.weights if huber.converged else None,
    )

    row = {
        "scenario": scenario,
        "anchor_count": int(n_anchors),
        "temporal_span_fraction": float(span_fraction),
        "seed": int(seed),
        **initial,
        "sigma_min_weighted_A": diag["sigma_min_weighted_A"],
        "condition_number_weighted_A": diag["condition_number_weighted_A"],
        "converged": bool(huber.converged),
        "iterations": int(huber.iterations),
        "solver_failure": not bool(huber.converged),
        "status": huber.status,
        "final_relative_parameter_change": huber.final_relative_parameter_change,
        "rmse_ms": np.nan,
        "mae_ms": np.nan,
        "p95_abs_error_ms": np.nan,
        "gamma_rmse_ppm": np.nan,
        "kappa_rmse_ms": np.nan,
    }

    if huber.converged:
        params = inverse_clock_parameters_from_solution(
            huber.z,
            n_events,
            n_sensors,
        )
        perf = evaluate_methods_on_identical_samples(
            methods={"Joint Huber": params},
            true_clock=true_clock,
            seed=seed + 55555,
            duration=config["duration"],
            jitter_ms=config["jitter_ms"],
            n_eval_samples=n_eval_samples,
        ).iloc[0]

        for col in [
            "rmse_ms",
            "mae_ms",
            "p95_abs_error_ms",
            "gamma_rmse_ppm",
            "kappa_rmse_ms",
        ]:
            row[col] = float(perf[col])

    return row
