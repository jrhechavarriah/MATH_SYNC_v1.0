from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from mathsync.g12_anchor_geometry import (
    solve_joint_huber_with_diagnostics,
    structural_diagnostics,
)
from mathsync.legacy_v1_synthetic import SENSORS


SENSOR_NAMES = [name for name, _ in SENSORS]
CONTAMINATION_AMPLITUDE_S = 0.020
CONTAMINATION_START_S = 3.0
CONTAMINATION_PERIOD_S = 114.0


def reference_order(reference_name: str) -> List[str]:
    if reference_name not in SENSOR_NAMES:
        raise ValueError(f"Unknown reference stream: {reference_name}")
    return [reference_name] + [name for name in SENSOR_NAMES if name != reference_name]


def apply_reference_contamination(
    observations: pd.DataFrame,
    reference_name: str,
) -> pd.DataFrame:
    out = observations.copy(deep=True)
    mask = out["sensor"].eq(reference_name)
    tau = out.loc[mask, "tau_true"].to_numpy(dtype=float)
    delta = CONTAMINATION_AMPLITUDE_S * np.sin(
        2.0 * np.pi * (tau - CONTAMINATION_START_S) / CONTAMINATION_PERIOD_S
    )
    out.loc[mask, "t_obs"] = (
        out.loc[mask, "t_obs"].to_numpy(dtype=float) + delta
    )
    return out


def remap_observations_for_reference(
    observations: pd.DataFrame,
    reference_name: str,
) -> Tuple[pd.DataFrame, List[str]]:
    order = reference_order(reference_name)
    index_map = {name: idx for idx, name in enumerate(order)}
    out = observations.copy(deep=True)
    out["sensor_idx"] = out["sensor"].map(index_map).astype(int)
    out = out.sort_values(["event", "sensor_idx"], kind="mergesort").reset_index(drop=True)
    return out, order


def extract_reference_coordinate_parameters(
    z: np.ndarray,
    n_events: int,
    sensor_order: List[str],
) -> Dict[str, Tuple[float, float]]:
    params: Dict[str, Tuple[float, float]] = {sensor_order[0]: (1.0, 0.0)}
    for sensor_idx in range(1, len(sensor_order)):
        offset = n_events + 2 * (sensor_idx - 1)
        params[sensor_order[sensor_idx]] = (
            float(z[offset]),
            float(z[offset + 1]),
        )
    return params


def canonicalize_reference_parameters(
    params_reference_coordinate: Dict[str, Tuple[float, float]],
    true_reference_clock: Tuple[float, float],
) -> Dict[str, Tuple[float, float]]:
    a_ref, b_ref = true_reference_clock
    if a_ref == 0:
        raise ValueError("Reference clock slope cannot be zero.")
    return {
        sensor_name: (
            float(gamma_ref / a_ref),
            float((kappa_ref - b_ref) / a_ref),
        )
        for sensor_name, (gamma_ref, kappa_ref)
        in params_reference_coordinate.items()
    }


def evaluate_canonical_parameters_all_streams(
    estimated_params: Dict[str, Tuple[float, float]],
    true_clock: Dict[str, Tuple[float, float]],
    *,
    seed: int,
    duration: float,
    jitter_ms: float,
    n_eval_samples: int,
) -> dict:
    rng = np.random.default_rng(seed)
    all_errors_ms = []
    gamma_errors_ppm = []
    kappa_errors_ms = []

    for sensor_name, _ in SENSORS:
        tau_true = rng.uniform(0.0, duration, n_eval_samples)
        a_i, b_i = true_clock[sensor_name]
        local_timestamp = (
            a_i * tau_true
            + b_i
            + rng.normal(0.0, jitter_ms / 1000.0, n_eval_samples)
        )
        gamma_hat, kappa_hat = estimated_params[sensor_name]
        tau_hat = gamma_hat * local_timestamp + kappa_hat
        all_errors_ms.extend((tau_hat - tau_true) * 1000.0)

        gamma_true = 1.0 / a_i
        kappa_true = -b_i / a_i
        gamma_errors_ppm.append((gamma_hat - gamma_true) * 1e6)
        kappa_errors_ms.append((kappa_hat - kappa_true) * 1000.0)

    errors = np.asarray(all_errors_ms, dtype=float)
    return {
        "rmse_ms": float(np.sqrt(np.mean(errors**2))),
        "mae_ms": float(np.mean(np.abs(errors))),
        "p95_abs_error_ms": float(np.quantile(np.abs(errors), 0.95)),
        "gamma_rmse_ppm": float(
            np.sqrt(np.mean(np.asarray(gamma_errors_ppm, dtype=float) ** 2))
        ),
        "kappa_rmse_ms": float(
            np.sqrt(np.mean(np.asarray(kappa_errors_ms, dtype=float) ** 2))
        ),
    }


def solve_reference_trial(
    *,
    observations: pd.DataFrame,
    true_clock: Dict[str, Tuple[float, float]],
    reference_name: str,
    contaminated: bool,
    seed: int,
    duration: float,
    jitter_ms: float,
    n_eval_samples: int,
) -> dict:
    work = observations.copy(deep=True)

    if contaminated:
        work = apply_reference_contamination(work, reference_name)

    work, order = remap_observations_for_reference(work, reference_name)
    n_events = int(work["event"].max()) + 1
    n_sensors = len(SENSORS)

    huber = solve_joint_huber_with_diagnostics(work, n_events, n_sensors)

    from mathsync.legacy_v1_synthetic import build_design_matrix
    A, _ = build_design_matrix(work, n_events, n_sensors)

    diag = structural_diagnostics(
        A,
        weights=huber.weights if huber.converged else None,
    )

    row = {
        "reference": reference_name,
        "condition": (
            "reference_systematically_contaminated"
            if contaminated
            else "reference_clean"
        ),
        "seed": int(seed),
        "n_observations": int(diag["n_observations"]),
        "n_parameters": int(diag["n_parameters"]),
        "rank_A": int(diag["rank_A"]),
        "identifiable": bool(diag["identifiable"]),
        "sigma_min_weighted_A": float(diag["sigma_min_weighted_A"])
        if np.isfinite(diag["sigma_min_weighted_A"]) else np.nan,
        "condition_number_weighted_A": float(diag["condition_number_weighted_A"])
        if np.isfinite(diag["condition_number_weighted_A"]) else np.inf,
        "iterations": int(huber.iterations),
        "converged": bool(huber.converged),
        "solver_failure": not bool(huber.converged),
        "status": huber.status,
        "final_relative_parameter_change":
            huber.final_relative_parameter_change,
        "rmse_ms": np.nan,
        "mae_ms": np.nan,
        "p95_abs_error_ms": np.nan,
        "gamma_rmse_ppm": np.nan,
        "kappa_rmse_ms": np.nan,
    }

    if not huber.converged:
        return row

    params_ref = extract_reference_coordinate_parameters(
        huber.z, n_events, order
    )
    params_can = canonicalize_reference_parameters(
        params_ref, true_clock[reference_name]
    )

    row.update(
        evaluate_canonical_parameters_all_streams(
            params_can,
            true_clock,
            seed=seed + 55555,
            duration=duration,
            jitter_ms=jitter_ms,
            n_eval_samples=n_eval_samples,
        )
    )
    return row
