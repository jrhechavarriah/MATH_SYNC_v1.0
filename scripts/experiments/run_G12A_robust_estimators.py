from __future__ import annotations

# IMPORTANT:
# This runner is intentionally provided but should not be executed until all
# G12-A tests pass. It does not modify the legacy core or the public release.

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mathsync.g12_estimators import (
    solve_joint_lad_from_matrix,
    solve_joint_tukey_from_matrix,
    solve_joint_ransac_from_matrix,
)
from mathsync.legacy_v1_synthetic import (
    N_EVAL_SAMPLES_PER_SENSOR,
    N_REPLICATES,
    SCENARIOS,
    SEED,
    SENSORS,
    build_design_matrix,
    evaluate_methods_on_identical_samples,
    inverse_clock_parameters_from_solution,
    simulate_anchor_observations,
    solve_joint_huber,
    solve_joint_ols,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "g12" / "g12a"
REPORTS = ROOT / "reports" / "g12"
SPEC = ROOT / "config" / "G12A_ESTIMATOR_SPEC.json"


def _params(z, n_events, n_sensors):
    return inverse_clock_parameters_from_solution(z, n_events, n_sensors)


def run_trial(seed: int, scenario: str, config: dict) -> pd.DataFrame:
    tau, observations, true_clock = simulate_anchor_observations(
        seed=seed,
        **config,
    )
    n_events = len(tau)
    n_sensors = len(SENSORS)
    A, b = build_design_matrix(observations, n_events, n_sensors)

    methods = {}
    diagnostics = {}

    t0 = time.perf_counter()
    z_ols = solve_joint_ols(observations, n_events, n_sensors)
    diagnostics["Joint OLS"] = {
        "runtime_s": time.perf_counter() - t0,
        "converged": True,
        "status": "legacy_ok",
    }
    methods["Joint OLS"] = _params(z_ols, n_events, n_sensors)

    t0 = time.perf_counter()
    z_huber = solve_joint_huber(observations, n_events, n_sensors)
    diagnostics["Joint Huber"] = {
        "runtime_s": time.perf_counter() - t0,
        "converged": True,
        "status": "legacy_ok",
    }
    methods["Joint Huber"] = _params(z_huber, n_events, n_sensors)

    t0 = time.perf_counter()
    lad = solve_joint_lad_from_matrix(A, b)
    diagnostics["Joint LAD"] = {
        "runtime_s": time.perf_counter() - t0,
        "converged": lad.converged,
        "status": lad.status,
        "iterations": lad.iterations,
    }
    if lad.converged:
        methods["Joint LAD"] = _params(lad.z, n_events, n_sensors)

    t0 = time.perf_counter()
    tukey = solve_joint_tukey_from_matrix(A, b)
    diagnostics["Joint Tukey"] = {
        "runtime_s": time.perf_counter() - t0,
        "converged": tukey.converged,
        "status": tukey.status,
        "iterations": tukey.iterations,
    }
    if tukey.converged:
        methods["Joint Tukey"] = _params(tukey.z, n_events, n_sensors)

    t0 = time.perf_counter()
    ransac = solve_joint_ransac_from_matrix(A, b, seed=seed)
    diagnostics["Joint RANSAC"] = {
        "runtime_s": time.perf_counter() - t0,
        "converged": ransac.converged,
        "status": ransac.status,
        "iterations": ransac.iterations,
        "inlier_count": ransac.inlier_count,
        "threshold": ransac.threshold,
    }
    if ransac.converged:
        methods["Joint RANSAC"] = _params(ransac.z, n_events, n_sensors)

    evaluated = evaluate_methods_on_identical_samples(
        methods=methods,
        true_clock=true_clock,
        seed=seed + 55555,
        duration=config["duration"],
        jitter_ms=config["jitter_ms"],
        n_eval_samples=N_EVAL_SAMPLES_PER_SENSOR,
    )

    for method_name, diag in diagnostics.items():
        if method_name in set(evaluated["method"]):
            for key, value in diag.items():
                evaluated.loc[evaluated["method"] == method_name, key] = value

    # Preserve failed estimators in the audit trail.
    successful = set(evaluated["method"])
    failed_rows = []
    for method_name, diag in diagnostics.items():
        if method_name not in successful:
            failed_rows.append({
                "method": method_name,
                "rmse_ms": np.nan,
                "mae_ms": np.nan,
                "p95_abs_error_ms": np.nan,
                "gamma_rmse_ppm": np.nan,
                "kappa_rmse_ms": np.nan,
                **diag,
            })
    if failed_rows:
        evaluated = pd.concat(
            [evaluated, pd.DataFrame(failed_rows)],
            ignore_index=True,
            sort=False,
        )

    evaluated["scenario"] = scenario
    evaluated["seed"] = seed
    return evaluated


def main() -> int:
    if not SPEC.exists():
        raise RuntimeError("Frozen G12A_ESTIMATOR_SPEC.json not found.")

    spec = json.loads(SPEC.read_text(encoding="utf-8-sig"))
    if spec.get("status") != "FROZEN_BEFORE_G12A_OUTCOME_INSPECTION":
        raise RuntimeError("G12-A specification is not frozen.")

    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    frames = []
    running_seed = SEED + 1_000_000
    for scenario, config in SCENARIOS.items():
        for replicate in range(N_REPLICATES):
            trial = run_trial(running_seed, scenario, config)
            trial["replicate"] = replicate
            frames.append(trial)
            running_seed += 1

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(
        OUT / "G12A_robust_estimator_results.csv",
        index=False,
        float_format="%.15g",
    )

    # No manuscript interpretation is performed here.
    audit = {
        "gate": "G12A_EXECUTION_AUDIT",
        "n_rows": int(len(results)),
        "scenarios": sorted(results["scenario"].unique().tolist()),
        "methods_observed": sorted(results["method"].unique().tolist()),
        "failed_rows": int(results["rmse_ms"].isna().sum()),
        "results_file": str(OUT / "G12A_robust_estimator_results.csv"),
    }
    (REPORTS / "G12A_execution_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

