from __future__ import annotations

import numpy as np

from mathsync.g12_reference_stream import solve_reference_trial
from mathsync.legacy_v1_synthetic import (
    N_EVAL_SAMPLES_PER_SENSOR,
    SCENARIOS,
    SENSORS,
    simulate_anchor_observations,
)


def test_reference_trial_does_not_mutate_source_observations():
    seed = 23260821
    cfg = dict(SCENARIOS["clean"])
    _, observations, true_clock = simulate_anchor_observations(
        seed=seed, **cfg
    )
    before = observations.copy(deep=True)

    solve_reference_trial(
        observations=observations,
        true_clock=true_clock,
        reference_name="EEG",
        contaminated=True,
        seed=seed,
        duration=cfg["duration"],
        jitter_ms=cfg["jitter_ms"],
        n_eval_samples=N_EVAL_SAMPLES_PER_SENSOR,
    )

    assert observations.equals(before)


def test_clean_reference_trial_returns_complete_metric_schema():
    seed = 23260822
    cfg = dict(SCENARIOS["clean"])
    _, observations, true_clock = simulate_anchor_observations(
        seed=seed, **cfg
    )

    row = solve_reference_trial(
        observations=observations,
        true_clock=true_clock,
        reference_name=SENSORS[0][0],
        contaminated=False,
        seed=seed,
        duration=cfg["duration"],
        jitter_ms=cfg["jitter_ms"],
        n_eval_samples=N_EVAL_SAMPLES_PER_SENSOR,
    )

    required = {
        "rmse_ms",
        "mae_ms",
        "p95_abs_error_ms",
        "gamma_rmse_ppm",
        "kappa_rmse_ms",
        "sigma_min_weighted_A",
        "condition_number_weighted_A",
        "iterations",
        "converged",
        "solver_failure",
        "status",
    }
    assert required.issubset(row)
    assert row["converged"]
    assert np.isfinite(row["rmse_ms"])

