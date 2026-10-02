from __future__ import annotations

import numpy as np

from mathsync.g12_anchor_geometry import (
    evaluate_geometry_trial,
    simulate_geometry_observations,
    solve_joint_huber_with_diagnostics,
    structural_diagnostics,
)
from mathsync.legacy_v1_synthetic import (
    N_EVAL_SAMPLES_PER_SENSOR,
    SCENARIOS,
    SENSORS,
    build_design_matrix,
    simulate_anchor_observations,
    solve_joint_huber,
)


def test_rank_deficient_design_is_retained_not_crashed():
    cfg = dict(SCENARIOS["stress"])
    cfg["missing_prob"] = 1.0

    row = evaluate_geometry_trial(
        seed=22260821,
        scenario="stress",
        config=cfg,
        n_anchors=4,
        span_fraction=0.25,
        n_eval_samples=N_EVAL_SAMPLES_PER_SENSOR,
    )

    assert row["identifiable"] is False
    assert row["solver_failure"] is True
    assert row["status"] == "rank_deficient_design"
    assert np.isnan(row["rmse_ms"])


def test_huber_diagnostic_mirror_matches_legacy_on_legacy_problem():
    tau, observations, _ = simulate_anchor_observations(
        seed=21260821,
        **SCENARIOS["nominal"],
    )
    n_events = len(tau)
    n_sensors = len(SENSORS)

    legacy = solve_joint_huber(observations, n_events, n_sensors)
    diag = solve_joint_huber_with_diagnostics(
        observations,
        n_events,
        n_sensors,
    )

    assert diag.converged
    assert np.array_equal(legacy, diag.z)


def test_structural_diagnostics_full_rank_identity():
    A = np.eye(5)
    d = structural_diagnostics(A)
    assert d["n_observations"] == 5
    assert d["n_parameters"] == 5
    assert d["rank_A"] == 5
    assert d["identifiable"] is True
    assert np.isclose(d["sigma_min_weighted_A"], 1.0)
    assert np.isclose(d["condition_number_weighted_A"], 1.0)
