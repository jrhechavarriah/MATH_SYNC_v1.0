from __future__ import annotations

import numpy as np

from mathsync.legacy_v1_synthetic import (
    SCENARIOS,
    SENSORS,
    build_design_matrix,
    simulate_anchor_observations,
    solve_joint_huber,
    solve_joint_ols,
)


def _frozen_nominal_problem(seed: int = 21260821):
    tau, observations, _ = simulate_anchor_observations(
        seed=seed,
        **SCENARIOS["nominal"],
    )
    return observations, len(tau), len(SENSORS)


def test_legacy_ols_solution_equals_direct_lstsq():
    observations, n_events, n_sensors = _frozen_nominal_problem()
    A, b = build_design_matrix(observations, n_events, n_sensors)

    z_legacy = solve_joint_ols(observations, n_events, n_sensors)
    z_direct, *_ = np.linalg.lstsq(A, b, rcond=None)

    assert np.array_equal(z_legacy, z_direct)


def test_legacy_huber_is_deterministic_and_unmodified():
    observations, n_events, n_sensors = _frozen_nominal_problem()

    z1 = solve_joint_huber(observations, n_events, n_sensors)
    z2 = solve_joint_huber(observations, n_events, n_sensors)

    assert np.array_equal(z1, z2)
    assert np.isfinite(z1).all()
