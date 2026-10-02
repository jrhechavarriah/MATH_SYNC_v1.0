from __future__ import annotations

import numpy as np

from mathsync.g12_estimators import (
    solve_joint_lad_from_matrix,
    solve_joint_ransac_from_matrix,
    solve_joint_tukey_from_matrix,
)
from mathsync.legacy_v1_synthetic import (
    SCENARIOS,
    SENSORS,
    build_design_matrix,
    simulate_anchor_observations,
)


def _problem(seed: int, scenario: str):
    tau, observations, _ = simulate_anchor_observations(
        seed=seed,
        **SCENARIOS[scenario],
    )
    A, b = build_design_matrix(observations, len(tau), len(SENSORS))
    return A, b


def test_g12a_estimators_accept_same_frozen_joint_problem():
    A, b = _problem(21260821, "nominal")
    p = A.shape[1]

    lad = solve_joint_lad_from_matrix(A, b)
    tukey = solve_joint_tukey_from_matrix(A, b)
    ransac = solve_joint_ransac_from_matrix(A, b, seed=21260821)

    assert lad.z.shape == (p,)
    assert tukey.z.shape == (p,)
    assert ransac.z.shape == (p,)

    assert np.isfinite(lad.z).all()
    assert np.isfinite(tukey.z).all()
    assert np.isfinite(ransac.z).all()


def test_ransac_conditional_feasibility_on_frozen_nominal_structure():
    A, b = _problem(21260821, "nominal")
    result = solve_joint_ransac_from_matrix(A, b, seed=21260821)

    assert np.linalg.matrix_rank(A) == A.shape[1]
    assert result.status in {"ok", "no_valid_consensus"}
    # This is a feasibility gate, not a performance assertion.
    # If no valid consensus is found, G12-A must report RANSAC as infeasible
    # rather than retune its frozen settings.
