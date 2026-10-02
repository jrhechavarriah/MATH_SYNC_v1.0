from __future__ import annotations

import numpy as np

from mathsync.g12_estimators import (
    _mad_scale,
    _random_full_rank_subset,
    solve_joint_lad_from_matrix,
    solve_joint_tukey_from_matrix,
    solve_joint_ransac_from_matrix,
)


def test_mad_scale_floor():
    residual = np.zeros(10)
    assert _mad_scale(residual) == 1e-6


def test_lad_exact_recovery_on_consistent_system():
    A = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, -1.0],
    ])
    z_true = np.array([2.0, -3.0])
    b = A @ z_true

    result = solve_joint_lad_from_matrix(A, b)

    assert result.converged
    assert result.status == "ok"
    assert np.allclose(result.z, z_true, atol=1e-9)
    assert abs(result.objective or 0.0) <= 1e-9


def test_tukey_exact_recovery_on_consistent_system():
    A = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, -1.0],
        [-1.0, 2.0],
    ])
    z_true = np.array([1.25, -0.75])
    b = A @ z_true

    result = solve_joint_tukey_from_matrix(A, b)

    assert result.converged
    assert result.status == "ok"
    assert np.allclose(result.z, z_true, atol=1e-9)


def test_random_full_rank_subset_has_p_rows_and_full_rank():
    rng = np.random.default_rng(12345)
    A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ])

    subset = _random_full_rank_subset(A, rng)

    assert subset is not None
    assert len(subset) == A.shape[1]
    assert np.linalg.matrix_rank(A[subset, :]) == A.shape[1]


def test_ransac_is_deterministic_for_same_seed():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(40, 3))
    z_true = np.array([0.5, -1.2, 2.0])
    b = A @ z_true + rng.normal(0.0, 0.01, size=40)
    b[[3, 11, 25]] += np.array([1.5, -2.0, 1.0])

    r1 = solve_joint_ransac_from_matrix(A, b, seed=123)
    r2 = solve_joint_ransac_from_matrix(A, b, seed=123)

    assert r1.status == r2.status
    assert r1.converged == r2.converged
    assert r1.inlier_count == r2.inlier_count
    assert r1.threshold == r2.threshold
    assert np.array_equal(r1.z, r2.z)


def test_ransac_uses_same_problem_and_returns_finite_solution_when_feasible():
    rng = np.random.default_rng(17)
    A = rng.normal(size=(60, 4))
    z_true = np.array([1.0, -2.0, 0.75, 0.25])
    b = A @ z_true + rng.normal(0.0, 0.02, size=60)
    b[[1, 8, 30, 44]] += np.array([2.0, -1.5, 2.5, -2.0])

    result = solve_joint_ransac_from_matrix(A, b, seed=20260821)

    assert result.converged
    assert result.status == "ok"
    assert result.inlier_count is not None
    assert result.inlier_count >= A.shape[1]
    assert np.isfinite(result.z).all()
