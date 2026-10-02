from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class EstimatorResult:
    z: np.ndarray
    converged: bool
    iterations: int
    status: str
    objective: Optional[float] = None
    inlier_count: Optional[int] = None
    threshold: Optional[float] = None


def _mad_scale(residual: np.ndarray, minimum_scale: float = 1e-6) -> float:
    residual = np.asarray(residual, dtype=float)
    median_r = np.median(residual)
    mad = np.median(np.abs(residual - median_r))
    return float(max(1.4826 * mad, minimum_scale))


def solve_joint_lad_from_matrix(A: np.ndarray, b: np.ndarray) -> EstimatorResult:
    """
    Solve min_z sum_j |A_j z - b_j| exactly as a linear program.

    Variables are x = [z, u], where u_j >= |A_j z - b_j|.
    The z variables are unbounded; u_j >= 0.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
        raise ValueError("Incompatible A/b dimensions.")

    m, p = A.shape
    c = np.concatenate([np.zeros(p, dtype=float), np.ones(m, dtype=float)])

    # A z - u <= b
    # -A z - u <= -b
    eye = np.eye(m, dtype=float)
    A_ub = np.block([
        [A, -eye],
        [-A, -eye],
    ])
    b_ub = np.concatenate([b, -b])

    bounds = [(None, None)] * p + [(0.0, None)] * m

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        return EstimatorResult(
            z=np.full(p, np.nan, dtype=float),
            converged=False,
            iterations=int(getattr(result, "nit", 0) or 0),
            status=f"linprog_failed:{result.status}:{result.message}",
            objective=None,
        )

    z = np.asarray(result.x[:p], dtype=float)
    return EstimatorResult(
        z=z,
        converged=True,
        iterations=int(getattr(result, "nit", 0) or 0),
        status="ok",
        objective=float(result.fun),
    )


def solve_joint_tukey_from_matrix(
    A: np.ndarray,
    b: np.ndarray,
    *,
    tuning_constant: float = 4.685,
    minimum_scale: float = 1e-6,
    max_iterations: int = 30,
    relative_tolerance: float = 1e-10,
) -> EstimatorResult:
    """
    Tukey biweight M-estimation by IRLS on the frozen joint affine system.

    Frozen G12-A choices:
      - OLS initialization
      - c = 4.685
      - scale = 1.4826 * MAD, floored at 1e-6
      - max 30 iterations
      - relative tolerance 1e-10
      - w(u) = (1-u^2)^2 for |u| < 1, otherwise 0

    A rank-deficient weighted design is reported as a failure rather than
    silently regularized or retuned.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
        raise ValueError("Incompatible A/b dimensions.")

    m, p = A.shape
    z, *_ = np.linalg.lstsq(A, b, rcond=None)

    for iteration in range(1, max_iterations + 1):
        residual = A @ z - b
        scale = _mad_scale(residual, minimum_scale=minimum_scale)

        u = residual / (tuning_constant * scale)
        abs_u = np.abs(u)
        weights = np.zeros_like(abs_u)
        mask = abs_u < 1.0
        weights[mask] = (1.0 - u[mask] ** 2) ** 2

        positive = weights > 0.0
        if positive.sum() < p:
            return EstimatorResult(
                z=np.asarray(z, dtype=float),
                converged=False,
                iterations=iteration,
                status="weighted_design_insufficient_positive_rows",
            )

        sqrt_w = np.sqrt(weights)
        weighted_A = A * sqrt_w[:, None]
        weighted_b = b * sqrt_w

        if np.linalg.matrix_rank(weighted_A) < p:
            return EstimatorResult(
                z=np.asarray(z, dtype=float),
                converged=False,
                iterations=iteration,
                status="weighted_design_rank_deficient",
            )

        z_new, *_ = np.linalg.lstsq(weighted_A, weighted_b, rcond=None)

        if np.linalg.norm(z_new - z) <= relative_tolerance * (
            1.0 + np.linalg.norm(z)
        ):
            residual_new = A @ z_new - b
            # Tukey rho, up to an irrelevant multiplicative constant.
            scale_new = _mad_scale(residual_new, minimum_scale=minimum_scale)
            u_new = residual_new / (tuning_constant * scale_new)
            inside = np.abs(u_new) < 1.0
            rho = np.full_like(u_new, tuning_constant**2 / 6.0)
            rho[inside] = (tuning_constant**2 / 6.0) * (
                1.0 - (1.0 - u_new[inside] ** 2) ** 3
            )
            return EstimatorResult(
                z=np.asarray(z_new, dtype=float),
                converged=True,
                iterations=iteration,
                status="ok",
                objective=float(np.sum(rho)),
            )

        z = z_new

    return EstimatorResult(
        z=np.asarray(z, dtype=float),
        converged=False,
        iterations=max_iterations,
        status="max_iterations_reached",
    )


def _random_full_rank_subset(
    A: np.ndarray,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    Build a p-row random minimal subset by greedily accepting rows that
    increase rank. This preserves the frozen RANSAC requirement that the
    minimal subset contain p rows and have full column rank.
    """
    m, p = A.shape
    if m < p or np.linalg.matrix_rank(A) < p:
        return None

    order = rng.permutation(m)
    selected: list[int] = []
    current_rank = 0

    for idx in order:
        trial = selected + [int(idx)]
        rank = np.linalg.matrix_rank(A[trial, :])
        if rank > current_rank:
            selected.append(int(idx))
            current_rank = rank
            if current_rank == p:
                return np.asarray(selected, dtype=int)

    return None


def solve_joint_ransac_from_matrix(
    A: np.ndarray,
    b: np.ndarray,
    *,
    seed: int,
    residual_threshold_multiplier: float = 2.5,
    max_trials: int = 500,
    seed_offset: int = 712345,
    minimum_scale: float = 1e-6,
) -> EstimatorResult:
    """
    Deterministic RANSAC-style consensus estimator for the SAME joint A,b system.

    Implementation interpretation fixed before G12-A outcome inspection:
      1) Compute a non-oracle robust residual scale from the legacy OLS residual.
      2) threshold = 2.5 * robust MAD scale.
      3) For each trial, construct a random p-row full-rank minimal subset.
      4) Fit OLS on that subset.
      5) Select by maximum inlier count; break ties by minimum absolute
         inlier residual sum.
      6) Refit OLS on the winning consensus set.

    Injected `is_outlier` labels are never accepted by this API and therefore
    cannot influence the estimator.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
        raise ValueError("Incompatible A/b dimensions.")

    m, p = A.shape
    if m < p or np.linalg.matrix_rank(A) < p:
        return EstimatorResult(
            z=np.full(p, np.nan, dtype=float),
            converged=False,
            iterations=0,
            status="full_design_rank_deficient",
        )

    z_ols, *_ = np.linalg.lstsq(A, b, rcond=None)
    ols_residual = A @ z_ols - b
    scale = _mad_scale(ols_residual, minimum_scale=minimum_scale)
    threshold = float(residual_threshold_multiplier * scale)

    rng = np.random.default_rng(int(seed) + int(seed_offset))

    best_inliers: Optional[np.ndarray] = None
    best_count = -1
    best_abs_sum = np.inf
    trials_completed = 0

    for trial in range(1, max_trials + 1):
        subset = _random_full_rank_subset(A, rng)
        trials_completed = trial
        if subset is None:
            break

        z_candidate, *_ = np.linalg.lstsq(A[subset, :], b[subset], rcond=None)
        residual = np.abs(A @ z_candidate - b)
        inliers = residual <= threshold
        count = int(inliers.sum())

        if count < p:
            continue
        if np.linalg.matrix_rank(A[inliers, :]) < p:
            continue

        abs_sum = float(residual[inliers].sum())

        if (count > best_count) or (
            count == best_count and abs_sum < best_abs_sum
        ):
            best_inliers = inliers.copy()
            best_count = count
            best_abs_sum = abs_sum

    if best_inliers is None:
        return EstimatorResult(
            z=np.asarray(z_ols, dtype=float),
            converged=False,
            iterations=trials_completed,
            status="no_valid_consensus",
            inlier_count=None,
            threshold=threshold,
        )

    z_final, *_ = np.linalg.lstsq(
        A[best_inliers, :],
        b[best_inliers],
        rcond=None,
    )

    return EstimatorResult(
        z=np.asarray(z_final, dtype=float),
        converged=True,
        iterations=trials_completed,
        status="ok",
        objective=best_abs_sum,
        inlier_count=best_count,
        threshold=threshold,
    )
