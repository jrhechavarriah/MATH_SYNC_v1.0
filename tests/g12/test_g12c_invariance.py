from __future__ import annotations

import numpy as np
import pandas as pd

from mathsync.g12_anchor_geometry import solve_joint_huber_with_diagnostics
from mathsync.g12_reference_stream import (
    SENSOR_NAMES,
    canonicalize_reference_parameters,
    extract_reference_coordinate_parameters,
    remap_observations_for_reference,
)
from mathsync.legacy_v1_synthetic import SENSORS


def make_noiseless_affine_problem():
    tau = np.linspace(3.0, 117.0, 24)
    true_clock = {}
    rows = []

    for idx, (name, _) in enumerate(SENSORS):
        a = 1.0 + idx * 20e-6
        b = (idx - 2) * 0.025
        true_clock[name] = (a, b)
        t = a * tau + b

        for event, (tau_e, t_e) in enumerate(zip(tau, t)):
            rows.append((idx, name, event, float(tau_e), float(t_e), False))

    observations = pd.DataFrame(
        rows,
        columns=["sensor_idx", "sensor", "event", "tau_true", "t_obs", "is_outlier"],
    )
    return observations, true_clock, len(tau)


def test_clean_reference_change_is_gauge_equivalent_in_noiseless_system():
    observations, true_clock, n_events = make_noiseless_affine_problem()

    for ref in SENSOR_NAMES:
        remapped, order = remap_observations_for_reference(observations, ref)
        fit = solve_joint_huber_with_diagnostics(
            remapped, n_events, len(SENSORS)
        )
        assert fit.converged

        params_ref = extract_reference_coordinate_parameters(
            fit.z, n_events, order
        )
        params_can = canonicalize_reference_parameters(
            params_ref, true_clock[ref]
        )

        for sensor_name in SENSOR_NAMES:
            a_i, b_i = true_clock[sensor_name]
            gamma_true = 1.0 / a_i
            kappa_true = -b_i / a_i
            gamma_hat, kappa_hat = params_can[sensor_name]

            assert np.isclose(gamma_hat, gamma_true, rtol=0.0, atol=2e-10)
            assert np.isclose(kappa_hat, kappa_true, rtol=0.0, atol=2e-8)


def test_canonical_solutions_agree_across_references_noiseless():
    observations, true_clock, n_events = make_noiseless_affine_problem()
    solutions = []

    for ref in SENSOR_NAMES:
        remapped, order = remap_observations_for_reference(observations, ref)
        fit = solve_joint_huber_with_diagnostics(
            remapped, n_events, len(SENSORS)
        )
        params_ref = extract_reference_coordinate_parameters(
            fit.z, n_events, order
        )
        solutions.append(
            canonicalize_reference_parameters(
                params_ref, true_clock[ref]
            )
        )

    baseline = solutions[0]
    for solution in solutions[1:]:
        for sensor_name in SENSOR_NAMES:
            assert np.allclose(
                solution[sensor_name],
                baseline[sensor_name],
                rtol=0.0,
                atol=2e-8,
            )
