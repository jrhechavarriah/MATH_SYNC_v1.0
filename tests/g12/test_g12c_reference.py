from __future__ import annotations

import numpy as np
import pandas as pd

from mathsync.g12_reference_stream import (
    SENSOR_NAMES,
    apply_reference_contamination,
    canonicalize_reference_parameters,
    reference_order,
    remap_observations_for_reference,
)
from mathsync.legacy_v1_synthetic import SENSORS


def test_all_six_candidate_references_are_available():
    assert SENSOR_NAMES == [name for name, _ in SENSORS]
    assert len(SENSOR_NAMES) == 6


def test_reference_order_places_selected_stream_first():
    for ref in SENSOR_NAMES:
        order = reference_order(ref)
        assert order[0] == ref
        assert sorted(order) == sorted(SENSOR_NAMES)


def test_remap_sets_selected_reference_to_zero():
    rows = []
    for event in range(2):
        for idx, name in enumerate(SENSOR_NAMES):
            rows.append((idx, name, event, float(event), float(event), False))

    obs = pd.DataFrame(
        rows,
        columns=["sensor_idx", "sensor", "event", "tau_true", "t_obs", "is_outlier"],
    )

    remapped, order = remap_observations_for_reference(obs, "EEG")
    assert order[0] == "EEG"
    assert set(
        remapped.loc[remapped["sensor"].eq("EEG"), "sensor_idx"]
    ) == {0}


def test_contamination_changes_only_selected_reference():
    rows = []
    for event, tau in enumerate(np.linspace(3.0, 117.0, 24)):
        for idx, name in enumerate(SENSOR_NAMES):
            rows.append((idx, name, event, tau, tau, False))

    obs = pd.DataFrame(
        rows,
        columns=["sensor_idx", "sensor", "event", "tau_true", "t_obs", "is_outlier"],
    )

    out = apply_reference_contamination(obs, "PPG")
    changed = ~np.isclose(
        out["t_obs"].to_numpy(),
        obs["t_obs"].to_numpy(),
        atol=0.0,
        rtol=0.0,
    )

    assert changed.any()
    assert out.loc[changed, "sensor"].eq("PPG").all()
    diff = (
        out.loc[out["sensor"].eq("PPG"), "t_obs"].to_numpy()
        - obs.loc[obs["sensor"].eq("PPG"), "t_obs"].to_numpy()
    )
    assert np.max(np.abs(diff)) <= 0.020000000001


def test_canonical_parameter_transform_is_exact_algebraically():
    params = {
        "REF_EVENT": (2.0, 0.5),
        "EEG": (1.0, 0.0),
    }
    can = canonicalize_reference_parameters(
        params,
        (1.0001, 0.2),
    )
    assert np.isclose(can["REF_EVENT"][0], 2.0 / 1.0001)
    assert np.isclose(can["REF_EVENT"][1], 0.3 / 1.0001)

