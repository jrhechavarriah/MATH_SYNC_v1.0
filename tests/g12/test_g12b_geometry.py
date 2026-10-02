from __future__ import annotations

import numpy as np

from mathsync.g12_anchor_geometry import (
    SPAN_WINDOWS,
    make_anchor_times_geometry,
    simulate_geometry_observations,
)
from mathsync.legacy_v1_synthetic import SCENARIOS


def test_temporal_windows_are_frozen():
    assert SPAN_WINDOWS[0.25] == (45.75, 74.25)
    assert SPAN_WINDOWS[0.50] == (31.50, 88.50)
    assert SPAN_WINDOWS[1.00] == (3.00, 117.00)


def test_anchor_times_remain_inside_selected_window():
    for span in (0.25, 0.50, 1.00):
        rng = np.random.default_rng(123)
        tau = make_anchor_times_geometry(rng, 24, span)
        lo, hi = SPAN_WINDOWS[span]
        assert len(tau) == 24
        assert np.all(np.diff(tau) >= 0)
        assert tau.min() >= lo
        assert tau.max() <= hi


def test_geometry_simulation_is_deterministic():
    cfg = SCENARIOS["nominal"]
    a = simulate_geometry_observations(
        seed=22260821,
        n_anchors=8,
        span_fraction=0.50,
        drift_ppm=cfg["drift_ppm"],
        offset_s=cfg["offset_s"],
        jitter_ms=cfg["jitter_ms"],
        missing_prob=cfg["missing_prob"],
        outlier_frac=cfg["outlier_frac"],
        outlier_ms=cfg["outlier_ms"],
    )
    b = simulate_geometry_observations(
        seed=22260821,
        n_anchors=8,
        span_fraction=0.50,
        drift_ppm=cfg["drift_ppm"],
        offset_s=cfg["offset_s"],
        jitter_ms=cfg["jitter_ms"],
        missing_prob=cfg["missing_prob"],
        outlier_frac=cfg["outlier_frac"],
        outlier_ms=cfg["outlier_ms"],
    )

    assert np.array_equal(a[0], b[0])
    assert a[1].equals(b[1])
    assert a[2] == b[2]


def test_no_forced_minimum_nonreference_anchors():
    cfg = dict(SCENARIOS["stress"])
    cfg["missing_prob"] = 1.0
    tau, observations, _ = simulate_geometry_observations(
        seed=22260821,
        n_anchors=4,
        span_fraction=0.25,
        drift_ppm=cfg["drift_ppm"],
        offset_s=cfg["offset_s"],
        jitter_ms=cfg["jitter_ms"],
        missing_prob=cfg["missing_prob"],
        outlier_frac=cfg["outlier_frac"],
        outlier_ms=cfg["outlier_ms"],
    )

    # Reference stays fully observed; no non-reference row is forced back in.
    assert len(tau) == 4
    assert (observations["sensor_idx"] == 0).sum() == 4
    assert (observations["sensor_idx"] != 0).sum() == 0
