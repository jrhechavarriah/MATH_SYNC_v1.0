import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validation"
    / "run_public_data_validation.py"
)

spec = importlib.util.spec_from_file_location("public_data_validation", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_frozen_scenarios_and_replicates():
    assert list(mod.SCENARIOS) == ["clean", "nominal", "stress"]
    assert mod.COHORT_REPLICATES == 50
    assert mod.SCENARIOS["nominal"]["jitter_ms"] == 1.0
    assert mod.SCENARIOS["stress"]["outlier_frac"] == 0.15


def test_seed_bases_are_frozen():
    assert mod.RANDOM_SEED == 20260821
    assert mod.S0201_SESSION1_SEED_BASE == 30260821
    assert mod.COHORT_SEED_BASE == 40260821


def test_deterministic_eval_timestamps():
    ts = np.arange(10000, dtype=float) / 100.0
    first = mod.deterministic_eval_timestamps(ts, 10.0, 80.0, max_samples=500)
    second = mod.deterministic_eval_timestamps(ts, 10.0, 80.0, max_samples=500)
    assert np.array_equal(first, second)
    assert len(first) <= 500
    assert first[0] >= 10.0
    assert first[-1] <= 80.0


def test_general_estimator_exact_noiseless_affine_case():
    tau = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    # Sensor 1: t = tau
    # Sensor 2: t = 1.0001*tau + 0.2
    rows = []
    for e, t in enumerate(tau):
        rows.append({"sensor_idx": 0, "sensor": "REF_EVENT", "event": e, "t_obs": t})
        rows.append({
            "sensor_idx": 1,
            "sensor": "S2",
            "event": e,
            "t_obs": 1.0001 * t + 0.2,
        })
    obs = pd.DataFrame(rows)
    names = ["REF_EVENT", "S2"]

    z = mod.solve_joint_ols_general(obs, len(tau), names)
    params = mod.inverse_params_general(z, len(tau), names)
    gamma, kappa = params["S2"]

    assert abs(gamma - 1.0 / 1.0001) < 1e-10
    assert abs(kappa - (-0.2 / 1.0001)) < 1e-10


def test_manuscript_targets_are_complete():
    assert len(mod.MANUSCRIPT_TABLE2_4DP) == 12
    assert len(mod.MANUSCRIPT_TABLE3_4DP) == 6
    assert set(mod.MANUSCRIPT_TABLE4) == {"Harvard", "WESAD"}
