from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

SEED = 20260821
N_REPLICATES = 50
DURATION_S = 120.0
N_ANCHORS = 24
N_EVAL_SAMPLES_PER_SENSOR = 300

SENSORS: List[Tuple[str, float]] = [
    ("REF_EVENT", 60.0),
    ("EEG", 250.0),
    ("PPG", 64.0),
    ("EDA", 32.0),
    ("IMU", 100.0),
    ("HMD", 90.0),
]

BASELINE_CONFIG = dict(
    duration=DURATION_S,
    n_anchors=N_ANCHORS,
    drift_ppm=100.0,
    offset_s=0.25,
    jitter_ms=1.0,
    missing_prob=0.10,
    outlier_frac=0.05,
    outlier_ms=60.0,
)

SCENARIOS = {
    "clean": dict(
        duration=DURATION_S,
        n_anchors=N_ANCHORS,
        drift_ppm=100.0,
        offset_s=0.25,
        jitter_ms=0.1,
        missing_prob=0.0,
        outlier_frac=0.0,
        outlier_ms=60.0,
    ),
    "nominal": BASELINE_CONFIG.copy(),
    "stress": dict(
        duration=DURATION_S,
        n_anchors=N_ANCHORS,
        drift_ppm=500.0,
        offset_s=0.50,
        jitter_ms=5.0,
        missing_prob=0.30,
        outlier_frac=0.15,
        outlier_ms=100.0,
    ),
}


def make_anchor_times(
    rng: np.random.Generator,
    duration: float,
    n_anchors: int,
) -> np.ndarray:
    base = np.linspace(3.0, duration - 3.0, n_anchors)
    perturbation = rng.uniform(-0.8, 0.8, n_anchors)
    return np.sort(np.clip(base + perturbation, 1.0, duration - 1.0))


def simulate_anchor_observations(
    seed: int,
    duration: float,
    n_anchors: int,
    drift_ppm: float,
    offset_s: float,
    jitter_ms: float,
    missing_prob: float,
    outlier_frac: float,
    outlier_ms: float,
):
    rng = np.random.default_rng(seed)
    tau = make_anchor_times(rng, duration, n_anchors)

    rows = []
    true_clock: Dict[str, Tuple[float, float]] = {}

    for sensor_idx, (sensor_name, _) in enumerate(SENSORS):
        if sensor_idx == 0:
            a_i, b_i = 1.0, 0.0
        else:
            a_i = 1.0 + rng.uniform(-drift_ppm, drift_ppm) * 1e-6
            b_i = rng.uniform(-offset_s, offset_s)

        true_clock[sensor_name] = (a_i, b_i)

        observed = (
            a_i * tau
            + b_i
            + rng.normal(0.0, jitter_ms / 1000.0, n_anchors)
        )

        if sensor_idx == 0:
            keep = np.ones(n_anchors, dtype=bool)
            is_outlier = np.zeros(n_anchors, dtype=bool)
        else:
            keep = rng.random(n_anchors) >= missing_prob

            if keep.sum() < 5:
                forced = rng.choice(n_anchors, 5, replace=False)
                keep[forced] = True

            is_outlier = np.zeros(n_anchors, dtype=bool)
            candidates = np.where(keep)[0]
            n_outliers = int(round(outlier_frac * len(candidates)))

            if n_outliers > 0:
                idx = rng.choice(candidates, n_outliers, replace=False)
                is_outlier[idx] = True
                signs = rng.choice([-1.0, 1.0], size=n_outliers)
                magnitudes = rng.uniform(
                    0.5 * outlier_ms,
                    1.5 * outlier_ms,
                    size=n_outliers,
                ) / 1000.0
                observed[idx] += signs * magnitudes

        for event_idx in np.where(keep)[0]:
            rows.append(
                (
                    sensor_idx,
                    sensor_name,
                    int(event_idx),
                    float(tau[event_idx]),
                    float(observed[event_idx]),
                    bool(is_outlier[event_idx]),
                )
            )

    observations = pd.DataFrame(
        rows,
        columns=[
            "sensor_idx",
            "sensor",
            "event",
            "tau_true",
            "t_obs",
            "is_outlier",
        ],
    )
    return tau, observations, true_clock


def build_design_matrix(
    observations: pd.DataFrame,
    n_events: int,
    n_sensors: int,
):
    n_parameters = n_events + 2 * (n_sensors - 1)
    A = np.zeros((len(observations), n_parameters), dtype=float)
    b = np.zeros(len(observations), dtype=float)

    for row_idx, row in enumerate(observations.itertuples(index=False)):
        event_idx = row.event
        sensor_idx = row.sensor_idx
        t_obs = row.t_obs

        if sensor_idx == 0:
            A[row_idx, event_idx] = 1.0
            b[row_idx] = t_obs
        else:
            A[row_idx, event_idx] = 1.0
            offset = n_events + 2 * (sensor_idx - 1)
            A[row_idx, offset] = -t_obs
            A[row_idx, offset + 1] = -1.0

    return A, b


def solve_joint_ols(
    observations: pd.DataFrame,
    n_events: int,
    n_sensors: int,
):
    A, b = build_design_matrix(observations, n_events, n_sensors)
    z, *_ = np.linalg.lstsq(A, b, rcond=None)
    return z


def solve_joint_huber(
    observations: pd.DataFrame,
    n_events: int,
    n_sensors: int,
    max_iter: int = 30,
    huber_c: float = 1.345,
    tolerance: float = 1e-10,
):
    A, b = build_design_matrix(observations, n_events, n_sensors)
    z, *_ = np.linalg.lstsq(A, b, rcond=None)

    for _ in range(max_iter):
        residual = A @ z - b
        median_r = np.median(residual)
        mad = np.median(np.abs(residual - median_r))
        scale = max(1.4826 * mad, 1e-6)

        standardized = np.abs(residual) / (huber_c * scale)
        weights = np.ones_like(standardized)
        mask = standardized > 1.0
        weights[mask] = 1.0 / standardized[mask]

        sqrt_w = np.sqrt(weights)
        weighted_A = A * sqrt_w[:, None]
        weighted_b = b * sqrt_w

        z_new, *_ = np.linalg.lstsq(weighted_A, weighted_b, rcond=None)

        if np.linalg.norm(z_new - z) <= tolerance * (
            1.0 + np.linalg.norm(z)
        ):
            z = z_new
            break
        z = z_new

    return z


def inverse_clock_parameters_from_solution(
    z: np.ndarray,
    n_events: int,
    n_sensors: int,
):
    params = {SENSORS[0][0]: (1.0, 0.0)}

    for sensor_idx in range(1, n_sensors):
        offset = n_events + 2 * (sensor_idx - 1)
        gamma_i = float(z[offset])
        kappa_i = float(z[offset + 1])
        params[SENSORS[sensor_idx][0]] = (gamma_i, kappa_i)

    return params


def estimate_offset_only(observations: pd.DataFrame):
    reference = (
        observations[observations.sensor_idx == 0][["event", "t_obs"]]
        .set_index("event")
        .t_obs
    )

    params = {SENSORS[0][0]: (1.0, 0.0)}

    for sensor_idx in range(1, len(SENSORS)):
        subset = observations[
            observations.sensor_idx == sensor_idx
        ][["event", "t_obs"]]

        differences = [
            float(reference.loc[row.event] - row.t_obs)
            for row in subset.itertuples(index=False)
            if row.event in reference.index
        ]

        params[SENSORS[sensor_idx][0]] = (
            1.0,
            float(np.median(differences)) if differences else 0.0,
        )

    return params


def evaluate_methods_on_identical_samples(
    methods: Dict[str, Dict[str, Tuple[float, float]]],
    true_clock: Dict[str, Tuple[float, float]],
    seed: int,
    duration: float,
    jitter_ms: float,
    n_eval_samples: int,
):
    rng = np.random.default_rng(seed)
    evaluation_data = {}

    for sensor_idx, (sensor_name, _) in enumerate(SENSORS):
        if sensor_idx == 0:
            continue

        tau_true = rng.uniform(0.0, duration, n_eval_samples)
        a_i, b_i = true_clock[sensor_name]
        local_timestamp = (
            a_i * tau_true
            + b_i
            + rng.normal(0.0, jitter_ms / 1000.0, n_eval_samples)
        )
        evaluation_data[sensor_name] = (tau_true, local_timestamp)

    output_rows = []

    for method_name, estimated_params in methods.items():
        all_errors_ms = []
        gamma_errors_ppm = []
        kappa_errors_ms = []

        for sensor_idx, (sensor_name, _) in enumerate(SENSORS):
            if sensor_idx == 0:
                continue

            tau_true, local_timestamp = evaluation_data[sensor_name]
            gamma_hat, kappa_hat = estimated_params[sensor_name]
            tau_hat = gamma_hat * local_timestamp + kappa_hat

            all_errors_ms.extend((tau_hat - tau_true) * 1000.0)

            a_i, b_i = true_clock[sensor_name]
            gamma_true = 1.0 / a_i
            kappa_true = -b_i / a_i

            gamma_errors_ppm.append(
                (gamma_hat - gamma_true) * 1e6
            )
            kappa_errors_ms.append(
                (kappa_hat - kappa_true) * 1000.0
            )

        errors = np.asarray(all_errors_ms, dtype=float)

        output_rows.append(
            {
                "method": method_name,
                "rmse_ms": float(np.sqrt(np.mean(errors**2))),
                "mae_ms": float(np.mean(np.abs(errors))),
                "p95_abs_error_ms": float(
                    np.quantile(np.abs(errors), 0.95)
                ),
                "gamma_rmse_ppm": float(
                    np.sqrt(
                        np.mean(
                            np.asarray(gamma_errors_ppm) ** 2
                        )
                    )
                ),
                "kappa_rmse_ms": float(
                    np.sqrt(
                        np.mean(
                            np.asarray(kappa_errors_ms) ** 2
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(output_rows)


def run_single_trial(seed: int, config: dict):
    tau, observations, true_clock = simulate_anchor_observations(
        seed=seed,
        **config,
    )
    n_events = len(tau)
    n_sensors = len(SENSORS)

    methods = {
        "Raw": {
            name: (1.0, 0.0)
            for name, _ in SENSORS
        },
        "Offset-only": estimate_offset_only(observations),
    }

    z_ols = solve_joint_ols(
        observations,
        n_events,
        n_sensors,
    )
    methods["Joint OLS"] = (
        inverse_clock_parameters_from_solution(
            z_ols,
            n_events,
            n_sensors,
        )
    )

    z_huber = solve_joint_huber(
        observations,
        n_events,
        n_sensors,
    )
    methods["Joint Huber"] = (
        inverse_clock_parameters_from_solution(
            z_huber,
            n_events,
            n_sensors,
        )
    )

    return evaluate_methods_on_identical_samples(
        methods=methods,
        true_clock=true_clock,
        seed=seed + 55555,
        duration=config["duration"],
        jitter_ms=config["jitter_ms"],
        n_eval_samples=N_EVAL_SAMPLES_PER_SENSOR,
    )


def run_named_scenarios(
    n_replicates: int = N_REPLICATES,
):
    frames = []
    running_seed = SEED + 1_000_000

    for scenario_name, config in SCENARIOS.items():
        for replicate in range(n_replicates):
            trial = run_single_trial(
                running_seed,
                config,
            )
            trial["scenario"] = scenario_name
            trial["replicate"] = replicate
            trial["seed"] = running_seed
            frames.append(trial)
            running_seed += 1

    return pd.concat(
        frames,
        ignore_index=True,
    )


def summarize_scenarios(
    scenario_results: pd.DataFrame,
):
    rows = []

    for (scenario, method), group in (
        scenario_results.groupby(
            ["scenario", "method"],
            sort=True,
        )
    ):
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "n": len(group),
                "rmse_ms_median": group["rmse_ms"].median(),
                "rmse_ms_q25": group["rmse_ms"].quantile(0.25),
                "rmse_ms_q75": group["rmse_ms"].quantile(0.75),
                "mae_ms_median": group["mae_ms"].median(),
                "p95_abs_error_ms_median": group[
                    "p95_abs_error_ms"
                ].median(),
                "gamma_rmse_ppm_median": group[
                    "gamma_rmse_ppm"
                ].median(),
                "kappa_rmse_ms_median": group[
                    "kappa_rmse_ms"
                ].median(),
            }
        )

    return pd.DataFrame(rows)
