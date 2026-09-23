
"""
MATH-SYNC S1
Robust affine-clock alignment of heterogeneous multirate sensor streams.

This script generates the first controlled Monte Carlo battery for the
MATH-SYNC computational release. It is intentionally hardware-agnostic and fully
reproducible.

Outputs
-------
results/s1_monte_carlo_results.csv
results/s1_summary_by_factor.csv
results/s1_huber_pairwise_tests.csv
results/s1_scenario_summary.csv
figures/s1_rmse_vs_jitter.png
figures/s1_rmse_vs_outliers.png
figures/s1_rmse_vs_missingness.png
figures/s1_rmse_vs_drift.png
figures/s1_runtime_by_method.png
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, rankdata


SEED = 20260821
N_REPLICATES = 50
DURATION_S = 120.0
N_ANCHORS = 24
N_EVAL_SAMPLES_PER_SENSOR = 300

# The reference/event stream defines the temporal gauge.
# Native sensor rates are intentionally heterogeneous and are never resampled.
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

SWEEPS = {
    "jitter_ms": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "outlier_frac": [0.00, 0.02, 0.05, 0.10, 0.20],
    "missing_prob": [0.00, 0.05, 0.10, 0.20, 0.40],
    "drift_ppm": [0.0, 10.0, 50.0, 100.0, 250.0, 500.0],
}

SCENARIOS = {
    "clean": dict(
        duration=DURATION_S, n_anchors=N_ANCHORS, drift_ppm=100.0,
        offset_s=0.25, jitter_ms=0.1, missing_prob=0.0,
        outlier_frac=0.0, outlier_ms=60.0
    ),
    "nominal": BASELINE_CONFIG.copy(),
    "stress": dict(
        duration=DURATION_S, n_anchors=N_ANCHORS, drift_ppm=500.0,
        offset_s=0.50, jitter_ms=5.0, missing_prob=0.30,
        outlier_frac=0.15, outlier_ms=100.0
    ),
}


def make_anchor_times(rng: np.random.Generator,
                      duration: float,
                      n_anchors: int) -> np.ndarray:
    """Generate irregular but well-spread global anchor-event times."""
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
    """
    Simulate observed anchor timestamps under independent affine device clocks.

    Forward clock:
        t_i = a_i * tau + b_i + epsilon_i

    The reference/event stream uses a_ref = 1 and b_ref = 0 and is fully
    observed. Non-reference streams may lose anchor observations or contain
    timestamp outliers.
    """
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

            # Maintain a minimally identifiable sensor-to-reference connection.
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
    """
    Construct the linear inverse-clock problem.

    Unknown vector:
        z = [tau_1,...,tau_M, gamma_2,kappa_2,...,gamma_N,kappa_N]^T

    Reference equation:
        tau_e = t_ref,e

    Non-reference equation:
        tau_e - gamma_i t_i,e - kappa_i = 0
    """
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
    """Ordinary least-squares solution of the joint latent-time model."""
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
    """
    Huber M-estimation solved by iteratively reweighted least squares (IRLS).

    The robust scale is estimated from the median absolute deviation (MAD).
    No oracle knowledge of injected outliers is used by the estimator.
    """
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

        if np.linalg.norm(z_new - z) <= tolerance * (1.0 + np.linalg.norm(z)):
            z = z_new
            break
        z = z_new

    return z


def inverse_clock_parameters_from_solution(
    z: np.ndarray,
    n_events: int,
    n_sensors: int,
):
    """Extract gamma_i and kappa_i from the joint solution."""
    params = {SENSORS[0][0]: (1.0, 0.0)}

    for sensor_idx in range(1, n_sensors):
        offset = n_events + 2 * (sensor_idx - 1)
        gamma_i = float(z[offset])
        kappa_i = float(z[offset + 1])
        params[SENSORS[sensor_idx][0]] = (gamma_i, kappa_i)

    return params


def estimate_offset_only(observations: pd.DataFrame):
    """
    Robust offset-only baseline.

    The slope is fixed at one. The offset is estimated as the median timestamp
    difference against the fully observed reference/event stream.
    """
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
    """
    Evaluate all methods on exactly the same sampled timestamps.

    This pairing is essential for fair Monte Carlo comparisons.
    """
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

            gamma_errors_ppm.append((gamma_hat - gamma_true) * 1e6)
            kappa_errors_ms.append((kappa_hat - kappa_true) * 1000.0)

        errors = np.asarray(all_errors_ms, dtype=float)

        output_rows.append(
            {
                "method": method_name,
                "rmse_ms": float(np.sqrt(np.mean(errors**2))),
                "mae_ms": float(np.mean(np.abs(errors))),
                "p95_abs_error_ms": float(np.quantile(np.abs(errors), 0.95)),
                "gamma_rmse_ppm": float(
                    np.sqrt(np.mean(np.asarray(gamma_errors_ppm) ** 2))
                ),
                "kappa_rmse_ms": float(
                    np.sqrt(np.mean(np.asarray(kappa_errors_ms) ** 2))
                ),
            }
        )

    return pd.DataFrame(output_rows)


def run_single_trial(seed: int, config: dict):
    """Run all synchronization methods on one paired Monte Carlo realization."""
    tau, observations, true_clock = simulate_anchor_observations(
        seed=seed,
        **config,
    )
    n_events = len(tau)
    n_sensors = len(SENSORS)

    methods = {
        "Raw": {name: (1.0, 0.0) for name, _ in SENSORS},
        "Offset-only": estimate_offset_only(observations),
    }

    start = time.perf_counter()
    z_ols = solve_joint_ols(observations, n_events, n_sensors)
    ols_runtime = time.perf_counter() - start
    methods["Joint OLS"] = inverse_clock_parameters_from_solution(
        z_ols, n_events, n_sensors
    )

    start = time.perf_counter()
    z_huber = solve_joint_huber(observations, n_events, n_sensors)
    huber_runtime = time.perf_counter() - start
    methods["Joint Huber"] = inverse_clock_parameters_from_solution(
        z_huber, n_events, n_sensors
    )

    results = evaluate_methods_on_identical_samples(
        methods=methods,
        true_clock=true_clock,
        seed=seed + 55555,
        duration=config["duration"],
        jitter_ms=config["jitter_ms"],
        n_eval_samples=N_EVAL_SAMPLES_PER_SENSOR,
    )

    runtime_map = {
        "Raw": 0.0,
        "Offset-only": 0.0,
        "Joint OLS": ols_runtime,
        "Joint Huber": huber_runtime,
    }
    results["runtime_s"] = results["method"].map(runtime_map)

    return results


def run_sweep_battery(n_replicates: int = N_REPLICATES):
    """Run the complete S1 one-factor-at-a-time Monte Carlo battery."""
    frames = []
    running_seed = SEED

    for factor, levels in SWEEPS.items():
        for level in levels:
            config = BASELINE_CONFIG.copy()
            config[factor] = level

            for replicate in range(n_replicates):
                trial = run_single_trial(running_seed, config)
                trial["factor"] = factor
                trial["level"] = float(level)
                trial["replicate"] = replicate
                trial["seed"] = running_seed
                frames.append(trial)
                running_seed += 1

    return pd.concat(frames, ignore_index=True)


def run_named_scenarios(n_replicates: int = N_REPLICATES):
    """Run clean, nominal, and combined-stress sanity-check scenarios."""
    frames = []
    running_seed = SEED + 1_000_000

    for scenario_name, config in SCENARIOS.items():
        for replicate in range(n_replicates):
            trial = run_single_trial(running_seed, config)
            trial["scenario"] = scenario_name
            trial["replicate"] = replicate
            trial["seed"] = running_seed
            frames.append(trial)
            running_seed += 1

    return pd.concat(frames, ignore_index=True)


def bootstrap_median_ci(values, seed, n_boot=2000, alpha=0.05):
    """Percentile bootstrap confidence interval for the sample median."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    medians = np.median(values[indices], axis=1)
    return (
        float(np.quantile(medians, alpha / 2.0)),
        float(np.quantile(medians, 1.0 - alpha / 2.0)),
    )


def summarize_sweeps(results: pd.DataFrame):
    """Aggregate Monte Carlo results with deterministic bootstrap CIs."""
    rows = []
    counter = 0

    for (factor, level, method), group in results.groupby(
        ["factor", "level", "method"], sort=True
    ):
        lo, hi = bootstrap_median_ci(
            group["rmse_ms"].to_numpy(),
            seed=SEED + 2_000_000 + counter,
        )
        rows.append(
            {
                "factor": factor,
                "level": level,
                "method": method,
                "n": len(group),
                "rmse_ms_mean": group["rmse_ms"].mean(),
                "rmse_ms_median": group["rmse_ms"].median(),
                "rmse_ms_q25": group["rmse_ms"].quantile(0.25),
                "rmse_ms_q75": group["rmse_ms"].quantile(0.75),
                "rmse_ms_median_ci95_low": lo,
                "rmse_ms_median_ci95_high": hi,
                "mae_ms_median": group["mae_ms"].median(),
                "p95_abs_error_ms_median": group["p95_abs_error_ms"].median(),
                "gamma_rmse_ppm_median": group["gamma_rmse_ppm"].median(),
                "kappa_rmse_ms_median": group["kappa_rmse_ms"].median(),
                "runtime_s_median": group["runtime_s"].median(),
            }
        )
        counter += 1

    return pd.DataFrame(rows)


def paired_rank_biserial(x, y):
    """
    Matched-pairs rank-biserial effect size for improvement x - y.

    Positive values mean x tends to be larger than y.
    When x = OLS error and y = Huber error, positive values favor Huber.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]

    if len(d) == 0:
        return 0.0

    ranks = rankdata(np.abs(d))
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def holm_adjust(p_values):
    """Holm family-wise error-rate correction."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)

    return adjusted


def compare_huber_against_baselines(results: pd.DataFrame):
    """Paired Wilcoxon tests for Joint Huber versus OLS and offset-only."""
    rows = []

    for (factor, level), group in results.groupby(["factor", "level"], sort=True):
        pivot = group.pivot_table(
            index="seed",
            columns="method",
            values="rmse_ms",
            aggfunc="first",
        )

        for baseline in ["Joint OLS", "Offset-only"]:
            paired = pivot[[baseline, "Joint Huber"]].dropna()
            baseline_values = paired[baseline].to_numpy()
            huber_values = paired["Joint Huber"].to_numpy()

            if np.allclose(baseline_values, huber_values):
                p_value = 1.0
            else:
                _, p_value = wilcoxon(
                    baseline_values,
                    huber_values,
                    alternative="two-sided",
                    zero_method="wilcox",
                )

            improvement = baseline_values - huber_values
            lo, hi = bootstrap_median_ci(
                improvement,
                seed=SEED
                + 3_000_000
                + len(rows),
            )

            rows.append(
                {
                    "factor": factor,
                    "level": level,
                    "baseline": baseline,
                    "n_pairs": len(paired),
                    "median_baseline_rmse_ms": np.median(baseline_values),
                    "median_huber_rmse_ms": np.median(huber_values),
                    "median_improvement_ms": np.median(improvement),
                    "median_improvement_ci95_low": lo,
                    "median_improvement_ci95_high": hi,
                    "huber_win_rate": np.mean(huber_values < baseline_values),
                    "rank_biserial_baseline_minus_huber": paired_rank_biserial(
                        baseline_values,
                        huber_values,
                    ),
                    "wilcoxon_p_raw": p_value,
                }
            )

    tests = pd.DataFrame(rows)
    tests["wilcoxon_p_holm"] = holm_adjust(tests["wilcoxon_p_raw"].to_numpy())
    return tests


def summarize_scenarios(scenario_results: pd.DataFrame):
    rows = []

    for (scenario, method), group in scenario_results.groupby(
        ["scenario", "method"], sort=True
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
                "p95_abs_error_ms_median": group["p95_abs_error_ms"].median(),
                "gamma_rmse_ppm_median": group["gamma_rmse_ppm"].median(),
                "kappa_rmse_ms_median": group["kappa_rmse_ms"].median(),
            }
        )

    return pd.DataFrame(rows)


def plot_factor(summary: pd.DataFrame, factor: str, output_path: Path):
    """Create one standalone publication-oriented figure per corruption factor."""
    subset = summary[summary.factor == factor]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for method in ["Raw", "Offset-only", "Joint OLS", "Joint Huber"]:
        method_data = subset[subset.method == method].sort_values("level")
        ax.plot(
            method_data["level"],
            method_data["rmse_ms_median"],
            marker="o",
            label=method,
        )

    labels = {
        "jitter_ms": "Timestamp jitter standard deviation (ms)",
        "outlier_frac": "Timestamp-outlier fraction",
        "missing_prob": "Missing-anchor probability",
        "drift_ppm": "Maximum absolute clock drift (ppm)",
    }

    ax.set_xlabel(labels[factor])
    ax.set_ylabel("Median reconstructed-time RMSE (ms)")
    ax.set_title(f"MATH-SYNC S1: sensitivity to {factor.replace('_', ' ')}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_runtime(results: pd.DataFrame, output_path: Path):
    runtime = (
        results[results.method.isin(["Joint OLS", "Joint Huber"])]
        .groupby("method", as_index=False)["runtime_s"]
        .median()
    )

    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    ax.bar(runtime["method"], runtime["runtime_s"])
    ax.set_ylabel("Median estimator runtime per trial (s)")
    ax.set_title("MATH-SYNC S1: computational runtime")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(output_root=".", n_replicates=N_REPLICATES):
    output_root = Path(output_root)
    results_dir = output_root / "results"
    figures_dir = output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    sweep_results = run_sweep_battery(n_replicates=n_replicates)
    scenario_results = run_named_scenarios(n_replicates=n_replicates)

    summary = summarize_sweeps(sweep_results)
    pairwise_tests = compare_huber_against_baselines(sweep_results)
    scenario_summary = summarize_scenarios(scenario_results)

    sweep_results.to_csv(
        results_dir / "s1_monte_carlo_results.csv",
        index=False,
    )
    scenario_results.to_csv(
        results_dir / "s1_named_scenario_results.csv",
        index=False,
    )
    summary.to_csv(
        results_dir / "s1_summary_by_factor.csv",
        index=False,
    )
    pairwise_tests.to_csv(
        results_dir / "s1_huber_pairwise_tests.csv",
        index=False,
    )
    scenario_summary.to_csv(
        results_dir / "s1_scenario_summary.csv",
        index=False,
    )

    plot_factor(
        summary,
        "jitter_ms",
        figures_dir / "s1_rmse_vs_jitter.png",
    )
    plot_factor(
        summary,
        "outlier_frac",
        figures_dir / "s1_rmse_vs_outliers.png",
    )
    plot_factor(
        summary,
        "missing_prob",
        figures_dir / "s1_rmse_vs_missingness.png",
    )
    plot_factor(
        summary,
        "drift_ppm",
        figures_dir / "s1_rmse_vs_drift.png",
    )
    plot_runtime(
        sweep_results,
        figures_dir / "s1_runtime_by_method.png",
    )

    elapsed = time.perf_counter() - start

    metadata = {
        "seed": SEED,
        "n_replicates": n_replicates,
        "duration_s": DURATION_S,
        "n_anchors": N_ANCHORS,
        "n_eval_samples_per_sensor": N_EVAL_SAMPLES_PER_SENSOR,
        "sensors": [{"name": n, "rate_hz": r} for n, r in SENSORS],
        "baseline_config": BASELINE_CONFIG,
        "sweeps": SWEEPS,
        "scenarios": SCENARIOS,
        "elapsed_seconds": elapsed,
    }

    with open(results_dir / "s1_run_metadata.json", "w", encoding="utf-8") as f:
        import json
        json.dump(metadata, f, indent=2)

    return sweep_results, summary, pairwise_tests, scenario_summary




# ----------------------------------------------------------------------
# Standalone public entry point.
# ----------------------------------------------------------------------

def validate_public_s1(output_root):
    output_root = Path(output_root)
    scenario_path = output_root / "results" / "s1_scenario_summary.csv"
    if not scenario_path.exists():
        raise FileNotFoundError(scenario_path)

    scenario = pd.read_csv(scenario_path)
    pivot = scenario.pivot(
        index="method",
        columns="scenario",
        values="rmse_ms_median",
    )

    expected = {
        ("Joint OLS", "clean"): 0.1066,
        ("Joint Huber", "clean"): 0.1076,
        ("Joint Huber", "nominal"): 1.096,
        ("Joint Huber", "stress"): 5.819,
    }

    checks = {}
    for key, reference in expected.items():
        observed = float(pivot.loc[key[0], key[1]])
        passed = abs(observed - reference) <= 0.02
        checks[f"{key[0]}::{key[1]}"] = {
            "observed": observed,
            "reference": reference,
            "absolute_difference": abs(observed - reference),
            "pass": passed,
        }

    if not all(item["pass"] for item in checks.values()):
        raise RuntimeError(
            "Frozen S1 numerical sanity check failed: "
            + json.dumps(checks, indent=2)
        )

    return checks


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run the frozen MATH-SYNC S1 controlled Monte Carlo benchmark."
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory. Defaults to <project>/outputs/synthetic_s1.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=N_REPLICATES,
        help="Paired replicates per factor level/scenario. Frozen value: 50.",
    )
    args = parser.parse_args()

    if args.replicates != N_REPLICATES:
        raise SystemExit(
            f"Public release requires --replicates {N_REPLICATES}; "
            f"received {args.replicates}."
        )

    project_root = Path(__file__).resolve().parents[2]
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else project_root / "outputs" / "synthetic_s1"
    )

    print("MATH-SYNC S1 controlled benchmark")
    print(f"Seed: {SEED}")
    print(f"Replicates per factor/scenario: {N_REPLICATES}")
    print(f"Output root: {output_root}")

    main(output_root=output_root, n_replicates=N_REPLICATES)
    checks = validate_public_s1(output_root)

    print(json.dumps(checks, indent=2))
    print("S1-CONTROLLED-BENCHMARK: PASS")
