from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STRUCTURAL_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "reproducibility"
    / "public_source_structural_validation.json"
)

STRUCTURAL_MODULE_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "acquisition"
    / "validate_public_sources.py"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "reproducibility"
REPORT_PATH = REPORT_DIR / "public_data_validation.json"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "public_replay"
CACHE_ROOT = OUTPUT_ROOT / "participant_cache"
TABLE_ROOT = OUTPUT_ROOT / "tables"
SUPPLEMENT_ROOT = OUTPUT_ROOT / "supplementary"

HARVARD_EXTRACT_ROOT = PROJECT_ROOT / "data" / "raw" / "harvard_extracted"
WESAD_EXTRACT_ROOT = PROJECT_ROOT / "data" / "raw" / "WESAD_extracted"

RANDOM_SEED = 20260821
COHORT_REPLICATES = 50
MAX_EVAL_SAMPLES_PER_STREAM = 5000
COHORT_SEED_BASE = RANDOM_SEED + 20_000_000
S0201_SESSION1_SEED_BASE = RANDOM_SEED + 10_000_000

BOOTSTRAP_SEED = 20260825
BOOTSTRAP_REPLICATES = 20_000

FROZEN_HARVARD_COHORT = [
    "S0201",
    "S0113",
    "S0171",
    "S0175",
    "S0199",
    "S0133",
    "S0120",
    "S0207",
    "S0101",
    "S0127",
]

EXPECTED_WESAD_PARTICIPANTS = [
    "S2", "S3", "S4", "S5", "S6",
    "S7", "S8", "S9", "S10", "S11",
    "S13", "S14", "S15", "S16", "S17",
]

SCENARIOS = {
    "clean": dict(
        drift_ppm=100.0,
        offset_s=0.25,
        jitter_ms=0.1,
        missing_prob=0.0,
        outlier_frac=0.0,
        outlier_ms=60.0,
    ),
    "nominal": dict(
        drift_ppm=100.0,
        offset_s=0.25,
        jitter_ms=1.0,
        missing_prob=0.10,
        outlier_frac=0.05,
        outlier_ms=60.0,
    ),
    "stress": dict(
        drift_ppm=500.0,
        offset_s=0.50,
        jitter_ms=5.0,
        missing_prob=0.30,
        outlier_frac=0.15,
        outlier_ms=100.0,
    ),
}

WESAD_STREAMS = {
    "CHEST_ECG": ("chest", "ECG", 700.0),
    "WRIST_BVP": ("wrist", "BVP", 64.0),
    "WRIST_ACC": ("wrist", "ACC", 32.0),
    "WRIST_EDA": ("wrist", "EDA", 4.0),
}

METHOD_ORDER = ["Raw", "Offset-only", "Joint OLS", "Joint Huber"]
SCENARIO_ORDER = ["clean", "nominal", "stress"]

# Manuscript-aligned values from the submitted/revised article, compared only at the
# displayed precision. This is a manuscript consistency check, not a hidden tuning target.
MANUSCRIPT_TABLE2_4DP = {
    ("clean", "Joint Huber"): (0.1046, 0.1049, 0.0003, 0.2615),
    ("clean", "Joint OLS"): (0.1041, 0.1039, 0.0002, 0.1838),
    ("clean", "Offset-only"): (18.3961, 20.9974, 2.6012, 14.1400),
    ("clean", "Raw"): (139.2195, 117.5859, 21.6335, 15.5392),
    ("nominal", "Joint Huber"): (1.0576, 1.0529, 0.0047, 0.4402),
    ("nominal", "Joint OLS"): (4.1790, 4.1610, 0.0181, 0.4321),
    ("nominal", "Offset-only"): (18.5849, 18.9052, 0.3202, 1.7231),
    ("nominal", "Raw"): (129.8258, 125.7160, 4.1098, 3.1656),
    ("stress", "Joint Huber"): (5.7685, 5.6458, 0.1227, 2.1267),
    ("stress", "Joint OLS"): (15.5131, 14.9785, 0.5345, 3.4458),
    ("stress", "Offset-only"): (91.8079, 92.6730, 0.8651, 0.9423),
    ("stress", "Raw"): (305.0333, 322.3447, 17.3114, 5.6752),
}

MANUSCRIPT_TABLE3_4DP = {
    ("Harvard", "clean"):  (10, 0.1041, 0.1050, -0.0004, -0.0010, -0.0002, -0.7751, 0.1000, 0.0117),
    ("Harvard", "nominal"):(10, 4.0518, 1.0644,  2.9904,  2.8632,  3.0947, 73.7311, 1.0000, 0.0117),
    ("Harvard", "stress"): (10,15.1897, 5.7126,  9.4999,  8.9975,  9.6993, 62.3917, 1.0000, 0.0117),
    ("WESAD", "clean"):    (15, 0.1061, 0.1065, -0.0006, -0.0007, -0.0003, -0.4164, 0.1333, 0.0004),
    ("WESAD", "nominal"):  (15, 6.1334, 1.0997,  5.0383,  4.8202,  5.1600, 82.0709, 1.0000, 0.0004),
    ("WESAD", "stress"):   (15,18.8376, 6.1235, 12.8167, 11.9952, 13.2268, 67.4930, 1.0000, 0.0004),
}

MANUSCRIPT_TABLE4 = {
    "Harvard": (10, 26.00, 1337.20, 26, 26),
    "WESAD": (15, 16.00, 5547.75, 14, 16),
}


def load_structural_module():
    if not STRUCTURAL_MODULE_PATH.exists():
        raise RuntimeError(
            f"Required structural-validation module not found: {STRUCTURAL_MODULE_PATH}"
        )
    spec = importlib.util.spec_from_file_location("public_structural", STRUCTURAL_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STRUCTURAL = load_structural_module()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_json_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_seed(label: str) -> int:
    token = f"{BOOTSTRAP_SEED}|{label}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def require_structural_gate() -> dict[str, Any]:
    if not STRUCTURAL_REPORT.exists():
        raise RuntimeError(
            f"Required structural gate report not found: {STRUCTURAL_REPORT}"
        )
    report = load_json(STRUCTURAL_REPORT)
    if not report.get("pass") or not report.get("ready_for_public_data_replay"):
        raise RuntimeError("Structural source validation is not PASS.")
    return report


# -------------------------------------------------------------------------
# Frozen generalized estimator
# -------------------------------------------------------------------------

def build_design_matrix_general(
    observations: pd.DataFrame,
    n_events: int,
    sensor_names: list[str],
):
    n_sensors = len(sensor_names)
    n_parameters = n_events + 2 * (n_sensors - 1)

    A = np.zeros((len(observations), n_parameters), dtype=float)
    b = np.zeros(len(observations), dtype=float)

    for row_idx, row in enumerate(observations.itertuples(index=False)):
        event_idx = int(row.event)
        sensor_idx = int(row.sensor_idx)
        t_obs = float(row.t_obs)

        if sensor_idx == 0:
            A[row_idx, event_idx] = 1.0
            b[row_idx] = t_obs
        else:
            A[row_idx, event_idx] = 1.0
            offset = n_events + 2 * (sensor_idx - 1)
            A[row_idx, offset] = -t_obs
            A[row_idx, offset + 1] = -1.0

    return A, b


def solve_joint_ols_general(
    observations: pd.DataFrame,
    n_events: int,
    sensor_names: list[str],
):
    A, b = build_design_matrix_general(observations, n_events, sensor_names)
    z, *_ = np.linalg.lstsq(A, b, rcond=None)
    return z


def solve_joint_huber_general(
    observations: pd.DataFrame,
    n_events: int,
    sensor_names: list[str],
    max_iter: int = 30,
    huber_c: float = 1.345,
    tolerance: float = 1e-10,
):
    A, b = build_design_matrix_general(observations, n_events, sensor_names)
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

        z_new, *_ = np.linalg.lstsq(
            weighted_A,
            weighted_b,
            rcond=None,
        )

        if np.linalg.norm(z_new - z) <= tolerance * (
            1.0 + np.linalg.norm(z)
        ):
            z = z_new
            break

        z = z_new

    return z


def inverse_params_general(
    z: np.ndarray,
    n_events: int,
    sensor_names: list[str],
):
    params = {sensor_names[0]: (1.0, 0.0)}

    for sensor_idx in range(1, len(sensor_names)):
        offset = n_events + 2 * (sensor_idx - 1)
        params[sensor_names[sensor_idx]] = (
            float(z[offset]),
            float(z[offset + 1]),
        )

    return params


def offset_only_general(
    observations: pd.DataFrame,
    sensor_names: list[str],
):
    reference = (
        observations[
            observations["sensor_idx"] == 0
        ][["event", "t_obs"]]
        .set_index("event")["t_obs"]
    )

    params = {sensor_names[0]: (1.0, 0.0)}

    for sensor_idx in range(1, len(sensor_names)):
        subset = observations[
            observations["sensor_idx"] == sensor_idx
        ][["event", "t_obs"]]

        differences = [
            float(reference.loc[row.event] - row.t_obs)
            for row in subset.itertuples(index=False)
            if row.event in reference.index
        ]

        params[sensor_names[sensor_idx]] = (
            1.0,
            float(np.median(differences)) if differences else 0.0,
        )

    return params


# -------------------------------------------------------------------------
# Public timing schedule loaders
# -------------------------------------------------------------------------

def harvard_participant_dir(participant: str) -> Path:
    root = HARVARD_EXTRACT_ROOT / participant
    return STRUCTURAL.find_participant_directory(root, participant)


def load_harvard_schedule(
    participant: str,
    session: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, float, float]:
    participant_dir = harvard_participant_dir(participant)

    timestamp_map: dict[str, np.ndarray] = {}

    for stream in ("EEG", "NIRS", "PHYSIO", "SIM"):
        filename = STRUCTURAL.harvard_core_file(participant, session, stream)
        path = participant_dir / filename
        loader, public = STRUCTURAL.load_mat_public_variables(path)
        matches = STRUCTURAL.find_expected_arrays(public, STRUCTURAL.HARVARD_CORE_ROWS[stream])

        if len(matches) != 1:
            raise RuntimeError(
                f"{participant} session {session} {stream}: expected one frozen "
                f"core array, found {len(matches)}."
            )

        _, array = matches[0]
        diag = STRUCTURAL.diagnose_timestamp_vector(
            array,
            STRUCTURAL.HARVARD_CORE_RATES[stream],
        )
        timestamp_map[stream] = np.asarray(
            diag["timestamp_vector"],
            dtype=float,
        )

    overlap_start = max(float(ts[0]) for ts in timestamp_map.values())
    overlap_end = min(float(ts[-1]) for ts in timestamp_map.values())

    anchors = STRUCTURAL.build_harvard_anchor_times(
        participant_dir,
        participant,
        session,
        overlap_start,
        overlap_end,
    )

    return timestamp_map, anchors, overlap_start, overlap_end


def find_wesad_pickle(participant: str) -> Path:
    matches = list(
        WESAD_EXTRACT_ROOT.rglob(f"{participant}.pkl")
    )
    qualified = [
        path for path in matches
        if path.parent.name == participant
    ]
    if len(qualified) != 1:
        raise RuntimeError(
            f"Expected one extracted WESAD pickle for {participant}; "
            f"found {len(qualified)}."
        )
    return qualified[0]


def load_wesad_schedule(
    participant: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, float, float]:
    path = find_wesad_pickle(participant)

    with path.open("rb") as f:
        subject_data = pickle.load(f, encoding="latin1")

    signal = subject_data["signal"]
    labels = np.asarray(subject_data["label"]).reshape(-1)

    timestamp_map = {}
    for stream, (location, key, rate) in WESAD_STREAMS.items():
        array = np.asarray(signal[location][key])
        n_samples = int(array.shape[0])
        timestamp_map[stream] = (
            np.arange(n_samples, dtype=float) / float(rate)
        )

    if labels.size != len(timestamp_map["CHEST_ECG"]):
        raise RuntimeError(
            f"{participant}: WESAD label length does not match chest ECG."
        )

    overlap_start = 0.0
    overlap_end = min(float(ts[-1]) for ts in timestamp_map.values())

    transition_indices = np.where(labels[1:] != labels[:-1])[0] + 1
    anchor_times = transition_indices.astype(float) / 700.0
    anchor_times = np.unique(
        anchor_times[
            (anchor_times >= overlap_start)
            & (anchor_times <= overlap_end)
        ]
    )

    del subject_data
    del signal
    del labels

    return timestamp_map, anchor_times, overlap_start, overlap_end


# -------------------------------------------------------------------------
# Frozen replay
# -------------------------------------------------------------------------

def deterministic_eval_timestamps(
    timestamps: np.ndarray,
    overlap_start_local: float,
    overlap_end_local: float,
    max_samples: int | None = MAX_EVAL_SAMPLES_PER_STREAM,
):
    timestamps = np.asarray(timestamps, dtype=float)
    valid = timestamps[
        (timestamps >= overlap_start_local)
        & (timestamps <= overlap_end_local)
    ]

    if len(valid) < 3:
        raise RuntimeError("Too few timestamps in evaluation overlap.")

    if max_samples is None or len(valid) <= max_samples:
        return valid.copy()

    targets = np.linspace(
        overlap_start_local,
        overlap_end_local,
        max_samples,
    )

    indices = np.searchsorted(valid, targets, side="left")
    indices = np.clip(indices, 1, len(valid) - 1)

    left = valid[indices - 1]
    right = valid[indices]
    choose_left = (
        np.abs(targets - left)
        <= np.abs(right - targets)
    )
    indices[choose_left] -= 1

    indices = np.unique(indices)
    return valid[indices]


def replay_schedule_trial(
    timestamp_map: dict[str, np.ndarray],
    anchor_times: np.ndarray,
    overlap_start_local: float,
    overlap_end_local: float,
    seed: int,
    config: dict[str, float],
    max_eval_samples: int | None = MAX_EVAL_SAMPLES_PER_STREAM,
):
    rng = np.random.default_rng(seed)

    stream_names = list(timestamp_map.keys())
    sensor_names = ["REF_EVENT"] + stream_names
    anchor_times = np.asarray(anchor_times, dtype=float)

    rows = []
    for event_idx, tau_e in enumerate(anchor_times):
        rows.append(
            {
                "sensor_idx": 0,
                "sensor": "REF_EVENT",
                "event": int(event_idx),
                "t_obs": float(tau_e),
            }
        )

    eval_pairs = {}

    for sensor_idx, stream in enumerate(stream_names, start=1):
        eval_tau = deterministic_eval_timestamps(
            timestamp_map[stream],
            overlap_start_local,
            overlap_end_local,
            max_samples=max_eval_samples,
        )

        a_i = 1.0 + rng.uniform(
            -config["drift_ppm"],
            config["drift_ppm"],
        ) * 1e-6

        b_i = rng.uniform(
            -config["offset_s"],
            config["offset_s"],
        )

        eval_local = (
            a_i * eval_tau
            + b_i
            + rng.normal(
                0.0,
                config["jitter_ms"] / 1000.0,
                len(eval_tau),
            )
        )

        anchor_local = (
            a_i * anchor_times
            + b_i
            + rng.normal(
                0.0,
                config["jitter_ms"] / 1000.0,
                len(anchor_times),
            )
        )

        keep = (
            rng.random(len(anchor_times))
            >= config["missing_prob"]
        )

        if keep.sum() < 5:
            forced = rng.choice(
                len(anchor_times),
                5,
                replace=False,
            )
            keep[forced] = True

        candidates = np.where(keep)[0]
        n_out = int(round(config["outlier_frac"] * len(candidates)))

        if n_out > 0:
            selected_outliers = rng.choice(
                candidates,
                n_out,
                replace=False,
            )
            signs = rng.choice(
                [-1.0, 1.0],
                size=n_out,
            )
            magnitudes = rng.uniform(
                0.5 * config["outlier_ms"],
                1.5 * config["outlier_ms"],
                size=n_out,
            ) / 1000.0

            anchor_local[selected_outliers] += signs * magnitudes

        for event_idx in np.where(keep)[0]:
            rows.append(
                {
                    "sensor_idx": sensor_idx,
                    "sensor": stream,
                    "event": int(event_idx),
                    "t_obs": float(anchor_local[event_idx]),
                }
            )

        eval_pairs[stream] = (eval_tau, eval_local)

    observations = pd.DataFrame(rows)

    methods = {
        "Raw": {
            name: (1.0, 0.0)
            for name in sensor_names
        },
        "Offset-only": offset_only_general(
            observations,
            sensor_names,
        ),
    }

    z_ols = solve_joint_ols_general(
        observations,
        len(anchor_times),
        sensor_names,
    )
    methods["Joint OLS"] = inverse_params_general(
        z_ols,
        len(anchor_times),
        sensor_names,
    )

    z_huber = solve_joint_huber_general(
        observations,
        len(anchor_times),
        sensor_names,
    )
    methods["Joint Huber"] = inverse_params_general(
        z_huber,
        len(anchor_times),
        sensor_names,
    )

    stream_rows = []

    for method_name, params in methods.items():
        for stream in stream_names:
            tau_true, t_local = eval_pairs[stream]
            gamma_hat, kappa_hat = params[stream]

            tau_hat = gamma_hat * t_local + kappa_hat
            error_ms = (tau_hat - tau_true) * 1000.0

            stream_rows.append(
                {
                    "method": method_name,
                    "stream": stream,
                    "n_eval_samples": int(len(error_ms)),
                    "rmse_ms": float(
                        np.sqrt(np.mean(error_ms ** 2))
                    ),
                    "mae_ms": float(
                        np.mean(np.abs(error_ms))
                    ),
                    "p95_abs_error_ms": float(
                        np.quantile(np.abs(error_ms), 0.95)
                    ),
                }
            )

    return pd.DataFrame(stream_rows)


def run_schedule_replay_battery(
    dataset_name: str,
    participant: str,
    session: str | int,
    timestamp_map: dict[str, np.ndarray],
    anchor_times: np.ndarray,
    overlap_start_local: float,
    overlap_end_local: float,
    seed_base: int,
    replicates: int = COHORT_REPLICATES,
    max_eval_samples: int | None = MAX_EVAL_SAMPLES_PER_STREAM,
):
    frames = []
    running_seed = int(seed_base)

    for scenario_name in SCENARIO_ORDER:
        config = SCENARIOS[scenario_name]

        for replicate in range(replicates):
            trial = replay_schedule_trial(
                timestamp_map=timestamp_map,
                anchor_times=anchor_times,
                overlap_start_local=overlap_start_local,
                overlap_end_local=overlap_end_local,
                seed=running_seed,
                config=config,
                max_eval_samples=max_eval_samples,
            )

            trial["dataset"] = dataset_name
            trial["participant"] = participant
            trial["session"] = session
            trial["scenario"] = scenario_name
            trial["replicate"] = replicate
            trial["seed"] = running_seed

            frames.append(trial)
            running_seed += 1

    stream_results = pd.concat(frames, ignore_index=True)

    trial_results = (
        stream_results
        .groupby(
            [
                "dataset",
                "participant",
                "session",
                "scenario",
                "replicate",
                "seed",
                "method",
            ],
            as_index=False,
        )
        .agg(
            macro_rmse_ms=("rmse_ms", "mean"),
            macro_mae_ms=("mae_ms", "mean"),
            macro_p95_abs_error_ms=("p95_abs_error_ms", "mean"),
            worst_stream_rmse_ms=("rmse_ms", "max"),
        )
    )

    return stream_results, trial_results


def battery_spec(
    dataset_name: str,
    participant: str,
    session: str | int,
    timestamp_map: dict[str, np.ndarray],
    anchor_times: np.ndarray,
    overlap_start_local: float,
    overlap_end_local: float,
    seed_base: int,
    max_eval_samples: int | None,
):
    return {
        "dataset": dataset_name,
        "participant": participant,
        "session": str(session),
        "seed_base": int(seed_base),
        "replicates": COHORT_REPLICATES,
        "max_eval_samples": max_eval_samples,
        "scenarios": SCENARIOS,
        "overlap_start_local": float(overlap_start_local),
        "overlap_end_local": float(overlap_end_local),
        "anchors": [float(v) for v in np.asarray(anchor_times, dtype=float)],
        "streams": {
            name: {
                "n": int(len(values)),
                "first": float(values[0]),
                "last": float(values[-1]),
            }
            for name, values in timestamp_map.items()
        },
    }


def run_or_load_battery(
    dataset_name: str,
    participant: str,
    session: str | int,
    timestamp_map: dict[str, np.ndarray],
    anchor_times: np.ndarray,
    overlap_start_local: float,
    overlap_end_local: float,
    seed_base: int,
    max_eval_samples: int | None = MAX_EVAL_SAMPLES_PER_STREAM,
):
    cache_dir = (
        CACHE_ROOT
        / dataset_name.lower()
        / participant
        / f"session_{session}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    stream_path = cache_dir / "stream_results.csv"
    trial_path = cache_dir / "trial_results.csv"
    marker_path = cache_dir / "cache_manifest.json"

    spec = battery_spec(
        dataset_name,
        participant,
        session,
        timestamp_map,
        anchor_times,
        overlap_start_local,
        overlap_end_local,
        seed_base,
        max_eval_samples,
    )
    spec_hash = stable_json_hash(spec)

    if stream_path.exists() and trial_path.exists() and marker_path.exists():
        try:
            marker = load_json(marker_path)
            if marker.get("spec_sha256") == spec_hash:
                stream_df = pd.read_csv(stream_path)
                trial_df = pd.read_csv(trial_path)
                expected_trial_rows = (
                    len(SCENARIO_ORDER)
                    * COHORT_REPLICATES
                    * len(METHOD_ORDER)
                )
                if len(trial_df) == expected_trial_rows:
                    print(
                        f"  CACHE PASS {dataset_name} {participant} "
                        f"session {session}"
                    )
                    return stream_df, trial_df, True
        except Exception:
            pass

    t0 = time.perf_counter()
    stream_df, trial_df = run_schedule_replay_battery(
        dataset_name=dataset_name,
        participant=participant,
        session=session,
        timestamp_map=timestamp_map,
        anchor_times=anchor_times,
        overlap_start_local=overlap_start_local,
        overlap_end_local=overlap_end_local,
        seed_base=seed_base,
        max_eval_samples=max_eval_samples,
    )
    elapsed = time.perf_counter() - t0

    stream_df.to_csv(stream_path, index=False)
    trial_df.to_csv(trial_path, index=False)
    marker_path.write_text(
        json.dumps(
            {
                "spec_sha256": spec_hash,
                "spec": spec,
                "elapsed_seconds": elapsed,
                "stream_rows": len(stream_df),
                "trial_rows": len(trial_df),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"  RUN PASS {dataset_name} {participant} session {session} "
        f"| {elapsed:.1f} s"
    )
    return stream_df, trial_df, False


# -------------------------------------------------------------------------
# Participant-level inference and manuscript tables
# -------------------------------------------------------------------------

def paired_rank_biserial(x, y):
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]

    if len(d) == 0:
        return 0.0

    ranks = rankdata(np.abs(d))
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def holm_adjust(p_values):
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


def subject_level_from_trials(trial_results: pd.DataFrame) -> pd.DataFrame:
    return (
        trial_results
        .groupby(
            ["participant", "scenario", "method"],
            as_index=False,
        )
        .agg(
            participant_median_macro_rmse_ms=(
                "macro_rmse_ms",
                "median",
            ),
            participant_median_macro_mae_ms=(
                "macro_mae_ms",
                "median",
            ),
        )
    )


def subject_level_paired_tests(
    subject_level_df: pd.DataFrame,
    dataset_label: str,
) -> pd.DataFrame:
    rows = []

    for scenario_name in SCENARIO_ORDER:
        scenario_data = subject_level_df[
            subject_level_df["scenario"] == scenario_name
        ]

        pivot = scenario_data.pivot(
            index="participant",
            columns="method",
            values="participant_median_macro_rmse_ms",
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

            rows.append(
                {
                    "dataset": dataset_label,
                    "scenario": scenario_name,
                    "baseline": baseline,
                    "n_participants": len(paired),
                    "median_baseline_rmse_ms": float(
                        np.median(baseline_values)
                    ),
                    "median_huber_rmse_ms": float(
                        np.median(huber_values)
                    ),
                    "median_improvement_ms": float(
                        np.median(improvement)
                    ),
                    "huber_subject_win_rate": float(
                        np.mean(huber_values < baseline_values)
                    ),
                    "rank_biserial_baseline_minus_huber":
                        paired_rank_biserial(
                            baseline_values,
                            huber_values,
                        ),
                    "wilcoxon_p_raw": float(p_value),
                }
            )

    result = pd.DataFrame(rows)
    result["wilcoxon_p_holm"] = holm_adjust(
        result["wilcoxon_p_raw"].to_numpy()
    )
    return result


def participant_comparison_summary(
    subject_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    rows = []

    for scenario in SCENARIO_ORDER:
        pivot = (
            subject_df[
                subject_df["scenario"] == scenario
            ]
            .pivot(
                index="participant",
                columns="method",
                values="participant_median_macro_rmse_ms",
            )
        )

        for baseline in ["Joint OLS", "Offset-only"]:
            paired = pivot[[baseline, "Joint Huber"]].dropna()

            baseline_values = paired[baseline].to_numpy(dtype=float)
            huber_values = paired["Joint Huber"].to_numpy(dtype=float)
            improvement = baseline_values - huber_values

            rng = np.random.default_rng(
                stable_seed(
                    f"{dataset_name}|{scenario}|{baseline}"
                )
            )

            bootstrap_values = np.empty(
                BOOTSTRAP_REPLICATES,
                dtype=float,
            )

            for index in range(BOOTSTRAP_REPLICATES):
                bootstrap_values[index] = np.median(
                    rng.choice(
                        improvement,
                        size=len(improvement),
                        replace=True,
                    )
                )

            ci_low, ci_high = np.quantile(
                bootstrap_values,
                [0.025, 0.975],
            )

            test_row = test_df[
                (test_df["dataset"].str.upper() == dataset_name.upper())
                & (test_df["scenario"] == scenario)
                & (test_df["baseline"] == baseline)
            ]

            if len(test_row) != 1:
                raise RuntimeError(
                    f"Expected one test row for "
                    f"{dataset_name}/{scenario}/{baseline}."
                )

            test_row = test_row.iloc[0]

            baseline_median = float(np.median(baseline_values))
            huber_median = float(np.median(huber_values))

            rows.append(
                {
                    "dataset": dataset_name,
                    "scenario": scenario,
                    "baseline": baseline,
                    "n_participants": int(len(paired)),
                    "baseline_median_rmse_ms": baseline_median,
                    "huber_median_rmse_ms": huber_median,
                    "median_paired_improvement_ms": float(
                        np.median(improvement)
                    ),
                    "bootstrap_ci95_low_ms": float(ci_low),
                    "bootstrap_ci95_high_ms": float(ci_high),
                    "relative_rmse_reduction_pct": float(
                        100.0
                        * (baseline_median - huber_median)
                        / baseline_median
                    ),
                    "huber_subject_win_rate": float(
                        test_row["huber_subject_win_rate"]
                    ),
                    "rank_biserial_effect": float(
                        test_row[
                            "rank_biserial_baseline_minus_huber"
                        ]
                    ),
                    "wilcoxon_p_raw": float(
                        test_row["wilcoxon_p_raw"]
                    ),
                    "wilcoxon_p_holm": float(
                        test_row["wilcoxon_p_holm"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_table2(
    s1_trial_results: pd.DataFrame,
    s2_trial_results: pd.DataFrame,
) -> pd.DataFrame:
    s1 = (
        s1_trial_results
        .groupby(["scenario", "method"], as_index=False)["macro_rmse_ms"]
        .median()
        .rename(
            columns={
                "macro_rmse_ms":
                    "session1_median_macro_rmse_ms"
            }
        )
    )

    s2 = (
        s2_trial_results
        .groupby(["scenario", "method"], as_index=False)["macro_rmse_ms"]
        .median()
        .rename(
            columns={
                "macro_rmse_ms":
                    "session2_median_macro_rmse_ms"
            }
        )
    )

    table = s1.merge(s2, on=["scenario", "method"], how="inner")
    table["absolute_session_difference_ms"] = np.abs(
        table["session2_median_macro_rmse_ms"]
        - table["session1_median_macro_rmse_ms"]
    )
    table["relative_session_difference_pct"] = (
        100.0
        * table["absolute_session_difference_ms"]
        / table["session1_median_macro_rmse_ms"]
    )

    scenario_rank = {name: i for i, name in enumerate(SCENARIO_ORDER)}
    method_rank = {
        "Joint Huber": 0,
        "Joint OLS": 1,
        "Offset-only": 2,
        "Raw": 3,
    }

    table["_scenario_rank"] = table["scenario"].map(scenario_rank)
    table["_method_rank"] = table["method"].map(method_rank)

    return (
        table
        .sort_values(["_scenario_rank", "_method_rank"])
        .drop(columns=["_scenario_rank", "_method_rank"])
        .reset_index(drop=True)
    )


def build_table3(
    harvard_subject: pd.DataFrame,
    wesad_subject: pd.DataFrame,
    subject_tests: pd.DataFrame,
) -> pd.DataFrame:
    full = pd.concat(
        [
            participant_comparison_summary(
                harvard_subject,
                subject_tests,
                "Harvard",
            ),
            participant_comparison_summary(
                wesad_subject,
                subject_tests,
                "WESAD",
            ),
        ],
        ignore_index=True,
    )

    return (
        full[full["baseline"] == "Joint OLS"]
        .reset_index(drop=True)
    )


def build_table4(structural_report: dict[str, Any]) -> pd.DataFrame:
    h_counts = structural_report["harvard"]["session1_anchor_counts"]
    h_median_overlap = structural_report["harvard"]["session1_median_overlap_s"]

    w_participants = structural_report["wesad"]["participants"]
    w_counts = [
        int(w_participants[p]["anchor_count"])
        for p in EXPECTED_WESAD_PARTICIPANTS
    ]
    w_overlaps = [
        float(w_participants[p]["overlap_duration_s"])
        for p in EXPECTED_WESAD_PARTICIPANTS
    ]

    return pd.DataFrame(
        [
            {
                "Dataset": "Harvard",
                "N": 10,
                "Median anchors": float(np.median(h_counts)),
                "Median overlap (s)": float(h_median_overlap),
                "Min anchors": int(min(h_counts)),
                "Max anchors": int(max(h_counts)),
            },
            {
                "Dataset": "WESAD",
                "N": 15,
                "Median anchors": float(np.median(w_counts)),
                "Median overlap (s)": float(np.median(w_overlaps)),
                "Min anchors": int(min(w_counts)),
                "Max anchors": int(max(w_counts)),
            },
        ]
    )


# -------------------------------------------------------------------------
# Manuscript-alignment checks
# -------------------------------------------------------------------------

def rounded_equal(observed: float, expected: float, digits: int) -> bool:
    return round(float(observed), digits) == round(float(expected), digits)


def check_table2(table2: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []

    for record in table2.itertuples(index=False):
        key = (record.scenario, record.method)
        expected = MANUSCRIPT_TABLE2_4DP[key]
        observed = (
            record.session1_median_macro_rmse_ms,
            record.session2_median_macro_rmse_ms,
            record.absolute_session_difference_ms,
            record.relative_session_difference_pct,
        )
        matches = [
            rounded_equal(o, e, 4)
            for o, e in zip(observed, expected)
        ]
        rows.append(
            {
                "scenario": key[0],
                "method": key[1],
                "observed_4dp": [round(float(v), 4) for v in observed],
                "expected_4dp": list(expected),
                "match": bool(all(matches)),
            }
        )

    return rows


def check_table3(table3: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []

    for record in table3.itertuples(index=False):
        key = (record.dataset, record.scenario)
        expected = MANUSCRIPT_TABLE3_4DP[key]
        observed = (
            int(record.n_participants),
            record.baseline_median_rmse_ms,
            record.huber_median_rmse_ms,
            record.median_paired_improvement_ms,
            record.bootstrap_ci95_low_ms,
            record.bootstrap_ci95_high_ms,
            record.relative_rmse_reduction_pct,
            record.huber_subject_win_rate,
            record.wilcoxon_p_holm,
        )

        matches = [int(observed[0]) == int(expected[0])]
        matches.extend(
            rounded_equal(o, e, 4)
            for o, e in zip(observed[1:], expected[1:])
        )

        rows.append(
            {
                "dataset": key[0],
                "scenario": key[1],
                "observed_4dp": [
                    int(observed[0]),
                    *[round(float(v), 4) for v in observed[1:]],
                ],
                "expected_4dp": list(expected),
                "match": bool(all(matches)),
            }
        )

    return rows


def check_table4(table4: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []

    for record in table4.itertuples(index=False):
        dataset = record.Dataset
        expected = MANUSCRIPT_TABLE4[dataset]
        observed = (
            int(record.N),
            float(getattr(record, "_2")),  # Median anchors
            float(getattr(record, "_3")),  # Median overlap (s)
            int(getattr(record, "_4")),    # Min anchors
            int(getattr(record, "_5")),    # Max anchors
        )
        match = (
            observed[0] == expected[0]
            and rounded_equal(observed[1], expected[1], 2)
            and rounded_equal(observed[2], expected[2], 2)
            and observed[3] == expected[3]
            and observed[4] == expected[4]
        )
        rows.append(
            {
                "dataset": dataset,
                "observed": observed,
                "expected": expected,
                "match": bool(match),
            }
        )

    return rows


def check_claim_directions(table3: pd.DataFrame) -> dict[str, Any]:
    nominal_stress = table3[
        table3["scenario"].isin(["nominal", "stress"])
    ]
    clean = table3[table3["scenario"] == "clean"]

    return {
        "nominal_stress_huber_better_all_dataset_scenarios": bool(
            (
                nominal_stress["huber_median_rmse_ms"]
                < nominal_stress["baseline_median_rmse_ms"]
            ).all()
        ),
        "nominal_stress_participant_win_rate_100pct": bool(
            np.allclose(
                nominal_stress["huber_subject_win_rate"].to_numpy(),
                1.0,
                rtol=0,
                atol=0,
            )
        ),
        "clean_absolute_median_difference_below_0_001_ms": bool(
            (
                np.abs(
                    clean["baseline_median_rmse_ms"]
                    - clean["huber_median_rmse_ms"]
                )
                < 0.001
            ).all()
        ),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    SUPPLEMENT_ROOT.mkdir(parents=True, exist_ok=True)

    structural_report = require_structural_gate()

    print()
    print("MATH-SYNC public-data validation")
    print("Frozen scenarios: clean, nominal, stress")
    print("Replicates per scenario: 50")
    print("Primary cohort outcome: participant-level median macro-RMSE")
    print()

    # -------------------------------------------------------------
    # S0201 Session 1 full-overlap replay used for Table 2.
    # max_eval_samples=None preserves the full native schedule.
    # -------------------------------------------------------------
    print("[1/4] Harvard S0201 within-participant replication")

    s1_map, s1_anchors, s1_start, s1_end = load_harvard_schedule("S0201", 1)
    _, s0201_s1_trials, _ = run_or_load_battery(
        dataset_name="HARVARD_S0201_FULL",
        participant="S0201",
        session=1,
        timestamp_map=s1_map,
        anchor_times=s1_anchors,
        overlap_start_local=s1_start,
        overlap_end_local=s1_end,
        seed_base=S0201_SESSION1_SEED_BASE,
        max_eval_samples=None,
    )

    s2_map, s2_anchors, s2_start, s2_end = load_harvard_schedule("S0201", 2)
    _, s0201_s2_trials, _ = run_or_load_battery(
        dataset_name="HARVARD_S0201_SESSION2",
        participant="S0201",
        session=2,
        timestamp_map=s2_map,
        anchor_times=s2_anchors,
        overlap_start_local=s2_start,
        overlap_end_local=s2_end,
        seed_base=COHORT_SEED_BASE,
        max_eval_samples=MAX_EVAL_SAMPLES_PER_STREAM,
    )

    table2 = build_table2(s0201_s1_trials, s0201_s2_trials)
    table2.to_csv(
        TABLE_ROOT / "Table_2_S0201_within_participant_replication.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Harvard N=10 cohort.
    # -------------------------------------------------------------
    print()
    print("[2/4] Harvard 10-participant public-data replay")

    harvard_trial_frames = []

    for participant_index, participant in enumerate(FROZEN_HARVARD_COHORT):
        print(
            f"Harvard {participant_index + 1}/"
            f"{len(FROZEN_HARVARD_COHORT)}: {participant}"
        )

        timestamp_map, anchors, overlap_start, overlap_end = (
            load_harvard_schedule(participant, 1)
        )

        _, trials, _ = run_or_load_battery(
            dataset_name="HARVARD",
            participant=participant,
            session=1,
            timestamp_map=timestamp_map,
            anchor_times=anchors,
            overlap_start_local=overlap_start,
            overlap_end_local=overlap_end,
            seed_base=(
                COHORT_SEED_BASE
                + 1_000_000
                + participant_index * 10_000
            ),
            max_eval_samples=MAX_EVAL_SAMPLES_PER_STREAM,
        )
        harvard_trial_frames.append(trials)

    harvard_trials = pd.concat(
        harvard_trial_frames,
        ignore_index=True,
    )
    harvard_subject = subject_level_from_trials(harvard_trials)
    harvard_tests = subject_level_paired_tests(
        harvard_subject,
        "HARVARD",
    )

    harvard_trials.to_csv(
        SUPPLEMENT_ROOT / "harvard_cohort_trial_results.csv",
        index=False,
    )
    harvard_subject.to_csv(
        SUPPLEMENT_ROOT / "harvard_cohort_subject_level.csv",
        index=False,
    )
    harvard_tests.to_csv(
        SUPPLEMENT_ROOT / "harvard_cohort_subject_level_tests.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # WESAD N=15 cohort.
    # -------------------------------------------------------------
    print()
    print("[3/4] WESAD 15-participant external replay")

    wesad_trial_frames = []

    for subject_index, participant in enumerate(EXPECTED_WESAD_PARTICIPANTS):
        print(
            f"WESAD {subject_index + 1}/"
            f"{len(EXPECTED_WESAD_PARTICIPANTS)}: {participant}"
        )

        timestamp_map, anchors, overlap_start, overlap_end = (
            load_wesad_schedule(participant)
        )

        _, trials, _ = run_or_load_battery(
            dataset_name="WESAD",
            participant=participant,
            session="SESSION",
            timestamp_map=timestamp_map,
            anchor_times=anchors,
            overlap_start_local=overlap_start,
            overlap_end_local=overlap_end,
            seed_base=(
                COHORT_SEED_BASE
                + 5_000_000
                + subject_index * 10_000
            ),
            max_eval_samples=MAX_EVAL_SAMPLES_PER_STREAM,
        )
        wesad_trial_frames.append(trials)

    wesad_trials = pd.concat(
        wesad_trial_frames,
        ignore_index=True,
    )
    wesad_subject = subject_level_from_trials(wesad_trials)
    wesad_tests = subject_level_paired_tests(
        wesad_subject,
        "WESAD",
    )

    wesad_trials.to_csv(
        SUPPLEMENT_ROOT / "wesad_trial_results.csv",
        index=False,
    )
    wesad_subject.to_csv(
        SUPPLEMENT_ROOT / "wesad_subject_level.csv",
        index=False,
    )
    wesad_tests.to_csv(
        SUPPLEMENT_ROOT / "wesad_subject_level_tests.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Integrated manuscript-aligned summaries.
    # -------------------------------------------------------------
    print()
    print("[4/4] Participant-level inference + manuscript consistency")

    subject_tests = pd.concat(
        [harvard_tests, wesad_tests],
        ignore_index=True,
    )
    subject_tests.to_csv(
        SUPPLEMENT_ROOT / "subject_level_replication_tests.csv",
        index=False,
    )

    table3 = build_table3(
        harvard_subject,
        wesad_subject,
        subject_tests,
    )
    table3.to_csv(
        TABLE_ROOT / "Table_3_Huber_vs_OLS_replication.csv",
        index=False,
    )

    table4 = build_table4(structural_report)
    table4.to_csv(
        TABLE_ROOT / "Table_4_public_dataset_QC_summary.csv",
        index=False,
    )

    table2_check = check_table2(table2)
    table3_check = check_table3(table3)

    # Avoid namedtuple issues for columns with spaces by using row-wise dicts.
    table4_check = []
    for _, row in table4.iterrows():
        dataset = row["Dataset"]
        observed = (
            int(row["N"]),
            float(row["Median anchors"]),
            float(row["Median overlap (s)"]),
            int(row["Min anchors"]),
            int(row["Max anchors"]),
        )
        expected = MANUSCRIPT_TABLE4[dataset]
        match = (
            observed[0] == expected[0]
            and rounded_equal(observed[1], expected[1], 2)
            and rounded_equal(observed[2], expected[2], 2)
            and observed[3] == expected[3]
            and observed[4] == expected[4]
        )
        table4_check.append(
            {
                "dataset": dataset,
                "observed": observed,
                "expected": expected,
                "match": bool(match),
            }
        )

    claim_check = check_claim_directions(table3)

    manuscript_match = (
        all(row["match"] for row in table2_check)
        and all(row["match"] for row in table3_check)
        and all(row["match"] for row in table4_check)
    )
    claims_pass = all(claim_check.values())

    consistency = {
        "table2_display_precision_match": table2_check,
        "table3_display_precision_match": table3_check,
        "table4_display_precision_match": table4_check,
        "claim_direction_checks": claim_check,
        "all_manuscript_displayed_values_match": bool(manuscript_match),
        "all_primary_claim_directions_pass": bool(claims_pass),
    }

    (OUTPUT_ROOT / "public_replay_claim_check.json").write_text(
        json.dumps(consistency, indent=2),
        encoding="utf-8",
    )

    report = {
        "stage": "public_data_validation",
        "random_seed": RANDOM_SEED,
        "cohort_replicates_per_scenario": COHORT_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "harvard_participants": FROZEN_HARVARD_COHORT,
        "wesad_participants": EXPECTED_WESAD_PARTICIPANTS,
        "scenarios": SCENARIOS,
        "table2_path": str(
            TABLE_ROOT / "Table_2_S0201_within_participant_replication.csv"
        ),
        "table3_path": str(
            TABLE_ROOT / "Table_3_Huber_vs_OLS_replication.csv"
        ),
        "table4_path": str(
            TABLE_ROOT / "Table_4_public_dataset_QC_summary.csv"
        ),
        "consistency_check_path": str(
            OUTPUT_ROOT / "public_replay_claim_check.json"
        ),
        "manuscript_displayed_values_match": bool(manuscript_match),
        "primary_claim_directions_pass": bool(claims_pass),
        "pass": bool(manuscript_match and claims_pass),
        "ready_for_release_freeze": bool(manuscript_match and claims_pass),
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print("Reproduced Table 2:")
    print(
        table2[
            [
                "scenario",
                "method",
                "session1_median_macro_rmse_ms",
                "session2_median_macro_rmse_ms",
                "absolute_session_difference_ms",
                "relative_session_difference_pct",
            ]
        ].to_string(index=False)
    )

    print()
    print("Reproduced Table 3 (Joint Huber vs Joint OLS):")
    print(
        table3[
            [
                "dataset",
                "scenario",
                "n_participants",
                "baseline_median_rmse_ms",
                "huber_median_rmse_ms",
                "median_paired_improvement_ms",
                "bootstrap_ci95_low_ms",
                "bootstrap_ci95_high_ms",
                "relative_rmse_reduction_pct",
                "huber_subject_win_rate",
                "wilcoxon_p_holm",
            ]
        ].to_string(index=False)
    )

    print()
    print("Reproduced Table 4:")
    print(table4.to_string(index=False))

    print()
    print(
        "Table 2 displayed-value agreement: "
        f"{sum(row['match'] for row in table2_check)}/"
        f"{len(table2_check)}"
    )
    print(
        "Table 3 displayed-value agreement: "
        f"{sum(row['match'] for row in table3_check)}/"
        f"{len(table3_check)}"
    )
    print(
        "Table 4 displayed-value agreement: "
        f"{sum(row['match'] for row in table4_check)}/"
        f"{len(table4_check)}"
    )
    print(
        "Primary claim directions: "
        f"{'PASS' if claims_pass else 'FAIL'}"
    )

    if manuscript_match and claims_pass:
        print()
        print("PUBLIC-DATA-VALIDATION: PASS")
        print("Ready for release freeze: YES")
        print(f"Gate report: {REPORT_PATH}")
        return 0

    print()
    print("PUBLIC-DATA-VALIDATION: FAIL")
    print("Ready for release freeze: NO")
    print(f"Gate report: {REPORT_PATH}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
