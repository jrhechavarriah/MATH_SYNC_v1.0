from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PUBLIC_REPLAY_ROOT = PROJECT_ROOT / "outputs" / "public_replay"
PUBLIC_TABLES = PUBLIC_REPLAY_ROOT / "tables"
PUBLIC_SUPPLEMENTARY = PUBLIC_REPLAY_ROOT / "supplementary"

S1_ROOT = PROJECT_ROOT / "outputs" / "synthetic_s1"

MANUSCRIPT_ROOT = PROJECT_ROOT / "outputs" / "manuscript"
MANUSCRIPT_TABLES = MANUSCRIPT_ROOT / "tables"
MANUSCRIPT_FIGURES = MANUSCRIPT_ROOT / "figures"
MANUSCRIPT_SUPPLEMENTARY = MANUSCRIPT_ROOT / "supplementary"

FIGURES_ZIP = MANUSCRIPT_ROOT / "Figures.zip"
SUPPLEMENTARY_ZIP = MANUSCRIPT_ROOT / "Supplementary.zip"

BOOTSTRAP_SEED = 20260825
BOOTSTRAP_REPLICATES = 20_000

EXPECTED_HARVARD_PARTICIPANTS = [
    "S0201", "S0113", "S0171", "S0175", "S0199",
    "S0133", "S0120", "S0207", "S0101", "S0127",
]

EXPECTED_WESAD_PARTICIPANTS = [
    "S2", "S3", "S4", "S5", "S6",
    "S7", "S8", "S9", "S10", "S11",
    "S13", "S14", "S15", "S16", "S17",
]

EXPECTED_SCENARIOS = ["clean", "nominal", "stress"]
EXPECTED_METHODS = ["Raw", "Offset-only", "Joint OLS", "Joint Huber"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(label: str) -> int:
    token = f"{BOOTSTRAP_SEED}|{label}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_inputs():
    table2 = pd.read_csv(
        require_file(
            PUBLIC_TABLES
            / "Table_2_S0201_within_participant_replication.csv"
        )
    )

    table3 = pd.read_csv(
        require_file(
            PUBLIC_TABLES
            / "Table_3_Huber_vs_OLS_replication.csv"
        )
    )

    table4 = pd.read_csv(
        require_file(
            PUBLIC_TABLES
            / "Table_4_public_dataset_QC_summary.csv"
        )
    )

    harvard_subject = pd.read_csv(
        require_file(
            PUBLIC_SUPPLEMENTARY
            / "harvard_cohort_subject_level.csv"
        )
    )
    wesad_subject = pd.read_csv(
        require_file(
            PUBLIC_SUPPLEMENTARY
            / "wesad_subject_level.csv"
        )
    )

    harvard_tests = pd.read_csv(
        require_file(
            PUBLIC_SUPPLEMENTARY
            / "harvard_cohort_subject_level_tests.csv"
        )
    )
    wesad_tests = pd.read_csv(
        require_file(
            PUBLIC_SUPPLEMENTARY
            / "wesad_subject_level_tests.csv"
        )
    )

    subject_tests = pd.concat(
        [harvard_tests, wesad_tests],
        ignore_index=True,
    )

    return {
        "table2": table2,
        "table3": table3,
        "table4": table4,
        "harvard_subject": harvard_subject,
        "wesad_subject": wesad_subject,
        "subject_tests": subject_tests,
    }


def verify_frozen_participants(
    subject_df: pd.DataFrame,
    expected: list[str],
    dataset_name: str,
) -> None:
    observed = set(subject_df["participant"].astype(str).unique())
    expected_set = set(expected)

    if observed != expected_set:
        raise RuntimeError(
            f"{dataset_name} frozen cohort mismatch. "
            f"Missing={sorted(expected_set - observed)}; "
            f"Unexpected={sorted(observed - expected_set)}"
        )

    if subject_df["participant"].nunique() != len(expected):
        raise RuntimeError(f"{dataset_name} participant count mismatch.")

    scenarios = set(subject_df["scenario"].astype(str).unique())
    methods = set(subject_df["method"].astype(str).unique())

    if scenarios != set(EXPECTED_SCENARIOS):
        raise RuntimeError(
            f"{dataset_name} scenario mismatch: {sorted(scenarios)}"
        )

    if methods != set(EXPECTED_METHODS):
        raise RuntimeError(
            f"{dataset_name} method mismatch: {sorted(methods)}"
        )


def participant_comparison_summary(
    subject_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    rows = []

    for scenario in EXPECTED_SCENARIOS:
        pivot = (
            subject_df[subject_df["scenario"] == scenario]
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
                stable_seed(f"{dataset_name}|{scenario}|{baseline}")
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
                    "Expected one statistical-test row for "
                    f"{dataset_name}/{scenario}/{baseline}; "
                    f"found {len(test_row)}."
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


def table1_validation_architecture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Evidence layer": "Formal theory",
                "Dataset / source": "Analytical formulation",
                "Independent unit": "Mathematical model",
                "Timing structure": "Inverse affine clock + temporal gauge",
                "Primary role": (
                    "Convexity, identifiability, conditioning, "
                    "and bounded residual-score contributions"
                ),
            },
            {
                "Evidence layer": "Synthetic ground truth",
                "Dataset / source": "MATH-SYNC S1 Monte Carlo",
                "Independent unit": "Controlled replicate",
                "Timing structure": (
                    "Six heterogeneous native-rate synthetic streams"
                ),
                "Primary role": (
                    "Known clock perturbations and exact reconstruction "
                    "ground truth"
                ),
            },
            {
                "Evidence layer": "Real public schedules",
                "Dataset / source": "Harvard S0201 Session 1",
                "Independent unit": "Session",
                "Timing structure": (
                    "EEG ~500 Hz; NIRS ~7.8 Hz; "
                    "Physio ~20 Hz; SIM ~60 Hz"
                ),
                "Primary role": (
                    "Transfer from synthetic schedules "
                    "to real LSL timing schedules"
                ),
            },
            {
                "Evidence layer": "Within-participant replication",
                "Dataset / source": "Harvard S0201 Session 2",
                "Independent unit": "Session",
                "Timing structure": "Same four frozen core streams",
                "Primary role": "Within-participant session replication",
            },
            {
                "Evidence layer": "Between-participant replication",
                "Dataset / source": "Harvard cohort",
                "Independent unit": "Participant (N=10)",
                "Timing structure": "Real LSL schedules + real event anchors",
                "Primary role": "Participant-level generalization",
            },
            {
                "Evidence layer": "External-dataset replication",
                "Dataset / source": "WESAD",
                "Independent unit": "Participant (N=15)",
                "Timing structure": (
                    "Chest ECG 700 Hz; wrist BVP 64 Hz; "
                    "ACC 32 Hz; EDA 4 Hz"
                ),
                "Primary role": "Cross-dataset and cross-device generalization",
            },
        ]
    )


def paired_participant_figure(
    subject_df: pd.DataFrame,
    dataset_name: str,
    scenario: str,
    output_path: Path,
) -> None:
    pivot = (
        subject_df[subject_df["scenario"] == scenario]
        .pivot(
            index="participant",
            columns="method",
            values="participant_median_macro_rmse_ms",
        )
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    for _, row in pivot.iterrows():
        ax.plot(
            [0, 1],
            [row["Joint OLS"], row["Joint Huber"]],
            marker="o",
            alpha=0.75,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Joint OLS", "Joint Huber"])
    ax.set_xlabel("Synchronization estimator")
    ax.set_ylabel("Participant-level median macro RMSE (ms)")
    ax.set_title(
        f"{dataset_name}: {scenario} "
        "paired participant-level reconstruction"
    )
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def cross_dataset_figure(
    harvard_subject: pd.DataFrame,
    wesad_subject: pd.DataFrame,
    scenario: str,
    output_path: Path,
) -> None:
    plot_rows = []

    for dataset_name, subject_df in [
        ("Harvard", harvard_subject),
        ("WESAD", wesad_subject),
    ]:
        subset = subject_df[
            subject_df["scenario"] == scenario
        ]

        for method_name in [
            "Offset-only",
            "Joint OLS",
            "Joint Huber",
        ]:
            values = subset[
                subset["method"] == method_name
            ]["participant_median_macro_rmse_ms"].to_numpy()

            plot_rows.append(
                {
                    "dataset": dataset_name,
                    "method": method_name,
                    "median_rmse_ms": float(np.median(values)),
                }
            )

    plot_df = pd.DataFrame(plot_rows)

    datasets = ["Harvard", "WESAD"]
    methods = ["Offset-only", "Joint OLS", "Joint Huber"]
    x = np.arange(len(datasets))
    width = 0.24

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    for index, method_name in enumerate(methods):
        values = (
            plot_df[plot_df["method"] == method_name]
            .set_index("dataset")
            .reindex(datasets)["median_rmse_ms"]
            .to_numpy()
        )

        ax.bar(
            x + (index - 1) * width,
            values,
            width=width,
            label=method_name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_xlabel("Public validation dataset")
    ax.set_ylabel("Median participant-level macro RMSE (ms)")
    ax.set_title(
        f"{scenario.capitalize()} contamination: "
        "cross-dataset replication"
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def figure1(
    table2: pd.DataFrame,
    output_path: Path,
) -> None:
    scenario_order = {"clean": 0, "nominal": 1, "stress": 2}

    huber = table2[
        table2["method"] == "Joint Huber"
    ].copy()
    huber["_scenario_order"] = huber["scenario"].map(
        scenario_order
    )
    huber = huber.sort_values("_scenario_order")

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x = np.arange(3)

    ax.plot(
        x,
        huber["session1_median_macro_rmse_ms"].to_numpy(),
        marker="o",
        label="Session 1",
    )
    ax.plot(
        x,
        huber["session2_median_macro_rmse_ms"].to_numpy(),
        marker="o",
        label="Session 2",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["Clean", "Nominal", "Stress"])
    ax.set_xlabel("Contamination scenario")
    ax.set_ylabel("Median macro RMSE (ms)")
    ax.set_title(
        "S0201 within-participant replication: Joint Huber"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def copy_supplementary_s1_figures() -> list[Path]:
    mapping = {
        S1_ROOT / "figures" / "s1_rmse_vs_drift.png":
            MANUSCRIPT_SUPPLEMENTARY
            / "Supplementary_Figure_S1_RMSE_vs_drift.png",
        S1_ROOT / "figures" / "s1_rmse_vs_jitter.png":
            MANUSCRIPT_SUPPLEMENTARY
            / "Supplementary_Figure_S2_RMSE_vs_jitter.png",
        S1_ROOT / "figures" / "s1_rmse_vs_missingness.png":
            MANUSCRIPT_SUPPLEMENTARY
            / "Supplementary_Figure_S3_RMSE_vs_missingness.png",
        S1_ROOT / "figures" / "s1_rmse_vs_outliers.png":
            MANUSCRIPT_SUPPLEMENTARY
            / "Supplementary_Figure_S4_RMSE_vs_outliers.png",
    }

    copied = []
    for source, destination in mapping.items():
        require_file(source)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def deterministic_zip(
    zip_path: Path,
    files: list[Path],
    base_dir: Path,
) -> None:
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(files):
            rel = path.relative_to(base_dir).as_posix()
            info = zipfile.ZipInfo(rel)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())

    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(
            f"ZIP integrity failure at {bad_member}: {zip_path}"
        )


def verify_key_values(
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    supplementary_s1: pd.DataFrame,
) -> dict:
    clean_huber = table2[
        (table2["scenario"] == "clean")
        & (table2["method"] == "Joint Huber")
    ]
    if len(clean_huber) != 1:
        raise RuntimeError("Table 2 clean/Joint Huber row unresolved.")

    table2_relative = float(
        clean_huber.iloc[0][
            "relative_session_difference_pct"
        ]
    )

    wesad_clean = table3[
        (table3["dataset"].str.upper() == "WESAD")
        & (table3["scenario"] == "clean")
        & (table3["baseline"] == "Joint OLS")
    ]
    if len(wesad_clean) != 1:
        raise RuntimeError("Table 3 WESAD clean row unresolved.")

    table3_reduction = float(
        wesad_clean.iloc[0]["relative_rmse_reduction_pct"]
    )
    table3_huber = float(
        wesad_clean.iloc[0]["huber_median_rmse_ms"]
    )

    supp_wesad_clean = supplementary_s1[
        (supplementary_s1["dataset"].str.upper() == "WESAD")
        & (supplementary_s1["scenario"] == "clean")
        & (supplementary_s1["baseline"] == "Joint OLS")
    ]
    if len(supp_wesad_clean) != 1:
        raise RuntimeError(
            "Supplementary Table S1 WESAD clean row unresolved."
        )

    supp_reduction = float(
        supp_wesad_clean.iloc[0][
            "relative_rmse_reduction_pct"
        ]
    )
    supp_huber = float(
        supp_wesad_clean.iloc[0]["huber_median_rmse_ms"]
    )

    checks = {
        "table2_clean_huber_relative_round4": round(
            table2_relative,
            4,
        ) == 0.2615,
        "table3_wesad_clean_reduction_round4": round(
            table3_reduction,
            4,
        ) == -0.4164,
        "table3_wesad_clean_huber_round6": round(
            table3_huber,
            6,
        ) == 0.106495,
        "supplementary_wesad_clean_reduction_matches_table3": (
            abs(supp_reduction - table3_reduction) < 1e-12
        ),
        "supplementary_wesad_clean_huber_matches_table3": (
            abs(supp_huber - table3_huber) < 1e-12
        ),
    }

    if not all(checks.values()):
        raise RuntimeError(
            "Manuscript-output value verification failed: "
            + json.dumps(checks, indent=2)
        )

    return checks


def generate() -> dict:
    inputs = load_inputs()

    table2 = inputs["table2"]
    table3 = inputs["table3"]
    table4 = inputs["table4"]
    harvard_subject = inputs["harvard_subject"]
    wesad_subject = inputs["wesad_subject"]
    subject_tests = inputs["subject_tests"]

    verify_frozen_participants(
        harvard_subject,
        EXPECTED_HARVARD_PARTICIPANTS,
        "Harvard",
    )
    verify_frozen_participants(
        wesad_subject,
        EXPECTED_WESAD_PARTICIPANTS,
        "WESAD",
    )

    for directory in [
        MANUSCRIPT_ROOT,
        MANUSCRIPT_TABLES,
        MANUSCRIPT_FIGURES,
        MANUSCRIPT_SUPPLEMENTARY,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # Main Tables 1–4.
    table1 = table1_validation_architecture()
    table1.to_csv(
        MANUSCRIPT_TABLES
        / "Table_1_validation_architecture.csv",
        index=False,
    )
    shutil.copy2(
        PUBLIC_TABLES
        / "Table_2_S0201_within_participant_replication.csv",
        MANUSCRIPT_TABLES
        / "Table_2_S0201_within_participant_replication.csv",
    )
    shutil.copy2(
        PUBLIC_TABLES
        / "Table_3_Huber_vs_OLS_replication.csv",
        MANUSCRIPT_TABLES
        / "Table_3_Huber_vs_OLS_replication.csv",
    )
    shutil.copy2(
        PUBLIC_TABLES
        / "Table_4_public_dataset_QC_summary.csv",
        MANUSCRIPT_TABLES
        / "Table_4_public_dataset_QC_summary.csv",
    )

    # Supplementary Table S1, regenerated from final participant-level results.
    supplementary_s1 = pd.concat(
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

    supplementary_s1_path = (
        MANUSCRIPT_SUPPLEMENTARY
        / "Supplementary_Table_S1_full_participant_comparisons.csv"
    )
    supplementary_s1.to_csv(
        supplementary_s1_path,
        index=False,
    )

    # Main Figures 1–7.
    figure1(
        table2,
        MANUSCRIPT_FIGURES
        / "Figure_1_S0201_within_participant_replication.png",
    )

    paired_participant_figure(
        harvard_subject,
        "Harvard",
        "nominal",
        MANUSCRIPT_FIGURES
        / "Figure_2_Harvard_nominal_paired_participants.png",
    )
    paired_participant_figure(
        harvard_subject,
        "Harvard",
        "stress",
        MANUSCRIPT_FIGURES
        / "Figure_3_Harvard_stress_paired_participants.png",
    )
    paired_participant_figure(
        wesad_subject,
        "WESAD",
        "nominal",
        MANUSCRIPT_FIGURES
        / "Figure_4_WESAD_nominal_paired_participants.png",
    )
    paired_participant_figure(
        wesad_subject,
        "WESAD",
        "stress",
        MANUSCRIPT_FIGURES
        / "Figure_5_WESAD_stress_paired_participants.png",
    )
    cross_dataset_figure(
        harvard_subject,
        wesad_subject,
        "nominal",
        MANUSCRIPT_FIGURES
        / "Figure_6_cross_dataset_nominal.png",
    )
    cross_dataset_figure(
        harvard_subject,
        wesad_subject,
        "stress",
        MANUSCRIPT_FIGURES
        / "Figure_7_cross_dataset_stress.png",
    )

    # Supplementary Figures S1–S4 from frozen S1.
    copy_supplementary_s1_figures()

    checks = verify_key_values(
        table2,
        table3,
        supplementary_s1,
    )

    figure_files = sorted(
        MANUSCRIPT_FIGURES.glob("Figure_*.png")
    )
    supplementary_files = sorted(
        MANUSCRIPT_SUPPLEMENTARY.iterdir()
    )

    if len(figure_files) != 7:
        raise RuntimeError(
            f"Expected 7 main figures; found {len(figure_files)}."
        )

    if len(
        list(
            MANUSCRIPT_SUPPLEMENTARY.glob(
                "Supplementary_Figure_S*.png"
            )
        )
    ) != 4:
        raise RuntimeError(
            "Expected Supplementary Figures S1–S4."
        )

    deterministic_zip(
        FIGURES_ZIP,
        figure_files,
        MANUSCRIPT_FIGURES,
    )
    deterministic_zip(
        SUPPLEMENTARY_ZIP,
        supplementary_files,
        MANUSCRIPT_SUPPLEMENTARY,
    )

    output_files = [
        path
        for path in MANUSCRIPT_ROOT.rglob("*")
        if path.is_file()
        and path.name not in {
            "SHA256SUMS.txt",
            "RESULTS_FREEZE.json",
        }
    ]

    results_freeze = {
        "release": "MATH-SYNC v1.0",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "inferential_unit": "participant",
        "primary_outcome": "equal-modality macro RMSE",
        "harvard_participants": EXPECTED_HARVARD_PARTICIPANTS,
        "wesad_participants": EXPECTED_WESAD_PARTICIPANTS,
        "main_tables": [
            path.name
            for path in sorted(
                MANUSCRIPT_TABLES.glob("Table_*.csv")
            )
        ],
        "main_figures": [
            path.name for path in figure_files
        ],
        "supplementary_files": [
            path.name for path in supplementary_files
        ],
        "key_value_checks": checks,
        "estimator_rerun_by_reporting_script": False,
        "source_public_replay_outputs_modified": False,
    }

    freeze_path = MANUSCRIPT_ROOT / "RESULTS_FREEZE.json"
    freeze_path.write_text(
        json.dumps(
            results_freeze,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    output_files.append(freeze_path)

    checksum_path = MANUSCRIPT_ROOT / "SHA256SUMS.txt"
    checksum_lines = [
        (
            f"{sha256_file(path)}  "
            f"{path.relative_to(MANUSCRIPT_ROOT).as_posix()}"
        )
        for path in sorted(output_files)
    ]
    checksum_path.write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )

    for line in checksum_path.read_text(
        encoding="utf-8"
    ).splitlines():
        digest, relative = line.split("  ", 1)
        path = MANUSCRIPT_ROOT / relative
        if sha256_file(path) != digest:
            raise RuntimeError(
                f"SHA-256 verification failed: {relative}"
            )

    return {
        "checks": checks,
        "main_figure_count": len(figure_files),
        "supplementary_figure_count": 4,
        "supplementary_table_count": 1,
        "figures_zip_sha256": sha256_file(FIGURES_ZIP),
        "supplementary_zip_sha256": sha256_file(
            SUPPLEMENTARY_ZIP
        ),
    }


def verify_existing() -> dict:
    inputs = load_inputs()

    supplementary_path = (
        MANUSCRIPT_SUPPLEMENTARY
        / "Supplementary_Table_S1_full_participant_comparisons.csv"
    )
    require_file(supplementary_path)

    supplementary_s1 = pd.read_csv(
        supplementary_path
    )

    checks = verify_key_values(
        inputs["table2"],
        inputs["table3"],
        supplementary_s1,
    )

    required = [
        MANUSCRIPT_FIGURES
        / "Figure_1_S0201_within_participant_replication.png",
        MANUSCRIPT_FIGURES
        / "Figure_2_Harvard_nominal_paired_participants.png",
        MANUSCRIPT_FIGURES
        / "Figure_3_Harvard_stress_paired_participants.png",
        MANUSCRIPT_FIGURES
        / "Figure_4_WESAD_nominal_paired_participants.png",
        MANUSCRIPT_FIGURES
        / "Figure_5_WESAD_stress_paired_participants.png",
        MANUSCRIPT_FIGURES
        / "Figure_6_cross_dataset_nominal.png",
        MANUSCRIPT_FIGURES
        / "Figure_7_cross_dataset_stress.png",
        MANUSCRIPT_SUPPLEMENTARY
        / "Supplementary_Figure_S1_RMSE_vs_drift.png",
        MANUSCRIPT_SUPPLEMENTARY
        / "Supplementary_Figure_S2_RMSE_vs_jitter.png",
        MANUSCRIPT_SUPPLEMENTARY
        / "Supplementary_Figure_S3_RMSE_vs_missingness.png",
        MANUSCRIPT_SUPPLEMENTARY
        / "Supplementary_Figure_S4_RMSE_vs_outliers.png",
        FIGURES_ZIP,
        SUPPLEMENTARY_ZIP,
        MANUSCRIPT_ROOT / "SHA256SUMS.txt",
        MANUSCRIPT_ROOT / "RESULTS_FREEZE.json",
    ]

    for path in required:
        require_file(path)

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate manuscript-aligned MATH-SYNC v1.0 "
            "tables, figures, and supplementary outputs."
        )
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing generated outputs without regenerating them.",
    )
    args = parser.parse_args()

    if args.verify_only:
        checks = verify_existing()
        print(json.dumps(checks, indent=2))
        print("MANUSCRIPT-OUTPUT-VERIFICATION: PASS")
        return 0

    result = generate()

    print("MATH-SYNC manuscript-output generation")
    print(
        f"Main figures: {result['main_figure_count']}/7"
    )
    print(
        "Supplementary figures: "
        f"{result['supplementary_figure_count']}/4"
    )
    print(
        "Supplementary tables: "
        f"{result['supplementary_table_count']}/1"
    )
    print(json.dumps(result["checks"], indent=2))
    print(
        "Figures.zip SHA-256: "
        f"{result['figures_zip_sha256']}"
    )
    print(
        "Supplementary.zip SHA-256: "
        f"{result['supplementary_zip_sha256']}"
    )
    print("MANUSCRIPT-OUTPUT-GENERATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
