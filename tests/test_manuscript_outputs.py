import importlib.util
import json
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reporting_outputs_and_key_values():
    manuscript = ROOT / "outputs" / "manuscript"

    required = [
        manuscript / "tables" / "Table_1_validation_architecture.csv",
        manuscript / "tables" / "Table_2_S0201_within_participant_replication.csv",
        manuscript / "tables" / "Table_3_Huber_vs_OLS_replication.csv",
        manuscript / "tables" / "Table_4_public_dataset_QC_summary.csv",
        manuscript / "supplementary" / "Supplementary_Table_S1_full_participant_comparisons.csv",
        manuscript / "Figures.zip",
        manuscript / "Supplementary.zip",
        manuscript / "RESULTS_FREEZE.json",
        manuscript / "SHA256SUMS.txt",
    ]
    for path in required:
        assert path.exists(), path

    table2 = pd.read_csv(
        manuscript
        / "tables"
        / "Table_2_S0201_within_participant_replication.csv"
    )
    row2 = table2[
        (table2["scenario"] == "clean")
        & (table2["method"] == "Joint Huber")
    ].iloc[0]
    assert round(float(row2["relative_session_difference_pct"]), 4) == 0.2615

    table3 = pd.read_csv(
        manuscript
        / "tables"
        / "Table_3_Huber_vs_OLS_replication.csv"
    )
    row3 = table3[
        (table3["dataset"].str.upper() == "WESAD")
        & (table3["scenario"] == "clean")
    ].iloc[0]
    assert round(float(row3["relative_rmse_reduction_pct"]), 4) == -0.4164
    assert round(float(row3["huber_median_rmse_ms"]), 6) == 0.106495

    supplementary = pd.read_csv(
        manuscript
        / "supplementary"
        / "Supplementary_Table_S1_full_participant_comparisons.csv"
    )
    row_s = supplementary[
        (supplementary["dataset"].str.upper() == "WESAD")
        & (supplementary["scenario"] == "clean")
        & (supplementary["baseline"] == "Joint OLS")
    ].iloc[0]

    assert abs(
        float(row_s["relative_rmse_reduction_pct"])
        - float(row3["relative_rmse_reduction_pct"])
    ) < 1e-12
    assert abs(
        float(row_s["huber_median_rmse_ms"])
        - float(row3["huber_median_rmse_ms"])
    ) < 1e-12


def test_main_figures_are_complete_300dpi_pngs():
    figures = sorted(
        (ROOT / "outputs" / "manuscript" / "figures").glob(
            "Figure_*.png"
        )
    )
    assert len(figures) == 7

    for path in figures:
        with Image.open(path) as image:
            assert image.width >= 2000
            assert image.height >= 1300
            dpi = image.info.get("dpi")
            assert dpi is not None
            assert abs(float(dpi[0]) - 300.0) < 1.0
            assert abs(float(dpi[1]) - 300.0) < 1.0


def test_supplementary_package_is_complete():
    manuscript = ROOT / "outputs" / "manuscript"
    supplementary_dir = manuscript / "supplementary"

    figures = sorted(
        supplementary_dir.glob("Supplementary_Figure_S*.png")
    )
    assert len(figures) == 4

    table = (
        supplementary_dir
        / "Supplementary_Table_S1_full_participant_comparisons.csv"
    )
    assert table.exists()

    with zipfile.ZipFile(
        manuscript / "Supplementary.zip",
        "r",
    ) as archive:
        names = set(archive.namelist())

    expected = {
        "Supplementary_Figure_S1_RMSE_vs_drift.png",
        "Supplementary_Figure_S2_RMSE_vs_jitter.png",
        "Supplementary_Figure_S3_RMSE_vs_missingness.png",
        "Supplementary_Figure_S4_RMSE_vs_outliers.png",
        "Supplementary_Table_S1_full_participant_comparisons.csv",
    }
    assert names == expected


def test_synthetic_script_preserves_frozen_design():
    module = load_module(
        ROOT / "scripts" / "synthetic" / "run_s1_benchmark.py",
        "math_sync_s1_public",
    )

    assert module.SEED == 20260821
    assert module.N_REPLICATES == 50
    assert module.N_ANCHORS == 24
    assert module.SCENARIOS["nominal"]["jitter_ms"] == 1.0
    assert module.SCENARIOS["stress"]["drift_ppm"] == 500.0


def test_release_metadata_lists_all_publication_artifacts():
    path = ROOT / "outputs" / "manuscript" / "RESULTS_FREEZE.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))

    assert metadata["release"] == "MATH-SYNC v1.0"
    assert len(metadata["main_tables"]) == 4
    assert len(metadata["main_figures"]) == 7
    assert len(metadata["supplementary_files"]) == 5
    assert all(metadata["key_value_checks"].values())
