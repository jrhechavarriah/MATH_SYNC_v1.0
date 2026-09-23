import importlib.util
import zipfile
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "acquisition"
    / "validate_public_sources.py"
)

spec = importlib.util.spec_from_file_location("public_structural_validation", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_harvard_frozen_cohort_exactly_10():
    assert len(mod.FROZEN_HARVARD_COHORT) == 10
    assert len(set(mod.FROZEN_HARVARD_COHORT)) == 10
    assert mod.FROZEN_HARVARD_COHORT[0] == "S0201"


def test_safe_member_target_rejects_traversal(tmp_path):
    try:
        mod.safe_member_target(tmp_path, "../escape.txt")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Traversal path was not rejected.")


def test_controlled_extract_and_crc(tmp_path):
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("S0001/example.txt", b"MATH-SYNC")

    dest = tmp_path / "out"
    rows = mod.controlled_extract(archive, dest)
    assert len(rows) == 1
    target = dest / "S0001" / "example.txt"
    assert target.read_bytes() == b"MATH-SYNC"


def test_timestamp_diagnosis_for_500_hz():
    ts = np.arange(1000, dtype=float) / 500.0
    array = np.vstack([ts] + [np.zeros_like(ts) for _ in range(8)])
    diag = mod.diagnose_timestamp_vector(array, 500.0)
    assert diag["strictly_increasing"] is True
    assert diag["rate_relative_error"] < 1e-9
    assert diag["reversal_step_count"] == 0


def test_wesad_anchor_range_matches_manuscript_qc():
    assert mod.WESAD_EXPECTED_ANCHOR_MIN == 14
    assert mod.WESAD_EXPECTED_ANCHOR_MAX == 16
