from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "config" / "G12C_REFERENCE_STREAM_SPEC.json"


def test_g12c_frozen_spec_exists_and_has_expected_design():
    cfg = json.loads(SPEC.read_text(encoding="utf-8-sig"))

    assert cfg["status"] == "FROZEN_BEFORE_G12C_OUTCOME_INSPECTION"
    assert cfg["candidate_references"] == [
        "REF_EVENT",
        "EEG",
        "PPG",
        "EDA",
        "IMU",
        "HMD",
    ]
    assert cfg["conditions"] == [
        "reference_clean",
        "reference_systematically_contaminated",
    ]
    assert cfg["systematic_contamination"]["amplitude_ms"] == 20.0
    assert cfg["expected_design"]["expected_trials"] == 600
    assert cfg["canonical_coordinate"]["required"] is True
    assert cfg["public_release_modified"] is False
    assert cfg["legacy_core_modified"] is False

