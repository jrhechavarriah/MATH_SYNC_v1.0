from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "config" / "G12B_ANCHOR_GEOMETRY_SPEC.json"


def test_g12b_frozen_spec_exists_and_has_expected_design():
    raw = SPEC.read_text(encoding="utf-8-sig")
    cfg = json.loads(raw)

    assert cfg["status"] == "FROZEN_BEFORE_G12B_OUTCOME_INSPECTION"
    assert cfg["design"]["anchor_counts"] == [4, 6, 8, 12, 16, 24]
    assert cfg["design"]["temporal_span_fractions"] == [0.25, 0.5, 1.0]
    assert cfg["design"]["replicates_per_configuration"] == 50
    assert cfg["design"]["expected_trials"] == 2700
    assert cfg["missingness"]["nonreference_forced_minimum_anchor_count"] is False
    assert cfg["public_release_modified"] is False
    assert cfg["legacy_core_modified"] is False
