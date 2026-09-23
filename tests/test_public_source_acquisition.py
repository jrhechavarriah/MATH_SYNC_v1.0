import importlib.util
import zipfile
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "acquisition"
    / "acquire_public_sources.py"
)

spec = importlib.util.spec_from_file_location("public_acquisition", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

def test_frozen_harvard_cohort():
    assert len(mod.FROZEN_HARVARD_COHORT) == 10
    assert len(set(mod.FROZEN_HARVARD_COHORT)) == 10
    assert mod.FROZEN_HARVARD_COHORT[0] == "S0201"

def test_manifest_extraction_from_nested_json():
    data = {
        "records": [
            {
                "participant": "S0201",
                "source": {
                    "filename": "S0201_physio.zip",
                    "size_bytes": 206095319,
                    "md5": "af5a73ca21a8430b0f1b797b689c375e",
                },
            }
        ]
    }
    original = mod.FROZEN_HARVARD_COHORT
    try:
        mod.FROZEN_HARVARD_COHORT = ["S0201"]
        rows = mod.extract_frozen_harvard_manifest(data)
    finally:
        mod.FROZEN_HARVARD_COHORT = original

    assert rows[0]["filename"] == "S0201_physio.zip"
    assert rows[0]["size_bytes"] == 206095319
    assert rows[0]["md5"] == "af5a73ca21a8430b0f1b797b689c375e"

def test_wesad_subject_detection(tmp_path):
    archive = tmp_path / "WESAD.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for subject in [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]:
            zf.writestr(f"WESAD/S{subject}/S{subject}.pkl", b"x")
    subjects = mod.wesad_pickle_subjects(archive)
    assert len(subjects) == 15
    assert subjects[0] == "S2"
    assert subjects[-1] == "S17"

def test_hash_helpers(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"MATH-SYNC")
    assert len(mod.md5_file(path)) == 32
    assert len(mod.sha256_file(path)) == 64
