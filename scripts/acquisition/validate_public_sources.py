from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.io import loadmat
except Exception as exc:
    raise RuntimeError(
        "scipy is required for Harvard MATLAB structural validation."
    ) from exc

try:
    import h5py
except Exception as exc:
    raise RuntimeError(
        "h5py is required as the MATLAB v7.3 fallback loader."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ACQUISITION_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "reproducibility"
    / "exact_source_acquisition.json"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "reproducibility"
REPORT_PATH = REPORT_DIR / "public_source_structural_validation.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "source_qc"
HARVARD_QC_CSV = OUTPUT_DIR / "harvard_structural_qc.csv"
WESAD_QC_CSV = OUTPUT_DIR / "wesad_structural_qc.csv"

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
HARVARD_ARCHIVE_ROOT = RAW_ROOT / "harvard_exact"
HARVARD_EXTRACT_ROOT = RAW_ROOT / "harvard_extracted"
WESAD_ARCHIVE = RAW_ROOT / "WESAD_external" / "WESAD.zip"
WESAD_EXTRACT_ROOT = RAW_ROOT / "WESAD_extracted"

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

HARVARD_CORE_ROWS = {
    "EEG": 9,
    "NIRS": 42,
    "PHYSIO": 5,
    "SIM": 31,
}

HARVARD_CORE_RATES = {
    "EEG": 500.0,
    "NIRS": 7.8125,
    "PHYSIO": 20.0,
    "SIM": 60.0,
}

HARVARD_RATE_RELATIVE_TOLERANCE = 0.05
HARVARD_SESSION1_EXPECTED_ANCHORS = 26

WESAD_EXPECTED_SUBJECTS = 15
WESAD_STREAMS = {
    "CHEST_ECG": ("chest", "ECG", 700.0),
    "WRIST_BVP": ("wrist", "BVP", 64.0),
    "WRIST_ACC": ("wrist", "ACC", 32.0),
    "WRIST_EDA": ("wrist", "EDA", 4.0),
}
WESAD_LABEL_RATE_HZ = 700.0
WESAD_EXPECTED_ANCHOR_MIN = 14
WESAD_EXPECTED_ANCHOR_MAX = 16


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_acquisition_report() -> dict[str, Any]:
    if not ACQUISITION_REPORT.exists():
        raise RuntimeError(
            f"Required acquisition report not found: {ACQUISITION_REPORT}"
        )

    report = load_json(ACQUISITION_REPORT)
    if not report.get("pass"):
        raise RuntimeError(
            "Exact-source acquisition report is not PASS."
        )
    return report


def safe_member_target(root: Path, member_name: str) -> Path:
    """
    Resolve a ZIP member under root and reject absolute paths and traversal.
    """
    normalized = member_name.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise RuntimeError(f"Unsafe absolute ZIP member path: {member_name}")

    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise RuntimeError(f"Unsafe ZIP traversal member: {member_name}")

    target = root.joinpath(*parts)
    root_resolved = root.resolve()
    target_parent = target.parent.resolve()

    try:
        target_parent.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"ZIP member escapes extraction root: {member_name}") from exc

    return target


def zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    # Unix file type is stored in the upper 16 bits when present.
    mode = (info.external_attr >> 16) & 0xFFFF
    return (mode & 0o170000) == 0o120000


def controlled_extract(
    archive: Path,
    destination: Path,
    member_predicate=None,
) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    rows = []

    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if zip_member_is_symlink(info):
                raise RuntimeError(
                    f"Symlink member is not accepted in controlled extraction: "
                    f"{info.filename}"
                )
            if member_predicate is not None and not member_predicate(info.filename):
                continue

            target = safe_member_target(destination, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)

            # Reuse only if exact uncompressed byte size and CRC match.
            reuse = False
            if target.exists() and target.stat().st_size == info.file_size:
                observed_crc = crc32_file(target)
                expected_crc = f"{info.CRC:08x}"
                reuse = observed_crc.lower() == expected_crc.lower()

            if not reuse:
                temp = target.with_suffix(target.suffix + ".part")
                if temp.exists():
                    temp.unlink()

                with zf.open(info, "r") as src, temp.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)

                if temp.stat().st_size != info.file_size:
                    raise RuntimeError(
                        f"Extracted byte-size mismatch for {info.filename}"
                    )

                observed_crc = crc32_file(temp)
                expected_crc = f"{info.CRC:08x}"
                if observed_crc.lower() != expected_crc.lower():
                    raise RuntimeError(
                        f"Extracted CRC32 mismatch for {info.filename}"
                    )

                if target.exists():
                    target.unlink()
                temp.replace(target)

            rows.append(
                {
                    "member_name": info.filename,
                    "target_path": str(target),
                    "uncompressed_bytes": int(info.file_size),
                    "crc32_hex": f"{info.CRC:08x}",
                    "reused_verified_extracted_file": reuse,
                }
            )

    return rows


def crc32_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    import zlib

    crc = 0
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            crc = zlib.crc32(block, crc)
    return f"{crc & 0xFFFFFFFF:08x}"


def find_participant_directory(extraction_root: Path, participant: str) -> Path:
    candidates = [
        extraction_root / participant,
        extraction_root / extraction_root.name / participant,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    matches = [
        path
        for path in extraction_root.rglob(participant)
        if path.is_dir()
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Extracted Harvard participant directory not found: {participant}"
        )
    # Deterministic choice only if one candidate contains required Session 1 files.
    qualified = []
    for path in matches:
        required = [
            path / f"EEG_{participant}_1.mat",
            path / f"NIRS_{participant}_1.mat",
            path / f"Physio_{participant}_1.mat",
            path / f"SIMlsl_{participant}_1.mat",
            path / f"EventTimes_{participant}_1.mat",
        ]
        if all(item.exists() for item in required):
            qualified.append(path)

    if len(qualified) == 1:
        return qualified[0]

    raise RuntimeError(
        f"Ambiguous extracted participant directory for {participant}: "
        f"{[str(p) for p in matches]}"
    )


def load_mat_public_variables(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        mat = loadmat(path, squeeze_me=False, struct_as_record=False)
        public = {
            key: value
            for key, value in mat.items()
            if not key.startswith("__")
        }
        return "SCIPY_MAT", public
    except (NotImplementedError, ValueError, OSError):
        public = {}
        with h5py.File(path, "r") as h5:
            for key in h5.keys():
                obj = h5[key]
                if isinstance(obj, h5py.Dataset):
                    public[key] = np.asarray(obj)
        return "HDF5_MAT73", public


def numeric_arrays(public: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    result = []
    for key, value in public.items():
        if not isinstance(value, np.ndarray):
            continue
        try:
            if np.issubdtype(value.dtype, np.number):
                result.append((key, np.asarray(value)))
        except TypeError:
            continue
    return result


def orient_rows(array: np.ndarray, expected_rows: int) -> np.ndarray | None:
    if array.ndim != 2:
        return None
    if array.shape[0] == expected_rows:
        return np.asarray(array)
    if array.shape[1] == expected_rows:
        return np.asarray(array).T
    return None


def find_expected_arrays(
    public: dict[str, Any],
    expected_rows: int,
) -> list[tuple[str, np.ndarray]]:
    matches = []
    for key, array in numeric_arrays(public):
        oriented = orient_rows(array, expected_rows)
        if oriented is not None:
            matches.append((key, oriented))
    return sorted(matches, key=lambda pair: pair[0])


def diagnose_timestamp_vector(
    array: np.ndarray,
    expected_rate: float,
) -> dict[str, Any]:
    raw = np.asarray(array[0, :], dtype=float).ravel()
    finite = raw[np.isfinite(raw)]

    if finite.size < 3:
        raise RuntimeError("Fewer than three finite timestamps.")

    diffs = np.diff(finite)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        raise RuntimeError("No positive timestamp increments.")

    median_positive_dt = float(np.median(positive))
    observed_rate = float(1.0 / median_positive_dt)
    rate_relative_error = abs(observed_rate - expected_rate) / expected_rate

    return {
        "samples": int(array.shape[1]),
        "finite_timestamp_count": int(finite.size),
        "strictly_increasing": bool(np.all(diffs > 0)),
        "duplicate_step_count": int(np.sum(diffs == 0)),
        "reversal_step_count": int(np.sum(diffs < 0)),
        "observed_rate_hz": observed_rate,
        "expected_rate_hz": expected_rate,
        "rate_relative_error": rate_relative_error,
        "start_time_s": float(finite[0]),
        "end_time_s": float(finite[-1]),
        "timestamp_vector": finite,
    }


def harvard_core_file(participant: str, session: int, stream: str) -> str:
    mapping = {
        "EEG": f"EEG_{participant}_{session}.mat",
        "NIRS": f"NIRS_{participant}_{session}.mat",
        "PHYSIO": f"Physio_{participant}_{session}.mat",
        "SIM": f"SIMlsl_{participant}_{session}.mat",
    }
    return mapping[stream]


def build_harvard_anchor_times(
    participant_dir: Path,
    participant: str,
    session: int,
    overlap_start: float,
    overlap_end: float,
) -> np.ndarray:
    event_path = participant_dir / f"EventTimes_{participant}_{session}.mat"
    if not event_path.exists():
        raise FileNotFoundError(event_path)

    _, public = load_mat_public_variables(event_path)

    if "EventTime" not in public:
        raise RuntimeError(f"{event_path.name}: EventTime variable not found.")
    if "BrakeLights_On" not in public:
        raise RuntimeError(f"{event_path.name}: BrakeLights_On variable not found.")

    event_time = np.asarray(public["EventTime"], dtype=float)
    brake = np.asarray(public["BrakeLights_On"], dtype=float).reshape(-1)

    values = []
    for value in event_time.reshape(-1):
        if np.isfinite(value):
            values.append(float(value))
    for value in brake:
        if np.isfinite(value):
            values.append(float(value))

    anchors = np.asarray(
        sorted(
            {
                value
                for value in values
                if overlap_start <= value <= overlap_end
            }
        ),
        dtype=float,
    )

    if anchors.size < 6:
        raise RuntimeError(
            f"{participant} session {session}: only {anchors.size} unique "
            "absolute event anchors remain after the frozen overlap rule."
        )

    return anchors


def validate_harvard_session(
    participant_dir: Path,
    participant: str,
    session: int,
) -> dict[str, Any]:
    stream_rows = []
    timestamp_map = {}

    for stream in ("EEG", "NIRS", "PHYSIO", "SIM"):
        filename = harvard_core_file(participant, session, stream)
        path = participant_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)

        loader, public = load_mat_public_variables(path)
        matches = find_expected_arrays(public, HARVARD_CORE_ROWS[stream])

        if len(matches) != 1:
            raise RuntimeError(
                f"{participant} session {session} {stream}: expected exactly "
                f"one {HARVARD_CORE_ROWS[stream]} x n numeric array; "
                f"found {len(matches)}."
            )

        variable, array = matches[0]
        diag = diagnose_timestamp_vector(
            array,
            HARVARD_CORE_RATES[stream],
        )

        admissible = (
            diag["strictly_increasing"]
            and diag["rate_relative_error"] <= HARVARD_RATE_RELATIVE_TOLERANCE
        )
        if not admissible:
            raise RuntimeError(
                f"{participant} session {session} {stream}: core timestamp "
                f"schedule failed structural QC: {diag}"
            )

        timestamp_map[stream] = diag.pop("timestamp_vector")
        stream_rows.append(
            {
                "participant": participant,
                "session": session,
                "stream": stream,
                "filename": filename,
                "mat_loader": loader,
                "variable": variable,
                "expected_rows": HARVARD_CORE_ROWS[stream],
                **diag,
                "admissible": True,
            }
        )

    overlap_start = max(float(ts[0]) for ts in timestamp_map.values())
    overlap_end = min(float(ts[-1]) for ts in timestamp_map.values())
    overlap_duration = overlap_end - overlap_start

    if overlap_duration <= 0:
        raise RuntimeError(
            f"{participant} session {session}: core streams have no positive overlap."
        )

    anchors = build_harvard_anchor_times(
        participant_dir,
        participant,
        session,
        overlap_start,
        overlap_end,
    )

    if session == 1 and anchors.size != HARVARD_SESSION1_EXPECTED_ANCHORS:
        raise RuntimeError(
            f"{participant} session 1: expected "
            f"{HARVARD_SESSION1_EXPECTED_ANCHORS} anchors; "
            f"observed {anchors.size}."
        )

    return {
        "participant": participant,
        "session": session,
        "participant_directory": str(participant_dir),
        "stream_rows": stream_rows,
        "overlap_start_s": overlap_start,
        "overlap_end_s": overlap_end,
        "overlap_duration_s": overlap_duration,
        "anchor_count": int(anchors.size),
        "first_anchor_s": float(anchors[0]),
        "last_anchor_s": float(anchors[-1]),
    }


def validate_s0201_pupil_policy(participant_dir: Path) -> dict[str, Any]:
    path = participant_dir / "PupilData_S0201_1.mat"
    if not path.exists():
        raise FileNotFoundError(path)

    loader, public = load_mat_public_variables(path)
    matches = find_expected_arrays(public, 3)

    if len(matches) < 2:
        raise RuntimeError(
            "PupilData_S0201_1.mat does not contain two published 3 x n arrays."
        )

    reversal_counts = []
    for key, array in matches[:2]:
        ts = np.asarray(array[0, :], dtype=float).ravel()
        ts = ts[np.isfinite(ts)]
        if ts.size < 3:
            raise RuntimeError(f"S0201 pupil array {key} has too few timestamps.")
        reversal_counts.append(int(np.sum(np.diff(ts) < 0)))

    if not all(count > 0 for count in reversal_counts):
        raise RuntimeError(
            "The two S0201 Session 1 pupil arrays were expected to exhibit "
            f"raw timestamp reversals; observed counts={reversal_counts}."
        )

    audio_path = participant_dir / "AudioEvents_S0201_1.mat"
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    return {
        "pupil_mat_loader": loader,
        "pupil_array_count": len(matches),
        "first_two_pupil_reversal_counts": reversal_counts,
        "audio_events_present": True,
    }


def acquisition_harvard_manifest(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("harvard", {}).get("archives", [])
    manifest = {row["filename"]: row for row in rows}
    if len(manifest) != 10:
        raise RuntimeError(
            f"Acquisition report does not contain 10 Harvard archive records: "
            f"{len(manifest)}"
        )
    return manifest


def process_harvard(report: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = acquisition_harvard_manifest(report)

    qc_rows = []
    participant_reports = {}

    for index, participant in enumerate(FROZEN_HARVARD_COHORT, start=1):
        filename = f"{participant}_physio.zip"
        archive = HARVARD_ARCHIVE_ROOT / filename

        if not archive.exists():
            raise FileNotFoundError(archive)

        expected_sha = manifest[filename]["sha256"]
        observed_sha = sha256_file(archive)
        if observed_sha.lower() != str(expected_sha).lower():
            raise RuntimeError(
                f"{filename}: SHA-256 differs from acquisition report."
            )

        destination = HARVARD_EXTRACT_ROOT / participant
        print(
            f"[Harvard {index}/{len(FROZEN_HARVARD_COHORT)}] "
            f"Controlled extraction + structural validation: {participant}"
        )

        extracted = controlled_extract(archive, destination)
        participant_dir = find_participant_directory(destination, participant)

        session1 = validate_harvard_session(
            participant_dir,
            participant,
            1,
        )

        session_reports = [session1]
        if participant == "S0201":
            session2 = validate_harvard_session(
                participant_dir,
                participant,
                2,
            )
            session_reports.append(session2)
            s0201_pupil = validate_s0201_pupil_policy(participant_dir)
        else:
            s0201_pupil = None

        participant_reports[participant] = {
            "archive": filename,
            "archive_sha256": observed_sha,
            "extracted_member_count": len(extracted),
            "participant_directory": str(participant_dir),
            "sessions": session_reports,
            "s0201_pupil_policy_check": s0201_pupil,
        }

        for session_report in session_reports:
            for row in session_report["stream_rows"]:
                qc_rows.append(
                    {
                        **row,
                        "overlap_duration_s": session_report["overlap_duration_s"],
                        "anchor_count": session_report["anchor_count"],
                    }
                )

        print(
            f"  PASS {participant} | "
            f"Session 1 anchors={session1['anchor_count']} | "
            f"overlap={session1['overlap_duration_s']:.3f} s"
        )

    return qc_rows, participant_reports


def wesad_pickle_members(archive: Path) -> list[dict[str, str]]:
    rows = []
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            match = re.search(
                r"(?:^|/)(S\d+)/(S\d+)\.pkl$",
                info.filename,
            )
            if match and match.group(1) == match.group(2):
                rows.append(
                    {
                        "participant": match.group(1),
                        "member": info.filename,
                    }
                )

    unique = {}
    for row in rows:
        unique.setdefault(row["participant"], row["member"])

    result = [
        {"participant": participant, "member": member}
        for participant, member in unique.items()
    ]
    result.sort(
        key=lambda row: int(re.search(r"\d+", row["participant"]).group())
    )
    return result


def process_wesad(report: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not WESAD_ARCHIVE.exists():
        raise FileNotFoundError(WESAD_ARCHIVE)

    expected_sha = report.get("wesad", {}).get("sha256")
    observed_sha = sha256_file(WESAD_ARCHIVE)
    if not expected_sha or observed_sha.lower() != str(expected_sha).lower():
        raise RuntimeError(
            "WESAD.zip SHA-256 differs from acquisition report."
        )

    manifest = wesad_pickle_members(WESAD_ARCHIVE)
    if len(manifest) != WESAD_EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Expected {WESAD_EXPECTED_SUBJECTS} WESAD participant pickle files; "
            f"found {len(manifest)}."
        )

    allowed_members = {row["member"] for row in manifest}
    print(
        f"[WESAD] Controlled extraction of {len(allowed_members)} "
        "synchronized participant pickle files..."
    )
    controlled_extract(
        WESAD_ARCHIVE,
        WESAD_EXTRACT_ROOT,
        member_predicate=lambda name: name in allowed_members,
    )

    qc_rows = []
    participant_reports = {}

    for index, row in enumerate(manifest, start=1):
        participant = row["participant"]
        extracted_path = safe_member_target(
            WESAD_EXTRACT_ROOT,
            row["member"],
        )
        if not extracted_path.exists():
            raise FileNotFoundError(extracted_path)

        print(
            f"  [WESAD {index}/{len(manifest)}] Structural validation: "
            f"{participant}"
        )

        with extracted_path.open("rb") as f:
            subject_data = pickle.load(f, encoding="latin1")

        if not isinstance(subject_data, dict):
            raise RuntimeError(f"{participant}: WESAD pickle is not a dictionary.")
        if "signal" not in subject_data or "label" not in subject_data:
            raise RuntimeError(
                f"{participant}: WESAD pickle lacks signal and/or label."
            )

        signal = subject_data["signal"]
        if "chest" not in signal or "wrist" not in signal:
            raise RuntimeError(
                f"{participant}: WESAD signal lacks chest and/or wrist dictionaries."
            )

        labels = np.asarray(subject_data["label"]).reshape(-1)
        if labels.size < 3:
            raise RuntimeError(f"{participant}: label vector too short.")

        sample_counts = {}
        durations = {}

        for stream, (location, key, rate) in WESAD_STREAMS.items():
            container = signal[location]
            if key not in container:
                raise RuntimeError(
                    f"{participant}: missing WESAD {location}/{key}."
                )

            array = np.asarray(container[key])
            if array.ndim == 0:
                raise RuntimeError(
                    f"{participant} {stream}: scalar array is invalid."
                )

            n_samples = int(array.shape[0])
            if n_samples < 3:
                raise RuntimeError(
                    f"{participant} {stream}: fewer than three samples."
                )

            sample_counts[stream] = n_samples
            durations[stream] = (n_samples - 1) / rate

        if labels.size != sample_counts["CHEST_ECG"]:
            raise RuntimeError(
                f"{participant}: label length {labels.size} does not match "
                f"chest ECG length {sample_counts['CHEST_ECG']}."
            )

        overlap_end = min(durations.values())

        transition_indices = np.where(labels[1:] != labels[:-1])[0] + 1
        transition_times = transition_indices.astype(float) / WESAD_LABEL_RATE_HZ
        transition_times = np.unique(
            transition_times[
                (transition_times >= 0.0)
                & (transition_times <= overlap_end)
            ]
        )

        anchor_count = int(transition_times.size)
        if not (
            WESAD_EXPECTED_ANCHOR_MIN
            <= anchor_count
            <= WESAD_EXPECTED_ANCHOR_MAX
        ):
            raise RuntimeError(
                f"{participant}: WESAD structural anchor count {anchor_count} "
                f"is outside the manuscript-aligned range "
                f"{WESAD_EXPECTED_ANCHOR_MIN}–{WESAD_EXPECTED_ANCHOR_MAX}."
            )

        qc = {
            "participant": participant,
            "pickle_path": str(extracted_path),
            "label_samples": int(labels.size),
            "chest_ecg_samples": sample_counts["CHEST_ECG"],
            "wrist_bvp_samples": sample_counts["WRIST_BVP"],
            "wrist_acc_samples": sample_counts["WRIST_ACC"],
            "wrist_eda_samples": sample_counts["WRIST_EDA"],
            "overlap_duration_s": float(overlap_end),
            "anchor_count": anchor_count,
            "first_anchor_s": (
                float(transition_times[0])
                if transition_times.size
                else None
            ),
            "last_anchor_s": (
                float(transition_times[-1])
                if transition_times.size
                else None
            ),
            "structural_pass": True,
        }
        qc_rows.append(qc)
        participant_reports[participant] = qc

        print(
            f"    PASS {participant} | anchors={anchor_count} | "
            f"overlap={overlap_end:.3f} s"
        )

        del subject_data
        del signal
        del labels

    return qc_rows, participant_reports


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HARVARD_EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    WESAD_EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

    report_b = load_acquisition_report()

    print()
    print("Controlled source extraction + structural validation")
    print("Estimator execution performed: NO")
    print("Scientific result generation performed: NO")
    print()

    harvard_qc_rows, harvard_reports = process_harvard(report_b)
    write_csv(HARVARD_QC_CSV, harvard_qc_rows)

    wesad_qc_rows, wesad_reports = process_wesad(report_b)
    write_csv(WESAD_QC_CSV, wesad_qc_rows)

    harvard_session1 = [
        report["sessions"][0]
        for report in harvard_reports.values()
    ]
    harvard_anchor_counts = [
        int(item["anchor_count"])
        for item in harvard_session1
    ]
    harvard_overlaps = [
        float(item["overlap_duration_s"])
        for item in harvard_session1
    ]

    wesad_anchor_counts = [
        int(row["anchor_count"])
        for row in wesad_qc_rows
    ]
    wesad_overlaps = [
        float(row["overlap_duration_s"])
        for row in wesad_qc_rows
    ]

    final_report = {
        "stage": "public_source_structural_validation",
        "created_at_utc": utc_now(),
        "estimator_execution_performed": False,
        "scientific_result_generation_performed": False,
        "harvard": {
            "participants_expected": 10,
            "participants_validated": len(harvard_reports),
            "session1_anchor_counts": harvard_anchor_counts,
            "session1_min_anchor_count": min(harvard_anchor_counts),
            "session1_max_anchor_count": max(harvard_anchor_counts),
            "session1_median_anchor_count": float(
                np.median(harvard_anchor_counts)
            ),
            "session1_median_overlap_s": float(
                np.median(harvard_overlaps)
            ),
            "s0201_session2_validated": (
                len(harvard_reports["S0201"]["sessions"]) == 2
            ),
            "s0201_pupil_reversal_policy_confirmed": bool(
                harvard_reports["S0201"]["s0201_pupil_policy_check"]
            ),
            "qc_csv": str(HARVARD_QC_CSV),
            "participants": harvard_reports,
        },
        "wesad": {
            "participants_expected": WESAD_EXPECTED_SUBJECTS,
            "participants_validated": len(wesad_reports),
            "min_anchor_count": min(wesad_anchor_counts),
            "max_anchor_count": max(wesad_anchor_counts),
            "median_anchor_count": float(
                np.median(wesad_anchor_counts)
            ),
            "median_overlap_s": float(
                np.median(wesad_overlaps)
            ),
            "qc_csv": str(WESAD_QC_CSV),
            "participants": wesad_reports,
        },
    }

    pass_condition = (
        len(harvard_reports) == 10
        and len(wesad_reports) == WESAD_EXPECTED_SUBJECTS
        and all(
            count == HARVARD_SESSION1_EXPECTED_ANCHORS
            for count in harvard_anchor_counts
        )
        and min(wesad_anchor_counts) >= WESAD_EXPECTED_ANCHOR_MIN
        and max(wesad_anchor_counts) <= WESAD_EXPECTED_ANCHOR_MAX
        and final_report["harvard"]["s0201_session2_validated"]
        and final_report["harvard"]["s0201_pupil_reversal_policy_confirmed"]
    )

    final_report["pass"] = bool(pass_condition)
    final_report["ready_for_public_data_replay"] = bool(pass_condition)

    REPORT_PATH.write_text(
        json.dumps(final_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print(
        "Harvard participants structurally validated: "
        f"{len(harvard_reports)}/10"
    )
    print(
        "Harvard Session 1 anchors: "
        f"min={min(harvard_anchor_counts)}, "
        f"median={np.median(harvard_anchor_counts):.1f}, "
        f"max={max(harvard_anchor_counts)}"
    )
    print(
        "Harvard Session 1 median overlap: "
        f"{np.median(harvard_overlaps):.2f} s"
    )
    print(
        "S0201 Session 2 structural validation: "
        f"{'PASS' if final_report['harvard']['s0201_session2_validated'] else 'FAIL'}"
    )
    print(
        "S0201 pupil timestamp-reversal policy check: "
        f"{'PASS' if final_report['harvard']['s0201_pupil_reversal_policy_confirmed'] else 'FAIL'}"
    )
    print(
        "WESAD participants structurally validated: "
        f"{len(wesad_reports)}/{WESAD_EXPECTED_SUBJECTS}"
    )
    print(
        "WESAD anchors: "
        f"min={min(wesad_anchor_counts)}, "
        f"median={np.median(wesad_anchor_counts):.1f}, "
        f"max={max(wesad_anchor_counts)}"
    )
    print(
        "WESAD median overlap: "
        f"{np.median(wesad_overlaps):.2f} s"
    )
    print("Estimator execution performed: NO")
    print("Scientific result generation performed: NO")
    print()

    if pass_condition:
        print("PUBLIC-SOURCE-STRUCTURAL-VALIDATION: PASS")
        print("Ready for public-data replay: YES")
        print(f"Gate report: {REPORT_PATH}")
        return 0

    print("PUBLIC-SOURCE-STRUCTURAL-VALIDATION: FAIL")
    print("Ready for public-data replay: NO")
    print(f"Gate report: {REPORT_PATH}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
