from __future__ import annotations

import hashlib
import json
import re
import subprocess
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "config" / "public_source_manifest.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "reproducibility" / "exact_source_acquisition.json"

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
HARVARD_ROOT = RAW_ROOT / "harvard_exact"
WESAD_ROOT = RAW_ROOT / "WESAD_external"
WESAD_ARCHIVE = WESAD_ROOT / "WESAD.zip"

HARVARD_DOI = "10.7910/DVN/HMZ5RG"
HARVARD_DATASET_API = (
    "https://dataverse.harvard.edu/api/datasets/:persistentId/"
    f"?persistentId=doi:{HARVARD_DOI}"
)
HARVARD_DATAFILE_URL = "https://dataverse.harvard.edu/api/access/datafile/{datafile_id}"

WESAD_SHARE_ID = "HGdUkoNlW1Ub0Gx"
WESAD_SOURCE_URLS = [
    f"https://uni-siegen.sciebo.de/s/{WESAD_SHARE_ID}/download/WESAD.zip",
    f"https://uni-siegen.sciebo.de/s/{WESAD_SHARE_ID}/download",
]

FROZEN_HARVARD_COHORT = [
    "S0201", "S0113", "S0171", "S0175", "S0199",
    "S0133", "S0120", "S0207", "S0101", "S0127",
]

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def hash_file(path: Path, algorithm: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()

def md5_file(path: Path) -> str:
    return hash_file(path, "md5")

def sha256_file(path: Path) -> str:
    return hash_file(path, "sha256")

def zip_crc_pass(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False

def wesad_pickle_subjects(path: Path) -> list[str]:
    subjects = set()
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            m = re.search(r"(?:^|/)(S\d+)/(S\d+)\.pkl$", name)
            if m and m.group(1) == m.group(2):
                subjects.add(m.group(1))
    return sorted(subjects, key=lambda s: int(s[1:]))

def _scalar_values(node: Any):
    if isinstance(node, dict):
        for value in node.values():
            yield from _scalar_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from _scalar_values(value)
    else:
        yield node

def _walk_dicts(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_dicts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_dicts(value)

def _first_md5(values) -> str | None:
    for value in values:
        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{32}", value.strip()):
            return value.strip().lower()
    return None

def _normalized_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())

def _md5_from_record(node: dict[str, Any]) -> str | None:
    candidates = []
    for subnode in _walk_dicts(node):
        for key, value in subnode.items():
            normalized = _normalized_key(key)
            if "md5" not in normalized:
                continue
            if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{32}", value.strip()):
                candidates.append(value.strip().lower())

    unique = sorted(set(candidates))
    return unique[0] if len(unique) == 1 else None

def _size_from_record(node: dict[str, Any]) -> int | None:
    preferred = []
    fallback = []

    for subnode in _walk_dicts(node):
        for key, value in subnode.items():
            normalized = _normalized_key(key)

            if "id" in normalized:
                continue
            if not ("size" in normalized or "bytes" in normalized or "filesize" in normalized):
                continue

            number = None
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                number = value
            elif isinstance(value, str) and value.isdigit():
                number = int(value)

            if number is None or number <= 1_000_000:
                continue

            if normalized in {
                "sizebytes",
                "filesize",
                "archivesize",
                "selectedarchivesize",
                "expectedsize",
                "expectedsizebytes",
            }:
                preferred.append(number)
            else:
                fallback.append(number)

    preferred_unique = sorted(set(preferred))
    fallback_unique = sorted(set(fallback))

    if len(preferred_unique) == 1:
        return preferred_unique[0]
    if len(preferred_unique) > 1:
        return max(preferred_unique)
    if len(fallback_unique) == 1:
        return fallback_unique[0]
    if len(fallback_unique) > 1:
        return max(fallback_unique)
    return None

def _record_contains_filename(node: dict[str, Any], filename: str) -> bool:
    for key, value in node.items():
        normalized = _normalized_key(key)
        if isinstance(value, (dict, list)):
            continue
        if "filename" in normalized or normalized in {"name", "archive", "selectedarchive"}:
            if str(value) == filename:
                return True
    return False

def extract_frozen_harvard_manifest(config: Any) -> list[dict[str, Any]]:
    records = []

    for participant in FROZEN_HARVARD_COHORT:
        filename = f"{participant}_physio.zip"
        candidates = []

        for node in _walk_dicts(config):
            if not _record_contains_filename(node, filename):
                continue

            md5 = _md5_from_record(node)
            size = _size_from_record(node)
            if md5 and size:
                candidates.append((md5, size))

        unique = sorted(set(candidates))
        if len(unique) != 1:
            raise RuntimeError(
                f"Could not recover one unambiguous frozen record for {filename}; "
                f"candidates={unique}"
            )

        md5, size = unique[0]
        records.append({
            "participant": participant,
            "filename": filename,
            "size_bytes": int(size),
            "md5": md5,
        })

    return records

def fetch_json(url: str, timeout: int = 60) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "MATH-SYNC/2.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

def fetch_harvard_metadata() -> list[dict[str, Any]]:
    payload = fetch_json(HARVARD_DATASET_API)
    files = payload["data"]["latestVersion"]["files"]
    rows = []
    for item in files:
        data_file = item.get("dataFile", {})
        rows.append({
            "filename": data_file.get("filename"),
            "size_bytes": int(data_file.get("filesize", -1)),
            "md5": str(data_file.get("md5", "")).lower(),
            "datafile_id": data_file.get("id"),
            "restricted": bool(item.get("restricted", False)),
        })
    return rows

def match_harvard_manifest(frozen: list[dict[str, Any]], remote: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remote_by_name = {row["filename"]: row for row in remote}
    matched = []
    for expected in frozen:
        observed = remote_by_name.get(expected["filename"])
        if observed is None:
            raise RuntimeError(f"Harvard file missing: {expected['filename']}")
        if observed["restricted"]:
            raise RuntimeError(f"Harvard file is restricted: {expected['filename']}")
        if observed["size_bytes"] != expected["size_bytes"]:
            raise RuntimeError(f"Harvard byte-size mismatch: {expected['filename']}")
        if observed["md5"] != expected["md5"]:
            raise RuntimeError(f"Harvard MD5 mismatch: {expected['filename']}")
        matched.append({**expected, "datafile_id": observed["datafile_id"]})
    return matched

def load_frozen_harvard_expectations() -> list[dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        raise RuntimeError(f"Public source manifest not found: {MANIFEST_PATH}")
    return extract_frozen_harvard_manifest(load_json(MANIFEST_PATH))

def validate_remote_harvard_manifest(
    frozen: list[dict[str, Any]],
    remote: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return match_harvard_manifest(frozen, remote)

def run_curl(url: str, output_part: Path) -> None:
    output_part.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "-fL", "-C", "-",
        "--retry", "5",
        "--retry-delay", "5",
        "--connect-timeout", "45",
        "--max-time", "21600",
        "-o", str(output_part),
        url,
    ]
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"Download command failed with code {completed.returncode}: {url}")

def verify_harvard_archive(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if path.stat().st_size != expected["size_bytes"]:
        raise RuntimeError(f"Byte-size mismatch after download: {path.name}")
    md5 = md5_file(path)
    if md5 != expected["md5"]:
        raise RuntimeError(f"MD5 mismatch after download: {path.name}")
    if not zip_crc_pass(path):
        raise RuntimeError(f"ZIP CRC failed: {path.name}")
    return {
        "participant": expected["participant"],
        "filename": expected["filename"],
        "size_bytes": expected["size_bytes"],
        "md5": md5,
        "sha256": sha256_file(path),
        "zip_crc_pass": True,
        "datafile_id": expected["datafile_id"],
    }

def acquire_harvard(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    HARVARD_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    for index, record in enumerate(records, start=1):
        final = HARVARD_ROOT / record["filename"]
        part = final.with_suffix(final.suffix + ".part")
        reuse = False

        if final.exists():
            try:
                verified = verify_harvard_archive(final, record)
                reuse = True
            except Exception:
                invalid = final.with_suffix(final.suffix + ".invalid")
                if invalid.exists():
                    invalid.unlink()
                final.replace(invalid)

        if not reuse:
            print(f"[Harvard {index}/{len(records)}] Acquiring {record['filename']}...")
            url = HARVARD_DATAFILE_URL.format(datafile_id=record["datafile_id"])
            run_curl(url, part)
            if final.exists():
                final.unlink()
            part.replace(final)
            verified = verify_harvard_archive(final, record)

        print(
            f"  PASS {record['filename']} | MD5={verified['md5']} | "
            f"SHA256={verified['sha256']}"
        )
        results.append(verified)
    return results

def verify_wesad(path: Path) -> dict[str, Any]:
    if not zip_crc_pass(path):
        raise RuntimeError("WESAD ZIP CRC failed.")
    subjects = wesad_pickle_subjects(path)
    if len(subjects) != 15:
        raise RuntimeError(f"Expected 15 synchronized WESAD participant pickles; found {len(subjects)}.")
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "md5": md5_file(path),
        "sha256": sha256_file(path),
        "zip_crc_pass": True,
        "synchronized_participant_pickles": len(subjects),
        "participants": subjects,
    }

def acquire_wesad() -> dict[str, Any]:
    WESAD_ROOT.mkdir(parents=True, exist_ok=True)
    if WESAD_ARCHIVE.exists():
        try:
            return verify_wesad(WESAD_ARCHIVE)
        except Exception:
            invalid = WESAD_ARCHIVE.with_suffix(".zip.invalid")
            if invalid.exists():
                invalid.unlink()
            WESAD_ARCHIVE.replace(invalid)

    part = WESAD_ARCHIVE.with_suffix(".zip.part")
    last_error = None
    for url in WESAD_SOURCE_URLS:
        try:
            print("[WESAD] Acquiring official public archive...")
            run_curl(url, part)
            if WESAD_ARCHIVE.exists():
                WESAD_ARCHIVE.unlink()
            part.replace(WESAD_ARCHIVE)
            return verify_wesad(WESAD_ARCHIVE)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"WESAD acquisition failed: {last_error}")

def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print()
    print("MATH-SYNC exact public-source acquisition")
    print("Scientific processing performed: NO")
    print()

    frozen = load_frozen_harvard_expectations()
    remote = fetch_harvard_metadata()
    matched = match_harvard_manifest(frozen, remote)

    print(f"Frozen Harvard source records recovered: {len(frozen)}/10")
    print(f"Harvard archives matched by filename + size + MD5 + unrestricted: {len(matched)}/10")

    harvard = acquire_harvard(matched)
    wesad = acquire_wesad()

    passed = len(harvard) == 10 and wesad["synchronized_participant_pickles"] == 15
    report = {
        "stage": "exact_public_source_acquisition",
        "created_at_utc": utc_now(),
        "scientific_processing_performed": False,
        "harvard": {"doi": HARVARD_DOI, "archives": harvard},
        "wesad": wesad,
        "pass": passed,
        "ready_for_controlled_extraction": passed,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"Harvard exact archives acquired: {len(harvard)}/10")
    print("Harvard MD5 verified: 10/10")
    print("Harvard SHA-256 recorded: 10/10")
    print("Harvard ZIP CRC: PASS")
    print("WESAD archive acquired: YES")
    print("WESAD MD5 recorded: YES")
    print("WESAD SHA-256 recorded: YES")
    print(f"WESAD synchronized participant pickles: {wesad['synchronized_participant_pickles']}/15")
    print("WESAD ZIP CRC: PASS")
    print("Scientific processing performed: NO")
    print()
    print("EXACT-PUBLIC-SOURCE-ACQUISITION: PASS")
    print("Ready for controlled extraction: YES")
    print(f"Report: {REPORT_PATH}")
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
