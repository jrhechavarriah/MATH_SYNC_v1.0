from __future__ import annotations

import importlib.util
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACQUISITION_SCRIPT = PROJECT_ROOT / "scripts" / "acquisition" / "acquire_public_sources.py"
REPORT_PATH = PROJECT_ROOT / "reports" / "reproducibility" / "public_source_preflight.json"

def load_acquisition_module():
    spec = importlib.util.spec_from_file_location("math_sync_public_acquisition", ACQUISITION_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def url_reachable(url: str) -> bool:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MATH-SYNC/2.0", "Range": "bytes=0-0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            response.read(1)
        return True
    except Exception:
        return False

def main() -> int:
    module = load_acquisition_module()
    frozen = module.load_frozen_harvard_expectations()
    remote = module.fetch_harvard_metadata()
    matched = module.validate_remote_harvard_manifest(frozen, remote)

    wesad = [
        {"url": url, "reachable": url_reachable(url)}
        for url in module.WESAD_SOURCE_URLS
    ]
    passed = len(matched) == 10 and any(row["reachable"] for row in wesad)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "stage": "public_source_preflight",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "harvard_exact_metadata_matches": len(matched),
                "wesad_endpoints": wesad,
                "downloads_performed": False,
                "pass": passed,
                "ready_for_source_acquisition": passed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("MATH-SYNC public-source preflight")
    print(f"Harvard exact metadata matches: {len(matched)}/10")
    print(f"WESAD public distribution reachable: {'YES' if any(row['reachable'] for row in wesad) else 'NO'}")
    print("Downloads performed: NO")
    print()
    print(f"PUBLIC-SOURCE-PREFLIGHT: {'PASS' if passed else 'FAIL'}")
    print(f"Ready for source acquisition: {'YES' if passed else 'NO'}")
    print(f"Report: {REPORT_PATH}")
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
