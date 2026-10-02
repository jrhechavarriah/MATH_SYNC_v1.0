from __future__ import annotations

# G12-B runner. Do not execute until G12-B tests pass and implementation is frozen.

import json
import time
from pathlib import Path

import pandas as pd

from mathsync.g12_anchor_geometry import evaluate_geometry_trial
from mathsync.legacy_v1_synthetic import (
    N_EVAL_SAMPLES_PER_SENSOR,
    SCENARIOS,
    SEED,
)


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "config" / "G12B_ANCHOR_GEOMETRY_SPEC.json"
OUT = ROOT / "results" / "g12" / "g12b"
REPORTS = ROOT / "reports" / "g12"


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8-sig"))
    if spec.get("status") != "FROZEN_BEFORE_G12B_OUTCOME_INSPECTION":
        raise RuntimeError("G12-B specification is not frozen.")

    anchor_counts = [int(x) for x in spec["design"]["anchor_counts"]]
    spans = [float(x) for x in spec["design"]["temporal_span_fractions"]]
    scenarios = list(spec["design"]["scenarios"])
    n_rep = int(spec["design"]["replicates_per_configuration"])

    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    rows = []
    start = time.perf_counter()

    # Frozen implementation-detail seed mapping:
    # the same replicate seed is reused across all geometries within a scenario.
    # This is fixed before outcome inspection.
    scenario_seed_bases = {
        scenario: SEED + 2_000_000 + idx * 10_000
        for idx, scenario in enumerate(scenarios)
    }

    total = len(scenarios) * len(anchor_counts) * len(spans) * n_rep
    completed = 0

    for scenario in scenarios:
        cfg = dict(SCENARIOS[scenario])
        for n_anchors in anchor_counts:
            for span in spans:
                for replicate in range(n_rep):
                    seed = scenario_seed_bases[scenario] + replicate
                    row = evaluate_geometry_trial(
                        seed=seed,
                        scenario=scenario,
                        config=cfg,
                        n_anchors=n_anchors,
                        span_fraction=span,
                        n_eval_samples=N_EVAL_SAMPLES_PER_SENSOR,
                    )
                    row["replicate"] = replicate
                    rows.append(row)
                    completed += 1

                    if completed % 100 == 0 or completed == total:
                        print(
                            f"G12-B progress: {completed}/{total} "
                            f"({100.0 * completed / total:.1f}%)",
                            flush=True,
                        )

    results = pd.DataFrame(rows)

    expected = int(spec["design"]["expected_trials"])
    if len(results) != expected:
        raise RuntimeError(
            f"Expected {expected} G12-B rows; observed {len(results)}."
        )

    results_path = OUT / "G12B_anchor_geometry_results.csv"
    results.to_csv(
        results_path,
        index=False,
        float_format="%.15g",
    )

    elapsed = time.perf_counter() - start

    audit = {
        "gate": "G12B_EXECUTION_AUDIT",
        "rows": int(len(results)),
        "expected_rows": expected,
        "identifiable_rows": int(results["identifiable"].sum()),
        "nonidentifiable_rows": int((~results["identifiable"]).sum()),
        "converged_rows": int(results["converged"].sum()),
        "solver_failure_rows": int(results["solver_failure"].sum()),
        "status_counts": {
            str(k): int(v)
            for k, v in results["status"].value_counts(dropna=False).items()
        },
        "elapsed_seconds": float(elapsed),
        "results_file": str(results_path),
        "public_release_modified": False,
    }

    audit_path = REPORTS / "G12B_execution_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
