from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from mathsync.g12_reference_stream import solve_reference_trial
from mathsync.legacy_v1_synthetic import (
    N_EVAL_SAMPLES_PER_SENSOR,
    SCENARIOS,
    SEED,
    simulate_anchor_observations,
)


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "config" / "G12C_REFERENCE_STREAM_SPEC.json"
OUT = ROOT / "results" / "g12" / "g12c"
REPORTS = ROOT / "reports" / "g12"


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8-sig"))

    if spec.get("status") != "FROZEN_BEFORE_G12C_OUTCOME_INSPECTION":
        raise RuntimeError("G12-C specification is not frozen.")

    references = list(spec["candidate_references"])
    conditions = list(spec["conditions"])
    n_replicates = int(spec["base_configuration"]["replicates"])

    if conditions != [
        "reference_clean",
        "reference_systematically_contaminated",
    ]:
        raise RuntimeError("Unexpected G12-C condition order.")

    cfg = dict(SCENARIOS["clean"])
    expected = int(spec["expected_design"]["expected_trials"])

    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    rows = []
    start = time.perf_counter()
    seed_base = SEED + 3_000_000
    completed = 0

    for replicate in range(n_replicates):
        seed = seed_base + replicate

        _, observations, true_clock = simulate_anchor_observations(
            seed=seed, **cfg
        )

        for reference in references:
            for condition in conditions:
                contaminated = (
                    condition
                    == "reference_systematically_contaminated"
                )

                row = solve_reference_trial(
                    observations=observations,
                    true_clock=true_clock,
                    reference_name=reference,
                    contaminated=contaminated,
                    seed=seed,
                    duration=cfg["duration"],
                    jitter_ms=cfg["jitter_ms"],
                    n_eval_samples=N_EVAL_SAMPLES_PER_SENSOR,
                )
                row["replicate"] = replicate
                rows.append(row)
                completed += 1

                if completed % 50 == 0 or completed == expected:
                    print(
                        f"G12-C progress: {completed}/{expected} "
                        f"({100.0 * completed / expected:.1f}%)",
                        flush=True,
                    )

    results = pd.DataFrame(rows)

    if len(results) != expected:
        raise RuntimeError(
            f"Expected {expected} rows; observed {len(results)}."
        )

    cell_counts = results.groupby(["reference", "condition"]).size()
    if not (cell_counts == n_replicates).all():
        raise RuntimeError(
            "G12-C reference-condition cell counts are not balanced."
        )

    results_path = OUT / "G12C_reference_stream_results.csv"
    results.to_csv(results_path, index=False, float_format="%.15g")

    elapsed = time.perf_counter() - start

    audit = {
        "gate": "G12C_EXECUTION_AUDIT",
        "rows": int(len(results)),
        "expected_rows": expected,
        "references": len(references),
        "conditions": len(conditions),
        "replicates": n_replicates,
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

    audit_path = REPORTS / "G12C_execution_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
