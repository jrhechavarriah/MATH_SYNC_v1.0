# MATH-SYNC

**Robust and Identifiable Affine-Clock Alignment of Heterogeneous Multirate Sensor Streams**

This repository contains the frozen, reproducible computational pipeline supporting the
MATH-SYNC study.

## Scientific scope

MATH-SYNC estimates a common temporal coordinate for heterogeneous multirate sensor
streams without forcing the signals onto a common sampling grid. The public validation
uses controlled synthetic experiments and semi-synthetic perturbation-and-replay on real
public timing schedules.

## Validation architecture

The computational evidence progresses through:

1. mathematical formulation of affine-clock alignment;
2. controlled synthetic validation with known temporal ground truth;
3. Harvard public-data schedule validation;
4. within-participant Harvard replication;
5. Harvard participant-level replication (`N = 10`);
6. WESAD external replication (`N = 15`);
7. participant-level inference and manuscript-aligned reporting.

## Public data

Raw source datasets are not redistributed in this repository.

- Harvard DOI: `10.7910/DVN/HMZ5RG`
- WESAD DOI: `10.24432/C57K5T`

## Public-data validation

After installing the packages specified by `requirements-lock.txt`, the public-data
workflow is:

```bash
python scripts/acquisition/preflight_public_sources.py
python scripts/acquisition/acquire_public_sources.py
python scripts/acquisition/validate_public_sources.py
python scripts/validation/run_public_data_validation.py
```

The acquisition and structural-validation stages verify source integrity before scientific
replay.

## Controlled S1 benchmark

The synthetic benchmark uses seed `20260821` and 50 paired replicates per factor level and
named scenario:

```bash
python scripts/synthetic/run_s1_benchmark.py
```

## Manuscript figures, tables, and supplementary materials

After the frozen public-data and S1 outputs are available:

```bash
python scripts/reporting/generate_manuscript_outputs.py
```

The generated publication package contains:

- Tables 1–4;
- Figures 1–7;
- Supplementary Table S1;
- Supplementary Figures S1–S4;
- `Figures.zip`;
- `Supplementary.zip`;
- SHA-256 integrity manifests.

The release already includes the frozen manuscript-aligned outputs under
`outputs/manuscript/`.

## Frozen statistical design

Population-level inference uses the participant as the independent unit. The primary
outcome is equal-modality macro-RMSE. Joint Huber is compared with Joint OLS and
Offset-only correction using paired Wilcoxon signed-rank tests, Holm family-wise
adjustment, participant-level win rates, matched-pairs rank-biserial effect sizes, and
deterministic percentile-bootstrap confidence intervals. The publication bootstrap uses
20,000 resamples and seed `20260825`.

## Integrity and data policy

Raw Harvard and WESAD archives, extracted source datasets, and participant caches are not
committed. File-level hashes are recorded in `SHA256SUMS.txt` and
`release_manifest.json`.

See:

- `docs/PUBLIC_DATA_SOURCES.md`
- `docs/REPRODUCIBILITY.md`
- `docs/MANUSCRIPT_OUTPUTS.md`
