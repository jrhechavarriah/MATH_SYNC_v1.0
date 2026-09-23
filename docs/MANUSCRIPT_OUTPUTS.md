# Manuscript and Supplementary Output Generation

The public release contains both the frozen manuscript-aligned artifacts and the scripts
required to regenerate them.

## Main Figures 1–7

Figures 1–7 are regenerated from the frozen public-data replay outputs:

```bash
python scripts/reporting/generate_manuscript_outputs.py
```

The generator uses the participant-level and within-participant result tables already
stored under `outputs/public_replay/`. It does not rerun or retune the estimator.

## Supplementary Figure S1–S4

Supplementary Figures S1–S4 originate from the controlled S1 Monte Carlo benchmark.
Regenerate the frozen S1 benchmark first:

```bash
python scripts/synthetic/run_s1_benchmark.py
```

Then regenerate the manuscript package:

```bash
python scripts/reporting/generate_manuscript_outputs.py
```

The S1 benchmark uses seed `20260821` and 50 paired replicates per factor level and named
scenario.

## Supplementary Table S1

Supplementary Table S1 is regenerated from the final Harvard and WESAD participant-level
results. The participant is the inferential unit. Deterministic 95% percentile-bootstrap
intervals use 20,000 resamples and seed `20260825`.

## Output directories

```text
outputs/manuscript/tables/
outputs/manuscript/figures/
outputs/manuscript/supplementary/
outputs/manuscript/Figures.zip
outputs/manuscript/Supplementary.zip
outputs/manuscript/RESULTS_FREEZE.json
outputs/manuscript/SHA256SUMS.txt
```

`Figures.zip` contains exactly Figures 1–7. `Supplementary.zip` contains Supplementary
Table S1 and Supplementary Figures S1–S4.

## Frozen manuscript checks

The reporting generator verifies the manuscript-critical values, including:

- S0201 clean Joint Huber relative session difference: `0.2615%` at four decimals.
- WESAD clean Huber-versus-OLS reduction: `-0.4164%` at four decimals.
- WESAD clean Huber median macro-RMSE: `0.106495 ms` at six decimals.

These checks are performed from the stored full-precision computational outputs.
