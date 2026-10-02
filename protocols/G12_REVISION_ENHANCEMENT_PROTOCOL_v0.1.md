# G12 Revision Enhancement Protocol v0.1
## MATH-SYNC reviewer-driven robustness analyses

**Status:** Frozen design specification before inspection of G12 outcomes  
**Purpose:** Address reviewer requests with prespecified computational analyses without changing the previously frozen Harvard/WESAD population-level results.

## 1. Scientific principle

G12 is a reviewer-driven robustness extension. It does not replace or retune the existing MATH-SYNC validation architecture. The previously reported estimator settings, contamination scenarios, participant cohorts, primary outcome definition, native-rate processing, and participant-level inferential unit remain unchanged.

No G12 factor level, estimator, metric, or reporting rule may be added, removed, or changed after outcome inspection without being documented as a post-freeze deviation.

## 2. Frozen elements inherited from the manuscript

- Huber tuning constant: c = 1.345
- Robust scale: MAD-based
- Maximum IRLS iterations: 30
- Existing clean / nominal / stress contamination definitions: unchanged
- Existing seed family and deterministic replay rules: unchanged
- Paired Monte Carlo design: 50 replicates per condition
- Native-rate timestamp representation: unchanged
- Equal-stream macro aggregation: unchanged
- Primary reconstruction outcome: macro-RMSE
- Harvard cohort: unchanged
- WESAD cohort: unchanged
- Population-level inferential unit: participant
- Existing manuscript results: immutable
- No tuning of an estimator to Harvard, WESAD, or a specific contamination scenario

## 3. G12-A — Alternative robust-estimator benchmark

### Objective
Evaluate whether the contamination-response pattern remains when Joint Huber is compared with representative robust estimators under the same affine synchronization problem.

### Estimators
1. Joint OLS
2. Joint Huber
3. Joint LAD / L1
4. Joint Tukey biweight
5. Joint RANSAC — conditional feasibility arm

### RANSAC feasibility rule
RANSAC may be included only if a mathematically consistent implementation can operate on the same joint latent-event affine system without changing the data-generating process, anchor definitions, or evaluation target. Feasibility must be established by unit tests before G12 outcome tables are inspected. If this condition fails, RANSAC is reported as not implemented for formulation-compatibility reasons; it must not be replaced post hoc by another estimator.

### Scenarios
- clean
- nominal
- stress

### Replication
- 50 paired realizations per condition
- identical generated samples across estimators

### Primary metric
- macro-RMSE

### Secondary metrics
- MAE
- 95th-percentile absolute error
- clock-rate error
- offset error
- convergence/failure indicator
- runtime

### Interpretation rule
The purpose is comparative robustness characterization, not proof that Huber is universally superior.

## 4. G12-B — Anchor geometry sensitivity

### Objective
Empirically connect anchor geometry with rank, smallest singular value, numerical conditioning, and reconstruction error.

### Anchor-count levels
- 4
- 6
- 8
- 12
- 16
- 24

### Temporal-span levels
- 25%
- 50%
- 100%

Anchors remain irregularly distributed inside the selected temporal span.

### Contamination scenarios
- clean
- nominal
- stress

### Replication
- 50 paired realizations per configuration

### Recorded quantities
- rank(A)
- sigma_min(W^(1/2) A)
- condition number
- macro-RMSE
- MAE
- clock-rate error
- offset error
- identifiable yes/no
- solver failure yes/no

### Planned summaries
- RMSE versus sigma_min
- anchor-count × temporal-span response surface / heatmap
- non-identifiability or failure frequency by geometry

## 5. G12-C — Reference-stream sensitivity

### Objective
Distinguish gauge/reference choice from degradation caused by systematic contamination of the selected reference stream.

### Candidate references
All eligible synthetic streams:
- event/reference stream (~60 Hz)
- EEG (250 Hz)
- PPG (64 Hz)
- EDA (32 Hz)
- IMU (100 Hz)
- HMD (90 Hz)

### C1. Reference-choice analysis
Repeat the same controlled synchronization configuration under each eligible reference.

### C2. Reference-contamination analysis
For every eligible reference, compare:
- reference clean
- reference systematically contaminated

### Canonical comparison rule
All recovered timelines must be transformed to the same canonical temporal coordinate before cross-reference error comparisons.

### Recorded quantities
- macro-RMSE
- MAE
- clock-rate error
- offset error
- sigma_min
- condition number
- iterations
- convergence/failure indicator

## 6. G12-D — IRLS convergence, runtime, and scaling

### D1. Convergence diagnostics
Record:
- iterations to convergence
- converged yes/no
- final relative parameter change
- final objective
- macro-RMSE
- runtime

### D2. Tolerance sensitivity
Frozen tolerance levels:
- 1e-4
- 1e-6
- 1e-8

All other Huber settings remain fixed.

### D3. Synthetic scaling
Stream counts:
- 5
- 10
- 20
- 40

Anchor counts:
- 8
- 16
- 32
- 64
- 128

Record:
- number of observations
- number of unknowns
- runtime
- iterations
- convergence/failure
- macro-RMSE
- optional peak memory if measured consistently

### Scope rule
No real-time or arbitrary large-scale performance claim will be made. Results characterize only the evaluated synthetic system sizes.

## 7. Analysis and reporting rules

1. Existing Harvard/WESAD results are not recomputed for selective replacement.
2. G12 analyses are reported separately as reviewer-driven robustness analyses.
3. Failed runs remain in the audit trail.
4. No estimator-specific seed selection is permitted.
5. No removal of unfavorable valid outcomes is permitted.
6. Primary figures and tables are generated programmatically from frozen result files.
7. All final G12 outputs receive SHA-256 checksums.
8. Any deviation from this protocol requires a dated `G12_DEVIATIONS.md` entry before rerunning affected analyses.

## 8. Planned manuscript integration

### Methods
Add `5.8 Reviewer-driven robustness analyses` with:
- 5.8.1 Alternative robust estimators
- 5.8.2 Anchor-geometry sensitivity
- 5.8.3 Reference-stream sensitivity
- 5.8.4 Convergence and computational scaling

### Results
Report only prespecified central findings in the main manuscript.

### Supplementary material
- Table S2 — Alternative estimator benchmark
- Table S3 — Anchor geometry sensitivity
- Table S4 — Reference-stream sensitivity
- Table S5 — Convergence/runtime/scaling
- supplementary figures for the principal sensitivity relationships

## 9. Reproducibility gate

Before scientific interpretation:
- configuration file parses successfully
- unit tests pass
- existing frozen MATH-SYNC tests pass
- deterministic rerun test passes
- output schemas validate
- manuscript tables/figures are generated only from frozen result artifacts

## 10. Freeze declaration

This protocol is intended to be committed before G12 scientific outcomes are inspected.

Suggested Git commit message:

`freeze G12 reviewer-driven robustness protocol`
