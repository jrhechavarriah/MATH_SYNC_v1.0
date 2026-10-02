# Changelog

All notable changes to the public MATH-SYNC computational release are documented here.

## v1.1 — 2026-10-02

### Added

This release extends the frozen MATH-SYNC v1.0 computational package with the
reviewer-driven G12 robustness and diagnostic analyses developed during manuscript
revision.

The original v1.0 validation architecture, results, scripts, and publication outputs
remain unchanged.

Added components include:

- G12-A: comparison of robust estimators under clean, nominal, and stress conditions;
- G12-B: anchor-count and temporal-span sensitivity analysis;
- G12-C: reference-stream sensitivity under clean and systematically contaminated
  reference conditions;
- G12-D: convergence-tolerance and computational-scaling diagnostics;
- Supplementary Tables S2-S5;
- frozen G12 experiment specifications and protocol;
- reusable MATH-SYNC modules supporting the G12 analyses;
- automated G12 tests.

### Reproducibility

The G12 analyses are distributed with:

- deterministic experiment configurations;
- frozen experiment specifications;
- retained failure outcomes;
- raw and summarized CSV results;
- Supplementary Tables S2-S5;
- automated tests;
- SHA-256 release integrity metadata.

The complete v1.1 test suite contains 52 tests:

- 19 inherited v1.0 regression tests;
- 33 G12 tests.

All 52 tests passed in the isolated v1.1 release candidate before publication.

### Provenance

The module `src/mathsync/legacy_v1_synthetic.py` is a modularized compatibility
representation of the synthetic S1 implementation already released publicly in
MATH-SYNC v1.0. Function signatures, experimental constants, and normalized
abstract-syntax-tree representations of the relevant S1 functions were verified
against the public v1.0 implementation before inclusion in v1.1.

### Preservation of v1.0

MATH-SYNC v1.0 remains an immutable public release representing the original
prespecified validation architecture.

The reviewer-driven G12 analyses were conducted subsequently under a separately
frozen protocol and are introduced for the first time in v1.1.

## v1.0 — 2026-09-23

Initial public computational release supporting the primary MATH-SYNC validation,
including controlled synthetic experiments, Harvard public-data schedule replay,
within-participant and participant-level replication, WESAD external replication,
manuscript-aligned figures and tables, automated tests, and integrity manifests.

Zenodo DOI: 10.5281/zenodo.22920440
