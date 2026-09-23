# Reproducibility

The public-data validation uses fixed public dataset definitions, a fixed Harvard cohort,
all 15 WESAD participants, fixed random seeds, fixed clean/nominal/stress scenarios,
50 paired replicates per scenario, fixed Huber tuning, deterministic native-timestamp
selection, participant-level inference, paired Wilcoxon tests with Holm adjustment, and
a deterministic 20,000-replicate participant bootstrap.

Harvard source archives are checked against frozen filename, byte-size, and MD5 metadata.
SHA-256 values are recorded locally and ZIP CRC validation is required. WESAD must contain
all 15 synchronized participant pickle files and pass ZIP CRC validation.

The controlled S1 benchmark can be regenerated with
`scripts/synthetic/run_s1_benchmark.py`. Manuscript Tables 1–4, Figures 1–7,
Supplementary Table S1, and Supplementary Figures S1–S4 can be regenerated with
`scripts/reporting/generate_manuscript_outputs.py` after the required frozen source
outputs are available.

Raw source archives, extracted raw data, and participant caches are not committed.
