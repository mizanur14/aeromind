# Experiment results index

Every reported number comes from one of these runs.
- Small reports (CSV/JSON/PNG/MD) are committed under `reports/`.
- Model bundles and checkpoints stay in the local `artifacts/` directory and are not
  committed.
- The dashboard reads the same paths through `configs/stage3/results_index.yaml`.

| Stage | Experiment | Evidence | Report directory | Summary |
|---|---|---|---|---|
| 1 | FD001 baselines, capped target (primary) | canonical validation + **official test (once)** | `reports/FD001/runs/fd001_baseline_20261117-105759/` | [RESULTS.md](RESULTS.md) |
| 1 | FD001 baselines, uncapped target | canonical validation | `reports/FD001/runs/fd001_uncapped_20261117-110014/` | [RESULTS.md](RESULTS.md) |
| 1 | Integrity audit | — | `reports/FD001/audit/stage1_audit.json` | [STAGE1_AUDIT.md](STAGE1_AUDIT.md) |
| 1 | EDA (pool and train scopes) | dataset | `reports/FD001/eda/` | — |
| 2 | FD001 comparison: Ridge, XGBoost, HGB, TCN, GRU, Transformer | 5-fold CV + canonical | `reports/FD001/stage2/s2_fd001_comparison_20261117-114027/` | [STAGE2_RESULTS.md §1](STAGE2_RESULTS.md) |
| 2 | FD001 uncertainty quantification | calibration → validation (and swap) | `reports/FD001/stage2/s2_fd001_uq_20261117-121004/` | [STAGE2_RESULTS.md §2](STAGE2_RESULTS.md) |
| 2 | FD001 ablations | CV folds cv0–cv2 | `reports/FD001/stage2/s2_fd001_ablation_20261117-121310/` | [STAGE2_RESULTS.md §3](STAGE2_RESULTS.md) |
| 2 | FD001 robustness and FD001 → FD003 shift | canonical validation | `reports/FD001/stage2/s2_fd001_robustness_20261117-120738/` | [STAGE2_RESULTS.md §4](STAGE2_RESULTS.md) |
| 2 | FD002 / FD003 / FD004 benchmarks | canonical validation (one fold) | `reports/FD00{2,3,4}/stage2/s2_fd00x_benchmark_*/` | [STAGE2_RESULTS.md §5](STAGE2_RESULTS.md) |
| 3 | Model registry (22 models, 4 calibrated) | metadata, copied from the reports above | `reports/registry/model_registry.json` | [MODEL_CARDS.md](MODEL_CARDS.md) |
| 3 | Serving benchmark (CPU) | latency | `reports/benchmarks/s3_benchmark_20261117-135548/` (and `latest/`) | [CPU_BENCHMARKING.md](CPU_BENCHMARKING.md) |

Superseded or diagnostic runs:
- `s2_fd001_uq_20261117-120505` is the first UQ run, without `hgb_quantile_uncapped_fit`;
  it is kept for traceability.
- `s2_fd001_realdata_smoke_*` is a 2-epoch smoke run, not a result.

**Official test set:** one evaluation in total, recorded in
`reports/test_set_ledger.json`. No Stage-2 or Stage-3 model has been evaluated on it.

**Not executed** (see [ROADMAP.md](ROADMAP.md)):
- Multi-seed deep runs.
- Ablations on folds cv3–cv4.
- Nested tuning.
- CV and UQ on FD002–FD004.
- Conditional-coverage methods.
