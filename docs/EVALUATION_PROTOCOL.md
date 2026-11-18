# Evaluation protocol

This document is the single reference for how every AeroMind number is produced. Details
live in [METHODOLOGY.md](METHODOLOGY.md) (Stage 1) and
[STAGE2_METHODOLOGY.md](STAGE2_METHODOLOGY.md) (Stage 2).

## Three kinds of evidence (never mixed)

| Evidence | Engines | Used for | Where it appears |
|---|---|---|---|
| **Cross-validation** | 85 development engines (FD001), 5 engine-level folds of 17 | Model comparison, ablations | Stage-2 reports, registry `cross_validation` |
| **Canonical validation** | 15 fixed validation engines | Stage-1 model selection; single-split benchmarks (FD002–FD004) | registry `canonical_validation` |
| **Calibration** | 15 fixed calibration engines | Only conformal thresholds (and the role-swap check) | registry `intervals` |
| **Official NASA test** | 100 FD001 test engines | One final evaluation of the Stage-1 selected model | `reports/test_set_ledger.json`, registry `official_test` |

The registry, the API and the dashboard label each metric with its evidence type. Test
metrics come only from the ledger; the registry build never reads test labels.

## Partitions

- Splits are engine-level, seeded (`data/manifests/fd00x_split_seed42.json`) and 70/15/15
  train / validation / calibration.
- An engine belongs to exactly one partition. Engine IDs are only meaningful within one
  subset.
- All fitted preprocessing (constant-sensor removal, condition clustering, scaler) is fitted
  on training engines only, and saved inside the model bundle.

## Targets and metrics

- Training target: `min(RUL, 125)`. Every metric is reported against capped truth
  (`_capped`) and against true RUL (`_true`).
- **Snapshots:** 10 seeded cut cycles per validation engine, one prediction each. Each
  engine counts equally, and the snapshots are identical for every model in a fold.
- Metrics:
  - MAE and RMSE.
  - PHM08 score with `d = pred − true`: `exp(−d/13) − 1` if `d < 0`, else `exp(d/10) − 1`.
    Reported as a mean per snapshot for validation, and as a sum over 100 engines for the
    official test, matching the challenge convention.
  - Near-end-of-life (near-EOL) RMSE: windows with true RUL ≤ 30.
- **Official test:** one prediction per test engine at its last observed cycle. This differs
  from the validation snapshot protocol, so the two are not comparable column-for-column.

## Variability and claims

- CV results are mean ± SD over folds, plus paired per-fold differences against the HGB
  reference.
- Folds share training engines, so the SD understates the real uncertainty. No significance
  tests are claimed.
- A model is not called better when the difference is within fold variability.
- Ablations use 3 folds and one seed; robustness and benchmarks use one fold. These are
  labelled as such everywhere.

## Prediction intervals

- **Method:** engine-level split conformal on absolute residuals against capped truth.
  One seeded random snapshot is drawn per calibration engine (n = 15), and the threshold is
  the ⌈(n+1)(1−α)⌉-th smallest score.
- **Guarantee:** marginal coverage ≥ 1−α for a snapshot drawn the same way from a new engine
  exchangeable with the calibration engines.
  - Not conditional on RUL range or on an individual engine.
  - Not valid under distribution shift.
  - Levels above n/(n+1) ≈ 0.94 are impossible.
- **Validation:** the same procedure, repeated over 500 draws in Stage 2, gave a mean
  coverage of 0.94–0.95 at 90 % on validation engines, with a 5th percentile of 0.83–0.87.
  The single seeded draw used by the served calibrator is one such draw.

## Serving protocol (Stage 3)

- A served prediction is computed by the registered bundle from the raw cycles in the
  request, using the bundle's own fitted pipeline.
- Served predictions are tested to equal the predictions saved when the Stage-2 run
  evaluated the same bundle, within 1e-4 cycles.
- The dashboard shows engine-level predictions only for **validation** engines, which were
  not used to fit the model or its calibrator, and for user uploads, which have no ground
  truth.
- The official test files are never read by the API, the dashboard or the registry build.
