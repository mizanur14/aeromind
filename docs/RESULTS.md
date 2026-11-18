# Stage-1 results — FD001

All numbers are copied from saved run files. Hardware: Intel i7-1165G7 (4C/8T), 16 GB RAM,
CPU only, Python 3.12.

- Primary run: `reports/FD001/runs/fd001_baseline_20261117-105759/` (cap 125)
- Secondary run: `reports/FD001/runs/fd001_uncapped_20261117-110014/` (uncapped target)
- Code state: the working tree on top of commit `1b25c6c` (recorded as `1b25c6c-dirty`, uncommitted Stage-1 changes)

## Validation (15 engines, 150 snapshots), primary run, cap = 125

Best grid point per family. RMSE is in cycles; `capped` truth = min(RUL, 125).

| Model | Snapshot RMSE (capped = target) | Snapshot RMSE (true) | Engine-weighted all-window RMSE (capped) | Near-EOL RMSE (true RUL ≤ 30) | Fit (s) | Single-row predict (ms) |
|---|---|---|---|---|---|---|
| **HistGradientBoosting** (lr 0.05, 15 leaves) | **11.26** | 40.90 | 12.88 | 4.41 | 2.4 | 2.68 |
| XGBoost (depth 5) | 11.52 | 40.98 | 12.67 | 4.23 | 3.6 | 0.21 |
| Random Forest (depth None, leaf 20) | 11.55 | 41.27 | 12.96 | 5.08 | 23.3 | 22.30 |
| Ridge (α 0.1) | 14.76 | 43.81 | 14.98 | 12.61 | 0.02 | 0.14 |
| Mean residual life (age only) | 25.24 | 47.90 | 25.94 | 35.63 | 0.00 | 0.01 |
| Constant median | 41.64 | 62.98 | 44.69 | 87.46 | 0.00 | 0.01 |

The training run took 124 s wall-clock (16 fits plus latency measurement). Peak process RSS
was 380 MiB. Saved bundle sizes range from 0.006 MiB (baselines) to 2.9 MiB (Random Forest).
Latencies are medians over repeats; the laptop throttles, so treat them as indicative.

## Official NASA test (100 engines, last observed cycle) — primary run

| Model | Role | RMSE (true) | MAE (true) | Bias (true) | PHM08 sum (true) | RMSE (capped) |
|---|---|---|---|---|---|---|
| **HistGradientBoosting** | **selected on validation** | **13.60** | **9.64** | −0.09 | **288.6** | 12.33 |
| XGBoost | post hoc | 13.76 | 9.62 | 0.19 | 301.5 | 12.53 |
| Random Forest | post hoc | 14.19 | 10.47 | 1.42 | 301.5 | 12.93 |
| Ridge | post hoc | 17.80 | 14.43 | −0.74 | 454.4 | 16.33 |
| Mean residual life | post hoc | 31.95 | 25.98 | −0.94 | 4077.3 | 30.91 |
| Constant median | post hoc | 49.28 | 38.74 | 26.48 | 150743.4 | 48.63 |

Bias is `pred − true`; positive means late (optimistic) predictions, which PHM08 penalises more.

## Capped vs uncapped training target (validation only)

| Run | Selected | Snapshot RMSE (true) | Snapshot RMSE (capped) | Near-EOL RMSE (true) |
|---|---|---|---|---|
| cap 125 | HistGradientBoosting | 40.90 | **11.26** | 4.41 |
| uncapped | Random Forest | **32.35** | 35.68 | 4.87 |

Training uncapped reduces error on early-life snapshots, where true RUL is far above 125.
It is much worse on the capped scale, and slightly worse near end of life. The secondary
run was **not** evaluated on the official test set, to keep the test set out of choices
between target formulations.

## Observations

- Learned models beat the age-only baseline by more than 2× on capped validation RMSE.
  Sensors carry substantial information beyond age.
- The three tree ensembles are statistically indistinguishable on 15 validation engines.
- Errors are largest for true RUL 125–400 (capped-model saturation) and smallest near failure
  (`validation_selected_error_by_rul_bin_*.csv`).
