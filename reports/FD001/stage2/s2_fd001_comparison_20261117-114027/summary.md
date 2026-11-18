# s2_fd001_comparison (FD001)

Validation only; official test set not used. `capped` = truth min(RUL, 125); `true` = uncapped RUL.
CV columns: mean ± SD over engine-level folds (folds share training engines, so SDs understate uncertainty).

| Model | Cap | Snapshot RMSE capped (CV) | Snapshot RMSE true (CV) | Snapshot MAE capped (CV) | PHM08/snapshot capped (CV) | Near-EOL RMSE true (CV) | Snapshot RMSE capped (canonical) |
|---|---|---|---|---|---|---|---|
| ridge | 125 | 16.65 ± 1.66 | 44.45 ± 5.04 | 12.84 ± 1.56 | 5.00 ± 1.31 | 11.96 ± 2.29 | 14.76 |
| xgboost | 125 | 13.71 ± 1.67 | 42.22 ± 5.07 | 9.64 ± 1.26 | 3.72 ± 1.11 | 4.53 ± 0.64 | 11.52 |
| hgb_reference | 125 | 13.97 ± 1.79 | 42.33 ± 4.93 | 9.85 ± 1.33 | 3.79 ± 1.18 | 4.86 ± 0.70 | 11.26 |
| tcn | 125 | 17.87 ± 2.75 | 44.44 ± 5.85 | 12.63 ± 1.88 | 9.44 ± 6.16 | 8.46 ± 1.47 | 14.07 |
| gru | 125 | 13.47 ± 2.49 | 42.58 ± 5.28 | 9.51 ± 1.59 | 3.68 ± 1.91 | 4.94 ± 1.33 | 13.49 |
| transformer | 125 | 13.37 ± 2.36 | 41.27 ± 5.44 | 8.81 ± 1.50 | 4.82 ± 3.02 | 4.46 ± 1.16 | 13.00 |

Paired per-fold difference in snapshot RMSE (capped) vs `hgb_reference` (negative = better):

| Model | Mean diff | SD | Min | Max | Folds better |
|---|---|---|---|---|---|
| ridge | +2.68 | 0.75 | +1.40 | +3.22 | 0/5 |
| xgboost | -0.26 | 0.20 | -0.59 | -0.05 | 5/5 |
| tcn | +3.90 | 1.52 | +2.56 | +6.48 | 0/5 |
| gru | -0.50 | 0.71 | -1.54 | +0.36 | 4/5 |
| transformer | -0.60 | 1.23 | -2.06 | +1.32 | 4/5 |

| Model | Params | Bundle MB | Fit s (mean) | Epochs (canonical) | µs/row batch | Single input ms | E2E engine ms | Peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| ridge | 167 | 0.007 | 1.4 | — | 0.3 | 0.19 | 62.2 | 307 |
| xgboost | — | 0.240 | 5.3 | — | 3.0 | 0.35 | 45.0 | 333 |
| hgb_reference | — | 0.482 | 4.7 | — | 8.6 | 8.84 | 45.3 | 294 |
| tcn | 17,665 | 0.065 | 43.5 | 29 | 42.6 | 1.39 | 24.9 | 609 |
| gru | 15,809 | 0.059 | 67.5 | 34 | 22.1 | 1.06 | 15.0 | 656 |
| transformer | 21,857 | 0.081 | 108.3 | 21 | 87.4 | 1.29 | 26.5 | 785 |
