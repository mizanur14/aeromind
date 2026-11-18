# s2_fd001_ablation (FD001)

Validation only; official test set not used. `capped` = truth min(RUL, 125); `true` = uncapped RUL.
CV columns: mean ± SD over engine-level folds (folds share training engines, so SDs understate uncertainty).

| Model | Cap | Snapshot RMSE capped (CV) | Snapshot RMSE true (CV) | Snapshot MAE capped (CV) | PHM08/snapshot capped (CV) | Near-EOL RMSE true (CV) | Snapshot RMSE capped (canonical) |
|---|---|---|---|---|---|---|---|
| gru | 125.0 | 11.91 ± 1.73 | 41.12 ± 2.90 | 8.54 ± 1.06 | 2.89 ± 1.72 | 4.28 ± 0.51 | — |
| gru__sensors_literature14 | 125.0 | 12.44 ± 1.44 | 41.04 ± 3.28 | 8.54 ± 0.75 | 3.59 ± 2.58 | 3.98 ± 0.55 | — |
| gru__sensors_trend12 | 125.0 | 12.40 ± 1.94 | 40.89 ± 3.20 | 8.81 ± 1.29 | 2.96 ± 1.74 | 4.68 ± 0.24 | — |
| gru__window20 | 125.0 | 13.79 ± 1.68 | 41.49 ± 2.81 | 9.60 ± 1.31 | 4.45 ± 2.52 | 5.34 ± 0.81 | — |
| gru__window50 | 125.0 | 10.13 ± 1.09 | 40.17 ± 3.55 | 7.22 ± 0.85 | 1.49 ± 0.26 | 3.62 ± 0.37 | — |
| gru__with_settings | 125.0 | 12.88 ± 1.71 | 41.66 ± 3.11 | 9.28 ± 1.19 | 3.36 ± 1.83 | 4.18 ± 0.38 | — |
| gru__capacity_small | 125.0 | 14.94 ± 1.56 | 42.26 ± 3.80 | 10.34 ± 0.96 | 6.27 ± 2.80 | 6.71 ± 3.80 | — |
| gru__capacity_large | 125.0 | 12.82 ± 2.75 | 40.65 ± 2.89 | 8.43 ± 1.35 | 5.91 ± 6.80 | 4.19 ± 0.39 | — |
| gru__engineered_inputs | 125.0 | 14.15 ± 0.39 | 42.15 ± 3.37 | 10.25 ± 0.31 | 3.71 ± 0.18 | 5.71 ± 0.38 | — |
| gru__uncapped_target | nan | 32.38 ± 4.88 | 33.28 ± 3.28 | 23.75 ± 4.50 | 329.73 ± 184.96 | 5.22 ± 0.71 | — |
| gru__loss_mae | 125.0 | 11.99 ± 1.51 | 41.44 ± 3.47 | 8.58 ± 0.83 | 2.85 ± 1.83 | 4.00 ± 0.25 | — |
| gru__loss_mse | 125.0 | 12.13 ± 1.38 | 40.88 ± 3.55 | 8.58 ± 0.74 | 3.36 ± 2.39 | 4.65 ± 0.54 | — |
| xgboost | 125.0 | 12.66 ± 1.10 | 40.83 ± 2.92 | 8.85 ± 0.80 | 3.32 ± 1.22 | 4.08 ± 0.18 | — |
| xgboost__sensors_literature14 | 125.0 | 12.86 ± 1.04 | 40.89 ± 2.69 | 9.00 ± 0.79 | 3.49 ± 1.25 | 4.22 ± 0.23 | — |
| xgboost__sensors_trend12 | 125.0 | 13.45 ± 2.39 | 41.68 ± 2.89 | 9.72 ± 1.54 | 3.87 ± 2.28 | 4.88 ± 0.46 | — |
| xgboost__window20 | 125.0 | 14.65 ± 0.89 | 41.59 ± 2.77 | 10.20 ± 0.70 | 4.94 ± 1.83 | 6.17 ± 0.83 | — |
| xgboost__window50 | 125.0 | 9.90 ± 0.36 | 39.86 ± 3.12 | 7.04 ± 0.26 | 1.57 ± 0.05 | 3.65 ± 0.80 | — |
| xgboost__with_settings | 125.0 | 12.70 ± 1.28 | 41.03 ± 2.94 | 8.94 ± 0.71 | 3.49 ± 1.69 | 4.29 ± 0.40 | — |
| xgboost__capacity_small | 125.0 | 12.41 ± 1.16 | 40.64 ± 2.96 | 8.80 ± 0.79 | 3.25 ± 1.45 | 4.78 ± 0.76 | — |
| xgboost__capacity_large | 125.0 | 13.25 ± 1.08 | 41.32 ± 2.93 | 9.23 ± 0.74 | 3.84 ± 1.41 | 4.43 ± 0.20 | — |
| xgboost__raw_inputs | 125.0 | 17.10 ± 1.81 | 42.63 ± 2.71 | 11.55 ± 0.93 | 8.37 ± 2.83 | 10.22 ± 1.81 | — |
| xgboost__uncapped_target | nan | 33.92 ± 3.03 | 32.64 ± 3.90 | 24.63 ± 2.01 | 434.89 ± 167.67 | 4.40 ± 0.54 | — |
| xgboost__loss_mae | 125.0 | 12.83 ± 1.03 | 40.72 ± 2.89 | 8.72 ± 0.78 | 3.77 ± 1.22 | 4.43 ± 0.37 | — |

| Model | Params | Bundle MB | Fit s (mean) | Epochs (canonical) | µs/row batch | Single input ms | E2E engine ms | Peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| gru | — | — | 57.1 | — | — | — | — | 649 |
| gru__sensors_literature14 | — | — | 55.6 | — | — | — | — | 653 |
| gru__sensors_trend12 | — | — | 65.1 | — | — | — | — | 650 |
| gru__window20 | — | — | 29.6 | — | — | — | — | 605 |
| gru__window50 | — | — | 154.9 | — | — | — | — | 739 |
| gru__with_settings | — | — | 44.9 | — | — | — | — | 651 |
| gru__capacity_small | — | — | 27.1 | — | — | — | — | 543 |
| gru__capacity_large | — | — | 286.9 | — | — | — | — | 866 |
| gru__engineered_inputs | — | — | 44.0 | — | — | — | — | 730 |
| gru__uncapped_target | — | — | 46.4 | — | — | — | — | 659 |
| gru__loss_mae | — | — | 68.5 | — | — | — | — | 650 |
| gru__loss_mse | — | — | 59.3 | — | — | — | — | 651 |
| xgboost | — | — | 4.4 | — | — | — | — | 330 |
| xgboost__sensors_literature14 | — | — | 4.1 | — | — | — | — | 330 |
| xgboost__sensors_trend12 | — | — | 4.0 | — | — | — | — | 311 |
| xgboost__window20 | — | — | 4.4 | — | — | — | — | 330 |
| xgboost__window50 | — | — | 4.7 | — | — | — | — | 331 |
| xgboost__with_settings | — | — | 5.1 | — | — | — | — | 344 |
| xgboost__capacity_small | — | — | 2.2 | — | — | — | — | 328 |
| xgboost__capacity_large | — | — | 14.9 | — | — | — | — | 467 |
| xgboost__raw_inputs | — | — | 0.5 | — | — | — | — | 241 |
| xgboost__uncapped_target | — | — | 4.3 | — | — | — | — | 332 |
| xgboost__loss_mae | — | — | 4.2 | — | — | — | — | 330 |
