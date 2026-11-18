# s2_fd004_benchmark (FD004)

Validation only; official test set not used. `capped` = truth min(RUL, 125); `true` = uncapped RUL.
CV columns: mean ± SD over engine-level folds (folds share training engines, so SDs understate uncertainty).

| Model | Cap | Snapshot RMSE capped (CV) | Snapshot RMSE true (CV) | Snapshot MAE capped (CV) | PHM08/snapshot capped (CV) | Near-EOL RMSE true (CV) | Snapshot RMSE capped (canonical) |
|---|---|---|---|---|---|---|---|
| ridge | 125 | — | — | — | — | — | 17.52 |
| xgboost | 125 | — | — | — | — | — | 14.42 |
| hgb_reference | 125 | — | — | — | — | — | 14.22 |
| gru | 125 | — | — | — | — | — | 15.55 |
| transformer | 125 | — | — | — | — | — | 14.81 |

| Model | Params | Bundle MB | Fit s (mean) | Epochs (canonical) | µs/row batch | Single input ms | E2E engine ms | Peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| ridge | 233 | 0.037 | 2.8 | — | 0.2 | 0.13 | 40.2 | 581 |
| xgboost | — | 0.272 | 9.9 | — | 1.3 | 0.24 | 41.6 | 532 |
| hgb_reference | — | 0.591 | 9.4 | — | 4.9 | 6.88 | 47.7 | 532 |
| gru | 16,961 | 0.091 | 107.0 | 15 | 22.9 | 1.20 | 23.4 | 822 |
| transformer | 22,049 | 0.111 | 318.1 | 24 | 75.0 | 1.36 | 37.4 | 946 |
