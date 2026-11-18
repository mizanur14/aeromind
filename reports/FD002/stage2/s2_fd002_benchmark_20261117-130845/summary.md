# s2_fd002_benchmark (FD002)

Validation only; official test set not used. `capped` = truth min(RUL, 125); `true` = uncapped RUL.
CV columns: mean ± SD over engine-level folds (folds share training engines, so SDs understate uncertainty).

| Model | Cap | Snapshot RMSE capped (CV) | Snapshot RMSE true (CV) | Snapshot MAE capped (CV) | PHM08/snapshot capped (CV) | Near-EOL RMSE true (CV) | Snapshot RMSE capped (canonical) |
|---|---|---|---|---|---|---|---|
| ridge | 125 | — | — | — | — | — | 16.50 |
| xgboost | 125 | — | — | — | — | — | 13.41 |
| hgb_reference | 125 | — | — | — | — | — | 12.91 |
| gru | 125 | — | — | — | — | — | 14.75 |
| transformer | 125 | — | — | — | — | — | 13.47 |

| Model | Params | Bundle MB | Fit s (mean) | Epochs (canonical) | µs/row batch | Single input ms | E2E engine ms | Peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| ridge | 233 | 0.034 | 3.1 | — | 0.2 | 0.13 | 41.6 | 505 |
| xgboost | — | 0.269 | 10.3 | — | 1.4 | 0.32 | 42.6 | 491 |
| hgb_reference | — | 0.579 | 9.3 | — | 5.2 | 6.35 | 48.8 | 484 |
| gru | 16,961 | 0.088 | 159.6 | 25 | 20.9 | 1.17 | 21.0 | 802 |
| transformer | 22,049 | 0.107 | 213.6 | 21 | 73.9 | 1.40 | 33.2 | 922 |
