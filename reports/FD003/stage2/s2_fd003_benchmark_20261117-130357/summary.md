# s2_fd003_benchmark (FD003)

Validation only; official test set not used. `capped` = truth min(RUL, 125); `true` = uncapped RUL.
CV columns: mean ± SD over engine-level folds (folds share training engines, so SDs understate uncertainty).

| Model | Cap | Snapshot RMSE capped (CV) | Snapshot RMSE true (CV) | Snapshot MAE capped (CV) | PHM08/snapshot capped (CV) | Near-EOL RMSE true (CV) | Snapshot RMSE capped (canonical) |
|---|---|---|---|---|---|---|---|
| ridge | 125 | — | — | — | — | — | 17.18 |
| xgboost | 125 | — | — | — | — | — | 12.86 |
| hgb_reference | 125 | — | — | — | — | — | 12.95 |
| gru | 125 | — | — | — | — | — | 15.51 |
| transformer | 125 | — | — | — | — | — | 11.56 |

| Model | Params | Bundle MB | Fit s (mean) | Epochs (canonical) | µs/row batch | Single input ms | E2E engine ms | Peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| ridge | 178 | 0.007 | 0.9 | — | 0.2 | 0.13 | 34.1 | 304 |
| xgboost | — | 0.238 | 5.0 | — | 1.5 | 0.27 | 37.2 | 334 |
| hgb_reference | — | 0.496 | 3.9 | — | 7.9 | 7.95 | 45.0 | 300 |
| gru | 16,001 | 0.059 | 46.5 | 17 | 22.3 | 1.04 | 17.8 | 662 |
| transformer | 21,889 | 0.081 | 214.6 | 40 | 73.9 | 1.28 | 30.6 | 785 |
