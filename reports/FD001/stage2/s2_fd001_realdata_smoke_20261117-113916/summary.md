# s2_fd001_realdata_smoke (FD001)

Validation only; official test set not used. `capped` = truth min(RUL, 125); `true` = uncapped RUL.
CV columns: mean ± SD over engine-level folds (folds share training engines, so SDs understate uncertainty).

| Model | Cap | Snapshot RMSE capped (CV) | Snapshot RMSE true (CV) | Snapshot MAE capped (CV) | PHM08/snapshot capped (CV) | Near-EOL RMSE true (CV) | Snapshot RMSE capped (canonical) |
|---|---|---|---|---|---|---|---|
| gru | 125 | — | — | — | — | — | 15.67 |
| hgb_reference | 125 | — | — | — | — | — | 11.26 |
| ridge | 125 | — | — | — | — | — | 14.76 |
| tcn | 125 | — | — | — | — | — | 15.54 |
| transformer | 125 | — | — | — | — | — | 16.18 |
| xgboost | 125 | — | — | — | — | — | 11.52 |

| Model | Params | Bundle MB | Fit s (mean) | Epochs (canonical) | µs/row batch | Single input ms | E2E engine ms | Peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| gru | 15,809 | 0.058 | 6.8 | 2 | 30.0 | 1.43 | 20.3 | 632 |
| hgb_reference | — | 0.482 | 5.3 | — | 10.6 | 10.46 | 53.8 | 283 |
| ridge | 167 | 0.007 | 1.1 | — | 0.2 | 0.16 | 41.2 | 278 |
| tcn | 17,665 | 0.065 | 6.4 | 2 | 38.9 | 1.20 | 16.7 | 586 |
| transformer | 21,857 | 0.081 | 11.8 | 2 | 99.1 | 1.40 | 30.3 | 738 |
| xgboost | — | 0.240 | 6.0 | — | 2.9 | 0.36 | 43.0 | 312 |
