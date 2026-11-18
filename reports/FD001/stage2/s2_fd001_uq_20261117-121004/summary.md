Calibration engines: 15; evaluation: validation engines (and role swap).

| Method | Model | Nominal | Calibrate → evaluate | Coverage (snapshots, capped) mean [p05, p95 over draws] | Coverage true RUL | Mean width (cycles) | Finite |
|---|---|---|---|---|---|---|---|
| conformal_pooled | hgb_reference | 0.90 | calibration → validation | 0.933  | 0.660 | 42.5 | 1.00 |
| conformal_engine | hgb_reference | 0.90 | calibration → validation | 0.953 [0.873, 0.987] | 0.696 | 53.9 | 1.00 |
| conformal_pooled | xgboost | 0.90 | calibration → validation | 0.933  | 0.660 | 40.0 | 1.00 |
| conformal_engine | xgboost | 0.90 | calibration → validation | 0.948 [0.873, 1.000] | 0.689 | 52.7 | 1.00 |
| conformal_pooled | gru | 0.90 | calibration → validation | 0.893  | 0.640 | 41.3 | 1.00 |
| conformal_engine | gru | 0.90 | calibration → validation | 0.944 [0.827, 0.993] | 0.704 | 58.3 | 1.00 |
| conformal_pooled | transformer | 0.90 | calibration → validation | 0.887  | 0.640 | 42.1 | 1.00 |
| conformal_engine | transformer | 0.90 | calibration → validation | 0.941 [0.833, 0.993] | 0.708 | 60.5 | 1.00 |
| quantile_raw | hgb_quantile | 0.90 | calibration → validation | 0.913  | 0.593 | 50.5 | 1.00 |
| cqr_pooled | hgb_quantile | 0.90 | calibration → validation | 0.913  | 0.593 | 51.2 | 1.00 |
| cqr_engine | hgb_quantile | 0.90 | calibration → validation | 0.959 [0.913, 1.000] | 0.651 | 60.1 | 1.00 |
| quantile_raw | hgb_quantile_uncapped_fit | 0.90 | calibration → validation | 0.913  | 0.593 | 30.3 | 1.00 |
| cqr_pooled | hgb_quantile_uncapped_fit | 0.90 | calibration → validation | 0.913  | 0.593 | 32.0 | 1.00 |
| cqr_engine | hgb_quantile_uncapped_fit | 0.90 | calibration → validation | 0.960 [0.913, 1.000] | 0.656 | 45.7 | 1.00 |
| quantile_raw | transformer_quantile | 0.90 | calibration → validation | 0.900  | 0.587 | 30.4 | 1.00 |
| cqr_pooled | transformer_quantile | 0.90 | calibration → validation | 0.900  | 0.587 | 30.4 | 1.00 |
| cqr_engine | transformer_quantile | 0.90 | calibration → validation | 0.942 [0.900, 0.993] | 0.642 | 38.8 | 1.00 |
| conformal_pooled | hgb_reference | 0.80 | calibration → validation | 0.827  | 0.547 | 28.7 | 1.00 |
| conformal_engine | hgb_reference | 0.80 | calibration → validation | 0.843 [0.679, 0.953] | 0.571 | 32.0 | 1.00 |
| conformal_pooled | xgboost | 0.80 | calibration → validation | 0.847  | 0.560 | 28.1 | 1.00 |
| conformal_engine | xgboost | 0.80 | calibration → validation | 0.839 [0.633, 0.947] | 0.570 | 30.4 | 1.00 |
| conformal_pooled | gru | 0.80 | calibration → validation | 0.760  | 0.513 | 28.3 | 1.00 |
| conformal_engine | gru | 0.80 | calibration → validation | 0.791 [0.546, 0.953] | 0.541 | 31.7 | 1.00 |
| conformal_pooled | transformer | 0.80 | calibration → validation | 0.780  | 0.520 | 26.4 | 1.00 |
| conformal_engine | transformer | 0.80 | calibration → validation | 0.793 [0.553, 0.947] | 0.537 | 30.8 | 1.00 |
| quantile_raw | hgb_quantile | 0.80 | calibration → validation | 0.793  | 0.493 | 25.0 | 1.00 |
| cqr_pooled | hgb_quantile | 0.80 | calibration → validation | 0.813  | 0.507 | 26.9 | 1.00 |
| cqr_engine | hgb_quantile | 0.80 | calibration → validation | 0.862 [0.793, 0.953] | 0.555 | 28.9 | 1.00 |
| quantile_raw | hgb_quantile_uncapped_fit | 0.80 | calibration → validation | 0.800  | 0.487 | 20.3 | 1.00 |
| cqr_pooled | hgb_quantile_uncapped_fit | 0.80 | calibration → validation | 0.840  | 0.527 | 23.1 | 1.00 |
| cqr_engine | hgb_quantile_uncapped_fit | 0.80 | calibration → validation | 0.862 [0.800, 0.927] | 0.556 | 25.8 | 1.00 |
| quantile_raw | transformer_quantile | 0.80 | calibration → validation | 0.827  | 0.533 | 23.8 | 1.00 |
| cqr_pooled | transformer_quantile | 0.80 | calibration → validation | 0.827  | 0.533 | 23.8 | 1.00 |
| cqr_engine | transformer_quantile | 0.80 | calibration → validation | 0.838 [0.827, 0.900] | 0.543 | 23.8 | 1.00 |
| conformal_pooled | hgb_reference | 0.90 | validation → calibration | 0.927  | 0.700 | 44.6 | 1.00 |
| conformal_engine | hgb_reference | 0.90 | validation → calibration | 0.946 [0.833, 1.000] | 0.736 | 57.6 | 1.00 |
| conformal_pooled | xgboost | 0.90 | validation → calibration | 0.920  | 0.693 | 43.1 | 1.00 |
| conformal_engine | xgboost | 0.90 | validation → calibration | 0.951 [0.840, 1.000] | 0.736 | 58.6 | 1.00 |
| conformal_pooled | gru | 0.90 | validation → calibration | 0.947  | 0.740 | 49.5 | 1.00 |
| conformal_engine | gru | 0.90 | validation → calibration | 0.971 [0.893, 1.000] | 0.784 | 62.5 | 1.00 |
| conformal_pooled | transformer | 0.90 | validation → calibration | 0.913  | 0.700 | 44.2 | 1.00 |
| conformal_engine | transformer | 0.90 | validation → calibration | 0.951 [0.853, 1.000] | 0.759 | 61.4 | 1.00 |
| quantile_raw | hgb_quantile | 0.90 | validation → calibration | 0.893  | 0.587 | 51.6 | 1.00 |
| cqr_pooled | hgb_quantile | 0.90 | validation → calibration | 0.893  | 0.587 | 51.6 | 1.00 |
| cqr_engine | hgb_quantile | 0.90 | validation → calibration | 0.952 [0.893, 0.987] | 0.650 | 58.6 | 1.00 |
| quantile_raw | hgb_quantile_uncapped_fit | 0.90 | validation → calibration | 0.853  | 0.520 | 31.6 | 1.00 |
| cqr_pooled | hgb_quantile_uncapped_fit | 0.90 | validation → calibration | 0.867  | 0.533 | 32.3 | 1.00 |
| cqr_engine | hgb_quantile_uncapped_fit | 0.90 | validation → calibration | 0.945 [0.853, 0.980] | 0.646 | 44.8 | 1.00 |
| quantile_raw | transformer_quantile | 0.90 | validation → calibration | 0.933  | 0.613 | 29.9 | 1.00 |
| cqr_pooled | transformer_quantile | 0.90 | validation → calibration | 0.933  | 0.613 | 29.9 | 1.00 |
| cqr_engine | transformer_quantile | 0.90 | validation → calibration | 0.963 [0.933, 1.000] | 0.669 | 40.0 | 1.00 |
| conformal_pooled | hgb_reference | 0.80 | validation → calibration | 0.793  | 0.547 | 29.8 | 1.00 |
| conformal_engine | hgb_reference | 0.80 | validation → calibration | 0.818 [0.599, 0.953] | 0.588 | 33.7 | 1.00 |
| conformal_pooled | xgboost | 0.80 | validation → calibration | 0.820  | 0.547 | 29.3 | 1.00 |
| conformal_engine | xgboost | 0.80 | validation → calibration | 0.820 [0.580, 0.953] | 0.574 | 33.5 | 1.00 |
| conformal_pooled | gru | 0.80 | validation → calibration | 0.853  | 0.627 | 32.6 | 1.00 |
| conformal_engine | gru | 0.80 | validation → calibration | 0.872 [0.733, 0.980] | 0.649 | 35.2 | 1.00 |
| conformal_pooled | transformer | 0.80 | validation → calibration | 0.833  | 0.573 | 29.4 | 1.00 |
| conformal_engine | transformer | 0.80 | validation → calibration | 0.823 [0.613, 0.953] | 0.589 | 32.1 | 1.00 |
| quantile_raw | hgb_quantile | 0.80 | validation → calibration | 0.680  | 0.407 | 24.3 | 1.00 |
| cqr_pooled | hgb_quantile | 0.80 | validation → calibration | 0.707  | 0.433 | 25.1 | 1.00 |
| cqr_engine | hgb_quantile | 0.80 | validation → calibration | 0.778 [0.680, 0.920] | 0.494 | 27.2 | 1.00 |
| quantile_raw | hgb_quantile_uncapped_fit | 0.80 | validation → calibration | 0.707  | 0.387 | 21.8 | 1.00 |
| cqr_pooled | hgb_quantile_uncapped_fit | 0.80 | validation → calibration | 0.720  | 0.400 | 22.8 | 1.00 |
| cqr_engine | hgb_quantile_uncapped_fit | 0.80 | validation → calibration | 0.788 [0.707, 0.940] | 0.480 | 25.9 | 1.00 |
| quantile_raw | transformer_quantile | 0.80 | validation → calibration | 0.880  | 0.567 | 23.8 | 1.00 |
| cqr_pooled | transformer_quantile | 0.80 | validation → calibration | 0.880  | 0.567 | 23.8 | 1.00 |
| cqr_engine | transformer_quantile | 0.80 | validation → calibration | 0.899 [0.880, 0.960] | 0.589 | 24.5 | 1.00 |
