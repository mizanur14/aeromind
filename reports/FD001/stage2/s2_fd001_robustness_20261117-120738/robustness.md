| Model | Scenario | Level | Runs | Snapshot RMSE capped | Δ vs clean | Snapshot RMSE true | Near-EOL RMSE true |
|---|---|---|---|---|---|---|---|
| ridge | clean | - | 1 | 14.76 | +0.00 | 43.81 | 12.61 |
| ridge | noise | 0.5sd | 3 | 16.69 ± 0.89 | +1.94 | 45.03 | 13.45 |
| ridge | noise | 1.0sd | 3 | 22.53 ± 1.21 | +7.78 | 48.54 | 18.80 |
| ridge | noise | 2.0sd | 3 | 33.23 ± 0.60 | +18.47 | 55.73 | 31.82 |
| ridge | dropout | sensor_2 | 1 | 17.91 | +3.15 | 48.81 | 12.67 |
| ridge | dropout | sensor_3 | 1 | 15.11 | +0.35 | 43.33 | 14.11 |
| ridge | dropout | sensor_4 | 1 | 17.88 | +3.13 | 48.08 | 15.98 |
| ridge | dropout | sensor_7 | 1 | 18.18 | +3.42 | 46.34 | 22.68 |
| ridge | dropout | sensor_8 | 1 | 14.33 | -0.43 | 41.58 | 10.58 |
| ridge | dropout | sensor_9 | 1 | 16.68 | +1.93 | 43.44 | 17.56 |
| ridge | dropout | sensor_11 | 1 | 17.44 | +2.69 | 46.29 | 20.72 |
| ridge | dropout | sensor_12 | 1 | 15.85 | +1.09 | 44.14 | 16.55 |
| ridge | dropout | sensor_13 | 1 | 14.65 | -0.11 | 41.89 | 11.21 |
| ridge | dropout | sensor_14 | 1 | 15.71 | +0.95 | 44.36 | 15.99 |
| ridge | dropout | sensor_15 | 1 | 19.20 | +4.44 | 48.87 | 20.43 |
| ridge | dropout | sensor_17 | 1 | 15.68 | +0.92 | 44.78 | 14.48 |
| ridge | dropout | sensor_20 | 1 | 15.64 | +0.88 | 44.86 | 14.87 |
| ridge | dropout | sensor_21 | 1 | 14.83 | +0.07 | 44.65 | 11.23 |
| ridge | missing | 5% | 3 | 14.81 ± 0.14 | +0.05 | 43.82 | 12.94 |
| ridge | missing | 20% | 3 | 15.39 ± 0.06 | +0.63 | 43.94 | 13.50 |
| ridge | missing | 50% | 3 | 17.38 ± 0.63 | +2.62 | 44.10 | 16.68 |
| ridge | shift | FD001->FD003 | 1 | 43.04 | +28.28 | 111.73 | 70.42 |
| xgboost | clean | - | 1 | 11.52 | +0.00 | 40.98 | 4.23 |
| xgboost | noise | 0.5sd | 3 | 21.75 ± 0.49 | +10.23 | 51.79 | 5.51 |
| xgboost | noise | 1.0sd | 3 | 30.10 ± 1.21 | +18.58 | 59.46 | 7.28 |
| xgboost | noise | 2.0sd | 3 | 37.04 ± 0.64 | +25.52 | 66.53 | 11.40 |
| xgboost | dropout | sensor_2 | 1 | 11.40 | -0.12 | 41.13 | 7.03 |
| xgboost | dropout | sensor_3 | 1 | 12.21 | +0.70 | 41.44 | 7.76 |
| xgboost | dropout | sensor_4 | 1 | 14.71 | +3.19 | 42.61 | 17.83 |
| xgboost | dropout | sensor_7 | 1 | 12.43 | +0.91 | 41.26 | 7.54 |
| xgboost | dropout | sensor_8 | 1 | 11.98 | +0.46 | 41.03 | 5.81 |
| xgboost | dropout | sensor_9 | 1 | 12.05 | +0.53 | 40.92 | 5.13 |
| xgboost | dropout | sensor_11 | 1 | 13.96 | +2.45 | 41.93 | 15.08 |
| xgboost | dropout | sensor_12 | 1 | 12.28 | +0.76 | 40.95 | 7.44 |
| xgboost | dropout | sensor_13 | 1 | 12.42 | +0.91 | 40.93 | 6.81 |
| xgboost | dropout | sensor_14 | 1 | 12.15 | +0.64 | 41.36 | 4.72 |
| xgboost | dropout | sensor_15 | 1 | 12.55 | +1.04 | 41.10 | 10.51 |
| xgboost | dropout | sensor_17 | 1 | 11.81 | +0.30 | 41.06 | 4.25 |
| xgboost | dropout | sensor_20 | 1 | 11.53 | +0.01 | 41.20 | 4.54 |
| xgboost | dropout | sensor_21 | 1 | 11.67 | +0.16 | 41.12 | 6.35 |
| xgboost | missing | 5% | 3 | 11.73 ± 0.35 | +0.22 | 41.08 | 4.36 |
| xgboost | missing | 20% | 3 | 12.14 ± 0.42 | +0.62 | 41.23 | 5.27 |
| xgboost | missing | 50% | 3 | 12.96 ± 1.31 | +1.44 | 41.77 | 7.95 |
| xgboost | shift | FD001->FD003 | 1 | 17.74 | +6.22 | 87.22 | 25.90 |
| hgb_reference | clean | - | 1 | 11.26 | +0.00 | 40.90 | 4.41 |
| hgb_reference | noise | 0.5sd | 3 | 21.30 ± 0.27 | +10.04 | 51.32 | 5.84 |
| hgb_reference | noise | 1.0sd | 3 | 25.89 ± 0.74 | +14.63 | 54.76 | 7.40 |
| hgb_reference | noise | 2.0sd | 3 | 30.99 ± 0.42 | +19.73 | 59.44 | 10.75 |
| hgb_reference | dropout | sensor_2 | 1 | 11.50 | +0.24 | 40.98 | 7.91 |
| hgb_reference | dropout | sensor_3 | 1 | 11.71 | +0.45 | 40.98 | 7.22 |
| hgb_reference | dropout | sensor_4 | 1 | 14.93 | +3.66 | 42.09 | 20.06 |
| hgb_reference | dropout | sensor_7 | 1 | 12.27 | +1.01 | 41.32 | 9.12 |
| hgb_reference | dropout | sensor_8 | 1 | 11.68 | +0.42 | 40.84 | 5.63 |
| hgb_reference | dropout | sensor_9 | 1 | 11.36 | +0.09 | 40.63 | 5.00 |
| hgb_reference | dropout | sensor_11 | 1 | 13.76 | +2.49 | 41.93 | 13.18 |
| hgb_reference | dropout | sensor_12 | 1 | 12.02 | +0.76 | 40.41 | 8.49 |
| hgb_reference | dropout | sensor_13 | 1 | 12.16 | +0.90 | 41.54 | 6.48 |
| hgb_reference | dropout | sensor_14 | 1 | 11.66 | +0.40 | 41.28 | 5.11 |
| hgb_reference | dropout | sensor_15 | 1 | 12.51 | +1.25 | 41.01 | 10.86 |
| hgb_reference | dropout | sensor_17 | 1 | 11.58 | +0.32 | 41.04 | 5.58 |
| hgb_reference | dropout | sensor_20 | 1 | 11.34 | +0.08 | 41.02 | 6.03 |
| hgb_reference | dropout | sensor_21 | 1 | 11.43 | +0.17 | 41.02 | 7.70 |
| hgb_reference | missing | 5% | 3 | 11.33 ± 0.36 | +0.07 | 40.94 | 4.66 |
| hgb_reference | missing | 20% | 3 | 12.00 ± 0.48 | +0.74 | 41.10 | 5.58 |
| hgb_reference | missing | 50% | 3 | 13.04 ± 1.19 | +1.78 | 41.61 | 8.51 |
| hgb_reference | shift | FD001->FD003 | 1 | 17.77 | +6.51 | 87.33 | 25.85 |
| tcn | clean | - | 1 | 14.07 | +0.00 | 42.43 | 6.39 |
| tcn | noise | 0.5sd | 3 | 14.40 ± 0.61 | +0.33 | 41.83 | 7.54 |
| tcn | noise | 1.0sd | 3 | 17.37 ± 1.04 | +3.29 | 42.99 | 14.42 |
| tcn | noise | 2.0sd | 3 | 28.69 ± 0.11 | +14.62 | 48.20 | 38.54 |
| tcn | dropout | sensor_2 | 1 | 13.75 | -0.33 | 41.89 | 10.26 |
| tcn | dropout | sensor_3 | 1 | 14.66 | +0.59 | 42.23 | 9.11 |
| tcn | dropout | sensor_4 | 1 | 15.45 | +1.38 | 42.81 | 7.56 |
| tcn | dropout | sensor_7 | 1 | 14.72 | +0.65 | 42.57 | 9.53 |
| tcn | dropout | sensor_8 | 1 | 14.59 | +0.51 | 42.27 | 8.90 |
| tcn | dropout | sensor_9 | 1 | 15.00 | +0.93 | 42.65 | 6.39 |
| tcn | dropout | sensor_11 | 1 | 15.18 | +1.11 | 44.06 | 11.42 |
| tcn | dropout | sensor_12 | 1 | 14.77 | +0.70 | 43.32 | 9.85 |
| tcn | dropout | sensor_13 | 1 | 17.00 | +2.93 | 43.53 | 14.53 |
| tcn | dropout | sensor_14 | 1 | 15.31 | +1.24 | 43.24 | 8.64 |
| tcn | dropout | sensor_15 | 1 | 14.09 | +0.01 | 42.83 | 9.97 |
| tcn | dropout | sensor_17 | 1 | 14.28 | +0.21 | 42.17 | 4.74 |
| tcn | dropout | sensor_20 | 1 | 14.79 | +0.71 | 42.75 | 6.20 |
| tcn | dropout | sensor_21 | 1 | 14.47 | +0.40 | 41.93 | 5.58 |
| tcn | missing | 5% | 3 | 13.82 ± 0.21 | -0.25 | 42.28 | 6.13 |
| tcn | missing | 20% | 3 | 13.80 ± 0.63 | -0.28 | 42.03 | 6.03 |
| tcn | missing | 50% | 3 | 14.02 ± 1.27 | -0.05 | 42.06 | 6.92 |
| tcn | shift | FD001->FD003 | 1 | 30.67 | +16.59 | 88.71 | 69.96 |
| gru | clean | - | 1 | 13.49 | +0.00 | 42.61 | 6.81 |
| gru | noise | 0.5sd | 3 | 16.01 ± 1.56 | +2.53 | 43.90 | 8.41 |
| gru | noise | 1.0sd | 3 | 21.89 ± 1.34 | +8.40 | 47.80 | 11.94 |
| gru | noise | 2.0sd | 3 | 31.06 ± 1.42 | +17.57 | 55.70 | 20.60 |
| gru | dropout | sensor_2 | 1 | 13.93 | +0.44 | 42.47 | 7.51 |
| gru | dropout | sensor_3 | 1 | 13.14 | -0.35 | 42.14 | 8.73 |
| gru | dropout | sensor_4 | 1 | 16.12 | +2.64 | 43.99 | 14.15 |
| gru | dropout | sensor_7 | 1 | 14.22 | +0.73 | 43.27 | 10.11 |
| gru | dropout | sensor_8 | 1 | 13.04 | -0.45 | 41.72 | 9.42 |
| gru | dropout | sensor_9 | 1 | 14.44 | +0.95 | 42.49 | 7.29 |
| gru | dropout | sensor_11 | 1 | 18.59 | +5.11 | 46.17 | 17.41 |
| gru | dropout | sensor_12 | 1 | 13.45 | -0.03 | 42.48 | 7.71 |
| gru | dropout | sensor_13 | 1 | 19.89 | +6.41 | 46.20 | 14.61 |
| gru | dropout | sensor_14 | 1 | 14.28 | +0.79 | 43.25 | 10.38 |
| gru | dropout | sensor_15 | 1 | 14.95 | +1.46 | 42.99 | 10.14 |
| gru | dropout | sensor_17 | 1 | 15.56 | +2.08 | 45.22 | 6.26 |
| gru | dropout | sensor_20 | 1 | 14.35 | +0.86 | 44.01 | 6.81 |
| gru | dropout | sensor_21 | 1 | 13.09 | -0.39 | 42.70 | 8.55 |
| gru | missing | 5% | 3 | 13.65 ± 0.22 | +0.16 | 42.69 | 6.80 |
| gru | missing | 20% | 3 | 14.42 ± 0.82 | +0.93 | 42.90 | 7.04 |
| gru | missing | 50% | 3 | 15.68 ± 1.09 | +2.20 | 43.37 | 8.04 |
| gru | shift | FD001->FD003 | 1 | 19.67 | +6.18 | 89.09 | 18.74 |
| transformer | clean | - | 1 | 13.00 | +0.00 | 41.72 | 6.10 |
| transformer | noise | 0.5sd | 3 | 14.71 ± 1.23 | +1.72 | 42.82 | 7.23 |
| transformer | noise | 1.0sd | 3 | 19.99 ± 1.02 | +7.00 | 47.03 | 9.24 |
| transformer | noise | 2.0sd | 3 | 32.83 ± 0.53 | +19.83 | 58.32 | 12.38 |
| transformer | dropout | sensor_2 | 1 | 14.02 | +1.02 | 41.95 | 11.31 |
| transformer | dropout | sensor_3 | 1 | 13.34 | +0.34 | 41.84 | 8.58 |
| transformer | dropout | sensor_4 | 1 | 15.18 | +2.19 | 42.99 | 14.56 |
| transformer | dropout | sensor_7 | 1 | 13.14 | +0.15 | 41.45 | 8.63 |
| transformer | dropout | sensor_8 | 1 | 13.01 | +0.02 | 41.21 | 6.45 |
| transformer | dropout | sensor_9 | 1 | 13.00 | +0.00 | 41.68 | 6.85 |
| transformer | dropout | sensor_11 | 1 | 14.62 | +1.62 | 42.15 | 11.60 |
| transformer | dropout | sensor_12 | 1 | 13.93 | +0.93 | 42.21 | 9.04 |
| transformer | dropout | sensor_13 | 1 | 15.78 | +2.79 | 43.76 | 16.08 |
| transformer | dropout | sensor_14 | 1 | 12.49 | -0.51 | 41.64 | 5.90 |
| transformer | dropout | sensor_15 | 1 | 15.10 | +2.10 | 43.54 | 12.74 |
| transformer | dropout | sensor_17 | 1 | 13.51 | +0.51 | 42.41 | 7.42 |
| transformer | dropout | sensor_20 | 1 | 13.14 | +0.14 | 41.74 | 6.09 |
| transformer | dropout | sensor_21 | 1 | 12.92 | -0.08 | 40.80 | 5.74 |
| transformer | missing | 5% | 3 | 13.02 ± 0.34 | +0.03 | 41.69 | 6.12 |
| transformer | missing | 20% | 3 | 13.37 ± 0.66 | +0.37 | 41.82 | 6.43 |
| transformer | missing | 50% | 3 | 14.99 ± 1.07 | +2.00 | 42.41 | 7.90 |
| transformer | shift | FD001->FD003 | 1 | 27.51 | +14.52 | 100.63 | 38.89 |
