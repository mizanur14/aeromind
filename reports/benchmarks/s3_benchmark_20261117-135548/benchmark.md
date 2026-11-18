# AeroMind serving benchmark (CPU)

Measured 2026-11-17T07:55:48+00:00 · x86_64 · 8 logical CPUs · Python 3.12.15 · registry built 2026-11-17T07:42:55+00:00

Protocol: median and p95 over 30 repeats after 3 warm-up calls; one full-history validation engine per subset (batch: all validation engines); CPU only; torch threads ≤ 2 per deep model.

| Model | Path | Request | p50 ms | p95 ms | Cold load s |
|---|---|---|---|---|---|
| fd001-s1-hgb | in_process | 1 engine × 259 cycles | 28.66 | 36.11 | 0.02 |
| fd001-s2-ridge | in_process | 1 engine × 259 cycles | 23.19 | 25.82 | 0.00 |
| fd001-s2-xgboost | in_process | 1 engine × 259 cycles | 24.93 | 26.42 | 0.02 |
| fd001-s2-hgb_reference | in_process | 1 engine × 259 cycles | 29.86 | 32.91 | 0.02 |
| fd001-s2-tcn | in_process | 1 engine × 259 cycles | 9.71 | 11.87 | 0.00 |
| fd001-s2-gru | in_process | 1 engine × 259 cycles | 9.99 | 10.93 | 0.00 |
| fd001-s2-transformer | in_process | 1 engine × 259 cycles | 21.65 | 29.34 | 0.00 |
| fd002-s2-ridge | in_process | 1 engine × 269 cycles | 31.34 | 43.62 | 0.00 |
| fd002-s2-xgboost | in_process | 1 engine × 269 cycles | 31.90 | 35.66 | 0.01 |
| fd002-s2-hgb_reference | in_process | 1 engine × 269 cycles | 37.12 | 42.80 | 0.03 |
| fd002-s2-gru | in_process | 1 engine × 269 cycles | 19.87 | 21.69 | 0.00 |
| fd002-s2-transformer | in_process | 1 engine × 269 cycles | 33.15 | 35.31 | 0.01 |
| fd003-s2-ridge | in_process | 1 engine × 424 cycles | 43.74 | 53.82 | 0.00 |
| fd003-s2-xgboost | in_process | 1 engine × 424 cycles | 41.26 | 43.34 | 0.01 |
| fd003-s2-hgb_reference | in_process | 1 engine × 424 cycles | 60.03 | 77.21 | 0.04 |
| fd003-s2-gru | in_process | 1 engine × 424 cycles | 20.45 | 23.69 | 0.00 |
| fd003-s2-transformer | in_process | 1 engine × 424 cycles | 43.83 | 52.68 | 0.00 |
| fd004-s2-ridge | in_process | 1 engine × 299 cycles | 70.03 | 110.00 | 0.00 |
| fd004-s2-xgboost | in_process | 1 engine × 299 cycles | 47.05 | 59.78 | 0.01 |
| fd004-s2-hgb_reference | in_process | 1 engine × 299 cycles | 50.65 | 53.40 | 0.03 |
| fd004-s2-gru | in_process | 1 engine × 299 cycles | 18.13 | 21.70 | 0.00 |
| fd004-s2-transformer | in_process | 1 engine × 299 cycles | 31.89 | 34.24 | 0.00 |
| fd001-s1-hgb | http | 1 engine × 259 cycles | 37.10 | 66.53 | — |
| fd001-s1-hgb | http_batch | 15 engines × 3160 cycles total | 865.09 | 1023.96 | — |
| fd001-s2-ridge | http | 1 engine × 259 cycles | 44.84 | 54.92 | — |
| fd001-s2-ridge | http_batch | 15 engines × 3160 cycles total | 486.72 | 512.22 | — |
| fd001-s2-xgboost | http | 1 engine × 259 cycles | 37.07 | 38.80 | — |
| fd001-s2-xgboost | http_batch | 15 engines × 3160 cycles total | 511.10 | 664.98 | — |
| fd001-s2-hgb_reference | http | 1 engine × 259 cycles | 53.31 | 64.10 | — |
| fd001-s2-hgb_reference | http_batch | 15 engines × 3160 cycles total | 730.15 | 992.74 | — |
| fd001-s2-tcn | http | 1 engine × 259 cycles | 22.01 | 23.45 | — |
| fd001-s2-tcn | http_batch | 15 engines × 3160 cycles total | 263.65 | 270.56 | — |
| fd001-s2-gru | http | 1 engine × 259 cycles | 22.44 | 24.73 | — |
| fd001-s2-gru | http_batch | 15 engines × 3160 cycles total | 267.52 | 271.57 | — |
| fd001-s2-transformer | http | 1 engine × 259 cycles | 31.59 | 34.90 | — |
| fd001-s2-transformer | http_batch | 15 engines × 3160 cycles total | 399.24 | 563.68 | — |
| fd002-s2-ridge | http | 1 engine × 269 cycles | 61.20 | 69.60 | — |
| fd002-s2-ridge | http_batch | 39 engines × 8166 cycles total | 2355.70 | 2671.12 | — |
| fd002-s2-xgboost | http | 1 engine × 269 cycles | 51.46 | 56.94 | — |
| fd002-s2-xgboost | http_batch | 39 engines × 8166 cycles total | 1798.43 | 2193.25 | — |
| fd002-s2-hgb_reference | http | 1 engine × 269 cycles | 59.83 | 62.72 | — |
| fd002-s2-hgb_reference | http_batch | 39 engines × 8166 cycles total | 2066.08 | 2399.97 | — |
| fd002-s2-gru | http | 1 engine × 269 cycles | 25.62 | 26.15 | — |
| fd002-s2-gru | http_batch | 39 engines × 8166 cycles total | 916.90 | 1056.53 | — |
| fd002-s2-transformer | http | 1 engine × 269 cycles | 43.80 | 53.00 | — |
| fd002-s2-transformer | http_batch | 39 engines × 8166 cycles total | 1386.24 | 1593.01 | — |
| fd003-s2-ridge | http | 1 engine × 424 cycles | 50.01 | 54.64 | — |
| fd003-s2-ridge | http_batch | 15 engines × 3688 cycles total | 580.80 | 677.67 | — |
| fd003-s2-xgboost | http | 1 engine × 424 cycles | 50.49 | 61.07 | — |
| fd003-s2-xgboost | http_batch | 15 engines × 3688 cycles total | 755.83 | 814.60 | — |
| fd003-s2-hgb_reference | http | 1 engine × 424 cycles | 56.38 | 66.81 | — |
| fd003-s2-hgb_reference | http_batch | 15 engines × 3688 cycles total | 716.28 | 1016.88 | — |
| fd003-s2-gru | http | 1 engine × 424 cycles | 29.06 | 31.64 | — |
| fd003-s2-gru | http_batch | 15 engines × 3688 cycles total | 290.33 | 299.96 | — |
| fd003-s2-transformer | http | 1 engine × 424 cycles | 47.46 | 57.11 | — |
| fd003-s2-transformer | http_batch | 15 engines × 3688 cycles total | 448.13 | 590.98 | — |
| fd004-s2-ridge | http | 1 engine × 299 cycles | 47.44 | 50.55 | — |
| fd004-s2-ridge | http_batch | 37 engines × 8652 cycles total | 1597.55 | 1639.48 | — |
| fd004-s2-xgboost | http | 1 engine × 299 cycles | 47.94 | 50.25 | — |
| fd004-s2-xgboost | http_batch | 37 engines × 8652 cycles total | 1646.04 | 1846.09 | — |
| fd004-s2-hgb_reference | http | 1 engine × 299 cycles | 57.21 | 60.62 | — |
| fd004-s2-hgb_reference | http_batch | 37 engines × 8652 cycles total | 2006.76 | 2089.24 | — |
| fd004-s2-gru | http | 1 engine × 299 cycles | 30.37 | 36.49 | — |
| fd004-s2-gru | http_batch | 37 engines × 8652 cycles total | 1277.67 | 1572.31 | — |
| fd004-s2-transformer | http | 1 engine × 299 cycles | 41.66 | 51.46 | — |
| fd004-s2-transformer | http_batch | 37 engines × 8652 cycles total | 1330.46 | 1393.55 | — |

Peak RSS of the benchmark process (all models loaded): 531 MB.
