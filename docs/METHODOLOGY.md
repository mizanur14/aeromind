# Experiment methodology (Stage 1)

## Partitions

| Partition | FD001 engines | Used for |
|---|---|---|
| train | 70 | fitting the feature pipeline (constant removal, scaler) and models |
| validation | 15 | hyperparameter and model selection, all reported validation metrics |
| calibration | 15 | **untouched** in Stage 1 (reserved for conformal / uncertainty calibration) |
| official test | 100 | one final evaluation of the validation-selected model |

The split is engine-level, seeded (42) and stored in `data/manifests/fd001_split_seed42.json`.

## Targets

- `rul`: uncapped true RUL; never modified.
- `target`: `min(rul, cap)` with `cap = 125` (primary, `fd001_baseline.yaml`) or uncapped
  (`fd001_uncapped.yaml`). The cap of 125 was fixed a priori from literature convention,
  not tuned on any data.

Every metric carries a suffix naming its truth:

| Suffix | Truth |
|---|---|
| `_true` | uncapped true RUL |
| `_capped` | `min(rul, 125)` (literature-style, the `capped_truth_cap` setting) |
| `_target` | the model's own training-target formulation |

Capped-truth errors are always lower, because they forgive under-prediction of healthy
engines. Never compare a `_capped` number with a `_true` number.

## Features (FD001 configuration: 166 columns)

- Raw values of 15 non-constant sensors (constant ones removed using training engines only).
- `cycle` (current age).
- For windows of 10 and 30 cycles: trailing mean, population std, OLS slope, min and max.
- Settings are excluded for FD001 (a single operating condition, so they carry only noise).
- **Short histories:** windows are trailing with `min_periods=1`. Before *w* cycles exist, a
  window uses all available cycles; std and slope are 0 for a single observation. The same
  rule applies to training rows and test engines (the shortest FD001 test engine has 31 cycles).
- Standard scaling is fitted on training-engine rows only.

## Models and grids

Constant median; mean residual life by age, `E[min(L−t, cap) | survived to t]` from
training engines (no sensors); Ridge; Random Forest; HistGradientBoosting (early stopping
off); XGBoost (`hist`). Grids are in `configs/fd001_baseline.yaml` (16 fits in total). Seeds
are injected by the factory. Predictions are clipped to `[0, cap]`.

A *persistence* baseline is not provided: no past RUL is observable at inference, so there
is nothing to persist.

## Validation protocol

1. **Snapshots (selection):** 10 seeded cut cycles per validation engine (150 in total),
   uniform over each engine's life. Every engine has equal weight. The selection metric is
   snapshot RMSE against the training-target truth (`snapshot_rmse_target`).
2. **All windows:** every validation row, reported row-weighted and engine-weighted (each
   engine's rows sum to weight 1, so long trajectories do not dominate).
3. **Near end of life:** rows with true RUL ≤ 30.
4. **Error by true-RUL bin**, per-engine tables, residual plots.

Snapshots are uniform over the whole life, so about a third of them (49 of 150) have true RUL above 125.
Uncapped (`_true`) snapshot errors are therefore dominated by early-life cases that a capped
model cannot represent. The official test truncation is undocumented, and its true RUL range
for FD001 is 7–145.

## Final test protocol

`aeromind test` computes features over each test engine's full observed history and keeps
the prediction at the last observed cycle (100 predictions), compared with `RUL_FD001.txt`.
Only the validation-selected model is the reported result. `--all-models` adds a table
marked post hoc. Each test evaluation is appended to `metadata.json["test_evaluations"]` and
to `reports/test_set_ledger.json`. The command requires `--confirm-final` and refuses a second
evaluation of the same run unless `--allow-repeat --reason ...` is given.

## Statistical caveat

With 15 validation engines, differences of a few tenths of a cycle between the tree
ensembles (11.26 vs 11.52 vs 11.55 snapshot RMSE) are not meaningful. No confidence intervals
are reported in Stage 1. Engine-level bootstrap intervals are planned with the Stage-2
uncertainty work.
