# Stage-2 methodology

Stage 2 adds temporal deep-learning models, repeated engine-level validation, uncertainty
quantification, ablations and robustness tests. The Stage-1 pipeline, configs and results are
unchanged. **No Stage-2 model has been evaluated on the official NASA test set.**

## 1. Data roles (FD001)

| Partition | Engines | Stage-2 use |
|---|---|---|
| train (canonical) | 70 | fitting (canonical fold) |
| validation (canonical) | 15 | canonical-fold evaluation; conformal coverage evaluation |
| development = train ∪ validation | 85 | engine-level 5-fold CV (17 engines per validation fold) |
| calibration | 15 | **only** conformal calibration (and the role-swap coverage check) |
| official test | 100 | **not used** in Stage 2 |

Deep models early-stop on an **inner** engine-level holdout (15 % of the fold's training
engines, seeded), so the fold's validation engines are never seen during training.

## 2. Identical protocol for every model

- Split manifest `data/manifests/fd001_split_seed42.json` (SHA-256 recorded per run).
- Training target: `min(RUL, 125)`. Metrics are reported against **capped truth**
  `min(RUL, 125)` (suffix `_capped`) and **uncapped true RUL** (suffix `_true`).
- **Snapshots:** 10 seeded cut cycles per validation engine, the same rows for every model
  in a fold (tested). Engine-weighted all-window metrics and near-end-of-life (true RUL ≤ 30)
  metrics are also reported.
- PHM08 is reported as mean per snapshot, because CV folds differ slightly in size.
- Predictions are clipped to `[0, cap]` for all models.

## 3. Models

| Model | Input | Size |
|---|---|---|
| Ridge, XGBoost, HistGradientBoosting (reference) | Stage-1 engineered features: 15 sensors, rolling stats over 10 and 30 cycles, cycle | Stage-1 validation-selected hyperparameters |
| TCN | window of 30 normalised raw sensors + cycle per step | 3 residual causal blocks, 32 channels, kernel 3, dilations 1/2/4 (receptive field 29) |
| GRU | same windows | 1 layer, 64 hidden units |
| Transformer | same windows | 2 pre-norm encoder layers, d_model 32, 4 heads, FF 64, learned positions, key-padding mask |

**Deep-model training:**

- AdamW (lr 1e-3, weight decay 1e-4), batch 256, up to 40 epochs.
- Huber loss (δ = 10 cycles) with targets scaled by 1/125.
- ReduceLROnPlateau, gradient-norm clipping at 1.0, early stopping (patience 8).
- Fixed seeds, deterministic kernels and 4 threads. The best inner-validation checkpoint is
  kept, and `last.pt` allows exact resume (tested).

**Windows:**

- Windows are causal and right-padded with explicit lengths.
- TCN reads the newest valid step. GRU uses packed sequences. The Transformer masks padding.
- Padding values provably do not affect outputs (tested).
- Early cycles use partial windows, exactly like the classical rolling features.

Deep architectures were fixed before results were seen and **were not tuned**. Classical
hyperparameters come from Stage-1 selection on the canonical validation engines. That gives
them a mild advantage on CV folds containing those engines; this is disclosed.

## 4. Repeated validation and variability

- Results are the mean ± SD over the 5 CV folds plus the canonical fold.
- Folds share most training engines, so the fold SD understates true uncertainty. No
  significance tests are claimed.
- Paired per-fold differences against the HistGradientBoosting reference are reported as
  mean, SD, range and "folds better".

## 5. Uncertainty quantification

**Intervals target the capped RUL**, the quantity the models estimate. Coverage of
uncapped true RUL is reported separately and is expected to fail early in life.

| Method | Calibration scores | Guarantee |
|---|---|---|
| `conformal_engine` | one randomly drawn snapshot per calibration engine (n = 15), repeated over 500 draws | Marginal coverage ≥ 1−α (and ≤ 1−α+1/16) for a snapshot drawn the same way from a new engine exchangeable with the calibration engines (Lei et al., 2018) |
| `conformal_pooled` | all ~3 000 calibration windows | **None**: windows within an engine are dependent; heuristic only |
| `quantile_raw` | none (HGB or deep quantile regression) | none |
| `cqr_engine` / `cqr_pooled` | CQR scores (Romano et al., 2019), same two schemes | as above |

**Implications of 15 calibration engines:**

- α < 1/16 cannot be achieved: the threshold is infinite, so there are no finite 95 % intervals.
- At α = 0.1 the threshold is the *largest* of 15 scores, so it varies strongly with the
  random snapshot draw. The p05–p95 range over draws is reported.
- The guarantee is marginal, averaged over engines and snapshot positions. It is not
  conditional on RUL range: coverage by RUL bin is a diagnostic only.
- It does not hold under distribution shift (other subsets, sensor faults); see Barber et al. (2023).

Coverage is evaluated on the 15 validation engines, then with roles swapped (calibrate on
validation, evaluate on calibration) as a consistency check.

## 6. Ablations

One factor at a time relative to a base model, on CV folds cv0–cv2. Each is paired per fold
with its base. Factors:

- sensor subsets
- window length 20 / 30 / 50
- operating settings on or off
- model capacity
- raw versus engineered per-step features
- capped versus uncapped target
- loss: Huber, MAE or MSE

There is no exhaustive grid search.

## 7. Robustness

- **Input corruption** of the canonical validation engines; the models are not retrained:
  - Gaussian noise of 0.5, 1 and 2 training SDs.
  - Each informative sensor frozen at its training mean.
  - 5 %, 20 % and 50 % of readings missing, with causal forward-fill imputation.
- **Separate benchmarks:** FD002, FD003 and FD004, each with its own engine split (canonical
  fold). FD002 and FD004 use per-operating-condition normalisation (k-means on settings,
  fitted on training engines).
- **Shift stress test:** FD001-trained models applied unchanged to FD003 validation engines,
  which add a fan fault mode. This is **not** transfer learning or zero-shot generalisation.

## References

See [REFERENCES.md](REFERENCES.md).
