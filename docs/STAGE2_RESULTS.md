# Stage-2 results

Every number below is copied from a saved run under `reports/<DS>/stage2/<run>/`. The full
artifacts, including model bundles and checkpoints, are in `artifacts/stage2/<run>/`.

All results are on **validation or calibration engines**. No Stage-2 model has been evaluated
on the official NASA test set; `reports/test_set_ledger.json` still holds only the Stage-1
entry. The protocol is described in [STAGE2_METHODOLOGY.md](STAGE2_METHODOLOGY.md).

**Metric conventions:**
- `capped` = error against `min(RUL, 125)`; `true` = error against uncapped RUL.
- "Near-EOL" = windows with true RUL ≤ 30.
- CV = mean ± SD over 5 engine-level folds of the 85 development engines (17 validation
  engines per fold, 10 snapshots per engine).
- The folds share training engines, so the SDs understate the real uncertainty. No
  significance tests are claimed.

Hardware: Intel i7-1165G7 (4C/8T), 16 GB RAM, CPU only, PyTorch 2.14.0+cpu, 4 threads per
model.

## Runs

| Experiment | Run directory | Wall time |
|---|---|---|
| FD001 model comparison (5-fold CV + canonical fold) | `s2_fd001_comparison_20261117-114027` | ≈ 24 min |
| FD001 uncertainty quantification | `s2_fd001_uq_20261117-121004` (supersedes `…-120505`, which lacks `hgb_quantile_uncapped_fit`) | ≈ 3 min |
| FD001 robustness | `s2_fd001_robustness_20261117-120738` | ≈ 2 min |
| FD001 ablations (folds cv0–cv2) | `s2_fd001_ablation_20261117-121310` | ≈ 51 min |
| FD003 / FD002 / FD004 benchmarks (canonical fold only) | `s2_fd003_benchmark_20261117-130357`, `s2_fd002_benchmark_20261117-130845`, `s2_fd004_benchmark_20261117-131540` | ≈ 5 / 7 / 9 min |
| Real-data smoke (2 epochs) | `s2_fd001_realdata_smoke_20261117-113916` | < 1 min |

## 1. FD001 model comparison

| Model | Snapshot RMSE capped (CV) | Snapshot RMSE true (CV) | Snapshot MAE capped (CV) | PHM08 / snapshot, capped (CV) | Near-EOL RMSE true (CV) | Snapshot RMSE capped (canonical) |
|---|---|---|---|---|---|---|
| Ridge | 16.65 ± 1.66 | 44.45 ± 5.04 | 12.84 ± 1.56 | 5.00 ± 1.31 | 11.96 ± 2.29 | 14.76 |
| XGBoost | 13.71 ± 1.67 | 42.22 ± 5.07 | 9.64 ± 1.26 | 3.72 ± 1.11 | 4.53 ± 0.64 | 11.52 |
| HGB (reference) | 13.97 ± 1.79 | 42.33 ± 4.93 | 9.85 ± 1.33 | 3.79 ± 1.18 | 4.86 ± 0.70 | 11.26 |
| TCN | 17.87 ± 2.75 | 44.44 ± 5.85 | 12.63 ± 1.88 | 9.44 ± 6.16 | 8.46 ± 1.47 | 14.07 |
| GRU | 13.47 ± 2.49 | 42.58 ± 5.28 | 9.51 ± 1.59 | 3.68 ± 1.91 | 4.94 ± 1.33 | 13.49 |
| Transformer | 13.37 ± 2.36 | 41.27 ± 5.44 | 8.81 ± 1.50 | 4.82 ± 3.02 | 4.46 ± 1.16 | 13.00 |

**Paired per-fold difference** in snapshot RMSE (capped) relative to HGB. Negative means
better than HGB.

| Model | Mean | SD | Min | Max | Folds better |
|---|---|---|---|---|---|
| XGBoost | −0.26 | 0.20 | −0.59 | −0.05 | 5/5 |
| GRU | −0.50 | 0.71 | −1.54 | +0.36 | 4/5 |
| Transformer | −0.60 | 1.23 | −2.06 | +1.32 | 4/5 |
| Ridge | +2.68 | 0.75 | +1.40 | +3.22 | 0/5 |
| TCN | +3.90 | 1.52 | +2.56 | +6.48 | 0/5 |

**Resources** (canonical fold; latency is the median over repeats):

| Model | Parameters | Bundle MB | Fit s (mean) | Epochs (canonical) | Batch µs / row | Single-input ms | Peak RSS MB |
|---|---|---|---|---|---|---|---|
| Ridge | 167 | 0.007 | 1.4 | — | 0.3 | 0.19 | 307 |
| XGBoost | — | 0.240 | 5.3 | — | 3.0 | 0.35 | 333 |
| HGB | — | 0.482 | 4.7 | — | 8.6 | 8.84 | 294 |
| TCN | 17,665 | 0.065 | 43.5 | 29 | 42.6 | 1.39 | 609 |
| GRU | 15,809 | 0.059 | 67.5 | 34 | 22.1 | 1.06 | 656 |
| Transformer | 21,857 | 0.081 | 108.3 | 21 | 87.4 | 1.29 | 785 |

Peak RSS is the whole process (Python, data and libraries; about 210 MB after data load),
measured in a separate process per model. All 15 deep CV fits stopped through the
patience rule.

**Interpretation:**
- **No deep model is demonstrably better than the tree baselines.** The GRU and Transformer
  have slightly lower CV means, but their paired SD (0.71 and 1.23) exceeds the mean
  difference, and each is worse than HGB on one fold.
- XGBoost's edge over HGB is consistent (5/5 folds) but small: 0.26 cycles, against fold
  SDs of about 1.7.
- The TCN and Ridge are clearly worse (0/5 folds).
- Deep models cost 10–20× more training time and about 2× more memory. Their single-input
  latency (≈ 1 ms) is lower than HGB's (8.8 ms) but higher than XGBoost's (0.35 ms).
- On the canonical fold the classical models look better (11.3–11.5 vs 13.0–13.5). Their
  hyperparameters were selected on exactly those 15 engines in Stage 1, so this is an
  expected optimistic bias, not evidence of superiority.

## 2. Uncertainty quantification (FD001)

**Setup:**
- Calibration uses the 15 held-out calibration engines; coverage is evaluated on the 15
  canonical validation engines.
- `conformal_engine` and `cqr_engine` use one random snapshot per calibration engine
  (n = 15), repeated over 500 draws. The bracketed range is p05–p95 over the draws.
- Intervals target capped RUL.

**Nominal 90 %, calibration → validation:**

| Method | Model | Coverage capped, mean [p05, p95] | Coverage true RUL | Mean width (cycles) |
|---|---|---|---|---|
| conformal_engine | HGB | 0.953 [0.873, 0.987] | 0.696 | 53.9 |
| conformal_engine | XGBoost | 0.948 [0.873, 1.000] | 0.689 | 52.7 |
| conformal_engine | GRU | 0.944 [0.827, 0.993] | 0.704 | 58.3 |
| conformal_engine | Transformer | 0.941 [0.833, 0.993] | 0.708 | 60.5 |
| conformal_pooled (heuristic) | HGB | 0.933 | 0.660 | 42.5 |
| conformal_pooled (heuristic) | Transformer | 0.887 | 0.640 | 42.1 |
| quantile_raw | hgb_quantile (fit on capped target) | 0.913 | 0.593 | 50.5 |
| quantile_raw | hgb_quantile_uncapped_fit | 0.913 | 0.593 | 30.3 |
| quantile_raw | transformer_quantile | 0.900 | 0.587 | 30.4 |
| cqr_engine | hgb_quantile_uncapped_fit | 0.960 [0.913, 1.000] | 0.656 | 45.7 |
| cqr_engine | transformer_quantile | 0.942 [0.900, 0.993] | 0.642 | 38.8 |

**Nominal 80 %, calibration → validation** (selected rows):

| Method | Model | Coverage capped | Width |
|---|---|---|---|
| conformal_engine | HGB | 0.843 [0.679, 0.953] | 32.0 |
| conformal_engine | Transformer | 0.793 [0.553, 0.947] | 30.8 |
| quantile_raw | hgb_quantile_uncapped_fit | 0.800 | 20.3 |
| quantile_raw | transformer_quantile | 0.827 | 23.8 |
| cqr_engine | transformer_quantile | 0.838 [0.827, 0.900] | 23.8 |

**Role swap (validation → calibration, 90 %):**
- conformal_engine: 0.946–0.971.
- quantile_raw: hgb_quantile_uncapped_fit 0.853, transformer_quantile 0.933.

At 80 %, quantile_raw hgb_quantile falls to 0.680 and hgb_quantile_uncapped_fit to 0.707.
**Uncalibrated quantile regression is not reliably calibrated across engine sets**;
conformalisation (CQR) recovers it at the cost of width.

The complete table, including every method, both nominal levels and both directions, is in
`reports/FD001/stage2/s2_fd001_uq_20261117-121004/summary.md`.

### Findings

1. **Engine-level conformal is conservative here (≈ 0.94–0.95 at 90 %), and its realised
   coverage depends strongly on which snapshot is drawn** (p05 0.83–0.87). With n = 15, the
   90 % threshold is the largest of the 15 scores, and the theory allows coverage of up to
   1 − α + 1/(n+1) ≈ 0.96. α < 1/16, including 95 % intervals, is unattainable.

2. **HGB quantile degeneracy at the cap (defect found and fixed).**
   - When HGB quantile models are fit directly on capped targets, the 95 % quantile is
     dominated by the point mass at 125.
   - For true RUL < 25 the mean interval width is **111.3 cycles**: an uninformative upper
     bound near the cap.
   - Fitting on uncapped RUL and clipping to [0, 125] estimates the same capped quantiles,
     because q_τ(min(Y, c)) = min(q_τ(Y), c). It reduces that width to **15.3 cycles**, with
     coverage of 0.928 against 0.933.
   - Overall width at 90 % falls from 50.5 to 30.3 at identical coverage (0.913).
   - The degenerate variant is kept in the tables as a documented finding.

3. **Coverage is not conditional on RUL.** Pooled conformal for HGB at 90 % covers:
   - 1.00 for true RUL in [0, 25)
   - 0.75–0.80 for [50, 125)
   - 0.97–1.00 for > 150, where the capped truth is easy

   The marginal guarantee averages these regions. Coverage by bin is a diagnostic, not a
   guarantee (`coverage_by_rul_bin.csv`).

4. **Against uncapped true RUL**, coverage is 0.59–0.71 by construction: early-life
   intervals cannot contain RUL > 125. These intervals must not be read as bounds on
   uncapped RUL.

5. **CQR pooled threshold of exactly 0 for transformer_quantile.** CQR and raw quantiles
   coincide for this model. This is not a bug: many calibration windows have a capped
   truth of 125 and an upper quantile clipped to 125, producing tied zero scores at the
   cap, and the pooled quantile of the scores lands on them.

## 3. Ablations (FD001, folds cv0–cv2)

These are one-factor variants of a GRU base and an XGBoost base. The base scores here
(11.91 and 12.66) are over cv0–cv2 only, so they differ from the 5-fold means in §1. Δ is
the paired difference against the base; negative means better.

| Base | Variant | RMSE capped | Δ capped (mean ± SD) | Folds better | RMSE true | Near-EOL RMSE true | Fit s |
|---|---|---|---|---|---|---|---|
| GRU | base (window 30, 15 sensors + cycle) | 11.91 ± 1.73 | — | — | 41.12 | 4.28 | 57 |
| GRU | literature 14 sensors | 12.44 | +0.53 ± 0.65 | 1/3 | 41.04 | 3.98 | 56 |
| GRU | trend-12 sensors | 12.40 | +0.49 ± 0.28 | 0/3 | 40.89 | 4.68 | 65 |
| GRU | window 20 | 13.79 | +1.88 ± 0.28 | 0/3 | 41.49 | 5.34 | 30 |
| GRU | **window 50** | **10.13** | **−1.78 ± 0.81** | **3/3** | 40.17 | 3.62 | 155 |
| GRU | with operating settings | 12.88 | +0.97 ± 0.41 | 0/3 | 41.66 | 4.18 | 45 |
| GRU | small (16 hidden) | 14.94 | +3.03 ± 2.88 | 0/3 | 42.26 | 6.71 | 27 |
| GRU | large (2 × 128) | 12.82 | +0.91 ± 1.28 | 1/3 | 40.65 | 4.19 | 287 |
| GRU | engineered per-step inputs | 14.15 | +2.24 ± 1.41 | 0/3 | 42.15 | 5.71 | 44 |
| GRU | uncapped target | 32.38 | +20.47 ± 3.18 | 0/3 | **33.28** | 5.22 | 46 |
| GRU | MAE loss | 11.99 | +0.08 ± 0.27 | 2/3 | 41.44 | 4.00 | 69 |
| GRU | MSE loss | 12.13 | +0.22 ± 0.59 | 1/3 | 40.88 | 4.65 | 59 |
| XGBoost | base (rolling 10/30) | 12.66 ± 1.10 | — | — | 40.83 | 4.08 | 4.4 |
| XGBoost | literature 14 sensors | 12.86 | +0.20 ± 0.22 | 1/3 | 40.89 | 4.22 | 4.1 |
| XGBoost | trend-12 sensors | 13.45 | +0.78 ± 1.34 | 1/3 | 41.68 | 4.88 | 4.0 |
| XGBoost | rolling 10/20 | 14.65 | +1.98 ± 0.25 | 0/3 | 41.59 | 6.17 | 4.4 |
| XGBoost | **rolling 10/50** | **9.90** | **−2.77 ± 0.94** | **3/3** | 39.86 | 3.65 | 4.7 |
| XGBoost | with operating settings | 12.70 | +0.04 ± 0.19 | 2/3 | 41.03 | 4.29 | 5.1 |
| XGBoost | max_depth 3 | 12.41 | −0.25 ± 0.13 | 3/3 | 40.64 | 4.78 | 2.2 |
| XGBoost | max_depth 8 | 13.25 | +0.59 ± 0.03 | 0/3 | 41.32 | 4.43 | 14.9 |
| XGBoost | raw inputs (no rolling features) | 17.10 | +4.44 ± 0.81 | 0/3 | 42.63 | 10.22 | 0.5 |
| XGBoost | uncapped target | 33.92 | +21.25 ± 3.90 | 0/3 | **32.64** | 4.40 | 4.3 |
| XGBoost | absolute-error objective | 12.83 | +0.17 ± 0.14 | 0/3 | 40.72 | 4.43 | 4.2 |

**Interpretation:**

- **History length is the largest controllable factor for both model families.** A
  50-cycle input window (GRU) or 50-cycle rolling statistics (XGBoost) beat 30 on all 3
  folds, by 1.8 and 2.8 cycles; 20 cycles is worse on all 3.
  - This was found *after* the comparison in §1, which used 30 for both families, so §1
    remains a like-for-like comparison.
  - Adopting window 50 would need re-confirmation on the remaining folds; it has **not**
    been re-selected or re-run on cv3–cv4.
  - A longer window costs the GRU 2.7× fit time.
- **Temporal context matters for XGBoost:** without rolling features, RMSE rises by
  4.4 cycles and near-EOL RMSE more than doubles.
- **Hand-engineered per-step statistics hurt the GRU** (+2.2): the recurrent model learns
  better from the normalised raw window.
- **Capping is a trade-off between estimands, not a free win.** Uncapped training improves
  RMSE against true RUL by about 8 cycles but worsens capped RMSE by about 20, because early
  in life RUL is weakly identifiable from healthy-looking sensors.
- **Effects below the fold SD, which are not distinguishable from zero with 3 folds:**
  loss choice, sensor subsets, operating settings (FD001 has a single operating condition)
  and the XGBoost depth changes. The large GRU is not better than the base and costs 5×
  the training time.

## 4. Robustness (FD001 canonical validation engines; models not retrained)

Values are snapshot RMSE (capped); noise and missing-reading entries are means over
3 seeds.

| Model | Clean | Noise 0.5 SD | Noise 1 SD | Noise 2 SD | Missing 5 % | Missing 20 % | Missing 50 % | Mean / worst Δ, one sensor frozen | FD001 → FD003 shift |
|---|---|---|---|---|---|---|---|---|---|
| Ridge | 14.76 | 16.69 | 22.53 | 33.23 | 14.81 | 15.39 | 17.38 | +1.61 / +4.44 (s15) | 43.04 |
| XGBoost | 11.52 | 21.75 | 30.10 | 37.04 | 11.73 | 12.14 | 12.96 | +0.85 / +3.19 (s4) | 17.74 |
| HGB | 11.26 | 21.30 | 25.89 | 30.99 | 11.33 | 12.00 | 13.04 | +0.87 / +3.66 (s4) | 17.77 |
| TCN | 14.07 | 14.40 | 17.37 | 28.69 | 13.82 | 13.80 | 14.02 | +0.79 / +2.93 (s13) | 30.67 |
| GRU | 13.49 | 16.01 | 21.89 | 31.06 | 13.65 | 14.42 | 15.68 | +1.45 / +6.41 (s13) | 19.67 |
| Transformer | 13.00 | 14.71 | 19.99 | 32.83 | 13.02 | 13.37 | 14.99 | +0.80 / +2.79 (s13) | 27.51 |

**Interpretation:**

- **The tree models are the most sensitive to additive noise.** At 0.5 training-SD,
  RMSE nearly doubles (11.5 → 21.8 for XGBoost), whereas the TCN rises by only 0.3 and
  the Transformer by 1.7.
  - A plausible explanation, not tested: the trees split on current-cycle values and
    short rolling statistics, while the deep models' input normalisation and temporal
    filtering average the noise out.
  - At 2 SD all models degrade to 29–37.
- **Missing readings with causal forward-fill are tolerated by every model.** At 50 %
  missing, Δ ranges from −0.05 (TCN) to +2.6 (Ridge).
- **A single frozen sensor costs about 1 cycle on average.** The worst case is the GRU
  with sensor 13 frozen (+6.4).
- **FD001 → FD003 shift** (a second fault mode, validation engines of the FD003 split):
  - Every model degrades.
  - XGBoost, HGB and the GRU degrade least (17.7–19.7). The GRU has the best near-EOL true
    RMSE under shift: 18.7, against about 25.9 for the trees.
  - The Transformer and TCN degrade badly (27.5 and 30.7), and Ridge collapses (43.0).
  - This is a stress test of the FD001-trained models, not a transfer or zero-shot
    protocol. FD003-trained models are in §5.

## 5. FD002–FD004 benchmarks (canonical fold only, one seed)

Each subset has its own 70/15/15 engine split. Validation engines: FD003 15, FD002 39,
FD004 37. FD002 and FD004 use per-operating-condition normalisation (6 k-means clusters on
the settings, fitted on training engines). Classical models use the FD001 Stage-1
hyperparameters, and deep models use the FD001 architecture; **nothing was tuned per
subset**. TCN was omitted after its FD001 result.

Values are canonical-fold snapshot RMSE (capped), with near-EOL RMSE (true) in parentheses.

| Model | FD003 | FD002 | FD004 |
|---|---|---|---|
| Ridge | 17.18 (13.33) | 16.50 (10.18) | 17.52 (13.97) |
| XGBoost | 12.86 (4.71) | 13.41 (5.31) | 14.42 (6.89) |
| HGB | 12.95 (5.20) | **12.91** (5.37) | **14.22** (7.03) |
| GRU | 15.51 (5.09) | 14.75 (4.95) | 15.55 (5.51) |
| Transformer | **11.56** (4.40) | 13.47 (4.31) | 14.81 (5.36) |

Fit times and peak memory:

| Model | FD003 fit s / RSS MB | FD002 | FD004 |
|---|---|---|---|
| Ridge | 0.9 / 304 | 3.1 / 505 | 2.8 / 581 |
| XGBoost | 5.0 / 334 | 10.3 / 491 | 9.9 / 532 |
| HGB | 3.9 / 300 | 9.3 / 484 | 9.4 / 532 |
| GRU | 46.5 / 662 | 159.6 / 802 | 107.0 / 822 |
| Transformer | 214.6 / 785 | 213.6 / 922 | 318.1 / 946 |

Single-input latency was similar to FD001: Ridge 0.13 ms, XGBoost 0.24–0.32 ms,
HGB 6.3–7.9 ms, GRU 1.04–1.20 ms, Transformer 1.28–1.40 ms.

**Interpretation:**
- These are single-fold, single-seed numbers. FD001 showed canonical-fold differences of
  1–2 cycles reversing under CV, so **no ranking is claimed**.
- On FD003 the Transformer ran to the 40-epoch limit without early stopping (best epoch
  34), so it may be under-trained.
- One consistent observation: the deep models have equal or lower near-EOL RMSE than the
  trees on FD002 and FD004 (4.3–5.5 vs 5.3–7.0), even where their overall capped RMSE is
  higher.

## 6. Error analysis

- **RMSE against true RUL (41–44 on FD001) is dominated by early-life snapshots.** True RUL
  there exceeds the cap, so every capped model under-predicts by design. Capped errors and
  near-EOL errors (≈ 4.5–5 cycles for the better models) are the meaningful accuracy
  measures.
- **The PHM08 score diverges from RMSE.** The Transformer has the lowest CV RMSE but a
  higher mean PHM08 score than HGB (4.82 vs 3.79). It has the largest positive bias
  (+1.64 cycles mean snapshot bias, capped, against +0.87 for HGB). Its penalty comes mostly
  from folds cv2 and cv3 (7.8 and 8.4 vs 4.7 and 5.3 for HGB). PHM08 penalises late
  (over-) predictions exponentially more, so a few large over-predictions dominate it.
- **Variation between folds (SD ≈ 1.7–2.8) is larger than most differences between
  models.** Folds cv2 and cv3 are hard for every model. With 17 validation engines per
  fold, individual engines move the metric substantially.
- **The TCN's poor result is not under-training:** all its fits early-stopped. Its
  receptive field (29) matches the window, and it reads only the newest step's
  representation. It was not tuned. This architecture is untuned, not ruled out.

## 7. Limitations (unresolved)

1. **Only 100 FD001 training engines.** Validation folds have 17 engines and calibration
   has 15. Fold SDs understate uncertainty because training sets overlap, and no
   significance tests are reported.
2. **The classical models' hyperparameters were selected on the canonical validation
   engines** (Stage 1), which are part of the CV folds. That gives them a mild advantage.
   Deep architectures were fixed a priori and not tuned. Neither family received a fair
   per-fold nested tuning.
3. **Ablations used 3 folds and one seed**, and the deep runs show seed sensitivity (not
   measured). The window-50 improvement is not yet confirmed on all folds or adopted.
4. **Conformal guarantees are marginal, over engines and the snapshot-drawing scheme.**
   They assume calibration and new engines are exchangeable. They do not hold under shift
   (FD003 faults, sensor failures), and there is no conditional coverage by RUL range.
   With n = 15, finite 95 % intervals are impossible.
5. **The validation snapshot distribution only approximates the official test truncation
   distribution**, which is undocumented. Test-set behaviour is unknown, and no Stage-2
   model has been tested.
6. **The FD002–FD004 benchmarks are one fold and one seed**, without per-subset tuning.
   Per-condition normalisation uses k = 6 clusters, an assumption that matches the known
   six conditions but was not validated per subset.
7. **Robustness corruptions are synthetic** (Gaussian, i.i.d. missing, constant-frozen
   sensor) and may not represent real sensor failure modes. C-MAPSS itself is simulated.
8. **Latency and memory were measured on one laptop that throttles.** Medians over repeats
   are reported, but the absolute values are machine-specific.

## 8. Completed vs not executed

**Completed:**
- Deep models (TCN, GRU, Transformer) pass the smoke tests and have genuine 5-fold + canonical FD001 runs.
- 6-model comparison with resources.
- UQ with 7 interval methods × 2 levels × role swap.
- 23-model ablation (3 folds).
- Robustness (noise, dropout, missing, FD001 → FD003 shift).
- FD002, FD003 and FD004 canonical benchmarks.

**Not executed:**
- Multi-seed repeats of deep models.
- Ablations on folds cv3–cv4.
- Nested hyperparameter tuning.
- CV on FD002–FD004.
- UQ on FD002–FD004.
- Quantile versions of GRU and TCN.
- Any official test-set evaluation.

## Reproduce

```bash
unset VIRTUAL_ENV; export OMP_WAIT_POLICY=PASSIVE
uv sync --locked --extra deep
uv run aeromind s2-smoke --no-isolate                                  # seconds, synthetic
uv run aeromind s2-compare    --config configs/stage2/fd001_comparison.yaml
uv run aeromind s2-uq         --config configs/stage2/fd001_uq.yaml
uv run aeromind s2-robustness --config configs/stage2/fd001_robustness.yaml
uv run aeromind s2-ablation   --config configs/stage2/fd001_ablation.yaml
for d in fd003 fd002 fd004; do uv run aeromind s2-compare --config configs/stage2/${d}_benchmark.yaml; done
```

`fd001_uq.yaml` and `fd001_robustness.yaml` reference the comparison run directory by name.
Edit `comparison_run` if you re-run the comparison.
