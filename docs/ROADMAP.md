# Research roadmap

Status as of v0.3.0. "Done" items have saved runs; everything else is planned, not started.

## Done

- **Stage 1:**
  - Validated data layer with SHA-256 provenance and engine-level splits.
  - Causal features and six classical baselines.
  - One ledgered official FD001 test evaluation.
  - Integrity audit.
- **Stage 2:**
  - TCN, GRU and Transformer models.
  - 5-fold engine-level CV on FD001.
  - Split-conformal and quantile UQ with CQR.
  - One-factor ablations (3 folds).
  - Robustness to noise, dropout, missing readings and shift.
  - FD002–FD004 canonical-fold benchmarks.
- **Stage 3:**
  - Hash-pinned model registry with conformal calibrators.
  - FastAPI service and Streamlit dashboard.
  - CPU benchmark, model cards and bundle export/import.

## Next (methodological priorities)

1. **Seed variability for deep models.** Repeat the FD001 CV with ≥ 3 seeds per
   architecture, to separate seed noise from architecture effects.
2. **Confirm the window-50 ablation.** Run it on folds cv3–cv4, and adopt it only if it holds.
3. **Fair tuning.** Nested, engine-level tuning for both classical and deep models, with the
   same budget.
4. **CV and UQ on FD002–FD004.** Five-fold comparisons, and calibration with each subset's
   own calibration engines.
5. **Conditional coverage.** Mondrian (RUL-bin-stratified) or locally weighted conformal
   intervals, evaluated per RUL range.
6. **Shift-aware uncertainty.** Detect out-of-distribution inputs (operating regime, sensor
   faults) and abstain or widen intervals. Evaluate on the FD001 → FD003 stress test.

## Later

- Physics-informed features, or hybrid models using the C-MAPSS health parameters (study
  only; still not a digital twin).
- Validation on another public run-to-failure dataset, for example N-CMAPSS, if its terms
  allow.
- A single final, ledgered test evaluation of the Stage-2 selection, after all of the above
  are frozen.

## Explicitly out of scope

- Operational deployment, certification, or real-fleet decision support.
- Automatic data download or redistribution of NASA data.
