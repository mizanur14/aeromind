# CLAUDE.md — AeroMind repository instructions

AeroMind: uncertainty-aware aircraft-engine Remaining Useful Life (RUL) prediction on
NASA C-MAPSS (FD001–FD004). These rules bind every contributor, human or AI.

## Research objectives

1. Leakage-free, reproducible RUL baselines on C-MAPSS (Stage 1).
2. Calibrated uncertainty for RUL (Stage 2: conformal / quantile methods using the
   reserved **calibration** engines).
3. A transparent research demonstration (Stage 3: registry, API, dashboard) that never
   presents simulation results as operational or certified.
4. Honest reporting: every number in a report must come from a saved run directory.

## Architecture

```
src/aeromind/
  cli.py              single argparse CLI (`aeromind ...` / `python -m aeromind ...`)
  config.py           typed YAML experiment configuration
  utils.py            logging, hashing, environment metadata, memory/timing
  data/cmapss.py      parsing, schema & trajectory validation, RUL labels, sensor metadata
  data/provenance.py  SHA-256 provenance manifests and verification
  data/splits.py      deterministic engine-level train/validation/calibration splits
  eda/                exploratory analysis functions (analysis.py) and report writer (report.py)
  features/           targets (capping), causal rolling features, fitted FeaturePipeline
  models/             baselines, model factory, bundle save/load
  evaluation/         metrics (MAE/RMSE/PHM08), snapshot & window evaluation, plots
  experiments/        Stage-1 train / evaluate / test / compare / audit workflows
  deep/               Stage-2 PyTorch models (optional `deep` extra): causal windows, TCN/GRU/
                      Transformer, deterministic trainer with checkpoints/resume, predictor bundle
  research/           Stage-2 experiments: engine-level CV comparison, UQ (quantile + conformal),
                      ablations, robustness; each model trained in its own process for memory
  serving/            Stage-3 registry (hash-pinned bundles, metrics from reports, conformal
                      calibrators), ModelService, CPU benchmark, bundle export/import, model
                      cards, hypothetical maintenance simulation, synthetic smoke
  api/                FastAPI app over ModelService (optional `api` extra)
  dashboard/          Streamlit app + Streamlit-free data loaders (optional `dashboard` extra)
configs/              experiment YAML files
data/raw/cmapss/      NASA files (never modified, never committed)
data/manifests/       split + provenance manifests (committed)
artifacts/            run directories with model bundles (gitignored)
reports/              small CSV/JSON/PNG summaries, registry, benchmarks, test ledger (committed)
examples/             API client example (builds requests from local NASA files)
scripts/              legacy Stage-0 scripts (kept working; CLI supersedes them)
```

## Environment

- Ubuntu, Python 3.12, `uv` (always `uv run ...`; `uv sync --locked --all-extras` for Stages 2–3).
- PyTorch 2.14.0+cpu from the explicit `pytorch-cpu` index; set `OMP_WAIT_POLICY=PASSIVE`.
- Intel i7-1165G7 (4C/8T), 16 GB RAM, **CPU only**. No CUDA dependency.
- The volume is NTFS (`UV_LINK_MODE=copy`). If a conda env is active, `unset VIRTUAL_ENV`.

## CPU-first constraints

- Stage 1 used classical models only; deep models (Stage 2) stay small and CPU-sized.
- Keep grids small; `n_jobs` ≤ 8. Single FD001 baseline run should take minutes, not hours.
- Laptop throttles: report latency as medians over repeats, never a single pass.

## Leakage prevention (non-negotiable)

1. Splits are **engine-level**. An engine ID belongs to exactly one of
   train / validation / calibration. IDs are only meaningful within one subset (FD00x);
   never join engines across subsets.
2. Official test labels (`RUL_FD00x.txt`) are read **only** by `aeromind test` (and by
   `aeromind validate` for integrity checks, without reporting label statistics). Training,
   tuning, feature selection, calibration and EDA never read them.
   - `aeromind test` requires `--confirm-final`, refuses to re-evaluate a run, and appends to
     `reports/test_set_ledger.json`. Read the ledger before any new test evaluation.
   - Never change features, models, grids, targets, calibration or the choice of Stage-2 base
     model in response to test numbers (including the post-hoc `--all-models` table).
   - Run `uv run aeromind audit <runs...>` before reporting results.
3. Every fitted transformation (constant-sensor removal, scaler, operating-condition
   clustering) is fitted on **training engines only**.
4. Features at cycle *t* use observations at cycles ≤ *t* of the same engine only:
   trailing windows, no centered windows, no backward fill, no cross-engine rolling.
5. `engine_id`, `rul`, and target columns are never model inputs.
6. Hyperparameters and model choice are selected on **validation** engines. Calibration
   engines are untouched until Stage 2.
7. EDA scope `pool` (all training-pool engines) is exploratory only; feature-selection
   decisions must come from scope `train` or from fitted pipeline steps.

## Stage-2 rules

- No Stage-2 model is evaluated on the official test set during development. Any future
  final evaluation goes through `aeromind test` and the ledger.
- Architecture choice, tuning and ablations use engine-level folds over train+validation
  engines (`research/spec.make_folds`); calibration engines are used only by `s2-uq` and the `s3-registry` calibrators.
- Deep models early-stop on an inner engine holdout of the fold's training engines.
- Every comparison uses the same split, target cap, snapshots and metrics; report `_capped`
  and `_true` separately and fold variability (mean ± SD, paired differences). Never claim a
  model is better from differences within fold variability.
- HistGradientBoosting is the Stage-1 *reference*, not a proven superior architecture.
- Smoke-test (`aeromind s2-smoke`, then a 2-epoch real-data run) before expensive training.

## Stage-3 rules

- Serve only bundles listed in `reports/registry/model_registry.json`; never bypass the
  SHA-256 check or load bundles from untrusted archives (use `aeromind s3-import`).
- Registry metrics are copied from run reports; official test metrics only from the ledger.
  Rebuild with `aeromind s3-registry`, then `aeromind s3-model-cards`; never hand-edit cards.
- Dashboard/API must label evidence (CV, canonical validation, calibration, official test,
  dataset ground truth, hypothetical) and attach provenance to every prediction.
- Engine-level dashboard predictions only for validation engines (or uploads without truth);
  never read official test files in serving code.
- Never fabricate predictions, intervals, confidence or savings; an unavailable model is an
  error (503), not a fallback.
- Model weights, NASA data, `dist/` archives and `.env` files are never committed.

## Testing standards

- `uv run pytest` must pass before claiming any phase complete; `uv run ruff check .` clean.
- Tests needing real NASA files are marked `real_data` and skip when files are absent (CI).
- New features need leakage tests (truncation invariance, engine-order invariance).
- `uv run aeromind smoke`, `s2-smoke` and `s3-smoke` exercise the pipelines on synthetic data in seconds.

## Model evaluation protocol

- Training target: piecewise-capped RUL `min(RUL, cap)` (default cap 125) or uncapped.
- Validation (model selection): engine-equal **snapshot** evaluation — a fixed, seeded set
  of cut cycles per validation engine; one prediction per snapshot. Selection metric is
  RMSE against the model's own training-target truth (capped if trained capped).
- Also reported: all-window metrics, row-weighted and engine-weighted.
- Every metric is reported against **true uncapped RUL** and against **capped truth**
  (column suffix `_true` / `_capped`). Never compare the two without saying so.
- Final NASA test: one prediction per test engine at its last observed cycle, compared
  with the official RUL. Run only after selection; each run is logged in metadata.
- PHM08 score: `d = pred − true`; `exp(−d/13) − 1` if `d < 0`, else `exp(d/10) − 1`; summed.

## Reproducibility

- Fixed seeds in config; deterministic splitting reproduces `fd001_split_seed42.json`.
- Each run saves resolved config, data SHA-256s, split-manifest hash, package versions,
  git commit, timings and peak memory in `metadata.json`.
- `data/manifests/cmapss_provenance.json` pins raw-file hashes; `aeromind validate` checks them.

## Documentation standards

- Typed, docstringed public functions. Ruff-clean.
- README commands must be ones that were actually run.
- Never invent metrics, citations or results. Report failures as failures.

## Known dataset limitations

- Simulated (C-MAPSS) data; not real flight data. Sensor noise is synthetic.
- `readme.txt` lists FD004 as 248 train / 249 test; the files contain 249 train / 248 test
  engines. It also says "26 sensors"; there are 21.
- Test trajectories are truncated at unknown, non-random points; the truncation
  distribution is not documented, so validation snapshots only approximate it.
- FD001 sensors 1, 5, 10, 16, 18, 19 (and setting 3) are constant; sensor 6 is near-constant.
- RUL early in life is weakly identifiable (engines look healthy), hence capped targets.
- Only 100 FD001 training engines: validation (15 engines) metrics have high variance.
- Raw files and model artifacts must not be committed or redistributed beyond NASA terms.
