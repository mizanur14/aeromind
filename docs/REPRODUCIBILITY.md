# Reproducibility guide

## Environment

```bash
cd aeromind
uv sync --locked          # exact versions from uv.lock (numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, xgboost 3.4.1)
uv sync --locked --all-extras   # + torch 2.14.0+cpu, fastapi 0.163.1, uvicorn 0.54.0, streamlit 1.67.0
```

Python 3.12 (`.python-version`). CPU only. If a conda environment is active, run
`unset VIRTUAL_ENV` first so uv uses the project `.venv`.

## Data integrity

```bash
uv run aeromind validate --dataset all      # must print "Provenance: all hashes match"
```

## Deterministic elements

| Element | Control |
|---|---|
| Engine split | `numpy.random.default_rng(42)`, sorted IDs. The regenerated manifest equals the committed one (tested) |
| Validation snapshots | `evaluation.snapshot_seed`. Order-independent (tested) |
| Models | `seed` injected as `random_state`; XGBoost `hist` on CPU |
| Features | deterministic; no fitted randomness for FD001 (k-means only when `n_conditions > 1`) |

Reloaded bundles reproduce the saved validation predictions exactly
(`report/reload_check.json`: max |Δ| < 1e-13; exactly 0 except Random Forest at 2.8e-14 / 5.7e-14). Re-training on the same machine and lock file
should reproduce the metrics. Multithreaded tree libraries may differ in the last
floating-point digits on other CPUs.

## Exact commands used for the Stage-1 results

```bash
uv run aeromind validate --dataset all --write-manifest
uv run aeromind split --dataset FD001 --seed 42
uv run aeromind eda --dataset FD001 --scope both
uv run aeromind train --config configs/fd001_baseline.yaml      # -> fd001_baseline_20261117-105759
uv run aeromind evaluate --run-dir artifacts/runs/fd001_baseline_20261117-105759
uv run aeromind train --config configs/fd001_uncapped.yaml      # -> fd001_uncapped_20261117-110014
uv run aeromind evaluate --run-dir artifacts/runs/fd001_uncapped_20261117-110014
uv run aeromind test --run-dir artifacts/runs/fd001_baseline_20261117-105759 --all-models
#   (run once, before the guard existed; today the same command needs --confirm-final and
#    is refused for this run because it is already in reports/test_set_ledger.json)
uv run aeromind compare artifacts/runs/fd001_baseline_20261117-105759 artifacts/runs/fd001_uncapped_20261117-110014 \
    --output reports/FD001/comparisons/fd001_capped_vs_uncapped.csv
uv run aeromind features --config configs/fd001_baseline.yaml
uv run aeromind audit artifacts/runs/fd001_baseline_20261117-105759 artifacts/runs/fd001_uncapped_20261117-110014
```

A new run gets a new timestamped directory. Pass `--run-dir` to choose one; existing
non-empty directories are never overwritten.

## Stage-2 environment and commands

```bash
unset VIRTUAL_ENV
uv sync --locked --extra deep        # adds torch 2.14.0+cpu from the explicit pytorch-cpu index
export OMP_WAIT_POLICY=PASSIVE
```

Deep training is deterministic for a fixed seed, thread count and machine:
- `torch.manual_seed` is set, with deterministic algorithms and a seeded DataLoader generator.
- `last.pt` stores the model, optimiser, scheduler and RNG states, so an interrupted
  training continues exactly when the same `--run-dir` is passed again. This is tested.
- The best inner-validation checkpoint is saved as `best.pt`.
- Each model is trained in its own spawned process. Re-running a comparison into the same
  `--run-dir` skips models whose results already exist.

The commands that produced the reported Stage-2 runs (see `docs/STAGE2_RESULTS.md`):

```bash
uv run aeromind s2-smoke --no-isolate
uv run aeromind s2-compare    --config configs/stage2/fd001_comparison.yaml   # -> s2_fd001_comparison_20261117-114027
uv run aeromind s2-robustness --config configs/stage2/fd001_robustness.yaml   # -> s2_fd001_robustness_20261117-120738
uv run aeromind s2-uq         --config configs/stage2/fd001_uq.yaml           # -> s2_fd001_uq_20261117-121004
uv run aeromind s2-ablation   --config configs/stage2/fd001_ablation.yaml     # -> s2_fd001_ablation_20261117-121310
uv run aeromind s2-compare    --config configs/stage2/fd003_benchmark.yaml    # -> s2_fd003_benchmark_20261117-130357
uv run aeromind s2-compare    --config configs/stage2/fd002_benchmark.yaml    # -> s2_fd002_benchmark_20261117-130845
uv run aeromind s2-compare    --config configs/stage2/fd004_benchmark.yaml    # -> s2_fd004_benchmark_20261117-131540
```

`fd001_uq.yaml` and `fd001_robustness.yaml` point at the comparison run directory. Run
metadata records package versions; `torch` has been tracked since after these runs, and
its version is pinned in `uv.lock`.

## Stage-3 commands (registry, serving, benchmark)

```bash
uv sync --locked --all-extras
uv run aeromind s3-registry --config configs/stage3/registry.yaml   # -> reports/registry/model_registry.json
uv run aeromind s3-model-cards                                      # -> docs/MODEL_CARDS.md
uv run aeromind s3-benchmark --repeats 30                           # -> reports/benchmarks/s3_benchmark_20261117-135548
uv run aeromind s3-smoke                                            # synthetic registry + API + dashboard
```

Deterministic elements of Stage 3:
- **Calibrator:** fixed seed (42) for the one-snapshot-per-engine draw. Rebuilding the
  registry from the same bundles and data reproduces the thresholds exactly.
- **Versions:** each registry entry pins its bundle by SHA-256, and `version` is the first
  12 hex digits.
- **Rebuilds:** a rebuilt registry differs only in `built_utc`, `git_commit` and package
  versions.

**Model artifacts without retraining:**
1. A maintainer exports the bundles:
   `uv run aeromind s3-export --output dist/aeromind-models.tar.gz` (22 bundles, a 4.5 MB archive).
2. They attach the archive to a GitHub Release, together with its SHA-256.
3. Users install it with `uv run aeromind s3-import aeromind-models.tar.gz`. Only files whose
   path and SHA-256 match the committed registry are written.

**Regenerating instead:**
- `uv run aeromind s2-compare --config configs/stage3/fd001_serving_models.yaml` (canonical
  fold only, about 5 minutes).
- Re-trained bundles have new hashes. Point `configs/stage3/registry.yaml` at the new run,
  then rebuild with `aeromind s3-registry`.
- The rebuilt registry then reflects your own runs. Reproducing the full CV metrics needs
  the 24-minute `fd001_comparison.yaml` run.

## What each run records

`metadata.json`: config name, seed, SHA-256 of the data files used, split-manifest SHA-256,
engine IDs per partition, package versions, git commit (with `-dirty`), timings, peak RSS,
selected model and parameters, and a log of every official-test evaluation.
