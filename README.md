# AeroMind

![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![CPU only](https://img.shields.io/badge/compute-CPU%20only-lightgrey)

**Uncertainty-aware Remaining Useful Life (RUL) prediction for turbofan engines on the NASA
C-MAPSS simulation benchmark**, with leakage-safe evaluation, calibrated prediction
intervals, a REST API and a research dashboard.

> **Read this first**
> - The data is **NASA's C-MAPSS simulation benchmark**: simulation-generated, not flight data.
> - AeroMind is an **independent research project**. NASA supplied the data but has **not
>   endorsed** this project.
> - Nothing here is **certified or validated for operational aviation maintenance or safety
>   decisions**.
> - AeroMind is **not a physical digital twin**: its models are statistical regressors
>   fitted to simulator output.
>
> See [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## What is in the box

| Stage | Contents |
|---|---|
| **1 · Baselines** | Validated data layer with SHA-256 provenance; seeded engine-level splits; causal features; six classical baselines; one ledgered official test evaluation; integrity audit |
| **2 · Research** | TCN, GRU and Transformer (PyTorch, CPU); 5-fold engine-level CV; split-conformal and quantile UQ; ablations; robustness; FD002–FD004 benchmarks |
| **3 · Platform** | Hash-pinned model registry with conformal calibrators; FastAPI service; Streamlit dashboard; CPU benchmark; model cards; bundle export/import |

## Results at a glance (FD001)

All numbers are in cycles and come from saved runs ([docs/RESULTS_INDEX.md](docs/RESULTS_INDEX.md)).

**Validation, 5-fold engine-level CV** (85 development engines, 17 per fold). Errors are
against capped RUL `min(RUL, 125)`; the paired Δ is per fold against the HGB reference.

| Model | Snapshot RMSE (mean ± SD) | Paired Δ vs HGB (folds better) | Near-EOL RMSE, true RUL ≤ 30 | Single-request CPU latency, p50 |
|---|---|---|---|---|
| HistGradientBoosting (reference) | 13.97 ± 1.79 | — | 4.86 | 29.9 ms |
| XGBoost | 13.71 ± 1.67 | −0.26 (5/5) | 4.53 | 24.9 ms |
| GRU | 13.47 ± 2.49 | −0.50 (4/5) | 4.94 | 10.0 ms |
| Transformer | 13.37 ± 2.36 | −0.60 (4/5) | 4.46 | 21.7 ms |
| TCN | 17.87 ± 2.75 | +3.90 (0/5) | 8.46 | 9.7 ms |
| Ridge | 16.65 ± 1.66 | +2.68 (0/5) | 11.96 | 23.2 ms |

- **No model is demonstrably best.** The differences among GRU, Transformer, XGBoost and HGB
  are within fold variability.
- Latency is the median for one full engine history (259 cycles) through the inference
  service, measured on an Intel i7-1165G7 laptop ([docs/CPU_BENCHMARKING.md](docs/CPU_BENCHMARKING.md)).

**Official NASA test set**, evaluated **once**, Stage 1 only, as recorded in
[`reports/test_set_ledger.json`](reports/test_set_ledger.json). The validation-selected HGB
scores RMSE 13.60 against true RUL, MAE 9.64 and PHM08 288.6 over the 100 test engines. No
Stage-2 or Stage-3 model has been evaluated on the test set.

**Prediction intervals.** Engine-level split conformal on the 15 calibration engines, at a
90 % nominal level:
- On validation engines, coverage of capped RUL is 0.94–0.95 on average.
- Coverage of the realised draw ranges from 0.83 to 1.00 between the 5th and 95th
  percentiles.
- Mean interval width is 53–61 cycles.

The guarantee is marginal and assumes new engines resemble the calibration engines
([docs/STAGE2_RESULTS.md](docs/STAGE2_RESULTS.md) §2).

## Quick start

```bash
git clone https://github.com/mizanur14/aeromind.git aeromind && cd aeromind
uv sync --locked --all-extras        # Python 3.12; core + PyTorch CPU + API + dashboard
uv run pytest                        # real-data tests skip without the NASA files
uv run aeromind s3-smoke             # synthetic end-to-end: registry, API, dashboard (~10 s)
```

[uv](https://docs.astral.sh/uv/) installs everything. The extras are optional:

| Extra | Adds |
|---|---|
| `deep` | PyTorch 2.14 CPU wheels |
| `api` | FastAPI, uvicorn, httpx |
| `dashboard` | Streamlit |

The core install (`uv sync --locked`) is enough for the Stage-1 pipeline.

## 1. Get the data (not included)

Download C-MAPSS from NASA, either from the
[NASA Open Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) or the
[PCoE Data Set Repository](https://www.nasa.gov/Intelligent-Systems-Division/Discovery-and-Systems-Health/Pcoe/Pcoe-Data-Set-Repository/).
Put the text files in `data/raw/cmapss/`, then check them:

```bash
uv run aeromind validate --dataset all      # checks schema and SHA-256 against data/manifests/
```

AeroMind never downloads or redistributes NASA data. See [docs/DATA.md](docs/DATA.md).

## 2. Get model artifacts (not in Git)

Model weights are kept out of the repository. The committed
[model registry](reports/registry/model_registry.json) lists 22 servable models, each
pinned by SHA-256. Choose one way to get compatible bundles:

| Option | Command | Time |
|---|---|---|
| **Import a release archive** (published with a GitHub Release by the maintainer) | `uv run aeromind s3-import aeromind-models.tar.gz` | seconds |
| **Regenerate the FD001 models** (canonical fold only) | `uv run aeromind s2-compare --config configs/stage3/fd001_serving_models.yaml`, then point `configs/stage3/registry.yaml` at the new run and `uv run aeromind s3-registry` | ~5 min |
| **Reproduce everything** | see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | ~2 h CPU |

- `s3-import` writes only files whose path and SHA-256 match the committed registry. Bundles
  are pickles, so never load bundles from a source you don't trust.
- Bundles load reliably only with the library versions pinned in `uv.lock` (scikit-learn
  1.9.1, XGBoost 3.4.1, PyTorch 2.14.0), so install with `uv sync --locked`.
- Without bundles, the API reports models as unavailable (HTTP 503) and the dashboard shows
  only its report-based sections. Nothing is ever faked.

## 3. Run the API

```bash
uv run aeromind serve                       # http://127.0.0.1:8000, interactive docs at /docs
uv run python examples/api_client.py        # predicts a held-out validation engine (needs NASA files)
curl -s -X POST http://127.0.0.1:8000/models/fd001-s2-gru/predict \
     -H 'Content-Type: application/json' -d @examples/request.json   # body written by the client
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status and the number of registered and available models |
| `GET /models` | Registry summary: availability, interval support, CV and canonical RMSE |
| `GET /models/{id}` | Full metadata: metrics by evidence type, inputs, calibrator, limitations, bundle hash |
| `POST /models/{id}/predict` | One engine's raw cycles → RUL at the last cycle, optional conformal intervals (`interval_levels: [0.8, 0.9]`) and per-cycle trajectory |
| `POST /models/{id}/predict/batch` | Up to 500 engines |

- **Inputs:** each request carries the raw C-MAPSS columns `cycle, setting_1..3,
  sensor_1..21` for every observed cycle. The bundle's own fitted preprocessing is applied.
- **Errors:** errors are typed JSON (`invalid_request`, `invalid_input`, `model_not_found`,
  `model_unavailable`, `interval_unavailable`).
- **Provenance:** every prediction carries the model version, the bundle SHA-256 and the
  source run.
- **Security:** the API has no authentication. Keep it on localhost.
- **Units:** inputs must use C-MAPSS units. Implausible but finite values are not detected
  ([docs/LIMITATIONS.md](docs/LIMITATIONS.md)).

## 4. Run the dashboard

```bash
uv run aeromind dashboard                   # http://127.0.0.1:8501
```

| Section | Shows |
|---|---|
| Overview | Objective, dataset, research status, model availability, limitations |
| Engine Health Intelligence | Held-out validation engine (or an uploaded CSV): sensor trajectories, predicted RUL at every cycle with a conformal band, ground truth (validation engines only) |
| Model Benchmarking | Model selector; CV and canonical MAE/RMSE/PHM08; per-fold spread; paired differences; latency, size, memory; the official test result separately labelled |
| Sensor Intelligence | Distributions early vs near failure; correlations; XGBoost gain importance; trajectories aligned at failure |
| Robustness & Experiments | Ablations; noise, dropout, missing readings and shift; interval calibration; reproducibility metadata, ledger, data hashes |
| Maintenance Simulation | **Hypothetical** threshold policies on validation engines with **user-assumed** costs |

Every view carries an evidence label: model result (validation), dataset ground truth,
official test (ledgered), or hypothetical illustration.

## Reproduce the research

```bash
uv run aeromind split --dataset FD001 --seed 42                          # reproduces the committed manifest
uv run aeromind train --config configs/fd001_baseline.yaml               # Stage 1 (~2 min)
uv run aeromind s2-compare --config configs/stage2/fd001_comparison.yaml # Stage 2 CV (~24 min)
uv run aeromind s2-uq --config configs/stage2/fd001_uq.yaml
uv run aeromind s3-registry && uv run aeromind s3-model-cards           # Stage 3 registry + cards
uv run aeromind s3-benchmark                                             # CPU latency
uv run aeromind audit artifacts/runs/<run_a> artifacts/runs/<run_b>      # integrity audit
```

The exact commands behind every reported run are in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). `aeromind --help` lists all commands.

## Leakage safeguards

- Engine-level train / validation / calibration splits. All fitted preprocessing sees only
  training engines and is saved inside each bundle.
- Causal features only: no centred windows, no backward fill. Truncation invariance and
  order invariance are unit-tested.
- Calibration engines are used only for conformal calibration.
- **Official test labels** are read only by `aeromind test`. It requires `--confirm-final`,
  refuses repeats, and writes to an append-only ledger. An AST audit checks every call site.

## Documentation

| Topic | Document |
|---|---|
| Architecture and diagram | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Dataset setup and citation | [docs/DATA.md](docs/DATA.md) |
| Technical methodology (overview) | [docs/TECHNICAL_METHODOLOGY.md](docs/TECHNICAL_METHODOLOGY.md) · Stage 1 [METHODOLOGY.md](docs/METHODOLOGY.md) · Stage 2 [STAGE2_METHODOLOGY.md](docs/STAGE2_METHODOLOGY.md) |
| Evaluation protocol | [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md) |
| Results | [docs/RESULTS_INDEX.md](docs/RESULTS_INDEX.md) · [RESULTS.md](docs/RESULTS.md) · [STAGE2_RESULTS.md](docs/STAGE2_RESULTS.md) · [STAGE1_AUDIT.md](docs/STAGE1_AUDIT.md) |
| Model cards | [docs/MODEL_CARDS.md](docs/MODEL_CARDS.md) |
| Reproducibility | [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) |
| CPU benchmarking | [docs/CPU_BENCHMARKING.md](docs/CPU_BENCHMARKING.md) |
| Limitations and ethics | [docs/LIMITATIONS.md](docs/LIMITATIONS.md) |
| Troubleshooting | [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |
| Roadmap | [docs/ROADMAP.md](docs/ROADMAP.md) |
| References (verified) | [docs/REFERENCES.md](docs/REFERENCES.md) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) · project rules [CLAUDE.md](CLAUDE.md) |

## Repository layout

```
src/aeromind/   data · features · models · evaluation · experiments (Stage 1)
                deep · research (Stage 2) · serving · api · dashboard (Stage 3)
configs/        YAML experiment, registry and results-index configs
data/manifests/ split and provenance manifests (committed); data/raw/ is local only
reports/        small run reports, registry, benchmarks, test-set ledger (committed)
artifacts/      model bundles and checkpoints (local only, gitignored)
docs/ tests/ examples/
```

## Citation

If you use the data, cite NASA's dataset and the C-MAPSS paper:

- A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA
  Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
- A. Saxena, K. Goebel, D. Simon, N. Eklund. "Damage Propagation Modeling for Aircraft Engine
  Run-to-Failure Simulation." *Proc. 1st Int. Conf. on Prognostics and Health Management
  (PHM08)*, Denver, CO, 2008.

Methods references are listed, verified, in [docs/REFERENCES.md](docs/REFERENCES.md).

## License

AeroMind's code and documentation are released under the [MIT License](LICENSE). The NASA
C-MAPSS data is not part of this repository and is subject to NASA's terms.
