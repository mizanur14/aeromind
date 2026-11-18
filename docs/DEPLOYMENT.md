# Deploying the public dashboard

The public dashboard is the Streamlit app in **public mode**: it shows the committed reports
and registry metadata only. It does not need the NASA files, model bundles, the FastAPI
service, PyTorch, XGBoost or scikit-learn, and it never loads a model.

## Entrypoint and files

| File | Role |
|---|---|
| `streamlit_app/streamlit_app.py` | Entrypoint. Adds `src/` to the import path, forces public mode, runs `aeromind.dashboard.app.main()` |
| `streamlit_app/requirements.txt` | Exact pins for the deployment (35 packages: the direct dependencies and all their transitive dependencies), every version taken from `uv.lock` |
| `streamlit_app/requirements.in` | The five direct dependencies the pins are generated from |
| `.streamlit/config.toml` | Theme and public-facing settings, read from the repository root |
| `reports/`, `configs/stage3/results_index.yaml`, `data/manifests/` | The committed content the dashboard displays |

### Streamlit Community Cloud settings

- **Repository / branch:** this repository, `main`.
- **Main file path:** `streamlit_app/streamlit_app.py`.
- **Python version:** 3.12 (advanced settings).
- **Secrets:** none. The public dashboard needs no secrets, tokens or environment variables.

Community Cloud looks for a dependency file first in the entrypoint's folder, then at the
repository root, using the first of `uv.lock`, `requirements.txt` or `pyproject.toml` it finds.
`streamlit_app/requirements.txt` is therefore used, and the root `uv.lock` (the full research
environment) is not. Apps start from the repository root, which is where `.streamlit/config.toml`
must live.

## Regenerating the pins

After changing `uv.lock`, regenerate the deployment pins so they stay identical to the lock.
The test `test_requirements_are_pinned_to_uv_lock_and_exclude_research_stack` fails if they drift.

```bash
uv export --locked --all-extras --no-hashes --no-emit-project --no-dev -o /tmp/aeromind-constraints.txt
uv pip compile streamlit_app/requirements.in -c /tmp/aeromind-constraints.txt --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 --no-header --no-annotate -o /tmp/aeromind-req.txt
```

Then replace the package lines of `streamlit_app/requirements.txt` with `/tmp/aeromind-req.txt`
and keep the header comment.

## Verifying locally before a deployment

```bash
# 1. Full research environment: lint and the whole test suite
uv sync --locked --all-extras
uv run ruff check .
uv run pytest -q

# 2. Deployment-equivalent environment: only the public requirements, fresh Python 3.12
uv venv /tmp/aeromind-public --python 3.12
uv pip install --python /tmp/aeromind-public/bin/python -r streamlit_app/requirements.txt pytest
PYTHONPATH=src /tmp/aeromind-public/bin/python -m pytest -q tests/test_dashboard_public.py
/tmp/aeromind-public/bin/python -m streamlit run streamlit_app/streamlit_app.py --server.headless true --server.port 8502
curl -sf http://127.0.0.1:8502/_stcore/health       # "ok"

# Or, from the research environment:
uv run aeromind dashboard --public
```

The CI job `public-dashboard` runs step 2 on every push.

## What public mode guarantees (tested)

- Only the six public sections are offered. The research pages (Engine Health, Sensor
  Intelligence, Maintenance Simulation) do not exist in public mode, even if bundles or NASA
  files are present on the host.
- Rendering every public section imports none of `torch`, `xgboost`, `joblib`, `sklearn` or
  `scipy`, and opens no NASA file (`train_`, `test_` or `RUL_FD00x.txt`), no model bundle and
  nothing under `artifacts/` or `data/raw/`. A Python audit hook checks this in a fresh
  interpreter.
- No upload widget is active, and nothing a visitor supplies is executed or deserialized.
- Missing or malformed committed files (registry, ledger, results index, report CSV/JSON) degrade to
  "not available" messages instead of crashing. Any unexpected error in a public section is logged
  on the server and shown to the visitor as a generic message, without traceback or paths.
- Captions show project-relative paths only. A test checks that no absolute local path, home
  directory or e-mail address appears on any page.
- Every chart value equals the committed run report it cites. Official-test values come only
  from the ledger, and no Stage-2 model carries a test result.
- `.streamlit/config.toml` hides developer menu options (`toolbarMode = "viewer"`), shows
  generic error messages (`showErrorDetails = "none"`; details go to the server log), keeps
  XSRF protection on, limits uploads to 2 MB and disables usage statistics.
  `aeromind dashboard` (research mode) re-enables full error details locally.

## Resource footprint (measured on the development laptop)

| | Value |
|---|---|
| Deployment environment (fresh Python 3.12 venv from `requirements.txt`) | about 385 MB on disk |
| Full research environment, dashboard extra only, for comparison | about 873 MB |
| Peak memory while rendering all six public sections headlessly | about 190 MB RSS |

These numbers are within Community Cloud's documented limits (690 MB–2.7 GB of memory).
Those limits can change without notice.

## Not part of public mode

- **The REST API.** It has no authentication, rate limiting or request-size limit. Run it only
  on localhost (`aeromind serve` binds to 127.0.0.1).
- **Live inference.** The *Prediction Demo* page describes the planned feature. Enabling it
  requires verified model files from a release asset installed through the registry's SHA-256
  check (`aeromind s3-import`), upload limits, and classical models only. That is a separate,
  future stage.
