# Contributing to AeroMind

Thank you for helping. AeroMind is a research codebase, and correctness and honest reporting
matter more than features. [CLAUDE.md](CLAUDE.md) contains the binding rules for every
contributor, human or AI. The essentials are below.

## Setup

```bash
git clone <your fork> && cd aeromind
uv sync --locked --all-extras        # core + deep (PyTorch CPU) + api + dashboard
uv run pytest                        # real-data tests skip without the NASA files
uv run ruff check .
uv run aeromind smoke && uv run aeromind s2-smoke --no-isolate && uv run aeromind s3-smoke
```

The NASA data is not in the repository; see [docs/DATA.md](docs/DATA.md).

## Non-negotiable research rules

1. **No leakage.**
   - Splits are engine-level.
   - Fitted transforms are fitted on training engines only.
   - Features are causal: they use cycles ≤ t of the same engine.
   - New features need truncation-invariance and engine-order-invariance tests.
2. **Calibration engines** are used only for conformal calibration (`s2-uq`, `s3-registry`).
3. **Official test set.**
   - Only `aeromind test` reads `RUL_FD00x.txt`. It requires `--confirm-final`, refuses
     repeats, and appends to `reports/test_set_ledger.json`.
   - Never select, tune or calibrate anything using test results.
   - Pull requests that evaluate on the test set need a written justification in the PR.
4. **Never invent numbers.**
   - Every metric in docs, cards, the dashboard or the API must come from a saved run or the
     registry.
   - State the evidence type (CV, canonical validation, calibration, official test) and the
     variability.
   - Do not claim a model is better from differences within fold variability.
5. **Never commit:**
   - NASA files
   - model bundles or checkpoints (`artifacts/`, `*.joblib`, `*.pt`)
   - credentials or `.env` files
   - large generated files

   Small reports under `reports/` are fine.

## Pull request checklist

- [ ] `uv run pytest` passes; `uv run ruff check .` is clean.
- [ ] The relevant smoke command passes (`smoke`, `s2-smoke`, `s3-smoke`).
- [ ] New public functions are typed and docstringed.
- [ ] New metrics or results reference their run directory. If the registry changed:
      `aeromind s3-registry`, then `aeromind s3-model-cards`.
- [ ] Docs and README commands you added were actually run.
- [ ] No test-set evaluation, or justification and ledger entry included.
- [ ] Dashboard changes pass `uv run pytest tests/test_dashboard_public.py` and
      `uv run aeromind dashboard --public` renders every section.

**Merging to `main` updates the live dashboard** at
<https://aeromind-research.streamlit.app/>: Streamlit Community Cloud redeploys on every push.
Merge only after CI passes. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Reporting issues

Please include:
- the command
- the full error
- `uv run python -c "import aeromind, sys; print(aeromind.__version__, sys.version)"`
- whether the NASA files validate (`aeromind validate --dataset all`)

Do not attach NASA data files or model bundles to issues.

## Scope

AeroMind is research and education software on simulated data. Contributions that present
it as operational, certified or safety-relevant tooling will not be accepted.
