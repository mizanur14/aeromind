# Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `warning: VIRTUAL_ENV=... does not match the project environment` | A conda env is active. Run `unset VIRTUAL_ENV`; uv uses `.venv`. |
| `Missing dataset file: data/raw/cmapss/train_FD001.txt` | Place the NASA files there (docs/DATA.md). Nothing is downloaded automatically. |
| `DatasetError: ... expected a nonempty 26-column dataset` / `inconsistent number of fields` | Corrupted or wrong file; re-extract and check `aeromind validate`. |
| `Provenance: train_FD001.txt: sha256 mismatch` | Local file differs from the recorded one. Restore it; only use `--write-manifest` for a deliberate data update. |
| `SplitError: ... Refusing to overwrite` | `aeromind split` would change existing engine IDs. Pass `--force` only if you intend to invalidate earlier runs. |
| `FileExistsError: Run directory ... already exists` | Choose a new `--run-dir` or omit it for a timestamped one. |
| `Split manifest not found` | Committed manifests exist for FD001–FD004 in `data/manifests/`. For a new seed, run `uv run aeromind split --dataset FD00x --seed N`. |
| Slow Random Forest / CPU at 100 % | Lower `training.n_jobs` or `n_estimators` in the config. |
| Real-data tests skipped | Expected without NASA files (e.g. in CI). Locally, check `data/raw/cmapss/`. |
| `uv sync` touches files slowly on this NTFS volume | `UV_LINK_MODE=copy` is expected here. |
| Split manifest hash no longer matches `metadata.json` | The legacy `scripts/create_splits.py` rewrites the manifest in the Stage-0 format (same IDs, no `fractions`/`source_sha256`). Re-run `uv run aeromind split --dataset FD001 --seed 42` to restore it. |
| `ModuleNotFoundError: torch` / deep tests skipped | Install the deep extra: `uv sync --locked --extra deep` (CPU wheel from the PyTorch index). |
| `aeromind serve` / `dashboard`: "needs the optional extra" | `uv sync --locked --extra api` and/or `--extra dashboard` (or `--all-extras`). |
| API `503 model_unavailable: bundle not found` | Model bundles are not in Git. Import a release archive (`aeromind s3-import`) or regenerate them (README, *Getting model artifacts*). |
| API `503 ... SHA-256 does not match the registry` | The local bundle differs from the registered one (re-trained or modified). It is never loaded. Rebuild the registry from your own runs with `aeromind s3-registry`, or restore the original bundle. |
| API `503 ... requires PyTorch` | A deep model was requested without the `deep` extra installed. |
| API `422 interval_unavailable` | The model has no calibrator (only the four FD001 Stage-2 models listed in the registry), or the level is not 0.80/0.90. |
| API `422 invalid_input: Missing or non-finite readings` | The models need complete readings. Impute causally (forward-fill per engine) before calling. |
| Dashboard: "No locally available model bundles" | Same as the 503 above; the report-based sections still work. |
| Dashboard: "NASA file ... not found" | Set `AEROMIND_DATA_DIR` or place the files in `data/raw/cmapss/`. |
| Dashboard: Engine Health / Sensor / Maintenance pages missing | Public mode shows only report-based sections. Use `uv run aeromind dashboard` (research mode). |
| Public dashboard shows a generic error | `showErrorDetails = "none"` hides details in the browser; read the server log, or run `uv run aeromind dashboard` locally for full errors. |
| `s3-registry` fails with `Model bundle not found` | The registry is built from local `artifacts/`. Only maintainers with the original runs (or after `s3-import`) can rebuild it unchanged. |
| High latency, CPU at 100 % while idle | `export OMP_WAIT_POLICY=PASSIVE` (PyTorch/OpenMP spin-waiting). |
