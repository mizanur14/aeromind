# Stage-1 research-integrity audit

Date: 2026-11-17 · Scope: runs `fd001_baseline_20261117-105759` (primary, cap 125) and
`fd001_uncapped_20261117-110014` (secondary) · Code base: commit `1b25c6c` plus uncommitted Stage-1 work.

**Verdict: Stage 1 is fit to build on.** All 41 automated checks pass. Six defects were found
and fixed, none of which affected any reported metric. No model was retrained, and no new
official test evaluation was run.

Re-run the audit at any time (no training, no test labels read, about 5 s):

```bash
uv run aeromind audit artifacts/runs/fd001_baseline_20261117-105759 artifacts/runs/fd001_uncapped_20261117-110014
```

Machine-readable report: `reports/FD001/audit/stage1_audit.json`.

## 1. Defects found and fixed

| # | Defect | Impact on results | Fix |
|---|---|---|---|
| D1 | `scripts/create_splits.py` rewrote the split manifest unconditionally in the Stage-0 format. Same IDs, but the file hash recorded by runs changed (this happened once and was restored) | None (IDs identical) | Retired: now a thin wrapper around `aeromind split`, which writes byte-identical output and refuses to change IDs without `--force` (tested) |
| D2 | `scripts/explore_data.py` and `analyze_sensors.py` loaded test labels through `load_cmapss` without needing them | None (labels unused) | Now use `load_train` and unlabelled test inputs |
| D3 | `aeromind validate` stored test-label min/max in its report object | None (not printed by the CLI; no decision used it) | Removed; validation checks label integrity only |
| D4 | The published `reports/FD001/runs/fd001_baseline_*/metadata.json` still said `official_test_labels_read: false` after the test evaluation | Documentation only | `run_test` now re-syncs the published metadata; corrected for both runs |
| D5 | No guard against repeated or exploratory test evaluation | None so far | `aeromind test` requires `--confirm-final`, refuses repeats per run, and appends to `reports/test_set_ledger.json` |
| D6 | The validation bar-chart title said `rmse_target`, which is ambiguous across capped and uncapped runs | Presentation | The title names the truth (e.g. "RMSE vs RUL capped at 125"); both charts re-rendered from saved CSVs |

A doc claim was also corrected: reload differences are < 1e-13, not ≤ 3e-14 (the uncapped
Random Forest shows 5.7e-14).

## 2. Official test set: use and protection

**Evaluations performed (complete list, also in `reports/test_set_ledger.json`):**

| # | When (UTC) | Run | Models | Selected (on validation) | RMSE true | MAE true | PHM08 | RMSE capped |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-17 05:02:43 | fd001_baseline_20261117-105759 | all six (`--all-models`, five post hoc) | HistGradientBoosting | 13.60 | 9.64 | 288.6 | 12.33 |

The secondary (uncapped) run was **never** evaluated on the test set.

**Verified, not merely asserted:**

- **Static check.** An AST scan finds label readers called only in `data/cmapss.py` (definitions and `validate`) and `experiments/test_eval.py`.
- **Runtime check.** A unit test deletes `RUL_FD001.txt` and forbids the readers, then trains successfully. Run metadata hashes only the train and test inputs, never the label file.
- **Timeline.** Both configs were last modified at 04:55:53, before training. Selection finished at 05:00:03 (primary) and 05:02:30 (secondary), both before the only test evaluation at 05:02:43. The configs in `configs/` are identical to the ones saved in each run.
- **No feedback loop.** After the test evaluation, no config, grid, feature, target or model changed. Post-test code changes were output, guard and documentation fixes only.
- **Feature selection.** The only data-driven step is constant-sensor removal, fitted on the 70 training engines (`feature_pipeline.json` → `fitted_engine_ids`). The Stage-0 full-pool reports in `reports/FD001/*.csv` are read by no code.
- **Calibration engines** appear in no prediction file and no fit.

**Exposure disclosure.** While writing the documentation after the final evaluation, the
developer looked at the test-label range (7–145) and the shortest test history (31 cycles).
Neither informed any modelling choice, and both postdate selection. The post-hoc
all-models test table has also been seen, so the rule below applies.

**Rule for Stage 2 and later:** the Stage-2 base model must be chosen on validation, which
currently selects HistGradientBoosting. The fact that XGBoost and Random Forest score nearly
the same on test must not influence that choice. Any new official test evaluation is a
ledger entry and must cite the existing ones.

## 3. Capped vs uncapped metrics

| Suffix | Truth | Example (primary, selected model) |
|---|---|---|
| `_target` | the run's training target: capped at 125 (primary) or uncapped (secondary) | validation snapshot RMSE 11.26 |
| `_capped` | min(RUL, 125) | validation 11.26 · test 12.33 |
| `_true` | uncapped true RUL | validation 40.90 · test 13.60 |

- Within a run, selection uses `_target`. Across runs, `compare` reports only `_true` and
  `_capped` columns, so runs with different targets are compared on a common truth.
- The large gap between validation `_true` (40.90) and test `_true` (13.60) comes from the
  protocol, not from leakage. A third of validation snapshots (49/150) have true RUL above
  125, which a model capped at 125 cannot reach. The test truncation distribution is different.

## 4. Protocol consistency and engine independence

- Engine partitions are disjoint and identical to the manifest (70/15/15). The manifest hash
  equals the one recorded by both runs.
- Within each run, every model and grid point is scored on the same 150 snapshots, exactly 10
  per validation engine. The snapshot set is identical across both runs.
- Feature set (166 columns) and split are identical across runs. Only the target differs.
- The selected model is the argmin of the recorded selection metric, and the per-family
  winners match the 16-row search log.

## 5. Runtime, memory, loading and inference

| Quantity | Primary | Secondary | Source |
|---|---|---|---|
| Training wall time (16 fits + latency) | 123.6 s | 136.1 s | `metadata.json` (sum of fits: 117.4 / 128.5 s) |
| Peak RSS, training process | 379.5 MiB | 407.8 MiB | `ru_maxrss`, whole process |
| Peak RSS, audit process (load data, 12 bundles, all predictions) | 323.7 MiB | | measured in this audit |
| Bundle load time | 0.6–75 ms (RF slowest) | 0.6–81 ms | measured in this audit |
| Max reload difference, validation predictions | 0 (RF 2.8e-14) | 0 (RF 5.7e-14) | measured in this audit |
| Max difference, saved test predictions recomputed from **unlabelled** test inputs | 0 (RF 2.8e-14) | n/a | measured in this audit |
| Recorded test metrics recomputed from the saved per-engine file | equal | n/a | arithmetic check; not a new evaluation |

Training determinism on real data was **not** re-checked, because retraining was out of scope
for this audit. Determinism of the estimators is unit-tested on synthetic data.

## 6. Test, lint and smoke results (this audit)

- `uv run pytest`: **94 passed**. Six new tests cover the test-set guard and ledger, CLI
  confirmation, audit pass plus tamper detection, the static label scan, validate
  label-statistics removal, and the retired split script being idempotent and refusing changes.
- `uv run ruff check .`: all checks passed.
- `uv run aeromind smoke`: OK. `uv run aeromind audit ...`: PASSED (41/41).

## 7. Remaining limitations and scientific concerns

1. **Small validation set.** 15 engines (150 snapshots). The three tree ensembles differ by
   at most 0.3 cycles and are statistically indistinguishable. No confidence intervals yet;
   engine-level bootstrap is planned for Stage 2.
2. **Optimistic validation score.** The selected model's validation score is the minimum
   over 16 fits on the same engines. The single test number is the less biased estimate.
3. **Snapshot vs test mismatch.** Snapshots are uniform over life, while the official test
   truncation is undocumented. Selection may therefore emphasise a different RUL regime than
   the test set.
4. **Community-level test reuse.** The cap of 125, the window lengths and the model families
   are literature conventions. Those conventions were developed by a field that has reused the
   public FD001 test set for years. Results are comparable to the literature, but this is not
   a pristine generalisation estimate.
5. **Resource numbers are indicative.** Peak RSS is whole-process; per-model memory is not
   isolated. Latencies are medians on a throttling laptop.
6. **No re-training check on real data**, for the reason given in §5.
7. Only FD001 has been modelled. FD002–FD004 pass validation but have no splits or runs.
8. **Uncommitted work.** None of the Stage-1 work is committed. The provenance of results
   rests on run metadata (`git_commit: 1b25c6c-dirty`) until a commit is approved.
