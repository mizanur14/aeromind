# Ethical and practical limitations

AeroMind is a research and teaching codebase. Read this before using any number, model or
interface from it.

## What AeroMind is not

- **Not an operational tool.** No model, interval or dashboard view is certified, qualified
  or validated for airworthiness, maintenance planning or any safety-related decision. Do
  not use it to decide whether a real engine, aircraft or component is safe.
- **Not a physical digital twin.** The models are statistical regressors fitted to
  simulator output. They have no thermodynamic or physical engine model, cannot simulate
  interventions, and have not been validated against a physical asset.
- **Not endorsed by NASA.** NASA supplied the C-MAPSS benchmark data. AeroMind is an
  independent project, and NASA has not reviewed or endorsed it.

## Data limitations

- C-MAPSS trajectories are **simulation-generated**: synthetic degradation, synthetic sensor
  noise, idealised operating regimes. Real fleets differ in sensors, maintenance history,
  missing data, sensor faults and failure definitions.
- There are few engines: FD001 has 100 training engines, giving 15–17 per validation fold
  and 15 for calibration. Metrics vary by about ±2 cycles RMSE between folds.
- Early-life RUL is not identifiable from healthy-looking sensors. Models predict
  `min(RUL, 125)`, and a prediction of 125 means "no detectable degradation", not
  "125 cycles left".
- The official test-set truncation rule is undocumented, so validation snapshots only
  approximate it.

## Modelling and uncertainty limitations

- Differences between the GRU, the Transformer, XGBoost and HGB on FD001 are within fold
  variability. No model is shown to be superior.
- Deep architectures were not tuned. Classical hyperparameters were selected on the
  canonical validation engines, a mild optimistic bias on folds that contain them.
- **Prediction intervals:**
  - They target capped RUL and have **marginal** coverage, on average over engines and
    snapshot positions.
  - Coverage differs by RUL range (about 0.75–1.0 per bin).
  - With 15 calibration engines the threshold is volatile, and 95 % intervals are impossible.
  - Coverage of uncapped true RUL is only 0.59–0.71.
  - Intervals are **invalid** under distribution shift: other subsets, new fault modes,
    faulty sensors.
- Models expect their training subset's operating regime and fault modes. On FD003
  engines, FD001 models' capped RMSE rises by 46 % (GRU) to 192 % (Ridge); see the Stage-2
  robustness results.
- Additive sensor noise of half a training SD roughly doubles the tree models' error.

## Interface limitations

- **No out-of-range detection.** Inputs must be complete and finite, and values that
  overflow are rejected (HTTP 422). Physically implausible but finite readings (wrong units,
  for example) are **not** detected, and can still yield confident-looking predictions.
  Check the units against the C-MAPSS files.
- The API has **no authentication, rate limiting or TLS**. Run it on localhost only, or
  behind a gateway that adds these.
- Model bundles are Python pickles. Load only bundles whose SHA-256 matches a registry you
  trust; `ModelService` and `aeromind s3-import` enforce this.
- The maintenance simulation is a **hypothetical illustration**:
  - Its failure and unused-cycle counts come from 15 simulated validation engines.
  - Its costs are user assumptions in arbitrary units.
  - It does not estimate savings, safety benefit or optimal policies.

## Responsible use

- Report AeroMind results with their evidence type (CV, canonical validation, calibration,
  official test) and their variability.
- Do not tune or select anything using the official test set. All test evaluations are
  ledgered in `reports/test_set_ledger.json`.
- Cite NASA and Saxena et al. (2008) for the data; see [DATA.md](DATA.md).
