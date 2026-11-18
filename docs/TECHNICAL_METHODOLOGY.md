# Technical methodology

An end-to-end description of how AeroMind turns NASA C-MAPSS files into served predictions.
Each step links to the detailed document.

## 1. Data ([DATA.md](DATA.md))

- **Parsing and validation.** The 26-column C-MAPSS files are parsed with schema, numeric,
  finiteness, integer and contiguous-cycle checks. Raw files are pinned by SHA-256 in
  `data/manifests/cmapss_provenance.json`.
- **Labels.** Training engines run to failure: `RUL = last_cycle − cycle`. Official test
  labels are read only by `aeromind test`.
- **Splits.** Seeded, engine-level 70/15/15 train / validation / calibration manifests for
  every subset.

## 2. Features ([METHODOLOGY.md](METHODOLOGY.md))

- **Fitted on training engines only:**
  - constant-sensor removal
  - optional per-operating-condition normalisation (k-means on the 3 settings; FD002/FD004)
  - a standard scaler
- **Causal per-engine features:**
  - classical models: trailing rolling statistics (mean, SD, slope, …) over 10 and 30 cycles
  - deep models: right-padded windows of the last 30 cycles with explicit lengths
- The fitted pipeline is saved inside each model bundle, so training and serving share the
  same preprocessing.

## 3. Models

| Family | Stage | Notes |
|---|---|---|
| Median, mean-residual-life, Ridge, Random Forest, HGB, XGBoost | 1 | grids selected on validation engines ([METHODOLOGY.md](METHODOLOGY.md)) |
| TCN, GRU, Transformer (PyTorch, CPU) | 2 | small, fixed a priori; Huber loss on targets scaled by 1/125; AdamW; early stopping on an inner engine holdout; deterministic, resumable ([STAGE2_METHODOLOGY.md](STAGE2_METHODOLOGY.md)) |

Target: `min(RUL, 125)`; predictions are clipped to [0, 125].

## 4. Evaluation ([EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md))

- Engine-balanced validation snapshots, and 5-fold engine-level CV with paired comparisons.
- Separate capped and true-RUL metrics, plus near-EOL metrics.
- One ledgered official test evaluation (Stage 1 only).

## 5. Uncertainty

**Stage 2 research** ([STAGE2_RESULTS.md §2](STAGE2_RESULTS.md)):
- Methods: split conformal (engine-level and pooled), HGB and Transformer quantile
  regression, and CQR.
- Diagnostics: coverage, width, coverage by RUL bin, and a role-swap check.

**Stage 3 serving:**
- Engine-level split conformal on the 15 calibration engines, one seeded snapshot per
  engine, at levels 0.80 and 0.90.
- Available for the four FD001 models whose coverage was evaluated in Stage 2.
- The registry stores each interval's guarantee statement and its validated coverage.

## 6. Serving (Stage 3; [ARCHITECTURE.md](ARCHITECTURE.md))

1. **Registry** (`aeromind s3-registry`). Per model it records:
   - bundle path, SHA-256 and version
   - input requirements
   - training engines
   - metrics copied from run reports, grouped by evidence type
   - resources
   - calibrator
   - limitations
2. **ModelService.** For each request it:
   - verifies the bundle hash before unpickling
   - validates one engine's raw cycles (complete, finite, unique positive integer cycles)
   - applies the bundle's own pipeline
   - returns the prediction at every observed cycle, with intervals, warnings and provenance
3. **Interfaces.** The FastAPI app and the Streamlit dashboard both call `ModelService`.
   - The dashboard shows predictions only for validation engines and user uploads.
   - The maintenance view is a labelled hypothetical simulation.
4. **Verification.** Served predictions equal the saved Stage-2 predictions (tested).
   Latency is benchmarked in process and over HTTP ([CPU_BENCHMARKING.md](CPU_BENCHMARKING.md)).

## 7. Limitations

See [LIMITATIONS.md](LIMITATIONS.md) and [STAGE2_RESULTS.md §7](STAGE2_RESULTS.md).
