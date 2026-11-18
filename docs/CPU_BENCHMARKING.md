# CPU benchmarking guide

AeroMind is CPU-only by design. This guide explains how latency and memory are measured, and
how to read the numbers.

## Two different latency measurements

| Measurement | Command | What it times | Where reported |
|---|---|---|---|
| **Model latency** (Stage 2) | `aeromind s2-compare` | `predict_inputs` on **one precomputed feature row or window**: the model alone | `resources.csv` (`single_input_ms`), registry `resources`, model cards |
| **Serving latency** (Stage 3) | `aeromind s3-benchmark` | A complete request for **one engine's full history**: input validation, the bundle's feature pipeline over every cycle, the model at every cycle, and the interval. In process, and over HTTP | `reports/benchmarks/<run>/benchmark.{json,md}` (`latest/` = most recent) |

They answer different questions. A model can be fast per row and still slow per request,
because the request recomputes the causal rolling features for the whole trajectory.

## Running

```bash
export OMP_WAIT_POLICY=PASSIVE
uv run aeromind s3-benchmark --repeats 30            # all locally available models, in-process + HTTP
uv run aeromind s3-benchmark --models fd001-s2-gru --no-http
```

**Protocol:**
- 3 warm-up calls, then the median (p50) and p95 over `--repeats` calls.
- PyTorch is limited to ≤ 2 threads per deep model.
- HTTP runs against a real `uvicorn` server on 127.0.0.1, started in a subprocess.
- The single-engine request uses the first validation engine of each subset; the batch uses
  all validation engines.
- Single passes are never reported: the test laptop throttles.

## Results (2026-11-17, Intel i7-1165G7 4C/8T, 16 GB, Python 3.12.15, PyTorch 2.14.0+cpu)

Source: `reports/benchmarks/s3_benchmark_*` (`latest/benchmark.md` has every row).

**FD001, one engine with 259 cycles:**

| Model | In-process p50 / p95 (ms) | HTTP p50 / p95 (ms) | HTTP batch, 15 engines / 3,160 cycles, p50 (ms) | Model-only latency, Stage 2 (ms) |
|---|---|---|---|---|
| Ridge | 23.2 / 25.8 | 44.8 / 54.9 | 487 | 0.19 |
| XGBoost | 24.9 / 26.4 | 37.1 / 38.8 | 511 | 0.35 |
| HGB (Stage 2) | 29.9 / 32.9 | 53.3 / 64.1 | 730 | 8.84 |
| HGB (Stage 1) | 28.7 / 36.1 | 37.1 / 66.5 | 865 | — |
| TCN | 9.7 / 11.9 | 22.0 / 23.5 | 264 | 1.39 |
| GRU | 10.0 / 10.9 | 22.4 / 24.7 | 268 | 1.06 |
| Transformer | 21.7 / 29.3 | 31.6 / 34.9 | 399 | 1.29 |

**Other subsets, in-process p50 for one engine:**
- FD002 (269 cycles): GRU 19.9 ms, classical models 31–37 ms.
- FD003 (424 cycles): GRU 20.5 ms, classical models 41–60 ms.
- FD004 (299 cycles): GRU 18.1 ms, classical models 47–70 ms.
- HTTP batches of the 37–39 FD002/FD004 validation engines take 0.9–2.4 s.

Peak RSS of the benchmark process with all 22 models loaded was 531 MB. Cold bundle loading
took ≤ 0.04 s per model; deep networks are rebuilt lazily on the first call, which falls
within the warm-up.

**Reading these numbers:**
- Per request, the deep models are **faster** than the classical ones, although per row they
  are slower. The classical bundles spend most of the request recomputing 166 rolling
  features in pandas.
- HTTP adds about 8–25 ms per single-engine request, for JSON encoding and validation of
  ~6,500 numbers (259 cycles × 25 fields). Batches amortise the connection, not the parsing.
- These are latencies on one throttling laptop. Treat them as orders of magnitude, not
  service-level guarantees.

## Memory

- **Training:** `peak_rss_mb` per model in Stage-2 `resources.csv`. Each model runs in its own
  spawned process, so the numbers are not contaminated by other models. The figure includes
  Python, pandas and the data (≈ 210 MB) — deep models 609–946 MB, classical models 294–581 MB.
- **Serving:** the benchmark process figure above.

## Tips

- Set `OMP_WAIT_POLICY=PASSIVE`; otherwise idle OpenMP threads spin and inflate latency.
- Do not run benchmarks while other CPU-heavy jobs run, and close the dashboard.
- For throughput, run several API workers (`uvicorn --workers N`); each loads its own models.
