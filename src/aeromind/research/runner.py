"""Fit / evaluate / benchmark any Stage-2 model under one protocol.

Every model (classical or deep) is trained on a fold's training engines with the same
target definition, evaluated on the fold's validation engines with the same seeded
snapshots, and benchmarked with the same latency procedure.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aeromind.evaluation.protocol import engine_balanced_weights, metrics_against_truths
from aeromind.features.pipeline import FeaturePipeline
from aeromind.features.targets import make_target
from aeromind.models.bundle import RULPredictor
from aeromind.models.factory import CYCLE_INPUT_MODELS, build_estimator
from aeromind.research.spec import ModelEntry
from aeromind.utils import timer

logger = logging.getLogger(__name__)


def fit_entry(
    entry: ModelEntry,
    name: str,
    train_df: pd.DataFrame,
    cap: int | None,
    seed: int,
    n_jobs: int = 4,
    checkpoint_dir: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train one model on training-engine rows. Returns (predictor, info)."""
    train_df = train_df.copy()
    train_df["target"] = make_target(train_df, cap)
    if entry.kind == "classical":
        with timer() as t:
            pipeline = FeaturePipeline(entry.feature_config(), seed).fit(train_df)
            kind = "cycle" if entry.family in CYCLE_INPUT_MODELS else "features"
            X = train_df[["cycle"]].to_numpy(float) if kind == "cycle" else pipeline.transform(train_df)
            est = build_estimator(entry.family, entry.params, seed, n_jobs)
            est.fit(X, train_df["target"].to_numpy())
        n_params = int(est.coef_.size + 1) if entry.family == "ridge" else None
        info = {"fit_seconds": t["seconds"], "n_parameters": n_params, "n_features": X.shape[1]}
        return RULPredictor(name, pipeline, est, kind, cap, entry.params), info

    from aeromind.deep.trainer import DeepSpec, fit_deep

    spec = DeepSpec.from_dict(entry.deep)
    with timer() as t:
        predictor, hist = fit_deep(
            spec, train_df, cap, seed, checkpoint_dir=checkpoint_dir, resume=checkpoint_dir is not None, name=name
        )
    info = {
        "fit_seconds": t["seconds"],
        "n_parameters": hist["n_parameters"],
        "n_features": predictor.n_channels,
        "window": spec.window,
        **{k: hist[k] for k in ("epochs_run", "best_epoch", "best_inner_val_rmse_target", "stopped_early", "train_window_mb")},
        "seconds_per_epoch": float(np.mean([c["seconds"] for c in hist["curve"]])),
        "curve": hist["curve"],
    }
    return predictor, info


def evaluate_frame(
    predictor: Any,
    val_df: pd.DataFrame,
    snapshot_index: pd.Index,
    target_cap: int | None,
    capped_truth_cap: int,
    near_eol: int,
    inputs: Any = None,
) -> tuple[dict[str, float], np.ndarray]:
    """Snapshot, engine-weighted all-window and near-EOL metrics (flat dict) + row predictions."""
    X = predictor.inputs(val_df) if inputs is None else inputs
    pred = predictor.predict_inputs(X)
    rul = val_df["rul"].to_numpy(float)
    pos = val_df.index.get_indexer(snapshot_index)
    out: dict[str, float] = {}
    for prefix, m in (
        ("snapshot", metrics_against_truths(rul[pos], pred[pos], target_cap, capped_truth_cap)),
        (
            "window_ew",
            metrics_against_truths(rul, pred, target_cap, capped_truth_cap, engine_balanced_weights(val_df["engine_id"])),
        ),
    ):
        out.update({f"{prefix}_{k}": v for k, v in m.items()})
    near = rul <= near_eol
    if near.any():
        m = metrics_against_truths(rul[near], pred[near], target_cap, capped_truth_cap)
        out.update({f"near_eol_{k}": v for k, v in m.items()})
    return out, pred


def _median_ms(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return float(np.median(times) * 1e3)


def measure_latency(predictor: Any, X: Any, val_df: pd.DataFrame, repeats: int) -> dict[str, float]:
    """Median CPU latencies (batch per row, one precomputed input, end-to-end one engine)."""
    engine = val_df[val_df.engine_id == val_df.engine_id.iloc[0]]
    predictor.predict_inputs(X[-1:])  # warm-up (lazy network build, thread pools)
    batch_ms = _median_ms(lambda: predictor.predict_inputs(X), max(3, repeats // 10))
    return {
        "batch_rows": len(X),
        "batch_us_per_row": batch_ms * 1e3 / len(X),
        "single_input_ms": _median_ms(lambda: predictor.predict_inputs(X[-1:]), repeats),
        "end_to_end_engine_ms": _median_ms(lambda: predictor.predict(engine)[-1], max(5, repeats // 3)),
        "end_to_end_engine_cycles": len(engine),
    }
