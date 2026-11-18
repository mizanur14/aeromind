"""Baseline training with validation-based model selection.

Workflow (``aeromind train --config ...``):

1. Load the training pool only (official test labels are never read here).
2. Apply the engine split; fit the FeaturePipeline on training engines.
3. For each model family and each grid point: fit on training engines, evaluate on the
   validation engines (snapshot + all-window protocols).
4. Pick the best grid point per family, then the overall best, by
   ``evaluation.selection_metric`` on the validation snapshots.
5. Save one bundle per family, predictions, tables, plots and ``metadata.json``.

Run layout::

    <run>/config.yaml  metadata.json
    <run>/models/<family>.joblib
    <run>/predictions/validation_{snapshots,windows}.csv
    <run>/report/...   (small CSV/JSON/PNG; copied to reports/<DS>/runs/<run>/)
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aeromind.config import ExperimentConfig, dump_config
from aeromind.data.provenance import dataset_hashes
from aeromind.evaluation import plots
from aeromind.evaluation.protocol import (
    engine_balanced_weights,
    error_by_rul_bin,
    metrics_against_truths,
    per_engine_table,
)
from aeromind.experiments.common import (
    DevelopmentData,
    load_development_data,
    new_run_dir,
    publish_reports,
)
from aeromind.features.pipeline import FeaturePipeline
from aeromind.models.bundle import RULPredictor, save_bundle
from aeromind.models.factory import CYCLE_INPUT_MODELS, build_estimator, expand_grid
from aeromind.utils import environment_info, peak_rss_mb, timer, write_json

logger = logging.getLogger(__name__)


def truth_label(metric_key: str, target_cap: int | None, capped_truth_cap: int) -> str:
    """Human-readable label naming the truth a metric is computed against."""
    name, _, truth = metric_key.rpartition("_")
    if truth == "target":
        truth = "capped" if target_cap is not None else "true"
        cap = target_cap
    else:
        cap = capped_truth_cap
    text = f"RUL capped at {cap}" if truth == "capped" else "uncapped true RUL"
    return f"{name.upper()} vs {text}"


SELECTION_KEYS = {
    "snapshot_rmse_target": "rmse_target",
    "snapshot_rmse_true": "rmse_true",
    "snapshot_rmse_capped": "rmse_capped",
}


def evaluate_on_validation(
    predictor: RULPredictor,
    X_val: np.ndarray,
    data: DevelopmentData,
) -> tuple[dict[str, Any], np.ndarray]:
    """Snapshot and all-window validation metrics plus the all-row predictions."""
    cfg = data.config
    val = data.validation
    pred = predictor.predict_inputs(X_val)
    rul = val["rul"].to_numpy(dtype=float)
    pos = val.index.get_indexer(data.snapshot_index)
    cap_t, cap_c = cfg.target.cap, cfg.evaluation.capped_truth_cap
    result: dict[str, Any] = {
        "snapshot": metrics_against_truths(rul[pos], pred[pos], cap_t, cap_c),
        "window_rows": metrics_against_truths(rul, pred, cap_t, cap_c),
        "window_engine_weighted": metrics_against_truths(
            rul, pred, cap_t, cap_c, weights=engine_balanced_weights(val["engine_id"])
        ),
    }
    near = rul <= cfg.evaluation.near_eol_threshold
    if near.any():
        result["near_eol_rows"] = metrics_against_truths(rul[near], pred[near], cap_t, cap_c)
    return result, pred


def _flatten(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def _median_seconds(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def measure_latency(
    predictor: RULPredictor, X_val: np.ndarray, data: DevelopmentData, repeats: int
) -> dict[str, float]:
    """Median CPU latencies: batch, single precomputed row, and end-to-end single engine.

    End-to-end = feature computation over one validation engine's full history plus a
    prediction at its last cycle (what an online monitor would do each cycle).
    """
    one_row = X_val[-1:]
    first_engine = data.validation["engine_id"].iloc[0]
    engine_df = data.validation[data.validation["engine_id"] == first_engine]
    batch_s = _median_seconds(lambda: predictor.predict_inputs(X_val), max(3, repeats // 10))
    row_s = _median_seconds(lambda: predictor.predict_inputs(one_row), repeats)
    e2e_s = _median_seconds(lambda: predictor.predict(engine_df)[-1], max(5, repeats // 5))
    return {
        "batch_rows": int(len(X_val)),
        "batch_ms": batch_s * 1e3,
        "batch_us_per_row": batch_s / len(X_val) * 1e6,
        "single_row_ms": row_s * 1e3,
        "end_to_end_engine_ms": e2e_s * 1e3,
        "end_to_end_engine_cycles": int(len(engine_df)),
    }


def run_training(
    config: ExperimentConfig,
    run_dir: str | Path | None = None,
    pool: pd.DataFrame | None = None,
    publish: bool = True,
) -> Path:
    """Train all configured model families, select on validation, save artifacts.

    Args:
        config: experiment configuration.
        run_dir: output directory (default: timestamped under ``paths.artifacts_dir``).
        pool: optional pre-loaded training pool (synthetic tests / smoke runs).
        publish: copy the small report files to ``reports/<dataset>/runs/<run>``.

    Returns:
        The run directory.
    """
    started = datetime.now(UTC)
    t_total = time.perf_counter()
    out = new_run_dir(config, run_dir)
    dump_config(config, out / "config.yaml")
    report_dir = out / "report"
    logger.info("Run directory: %s", out)

    data = load_development_data(config, pool)
    with timer() as t_feat:
        pipeline = FeaturePipeline(config.features, config.seed).fit(data.train)
        X_train = pipeline.transform(data.train)
        X_val = pipeline.transform(data.validation)
    logger.info("Features: %d columns in %.2fs", X_train.shape[1], t_feat["seconds"])
    # Leakage guard: the pipeline must have been fitted on training engines only.
    assert pipeline.fitted_engine_ids_ == data.manifest.train_engine_ids

    cycle_train = data.train[["cycle"]].to_numpy(dtype=float)
    cycle_val = data.validation[["cycle"]].to_numpy(dtype=float)
    sample_weight = (
        engine_balanced_weights(data.train["engine_id"]) if config.training.sample_weight == "engine_balanced" else None
    )
    sel_key = SELECTION_KEYS[config.evaluation.selection_metric]

    search_rows: list[dict[str, Any]] = []
    best: dict[str, dict[str, Any]] = {}
    for family, spec in config.models.items():
        for trial, params in enumerate(expand_grid(spec.params, spec.grid)):
            estimator = build_estimator(family, params, config.seed, config.training.n_jobs)
            kind = "cycle" if family in CYCLE_INPUT_MODELS else "features"
            Xtr, Xva = (cycle_train, cycle_val) if kind == "cycle" else (X_train, X_val)
            fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
            with timer() as t_fit:
                estimator.fit(Xtr, data.y_train, **fit_kwargs)
            predictor = RULPredictor(family, pipeline, estimator, kind, config.target.cap, params)
            metrics, pred = evaluate_on_validation(predictor, Xva, data)
            score = metrics["snapshot"][sel_key]
            row = {
                "model": family,
                "trial": trial,
                "params": repr(params),
                "fit_seconds": t_fit["seconds"],
                "peak_rss_mb_after_fit": peak_rss_mb(),
                **_flatten("snapshot", metrics["snapshot"]),
                **_flatten("window_rows", metrics["window_rows"]),
                **_flatten("window_engine_weighted", metrics["window_engine_weighted"]),
            }
            search_rows.append(row)
            logger.info(
                "%-24s trial %d  fit %.2fs  val snapshot RMSE(target)=%.2f RMSE(true)=%.2f  %s",
                family,
                trial,
                t_fit["seconds"],
                metrics["snapshot"]["rmse_target"],
                metrics["snapshot"]["rmse_true"],
                params,
            )
            if family not in best or score < best[family]["score"]:
                best[family] = {
                    "score": score,
                    "trial": trial,
                    "params": params,
                    "predictor": predictor,
                    "metrics": metrics,
                    "pred": pred,
                    "inputs": Xva,
                    "fit_seconds": t_fit["seconds"],
                }

    search = pd.DataFrame(search_rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    search.to_csv(report_dir / "search_results.csv", index=False)

    # ---- per-family best: save bundles, latency, comparison table
    comparison_rows = []
    snapshot_frames, window_frames = [], []
    for family, info in best.items():
        predictor: RULPredictor = info["predictor"]
        bundle_path = save_bundle(
            predictor,
            out / "models" / f"{family}.joblib",
            {"run": out.name, "model": family, "params": info["params"], "config_name": config.name},
        )
        latency = measure_latency(predictor, info["inputs"], data, config.evaluation.latency_repeats)
        info["latency"] = latency
        comparison_rows.append(
            {
                "model": family,
                "params": repr(info["params"]),
                "fit_seconds": info["fit_seconds"],
                "model_file_mb": bundle_path.stat().st_size / 2**20,
                **_flatten("snapshot", info["metrics"]["snapshot"]),
                **_flatten("window_engine_weighted", info["metrics"]["window_engine_weighted"]),
                **_flatten("window_rows", info["metrics"]["window_rows"]),
                **_flatten("near_eol", info["metrics"].get("near_eol_rows", {})),
                **{f"latency_{k}": v for k, v in latency.items()},
            }
        )
        frame = data.validation[["engine_id", "cycle", "rul", "target"]].copy()
        frame["pred"] = info["pred"]
        frame["model"] = family
        window_frames.append(frame)
        snapshot_frames.append(frame.loc[data.snapshot_index])

    comparison = pd.DataFrame(comparison_rows).sort_values(f"snapshot_{sel_key}").reset_index(drop=True)
    comparison.to_csv(report_dir / "validation_model_comparison.csv", index=False)
    selected = str(comparison.loc[0, "model"])
    logger.info("Selected model (validation %s): %s", config.evaluation.selection_metric, selected)

    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.concat(window_frames, ignore_index=True)
    windows.to_csv(pred_dir / "validation_windows.csv", index=False)
    snapshots = pd.concat(snapshot_frames, ignore_index=True)
    snapshots.to_csv(pred_dir / "validation_snapshots.csv", index=False)

    # ---- diagnostics for the selected model and the comparison
    sel = best[selected]
    sel_frame = windows[windows["model"] == selected]
    sel_snap = snapshots[snapshots["model"] == selected]
    bins = config.evaluation.rul_bins
    error_by_rul_bin(sel_snap["rul"], sel_snap["pred"], bins).to_csv(
        report_dir / "validation_selected_error_by_rul_bin_snapshots.csv", index=False
    )
    error_by_rul_bin(sel_frame["rul"], sel_frame["pred"], bins).to_csv(
        report_dir / "validation_selected_error_by_rul_bin_windows.csv", index=False
    )
    per_engine_table(sel_frame).to_csv(report_dir / "validation_selected_per_engine.csv", index=False)

    title = f"{config.dataset} validation — {selected}"
    plots.plot_pred_vs_true(
        sel_snap["rul"],
        sel_snap["pred"],
        f"{title} (snapshots)",
        report_dir / "val_pred_vs_true.png",
        config.target.cap,
    )
    plots.plot_residuals(
        sel_frame["rul"], sel_frame["pred"], f"{title} (all windows)", report_dir / "val_residuals.png"
    )
    plots.plot_metric_bars(
        comparison,
        f"snapshot_{sel_key}",
        f"{config.dataset} validation snapshots — {truth_label(sel_key, config.target.cap, config.evaluation.capped_truth_cap)}",
        report_dir / "val_model_comparison.png",
    )
    top3 = comparison["model"].head(3).tolist()
    plots.plot_error_by_bin(
        {
            m: error_by_rul_bin(windows.loc[windows.model == m, "rul"], windows.loc[windows.model == m, "pred"], bins)
            for m in top3
        },
        f"{config.dataset} validation (all windows): RMSE vs true RUL, top 3",
        report_dir / "val_error_by_rul_bin.png",
    )
    show = sorted(data.validation["engine_id"].unique())[:6]
    plots.plot_engine_trajectories(
        sel_frame, show, title, report_dir / "val_engine_trajectories.png", config.target.cap
    )

    summary = {
        "run": out.name,
        "dataset": config.dataset,
        "target_cap": config.target.cap,
        "capped_truth_cap": config.evaluation.capped_truth_cap,
        "selection_metric": config.evaluation.selection_metric,
        "selected_model": selected,
        "selected_params": sel["params"],
        "validation": {fam: {k: v for k, v in info["metrics"].items()} for fam, info in best.items()},
        "latency": {fam: info["latency"] for fam, info in best.items()},
    }
    write_json(report_dir / "validation_summary.json", summary)
    write_json(report_dir / "feature_pipeline.json", pipeline.describe())

    metadata = {
        "run": out.name,
        "config_name": config.name,
        "dataset": config.dataset,
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "total_seconds": time.perf_counter() - t_total,
        "feature_seconds": t_feat["seconds"],
        "peak_rss_mb": peak_rss_mb(),
        "seed": config.seed,
        "selected_model": selected,
        "selected_params": sel["params"],
        "selection_metric": config.evaluation.selection_metric,
        "selection_value": sel["score"],
        "models": {fam: f"models/{fam}.joblib" for fam in best},
        "split_manifest": str(config.split_manifest_path),
        "split_manifest_sha256": data.manifest_sha256,
        "engines": {
            "train": data.manifest.train_engine_ids,
            "validation": data.manifest.validation_engine_ids,
            "calibration_untouched": data.manifest.calibration_engine_ids,
        },
        "n_features": len(pipeline.feature_names_),
        "data_sha256": dataset_hashes(config.paths.raw_dir, config.dataset) if pool is None else {"synthetic": True},
        "official_test_labels_read": False,
        "test_evaluations": [],
        "environment": environment_info(),
    }
    write_json(out / "metadata.json", metadata)
    write_json(report_dir / "metadata.json", {k: v for k, v in metadata.items() if k != "engines"})

    if publish:
        dest = Path(config.paths.reports_dir) / config.dataset / "runs" / out.name
        publish_reports(report_dir, dest)
        logger.info("Report copied to %s", dest)
    logger.info("Training finished in %.1fs", metadata["total_seconds"])
    return out
