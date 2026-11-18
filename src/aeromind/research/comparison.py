"""Stage-2 model comparison with repeated engine-level validation.

``aeromind s2-compare --config configs/stage2/<name>.yaml``

For each model (in its own spawned process, so peak RSS is per model) and each fold
(``canonical`` = the Stage-1 70/15 split, ``cv0..cv4`` = engine-level 5-fold over the 85
development engines; calibration engines never used):

- fit on the fold's training engines (deep models early-stop on an inner engine holdout),
- evaluate on the fold's validation engines (seeded snapshots, engine-weighted windows,
  near-EOL rows) against true and capped RUL,
- on the canonical fold only: save the model bundle and measure CPU latency.

The official test set is never read.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aeromind.data.cmapss import load_train
from aeromind.data.splits import load_split, verify_split
from aeromind.evaluation.protocol import select_snapshots
from aeromind.models.bundle import save_bundle
from aeromind.research.runner import evaluate_frame, fit_entry, measure_latency
from aeromind.research.spec import Stage2Config, dump_stage2_config, make_folds
from aeromind.utils import environment_info, peak_rss_mb, read_json, setup_logging, sha256_file, write_json

logger = logging.getLogger(__name__)

KEY_METRICS = [
    "snapshot_rmse_capped",
    "snapshot_mae_capped",
    "snapshot_rmse_true",
    "snapshot_mae_true",
    "snapshot_phm08_mean_capped",
    "snapshot_phm08_mean_true",
    "window_ew_rmse_capped",
    "near_eol_rmse_true",
    "near_eol_mae_true",
]


def fold_frames(pool: pd.DataFrame, fold: dict[str, list[int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and validation rows of one fold."""
    train = pool[pool.engine_id.isin(fold["train"])].reset_index(drop=True)
    val = pool[pool.engine_id.isin(fold["validation"])].reset_index(drop=True)
    return train, val


def run_model(cfg: Stage2Config, name: str, run_dir: Path, pool: pd.DataFrame | None = None) -> dict[str, Any]:
    """Train/evaluate one model over all requested folds; write ``results/<name>.json``."""
    t_start = time.perf_counter()
    entry = cfg.models[name]
    cap = entry.resolved_cap(cfg.target_cap)
    pool = load_train(cfg.paths.raw_dir, cfg.dataset) if pool is None else pool
    rss_after_load = peak_rss_mb()
    manifest = load_split(cfg.split_manifest_path)
    verify_split(manifest, pool.engine_id.unique())
    folds = make_folds(manifest, cfg.cv.folds, cfg.cv.seed, cfg.cv.include_canonical)
    names = cfg.cv.run_folds or list(folds)
    ev = cfg.evaluation
    out: dict[str, Any] = {"model": name, "kind": entry.kind, "target_cap": cap, "folds": {}}
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for fold_name in names:
        train, val = fold_frames(pool, folds[fold_name])
        snaps = select_snapshots(val, ev.snapshots_per_engine, ev.snapshot_seed)
        ckpt = run_dir / "checkpoints" / name if (entry.kind == "deep" and fold_name == "canonical") else None
        predictor, info = fit_entry(entry, name, train, cap, cfg.seed, checkpoint_dir=ckpt)
        X = predictor.inputs(val)
        metrics, pred = evaluate_frame(predictor, val, snaps, cap, ev.capped_truth_cap, ev.near_eol_threshold, X)
        frame = val[["engine_id", "cycle", "rul"]].assign(pred=pred, model=name, fold=fold_name)
        frame["snapshot"] = frame.index.isin(snaps)
        frame.to_csv(pred_dir / f"{name}__{fold_name}.csv", index=False)
        record = {"metrics": metrics, "fit": {k: v for k, v in info.items() if k != "curve"}}
        if fold_name == "canonical":
            path = save_bundle(predictor, run_dir / "models" / f"{name}.joblib", {"model": name, "fold": "canonical"})
            record["bundle_mb"] = path.stat().st_size / 2**20
            record["latency"] = measure_latency(predictor, X, val, cfg.latency_repeats)
            if "curve" in info:
                pd.DataFrame(info["curve"]).to_csv(run_dir / "report" / f"curve_{name}.csv", index=False)
        out["folds"][fold_name] = record
        logger.info(
            "%-22s %-9s fit %6.1fs  snapshot RMSE capped %.2f true %.2f",
            name,
            fold_name,
            info["fit_seconds"],
            metrics["snapshot_rmse_capped"],
            metrics["snapshot_rmse_true"],
        )
    out["peak_rss_mb"] = peak_rss_mb()
    out["rss_after_data_load_mb"] = rss_after_load
    out["wall_seconds"] = time.perf_counter() - t_start
    write_json(run_dir / "results" / f"{name}.json", out)
    return out


def _worker(cfg_dict: dict, name: str, run_dir: str, log_level: str) -> None:
    setup_logging(log_level)
    try:
        run_model(Stage2Config.from_dict(cfg_dict), name, Path(run_dir))
    except Exception:  # pragma: no cover - surfaced via exit code
        traceback.print_exc()
        raise


def run_comparison(
    cfg: Stage2Config,
    run_dir: str | Path | None = None,
    isolate: bool = True,
    pool: pd.DataFrame | None = None,
    log_level: str = "INFO",
    models: list[str] | None = None,
) -> Path:
    """Run every model (each in a spawned process if ``isolate``) and aggregate the results."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(run_dir or Path(cfg.paths.artifacts_dir) / f"{cfg.name}_{stamp}")
    (run_dir / "report").mkdir(parents=True, exist_ok=True)
    dump_stage2_config(cfg, run_dir / "config.yaml")
    started = datetime.now(UTC).isoformat(timespec="seconds")
    for name in models or list(cfg.models):
        if (run_dir / "results" / f"{name}.json").is_file():
            logger.info("Skipping %s (results exist; resume)", name)
            continue
        if isolate:
            proc = mp.get_context("spawn").Process(target=_worker, args=(cfg.to_dict(), name, str(run_dir), log_level))
            proc.start()
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"Model {name} failed (exit code {proc.exitcode})")
        else:
            run_model(cfg, name, run_dir, pool)
    write_json(
        run_dir / "metadata.json",
        {
            "name": cfg.name,
            "dataset": cfg.dataset,
            "started_utc": started,
            "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "split_manifest_sha256": sha256_file(cfg.split_manifest_path),
            "official_test_labels_read": False,
            "isolated_processes": isolate,
            "environment": environment_info(),
        },
    )
    aggregate(run_dir)
    return run_dir


def aggregate(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Build fold-level, summary, paired-difference and resource tables (+ plots)."""
    from aeromind.research.spec import load_stage2_config

    run_dir = Path(run_dir)
    cfg = load_stage2_config(run_dir / "config.yaml")
    results = {p.stem: read_json(p) for p in sorted((run_dir / "results").glob("*.json"))}
    rows = []
    for name, r in results.items():
        for fold, rec in r["folds"].items():
            rows.append({"model": name, "kind": r["kind"], "target_cap": r["target_cap"], "fold": fold, **rec["metrics"], **{f"fit_{k}": v for k, v in rec["fit"].items() if not isinstance(v, (list, dict))}})
    folds = pd.DataFrame(rows)
    report = run_dir / "report"
    folds.to_csv(report / "fold_metrics.csv", index=False)
    order = [m for m in cfg.models if m in results]

    cv = folds[folds.fold.str.startswith("cv")]
    summary_rows = []
    for name in order:
        g = cv[cv.model == name]
        c = folds[(folds.model == name) & (folds.fold == "canonical")]
        row: dict[str, Any] = {"model": name, "kind": results[name]["kind"], "target_cap": results[name]["target_cap"], "cv_folds": len(g)}
        for m in KEY_METRICS:
            if m in g:
                row[f"{m}_cv_mean"] = g[m].mean()
                row[f"{m}_cv_sd"] = g[m].std(ddof=1) if len(g) > 1 else np.nan
            if m in c and len(c):
                row[f"{m}_canonical"] = float(c[m].iloc[0])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(report / "summary.csv", index=False)

    paired = pd.DataFrame()
    ref = cfg.reference_model
    if ref and ref in results and len(cv):
        piv = cv.pivot(index="fold", columns="model", values="snapshot_rmse_capped")
        prow = []
        for name in order:
            if name == ref or name not in piv:
                continue
            d = (piv[name] - piv[ref]).dropna()
            prow.append(
                {
                    "model": name,
                    "reference": ref,
                    "folds": len(d),
                    "mean_diff_rmse_capped": d.mean(),
                    "sd_diff": d.std(ddof=1) if len(d) > 1 else np.nan,
                    "min_diff": d.min(),
                    "max_diff": d.max(),
                    "folds_better_than_reference": int((d < 0).sum()),
                }
            )
        paired = pd.DataFrame(prow)
        paired.to_csv(report / "paired_vs_reference.csv", index=False)

    res_rows = []
    for name in order:
        r = results[name]
        can = r["folds"].get("canonical", {})
        fits = [rec["fit"]["fit_seconds"] for f, rec in r["folds"].items()]
        lat = can.get("latency", {})
        res_rows.append(
            {
                "model": name,
                "n_parameters": can.get("fit", {}).get("n_parameters"),
                "bundle_mb": can.get("bundle_mb"),
                "fit_seconds_mean": float(np.mean(fits)),
                "fit_seconds_canonical": can.get("fit", {}).get("fit_seconds"),
                "epochs_canonical": can.get("fit", {}).get("epochs_run"),
                "seconds_per_epoch": can.get("fit", {}).get("seconds_per_epoch"),
                "batch_us_per_row": lat.get("batch_us_per_row"),
                "single_input_ms": lat.get("single_input_ms"),
                "end_to_end_engine_ms": lat.get("end_to_end_engine_ms"),
                "peak_rss_mb": r["peak_rss_mb"],
                "rss_after_data_load_mb": r["rss_after_data_load_mb"],
                "process_wall_seconds": r["wall_seconds"],
            }
        )
    resources = pd.DataFrame(res_rows)
    resources.to_csv(report / "resources.csv", index=False)
    _plots(cfg, cv, order, run_dir)
    _markdown(cfg, summary, paired, resources, report / "summary.md")
    return {"folds": folds, "summary": summary, "paired": paired, "resources": resources}


def _plots(cfg: Stage2Config, cv: pd.DataFrame, order: list[str], run_dir: Path) -> None:
    from aeromind.evaluation import plots as P

    if cv.empty:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(order) + 1.4))
    for i, name in enumerate(order):
        v = cv.loc[cv.model == name, "snapshot_rmse_capped"].to_numpy()
        ax.scatter(v, np.full(len(v), i), s=26, color=P.PALETTE[0], alpha=0.75, linewidths=0)
        ax.plot([v.mean()], [i], marker="|", markersize=16, color=P.INK, mew=2)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set(xlabel=f"Validation snapshot RMSE vs RUL capped at {cfg.evaluation.capped_truth_cap} (cycles)", title=f"{cfg.dataset}: per-fold results (dots) and mean (bar)")
    ax.grid(axis="y", visible=False)
    P.save(fig, run_dir / "report" / "cv_rmse_by_fold.png")

    curves = sorted((run_dir / "report").glob("curve_*.csv"))
    if curves:
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        for i, path in enumerate(curves[:3]):
            c = pd.read_csv(path)
            ax.plot(c["epoch"], c["inner_val_rmse"], color=P.PALETTE[i], label=path.stem.removeprefix("curve_"))
        ax.set(xlabel="Epoch", ylabel="Inner early-stopping RMSE (cycles, target)", title=f"{cfg.dataset} canonical fold: learning curves")
        ax.legend()
        P.save(fig, run_dir / "report" / "learning_curves.png")


def _fmt(mean: float, sd: float) -> str:
    return f"{mean:.2f} ± {sd:.2f}" if pd.notna(sd) else f"{mean:.2f}"


def _markdown(cfg: Stage2Config, summary: pd.DataFrame, paired: pd.DataFrame, resources: pd.DataFrame, path: Path) -> None:
    lines = [
        f"# {cfg.name} ({cfg.dataset})",
        "",
        f"Validation only; official test set not used. `capped` = truth min(RUL, {cfg.evaluation.capped_truth_cap}); `true` = uncapped RUL.",
        "CV columns: mean ± SD over engine-level folds (folds share training engines, so SDs understate uncertainty).",
        "",
        "| Model | Cap | Snapshot RMSE capped (CV) | Snapshot RMSE true (CV) | Snapshot MAE capped (CV) | PHM08/snapshot capped (CV) | Near-EOL RMSE true (CV) | Snapshot RMSE capped (canonical) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        def g(m: str, r: pd.Series = r) -> str:
            return _fmt(r.get(f"{m}_cv_mean", np.nan), r.get(f"{m}_cv_sd", np.nan)) if pd.notna(r.get(f"{m}_cv_mean", np.nan)) else "—"

        can = r.get("snapshot_rmse_capped_canonical", np.nan)
        lines.append(
            f"| {r['model']} | {r['target_cap'] if pd.notna(r['target_cap']) else 'none'} | {g('snapshot_rmse_capped')} | {g('snapshot_rmse_true')} | {g('snapshot_mae_capped')} | {g('snapshot_phm08_mean_capped')} | {g('near_eol_rmse_true')} | {can:.2f} |"
            if pd.notna(can)
            else f"| {r['model']} | {r['target_cap']} | {g('snapshot_rmse_capped')} | {g('snapshot_rmse_true')} | {g('snapshot_mae_capped')} | {g('snapshot_phm08_mean_capped')} | {g('near_eol_rmse_true')} | — |"
        )
    if not paired.empty:
        lines += ["", f"Paired per-fold difference in snapshot RMSE (capped) vs `{paired.reference.iloc[0]}` (negative = better):", "",
                  "| Model | Mean diff | SD | Min | Max | Folds better |", "|---|---|---|---|---|---|"]
        for _, r in paired.iterrows():
            lines.append(f"| {r['model']} | {r['mean_diff_rmse_capped']:+.2f} | {r['sd_diff']:.2f} | {r['min_diff']:+.2f} | {r['max_diff']:+.2f} | {r['folds_better_than_reference']}/{r['folds']} |")
    lines += ["", "| Model | Params | Bundle MB | Fit s (mean) | Epochs (canonical) | µs/row batch | Single input ms | E2E engine ms | Peak RSS MB |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in resources.iterrows():
        def f(v: Any, fmt: str) -> str:
            return format(v, fmt) if v is not None and pd.notna(v) else "—"

        lines.append(
            f"| {r['model']} | {f(r['n_parameters'], ',.0f')} | {f(r['bundle_mb'], '.3f')} | {f(r['fit_seconds_mean'], '.1f')} | {f(r['epochs_canonical'], '.0f')} | {f(r['batch_us_per_row'], '.1f')} | {f(r['single_input_ms'], '.2f')} | {f(r['end_to_end_engine_ms'], '.1f')} | {f(r['peak_rss_mb'], '.0f')} |"
        )
    path.write_text("\n".join(lines) + "\n")
