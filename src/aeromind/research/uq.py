"""Uncertainty quantification: quantile regression and split conformal prediction.

Data roles (canonical split; official test never used):

- **train** (70 engines): point and quantile models are fitted here (point models are the
  canonical-fold bundles of a Stage-2 comparison run).
- **calibration** (15 engines): conformal calibration only.
- **validation** (15 engines): coverage / width evaluation.
- Role swap (sanity check): calibrate on validation engines, evaluate on calibration engines.

Methods (intervals for the *capped* RUL target, the quantity the models estimate; coverage
of uncapped true RUL is reported separately):

- ``conformal_engine``  split conformal with ONE randomly drawn snapshot per calibration
  engine (n = 15 exchangeable scores). Valid marginal coverage ≥ 1−α for a snapshot drawn
  the same way from a new exchangeable engine. With n = 15, α < 1/16 is impossible (infinite
  interval). Repeated over ``draws`` random snapshot choices to show variability.
- ``conformal_pooled``  all calibration windows as scores (n ≈ 3 000). **Heuristic**: windows
  of the same engine are dependent, so the finite-sample guarantee does not apply.
- ``quantile_raw``      uncalibrated central quantile interval [q_{α/2}, q_{1−α/2}].
- ``cqr_engine`` / ``cqr_pooled``  conformalized quantile regression (Romano et al., 2019)
  with the same two score-sampling schemes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aeromind.data.cmapss import load_train
from aeromind.data.splits import load_split, verify_split
from aeromind.evaluation.protocol import engine_balanced_weights, select_snapshots
from aeromind.features.pipeline import FeaturePipeline
from aeromind.features.targets import cap_rul, make_target
from aeromind.models.bundle import load_bundle
from aeromind.models.factory import build_estimator
from aeromind.research.runner import fit_entry
from aeromind.research.spec import ModelEntry, load_stage2_config
from aeromind.utils import environment_info, peak_rss_mb, read_json, timer, write_json

logger = logging.getLogger(__name__)


@dataclass
class UQConfig:
    """Uncertainty experiment configuration."""

    name: str
    comparison_run: str
    point_models: list[str]
    quantile_models: dict[str, dict[str, Any]] = field(default_factory=dict)
    quantiles: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    alphas: list[float] = field(default_factory=lambda: [0.1, 0.2])
    reliability_levels: list[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    draws: int = 500
    seed: int = 42
    artifacts_dir: str = "artifacts/stage2"

    @classmethod
    def load(cls, path: str | Path) -> UQConfig:
        """Load from YAML (unknown keys raise)."""
        payload = yaml.safe_load(Path(path).read_text())
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown UQ keys {sorted(unknown)}")
        return cls(**payload)


# ----------------------------------------------------------------------------- core maths
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold: the ⌈(n+1)(1−α)⌉-th smallest score (∞ if that exceeds n)."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(s[k - 1])


def one_per_engine(engine_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Positions of one uniformly drawn row per engine (engines in sorted order)."""
    ids = np.asarray(engine_ids)
    out = []
    for e in np.unique(ids):
        rows = np.flatnonzero(ids == e)
        out.append(rng.choice(rows))
    return np.asarray(out)


class ClassicalQuantileModel:
    """One HistGradientBoosting quantile regressor per level on a shared fitted pipeline."""

    def __init__(self, entry: ModelEntry, quantiles: list[float], seed: int) -> None:
        self.entry, self.quantiles, self.seed = entry, sorted(quantiles), seed
        self.pipeline: FeaturePipeline | None = None
        self.models: dict[float, Any] = {}

    def fit(self, train_df: pd.DataFrame, cap: int | None, fit_cap: int | None | str = "same") -> ClassicalQuantileModel:
        """Fit pipeline and per-quantile estimators on training engines.

        ``fit_cap=None`` fits on uncapped RUL and clips predictions at ``cap``. Because
        quantiles commute with monotone maps, q_τ(min(Y, c)) = min(q_τ(Y), c), so this estimates
        the capped-target quantiles while avoiding the point mass of targets at the cap, which
        makes gradient-boosted upper quantiles degenerate (observed for HGB on FD001).
        """
        self.pipeline = FeaturePipeline(self.entry.feature_config(), self.seed).fit(train_df)
        X = self.pipeline.transform(train_df)
        y = make_target(train_df, cap if fit_cap == "same" else fit_cap)
        for q in self.quantiles:
            params = {**self.entry.params, "loss": "quantile", "quantile": q}
            self.models[q] = build_estimator("hist_gradient_boosting", params, self.seed).fit(X, y)
        self.cap = cap
        return self

    def predict_quantiles(self, df: pd.DataFrame) -> np.ndarray:
        """(N, n_quantiles), sorted per row, clipped to [0, cap]."""
        X = self.pipeline.transform(df)
        out = np.column_stack([self.models[q].predict(X) for q in self.quantiles])
        return np.clip(np.sort(out, axis=1), 0, self.cap)


# ----------------------------------------------------------------------------- evaluation
def _coverage(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    inside = (y >= lo) & (y <= hi)
    return float(np.average(inside, weights=w))


def _interval_rows(
    method: str,
    model: str,
    alpha: float,
    roles: tuple[str, str],
    lo_fn,
    hi_fn,
    ev: dict[str, np.ndarray],
    draws: list[float] | None,
) -> dict[str, Any]:
    """Summarise an interval method. ``lo_fn/hi_fn(q)`` map a conformal threshold to bounds."""
    thresholds = draws if draws is not None else [0.0]
    cov_s, cov_w, cov_true, width = [], [], [], []
    for q in thresholds:
        lo, hi = lo_fn(q), hi_fn(q)
        s = ev["snap"]
        cov_s.append(_coverage(lo[s], hi[s], ev["y_cap"][s]))
        cov_true.append(_coverage(lo[s], hi[s], ev["y_true"][s]))
        cov_w.append(_coverage(lo, hi, ev["y_cap"], ev["w"]))
        width.append(float(np.mean(hi[s] - lo[s])))
    finite = np.isfinite(thresholds)
    arr = np.asarray
    return {
        "method": method,
        "model": model,
        "nominal": 1 - alpha,
        "calibrate_on": roles[0],
        "evaluate_on": roles[1],
        "draws": len(thresholds) if draws is not None else 0,
        "finite_fraction": float(finite.mean()),
        "threshold_median": float(np.median(thresholds)),
        "coverage_snapshot_capped_mean": float(arr(cov_s).mean()),
        "coverage_snapshot_capped_p05": float(np.percentile(cov_s, 5)),
        "coverage_snapshot_capped_p95": float(np.percentile(cov_s, 95)),
        "coverage_window_ew_capped_mean": float(arr(cov_w).mean()),
        "coverage_snapshot_true_mean": float(arr(cov_true).mean()),
        "width_snapshot_mean": float(np.mean(width)) if finite.all() else float("inf"),
        "width_snapshot_median_over_draws": float(np.median(width)),
    }


def run_uq(cfg: UQConfig, pool: pd.DataFrame | None = None) -> Path:
    """Execute the uncertainty experiment; returns the output directory."""
    comp_dir = Path(cfg.comparison_run)
    s2 = load_stage2_config(comp_dir / "config.yaml")
    cap, cap_c = s2.target_cap, s2.evaluation.capped_truth_cap
    pool = load_train(s2.paths.raw_dir, s2.dataset) if pool is None else pool
    manifest = load_split(s2.split_manifest_path)
    verify_split(manifest, pool.engine_id.unique())
    parts = {
        p: pool[pool.engine_id.isin(manifest.ids(p))].reset_index(drop=True)
        for p in ("train", "validation", "calibration")
    }
    out = Path(cfg.artifacts_dir) / f"{cfg.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    (out / "report").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))

    # Per-partition evaluation context.
    ctx: dict[str, dict[str, np.ndarray]] = {}
    for p in ("validation", "calibration"):
        df = parts[p]
        snaps = select_snapshots(df, s2.evaluation.snapshots_per_engine, s2.evaluation.snapshot_seed)
        ctx[p] = {
            "y_true": df["rul"].to_numpy(float),
            "y_cap": cap_rul(df["rul"], cap_c),
            "engine": df["engine_id"].to_numpy(),
            "w": engine_balanced_weights(df["engine_id"]),
            "snap": df.index.get_indexer(snaps),
        }

    # Point predictions from the comparison run's canonical-fold bundles.
    point: dict[str, dict[str, np.ndarray]] = {}
    for name in cfg.point_models:
        pred, _ = load_bundle(comp_dir / "models" / f"{name}.joblib")
        point[name] = {p: pred.predict(parts[p]) for p in ("validation", "calibration")}

    # Quantile models trained on the canonical training engines.
    qpred: dict[str, dict[str, np.ndarray]] = {}
    fit_info: dict[str, Any] = {}
    train = parts["train"].copy()
    for name, spec in cfg.quantile_models.items():
        entry = ModelEntry(**spec)
        with timer() as t:
            if entry.kind == "classical":
                fit_cap = "same" if entry.target_cap == "default" else entry.target_cap
                model = ClassicalQuantileModel(entry, cfg.quantiles, cfg.seed).fit(train, cap, fit_cap)
                qpred[name] = {p: model.predict_quantiles(parts[p]) for p in ("validation", "calibration")}
                info: dict[str, Any] = {}
            else:
                deep = {**entry.deep, "train": {**entry.deep.get("train", {}), "loss": "quantile", "quantiles": cfg.quantiles}}
                predictor, info = fit_entry(ModelEntry(kind="deep", deep=deep), name, train, cap, cfg.seed)
                qpred[name] = {p: predictor.predict_quantiles(predictor.inputs(parts[p])) for p in ("validation", "calibration")}
                info = {k: v for k, v in info.items() if k != "curve"}
        fit_info[name] = {"fit_seconds": t["seconds"], **info}
        logger.info("Quantile model %s fitted in %.1fs", name, t["seconds"])

    qidx = {q: i for i, q in enumerate(sorted(cfg.quantiles))}
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, Any]] = []
    for roles in (("calibration", "validation"), ("validation", "calibration")):
        cal, ev = ctx[roles[0]], ctx[roles[1]]
        draw_idx = [one_per_engine(cal["engine"], rng) for _ in range(cfg.draws)]
        for alpha in sorted(set(cfg.alphas) | {round(1 - lv, 10) for lv in cfg.reliability_levels}):
            main = alpha in cfg.alphas
            for name, pr in point.items():
                yhat_c, yhat_e = pr[roles[0]], pr[roles[1]]
                scores = np.abs(cal["y_cap"] - yhat_c)
                lo_f = lambda q, y=yhat_e: np.maximum(y - q, 0.0)  # noqa: E731
                hi_f = lambda q, y=yhat_e: y + q  # noqa: E731
                rows.append(_interval_rows("conformal_pooled", name, alpha, roles, lo_f, hi_f, ev, [conformal_quantile(scores, alpha)]))
                if main:
                    th = [conformal_quantile(scores[i], alpha) for i in draw_idx]
                    rows.append(_interval_rows("conformal_engine", name, alpha, roles, lo_f, hi_f, ev, th))
            lo_q, hi_q = round(alpha / 2, 10), round(1 - alpha / 2, 10)
            if lo_q not in qidx or hi_q not in qidx:
                continue
            for name, qp in qpred.items():
                lc, hc = qp[roles[0]][:, qidx[lo_q]], qp[roles[0]][:, qidx[hi_q]]
                le, he = qp[roles[1]][:, qidx[lo_q]], qp[roles[1]][:, qidx[hi_q]]
                scores = np.maximum(lc - cal["y_cap"], cal["y_cap"] - hc)
                rows.append(_interval_rows("quantile_raw", name, alpha, roles, lambda q, a=le: a, lambda q, b=he: b, ev, None))
                lo_f = lambda q, a=le: np.maximum(a - q, 0.0)  # noqa: E731
                hi_f = lambda q, b=he: b + q  # noqa: E731
                rows.append(_interval_rows("cqr_pooled", name, alpha, roles, lo_f, hi_f, ev, [conformal_quantile(scores, alpha)]))
                if main:
                    th = [conformal_quantile(scores[i], alpha) for i in draw_idx]
                    rows.append(_interval_rows("cqr_engine", name, alpha, roles, lo_f, hi_f, ev, th))
    table = pd.DataFrame(rows)
    table.to_csv(out / "report" / "intervals.csv", index=False)
    main_table = table[table.nominal.round(6).isin([round(1 - a, 6) for a in cfg.alphas])]
    main_table.to_csv(out / "report" / "intervals_main.csv", index=False)

    by_bin = _coverage_by_bin(cfg, s2, point, qpred, ctx, qidx, parts)
    by_bin.to_csv(out / "report" / "coverage_by_rul_bin.csv", index=False)
    _plots(cfg, table, by_bin, point, ctx, parts, out, cap_c)
    write_json(
        out / "metadata.json",
        {
            "comparison_run": str(comp_dir),
            "dataset": s2.dataset,
            "target_cap": cap,
            "capped_truth_cap": cap_c,
            "engines": {p: manifest.ids(p) for p in ("train", "validation", "calibration")},
            "quantile_fit": fit_info,
            "official_test_labels_read": False,
            "peak_rss_mb": peak_rss_mb(),
            "environment": environment_info(),
        },
    )
    return out


def _coverage_by_bin(cfg, s2, point, qpred, ctx, qidx, parts) -> pd.DataFrame:
    """Coverage and width by true-RUL bin on validation windows (calibrated on calibration engines), α=0.1."""
    alpha = 0.1
    cal, ev = ctx["calibration"], ctx["validation"]
    bins = s2.evaluation.rul_bins
    rows = []
    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, pr in point.items():
        q = conformal_quantile(np.abs(cal["y_cap"] - pr["calibration"]), alpha)
        variants[f"conformal_pooled:{name}"] = (np.maximum(pr["validation"] - q, 0), pr["validation"] + q)
    if 0.05 in qidx and 0.95 in qidx:
        for name, qp in qpred.items():
            lc, hc = qp["calibration"][:, qidx[0.05]], qp["calibration"][:, qidx[0.95]]
            q = conformal_quantile(np.maximum(lc - cal["y_cap"], cal["y_cap"] - hc), alpha)
            variants[f"cqr_pooled:{name}"] = (np.maximum(qp["validation"][:, qidx[0.05]] - q, 0), qp["validation"][:, qidx[0.95]] + q)
            variants[f"quantile_raw:{name}"] = (qp["validation"][:, qidx[0.05]], qp["validation"][:, qidx[0.95]])
    cut = pd.cut(ev["y_true"], bins=bins, right=False)
    for label, (lo, hi) in variants.items():
        frame = pd.DataFrame({"bin": cut, "in": (ev["y_cap"] >= lo) & (ev["y_cap"] <= hi), "in_true": (ev["y_true"] >= lo) & (ev["y_true"] <= hi), "width": hi - lo})
        g = frame.groupby("bin", observed=True)
        t = pd.DataFrame({"n": g.size(), "coverage_capped": g["in"].mean(), "coverage_true": g["in_true"].mean(), "width_mean": g["width"].mean()})
        t.index = t.index.astype(str)
        t = t.reset_index().rename(columns={"bin": "true_rul_bin"})
        t.insert(0, "method", label)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def _plots(cfg, table, by_bin, point, ctx, parts, out: Path, cap_c: int = 125) -> None:
    import matplotlib.pyplot as plt

    from aeromind.evaluation import plots as P

    rel = table[(table.calibrate_on == "calibration") & table.method.isin(["conformal_pooled", "quantile_raw", "cqr_pooled"])]
    labels = sorted({(m, n) for m, n in zip(rel.method, rel.model, strict=True)})[:3]
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.plot([0.45, 1], [0.45, 1], color=P.MUTED, ls="--", lw=1, label="ideal")
    for i, (m, n) in enumerate(labels):
        d = rel[(rel.method == m) & (rel.model == n)].sort_values("nominal")
        ax.plot(d.nominal, d.coverage_snapshot_capped_mean, marker="o", ms=6, color=P.PALETTE[i], label=f"{m} · {n}")
    ax.set(xlabel="Nominal coverage", ylabel="Empirical coverage (validation snapshots, capped RUL)", title="Calibration diagnostics")
    ax.legend(loc="upper left")
    P.save(fig, out / "report" / "reliability.png")

    methods = by_bin.method.unique()[:3]
    bins = by_bin.true_rul_bin.unique().tolist()
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    width = 0.8 / max(1, len(methods))
    for i, m in enumerate(methods):
        d = by_bin[by_bin.method == m].set_index("true_rul_bin").reindex(bins)
        ax.bar(np.arange(len(bins)) + (i - (len(methods) - 1) / 2) * width, d.coverage_capped, width * 0.92, color=P.PALETTE[i], label=m)
    ax.axhline(0.9, color=P.INK_2, ls=":", lw=1)
    ax.set_xticks(range(len(bins)), bins, rotation=30, ha="right")
    ax.set(ylim=(0, 1.05), xlabel="True RUL bin (cycles)", ylabel="Coverage of capped RUL", title="90% intervals: coverage by RUL range (validation windows)")
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    P.save(fig, out / "report" / "coverage_by_rul_bin.png")

    name = next(iter(point))
    cal = ctx["calibration"]
    q = conformal_quantile(np.abs(cal["y_cap"] - point[name]["calibration"]), 0.1)
    val = parts["validation"].assign(pred=point[name]["validation"])
    show = sorted(val.engine_id.unique())[:6]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.4), sharey=True, squeeze=False)
    for ax, e in zip(axes.ravel(), show, strict=False):
        d = val[val.engine_id == e]
        ax.fill_between(d.cycle, np.maximum(d.pred - q, 0), d.pred + q, color=P.PALETTE[0], alpha=0.18, linewidth=0, label="90% interval")
        ax.plot(d.cycle, d.pred, color=P.PALETTE[0], lw=1.4, label="prediction")
        ax.plot(d.cycle, np.minimum(d.rul, cap_c), color=P.INK_2, lw=1, ls=":", label="capped truth")
        ax.plot(d.cycle, d.rul, color=P.MUTED, lw=1, label="true RUL")
        ax.set_title(f"Engine {e}", fontsize=9)
    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle(f"{name}: pooled split-conformal 90% band (validation engines)", fontsize=11)
    P.save(fig, out / "report" / "interval_trajectories.png")


def summarise_uq(out_dir: str | Path) -> str:
    """Markdown table of the main (α in config) interval results."""
    out_dir = Path(out_dir)
    t = pd.read_csv(out_dir / "report" / "intervals_main.csv")
    meta = read_json(out_dir / "metadata.json")
    lines = [
        f"Calibration engines: {len(meta['engines']['calibration'])}; evaluation: validation engines (and role swap).",
        "",
        "| Method | Model | Nominal | Calibrate → evaluate | Coverage (snapshots, capped) mean [p05, p95 over draws] | Coverage true RUL | Mean width (cycles) | Finite |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in t.iterrows():
        rng_ = f"[{r.coverage_snapshot_capped_p05:.3f}, {r.coverage_snapshot_capped_p95:.3f}]" if r.draws > 1 else ""
        lines.append(
            f"| {r.method} | {r.model} | {r.nominal:.2f} | {r.calibrate_on} → {r.evaluate_on} | {r.coverage_snapshot_capped_mean:.3f} {rng_} | {r.coverage_snapshot_true_mean:.3f} | {r.width_snapshot_median_over_draws:.1f} | {r.finite_fraction:.2f} |"
        )
    text = "\n".join(lines) + "\n"
    (out_dir / "report" / "summary.md").write_text(text)
    return text
