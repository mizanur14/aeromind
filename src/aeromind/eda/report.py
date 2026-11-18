"""EDA report generation.

Two explicit scopes:

- ``pool``  — every engine in the training file. **Exploratory only.** Useful to understand
  the data, but must not drive feature selection, because it includes validation and
  calibration engines.
- ``train`` — only the training-split engine IDs from the split manifest. This is the
  **model-development** analysis; feature-selection decisions may use it.

Official test files are never read.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from aeromind.data.cmapss import SENSOR_COLUMNS, SENSOR_INFO, SETTING_COLUMNS, SUBSET_INFO, load_train
from aeromind.data.splits import load_split, split_frame, verify_split
from aeromind.eda import analysis as A
from aeromind.evaluation import plots as P
from aeromind.utils import write_json

logger = logging.getLogger(__name__)

SCOPES = {
    "pool": "Full training pool (all engines in train_FD00x.txt). EXPLORATORY ONLY — includes "
    "validation and calibration engines; do not use for feature selection.",
    "train": "Training-split engines only (from the split manifest). Model-development analysis.",
}

DIVERGING = LinearSegmentedColormap.from_list("aero_div", [P.PALETTE[0], "#e6e5e0", P.PALETTE[7]])


def _label(col: str) -> str:
    return f"{col} ({SENSOR_INFO[col][0]})" if col in SENSOR_INFO else col


def run_eda(
    raw_dir: str | Path,
    dataset: str,
    scope: str,
    manifest_path: str | Path | None,
    reports_dir: str | Path = "reports",
    cap: int = 125,
    pool: pd.DataFrame | None = None,
) -> Path:
    """Generate CSV/JSON/PNG EDA outputs under ``<reports_dir>/<dataset>/eda/<scope>/``."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {sorted(SCOPES)}")
    df = pool if pool is not None else load_train(raw_dir, dataset)
    if scope == "train":
        if manifest_path is None:
            raise ValueError("scope 'train' requires a split manifest")
        manifest = load_split(manifest_path)
        verify_split(manifest, df["engine_id"].unique())
        df = split_frame(df, manifest, "train").reset_index(drop=True)
    out = Path(reports_dir) / dataset / "eda" / scope
    out.mkdir(parents=True, exist_ok=True)
    logger.info("EDA %s/%s: %d engines, %d rows -> %s", dataset, scope, df.engine_id.nunique(), len(df), out)

    cols = SETTING_COLUMNS + SENSOR_COLUMNS
    life = A.lifetime_table(df)
    stats = A.sensor_statistics(df, cols)
    informative = [c for c in SENSOR_COLUMNS if not stats.loc[c, "constant"] and not stats.loc[c, "near_constant"]]
    rulc = A.rul_correlations(df, SENSOR_COLUMNS, cap)
    corr = A.correlation_matrix(df, informative)
    trends = A.within_engine_trends(df, informative)
    between = A.between_engine_variability(df, informative)
    n_cond = int(SUBSET_INFO[dataset]["conditions"])
    conds = A.operating_conditions(df, n_cond)
    early_late = A.early_late_comparison(df, informative)
    outliers = A.outlier_summary(df, informative)
    vif = A.variance_inflation(df, informative)
    pairs = A.high_correlation_pairs(corr, threshold=0.8)

    life.to_csv(out / "lifetimes.csv", index=False)
    stats.to_csv(out / "column_statistics.csv")
    rulc.to_csv(out / "sensor_rul_spearman.csv", index=False)
    corr.to_csv(out / "sensor_correlation_spearman.csv")
    trends.to_csv(out / "within_engine_trends.csv", index=False)
    between.to_csv(out / "between_engine_variability.csv", index=False)
    conds.to_csv(out / "operating_conditions.csv", index=False)
    early_late.to_csv(out / "early_vs_late_ks.csv", index=False)
    outliers.to_csv(out / "outliers_detrended.csv", index=False)
    vif.to_csv(out / "vif.csv", index=False)
    pairs.to_csv(out / "high_correlation_pairs.csv", index=False)

    _plots(df, life, rulc, corr, trends, early_late, vif, out, dataset, scope)

    summary = {
        "dataset": dataset,
        "scope": scope,
        "scope_description": SCOPES[scope],
        "engines": int(df.engine_id.nunique()),
        "rows": int(len(df)),
        "missing_values": int(df[cols].isna().sum().sum()),
        "lifetime": {k: float(v) for k, v in life["lifetime"].describe().items()},
        "constant_columns": stats.index[stats["constant"]].tolist(),
        "near_constant_columns": stats.index[stats["near_constant"]].tolist(),
        "informative_sensors": informative,
        "top_rul_spearman": dict(zip(rulc["column"].head(8), rulc["spearman_rul"].head(8).round(4), strict=True)),
        "strongest_within_engine_trends": dict(
            zip(trends["column"].head(8), trends["median_rho"].head(8).round(4), strict=True)
        ),
        "high_vif_columns_gt10": vif.loc[vif["vif"] > 10, "column"].tolist(),
        "high_spearman_pairs_abs_ge_0_8": pairs.to_dict(orient="records"),
        "operating_conditions": n_cond,
        "outlier_rule": "robust |z|>5 on per-engine quadratic-detrended residuals (descriptive only)",
        "total_outliers": int(outliers["outliers"].sum()),
        "notes": [
            "Row-level correlations weight long-lived engines more heavily.",
            "Correlations are associations, not causal effects.",
            "Official test files were not read.",
        ],
    }
    write_json(out / "summary.json", summary)
    return out


def _plots(df, life, rulc, corr, trends, early_late, vif, out: Path, dataset: str, scope: str) -> None:
    import matplotlib.pyplot as plt

    tag = f"{dataset} [{scope}]"

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(life["lifetime"], bins=20, color=P.PALETTE[0], edgecolor=P.SURFACE, linewidth=2)
    ax.set(xlabel="Lifetime (cycles to failure)", ylabel="Engines", title=f"{tag}: engine lifetimes")
    ax.grid(axis="x", visible=False)
    P.save(fig, out / "lifetimes.png")

    data = rulc.dropna().sort_values("spearman_rul")
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(data) + 1.2))
    ax.barh([_label(c) for c in data["column"]], data["spearman_rul"], color=P.PALETTE[0], height=0.6)
    ax.axvline(0, color=P.MUTED, lw=1)
    ax.set(xlabel="Spearman correlation with uncapped RUL (row level)", title=f"{tag}: sensor–RUL association")
    ax.grid(axis="y", visible=False)
    P.save(fig, out / "sensor_rul_spearman.png")

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman r")
    ax.set_title(f"{tag}: sensor correlations")
    P.save(fig, out / "sensor_correlation_spearman.png")

    top = trends["column"].head(6).tolist()
    engines = sorted(df.engine_id.unique())[:15]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), squeeze=False)
    for ax, col in zip(axes.ravel(), top, strict=False):
        for eid in engines:
            e = df[df.engine_id == eid]
            ax.plot(-e["rul"], e[col], color=P.MUTED, lw=0.7, alpha=0.7)
        med = df.assign(t=-df["rul"]).groupby("t")[col].median()
        med = med[med.index >= -250]
        ax.plot(med.index, med.to_numpy(), color=P.PALETTE[0], lw=1.8)
        ax.set_title(_label(col), fontsize=9)
        ax.set_xlabel("Cycles relative to failure")
    for ax in axes.ravel()[len(top) :]:
        ax.set_visible(False)
    fig.suptitle(f"{tag}: degradation trajectories (grey: 15 engines; blue: median)", fontsize=11)
    P.save(fig, out / "degradation_trajectories.png")

    top = early_late["column"].head(6).tolist()
    pos = df.groupby("engine_id").cumcount()
    n = df.groupby("engine_id")["cycle"].transform("size")
    early, late = df[pos < 0.2 * n], df[pos >= 0.8 * n]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), squeeze=False)
    for ax, col in zip(axes.ravel(), top, strict=False):
        bins = np.linspace(df[col].min(), df[col].max(), 40)
        ax.hist(early[col], bins=bins, color=P.PALETTE[0], alpha=0.6, label="first 20% of life")
        ax.hist(late[col], bins=bins, color=P.PALETTE[1], alpha=0.6, label="last 20% of life")
        ax.set_title(_label(col), fontsize=9)
        ax.grid(axis="x", visible=False)
    for ax in axes.ravel()[len(top) :]:
        ax.set_visible(False)
    axes[0, 0].legend()
    fig.suptitle(f"{tag}: early vs late-life distributions (largest KS)", fontsize=11)
    P.save(fig, out / "early_vs_late_distributions.png")

    data = vif.sort_values("vif")
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(data) + 1.2))
    ax.barh(data["column"], np.minimum(data["vif"], 1e4), color=P.PALETTE[0], height=0.6)
    ax.axvline(10, color=P.INK_2, lw=1, ls=":")
    ax.set_xscale("log")
    ax.set(xlabel="Variance inflation factor (log; dotted = 10)", title=f"{tag}: multicollinearity")
    ax.grid(axis="y", visible=False)
    P.save(fig, out / "vif.png")
