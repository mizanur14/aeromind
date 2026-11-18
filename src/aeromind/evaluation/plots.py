"""Static diagnostic plots (matplotlib, Agg backend).

Style: fixed categorical order (validated reference palette), thin marks, recessive grid,
one y-axis per chart, small multiples instead of crowded overlays.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#9b9a95"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"


def apply_style() -> None:
    """Set global rcParams once."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        }
    )


apply_style()


def save(fig: plt.Figure, path: str | Path) -> Path:
    """Save at 150 dpi and close."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pred_vs_true(rul_true, pred, title: str, path: str | Path, cap: int | None = None) -> Path:
    """Scatter of predicted vs true uncapped RUL with identity (and cap) reference."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    t = np.asarray(rul_true, float)
    ax.scatter(t, pred, s=14, color=PALETTE[0], alpha=0.65, linewidths=0, label="prediction")
    hi = max(float(t.max()), float(np.max(pred))) * 1.03
    ax.plot([0, hi], [0, hi], color=MUTED, lw=1, ls="--", label="perfect (y = x)")
    if cap is not None:
        ax.plot([0, cap, hi], [0, cap, cap], color=INK_2, lw=1, ls=":", label=f"capped truth ({cap})")
    ax.set(xlabel="True RUL (cycles, uncapped)", ylabel="Predicted RUL (cycles)", title=title)
    ax.legend(loc="upper left")
    return save(fig, path)


def plot_residuals(rul_true, pred, title: str, path: str | Path) -> Path:
    """Residual (pred − true) versus true uncapped RUL."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    t = np.asarray(rul_true, float)
    ax.scatter(t, np.asarray(pred) - t, s=12, color=PALETTE[0], alpha=0.6, linewidths=0)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set(xlabel="True RUL (cycles, uncapped)", ylabel="Residual: pred − true (cycles)", title=title)
    return save(fig, path)


def plot_metric_bars(table: pd.DataFrame, metric: str, title: str, path: str | Path) -> Path:
    """Horizontal bar chart of one metric per model (single series, sorted ascending)."""
    data = table.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(data) + 1.2))
    ax.barh(data["model"], data[metric], color=PALETTE[0], height=0.6)
    for y, v in enumerate(data[metric]):
        ax.text(v, y, f" {v:.1f}", va="center", fontsize=8, color=INK_2)
    ax.set(xlabel=metric, title=title)
    ax.grid(axis="y", visible=False)
    return save(fig, path)


def plot_error_by_bin(tables: dict[str, pd.DataFrame], title: str, path: str | Path, metric: str = "rmse") -> Path:
    """Grouped bars of an error metric per true-RUL bin for up to 3 models."""
    names = list(tables)[:3]
    bins = tables[names[0]]["true_rul_bin"].tolist()
    x = np.arange(len(bins))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for i, name in enumerate(names):
        tab = tables[name].set_index("true_rul_bin").reindex(bins)
        ax.bar(x + (i - (len(names) - 1) / 2) * width, tab[metric], width * 0.92, color=PALETTE[i], label=name)
    ax.set_xticks(x, bins, rotation=30, ha="right")
    ax.set(xlabel="True RUL bin (cycles, uncapped)", ylabel=f"{metric.upper()} (cycles)", title=title)
    ax.grid(axis="x", visible=False)
    ax.legend()
    return save(fig, path)


def plot_engine_trajectories(
    frame: pd.DataFrame, engine_ids: list[int], title: str, path: str | Path, cap: int | None = None
) -> Path:
    """Small multiples: true RUL vs predicted RUL over cycles for selected engines."""
    n = len(engine_ids)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), squeeze=False, sharey=True)
    for ax, eid in zip(axes.ravel(), engine_ids, strict=False):
        e = frame[frame["engine_id"] == eid]
        ax.plot(e["cycle"], e["rul"], color=MUTED, lw=1.2, label="true RUL")
        if cap is not None:
            ax.plot(e["cycle"], np.minimum(e["rul"], cap), color=INK_2, lw=1, ls=":", label="capped truth")
        ax.plot(e["cycle"], e["pred"], color=PALETTE[0], lw=1.6, label="prediction")
        ax.set_title(f"Engine {eid}", fontsize=9)
        ax.set_xlabel("Cycle")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel("RUL (cycles)")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(title, fontsize=11, color=INK)
    return save(fig, path)
