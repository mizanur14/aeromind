"""Evaluation protocols.

- **Snapshot evaluation** (internal validation): for each validation engine a fixed,
  seeded set of cut cycles; the prediction at a cut uses only history up to the cut
  (guaranteed by the causal features). Every engine contributes the same number of
  snapshots, so long trajectories do not dominate.
- **All-window evaluation**: every row of every engine, reported both row-weighted
  and engine-weighted (each engine's rows share total weight 1).
- **Final-cycle evaluation** (official test): the last observed row of each engine.

Each protocol reports metrics against three truths:

- ``true``    uncapped true RUL,
- ``capped``  ``min(true, capped_truth_cap)`` (literature convention),
- ``target``  the model's own training target formulation (capped if trained capped).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aeromind.evaluation.metrics import regression_metrics
from aeromind.features.targets import cap_rul


def select_snapshots(df: pd.DataFrame, n_per_engine: int, seed: int) -> pd.Index:
    """Deterministically pick up to ``n_per_engine`` rows per engine (uniform over cycles).

    Engines are processed in sorted ID order with one RNG, so the result depends only on
    the set of engines, their lengths and the seed.
    """
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for _, group in sorted(df.groupby("engine_id"), key=lambda kv: kv[0]):
        group = group.sort_values("cycle")  # row order must not affect the selection
        k = min(n_per_engine, len(group))
        chosen.append(np.sort(rng.choice(group.index.to_numpy(), size=k, replace=False)))
    return pd.Index(np.concatenate(chosen)) if chosen else pd.Index([])


def final_cycle_index(df: pd.DataFrame) -> pd.Index:
    """Index of the last observed row of each engine."""
    return df.groupby("engine_id")["cycle"].idxmax().sort_index().pipe(lambda s: pd.Index(s.to_numpy()))


def engine_balanced_weights(engine_ids: pd.Series | np.ndarray) -> np.ndarray:
    """Weights ``1 / n_rows(engine)`` so each engine has total weight 1."""
    ids = pd.Series(np.asarray(engine_ids))
    return (1.0 / ids.map(ids.value_counts())).to_numpy(dtype=float)


def metrics_against_truths(
    rul_true: np.ndarray,
    pred: np.ndarray,
    target_cap: int | None,
    capped_truth_cap: int,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Flat dict of metrics with ``_true``, ``_capped`` and ``_target`` suffixes."""
    truths = {
        "true": np.asarray(rul_true, dtype=float),
        "capped": cap_rul(rul_true, capped_truth_cap),
        "target": cap_rul(rul_true, target_cap),
    }
    out: dict[str, float] = {}
    for label, truth in truths.items():
        for key, value in regression_metrics(truth, pred, weights).items():
            if key == "n":
                out["n"] = value
            else:
                out[f"{key}_{label}"] = value
    return out


def error_by_rul_bin(rul_true: np.ndarray, pred: np.ndarray, bins: list[int]) -> pd.DataFrame:
    """MAE/RMSE/bias of predictions grouped by uncapped true-RUL bins (left-closed)."""
    frame = pd.DataFrame(
        {"rul": np.asarray(rul_true, float), "err": np.asarray(pred, float) - np.asarray(rul_true, float)}
    )
    frame["bin"] = pd.cut(frame["rul"], bins=bins, right=False)
    grouped = frame.groupby("bin", observed=True)["err"]
    table = pd.DataFrame(
        {
            "n": grouped.size(),
            "mae": grouped.apply(lambda e: float(np.abs(e).mean())),
            "rmse": grouped.apply(lambda e: float(np.sqrt((e**2).mean()))),
            "bias": grouped.mean(),
        }
    )
    table.index = table.index.astype(str)
    return table.reset_index().rename(columns={"bin": "true_rul_bin"})


def per_engine_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-engine error summary from a frame with ``engine_id, rul, pred`` columns."""
    err = frame["pred"] - frame["rul"]
    grouped = frame.assign(err=err).groupby("engine_id")
    return pd.DataFrame(
        {
            "n": grouped.size(),
            "mae": grouped["err"].apply(lambda e: float(np.abs(e).mean())),
            "rmse": grouped["err"].apply(lambda e: float(np.sqrt((e**2).mean()))),
            "bias": grouped["err"].mean(),
            "max_true_rul": grouped["rul"].max(),
        }
    ).reset_index()
