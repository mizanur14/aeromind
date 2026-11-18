"""Causal (trailing) per-engine rolling features.

Every feature at row *(engine e, cycle t)* depends only on rows of engine *e* with
cycle ≤ *t*. Windows are trailing with ``min_periods=1``: for ``t < w`` the window holds
the ``t`` observations available so far (a *partial window*). This is applied identically
to training, validation and test trajectories, so short test trajectories are handled the
same way as early cycles of training engines.

Definitions over the trailing window of ``n`` observations ``x``:

- ``mean``, ``min``, ``max``
- ``std``   population std (``ddof=0``), 0 for a single observation
- ``slope`` ordinary least-squares slope of ``x`` against cycle, 0 for a single observation
- ``diff1`` (optional) ``x_t − x_{t−1}``, 0 at the first cycle
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROLLING_STATS = ("mean", "std", "slope", "min", "max")


def check_trajectory_order(df: pd.DataFrame) -> None:
    """Require rows grouped by engine with strictly increasing cycles within each engine."""
    for col in ("engine_id", "cycle"):
        if col not in df.columns:
            raise KeyError(f"Frame must contain {col!r}")
    engine = df["engine_id"].to_numpy()
    cycle = df["cycle"].to_numpy()
    same = engine[1:] == engine[:-1]
    if np.any(same & (cycle[1:] <= cycle[:-1])):
        raise ValueError("Cycles must be strictly increasing within each engine")
    # Each engine must occupy one contiguous block.
    starts = engine[np.r_[True, ~same]]
    if len(starts) != len(np.unique(starts)):
        raise ValueError("Rows of each engine must be contiguous (sort by engine_id, cycle)")


def feature_name(column: str, stat: str, window: int | None = None) -> str:
    """Canonical feature name, e.g. ``sensor_11__slope_w30`` or ``sensor_11__diff1``."""
    return f"{column}__{stat}" if window is None else f"{column}__{stat}_w{window}"


def _grouped_rolling(frame: pd.DataFrame, keys: pd.Series, window: int):
    return frame.groupby(keys.to_numpy(), sort=False).rolling(window, min_periods=1)


def _undo_group_index(result: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    return result.droplevel(0).reindex(index)


def trailing_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
    stats: list[str] | tuple[str, ...] = ROLLING_STATS,
    include_diff: bool = False,
) -> pd.DataFrame:
    """Compute trailing rolling features per engine.

    Args:
        df: trajectories sorted by ``engine_id, cycle`` (validated).
        columns: measurement columns to summarise.
        windows: trailing window lengths in cycles (observations), each ≥ 2.
        stats: subset of :data:`ROLLING_STATS`.
        include_diff: add first differences.

    Returns:
        Feature frame aligned with ``df.index`` (no ID or target columns).
    """
    check_trajectory_order(df)
    bad = set(stats) - set(ROLLING_STATS)
    if bad:
        raise ValueError(f"Unknown rolling stats {sorted(bad)}")
    if any(w < 2 for w in windows):
        raise ValueError("Windows must be >= 2")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}")

    keys = df["engine_id"]
    x = df[columns].astype(float)
    parts: list[pd.DataFrame] = []

    # Shift values by each engine's first observation (a past value) to reduce
    # cancellation in the slope sums; slope is invariant to this shift.
    x0 = x - x.groupby(keys.to_numpy(), sort=False).transform("first")
    t = (df["cycle"] - df.groupby("engine_id", sort=False)["cycle"].transform("first")).astype(float)

    for w in windows:
        if any(s in stats for s in ("mean", "std", "min", "max")):
            roll = _grouped_rolling(x, keys, w)
            if "mean" in stats:
                parts.append(_undo_group_index(roll.mean(), df.index).add_suffix(f"__mean_w{w}"))
            if "std" in stats:
                std = _undo_group_index(roll.std(ddof=0), df.index).fillna(0.0)
                parts.append(std.clip(lower=0.0).add_suffix(f"__std_w{w}"))
            if "min" in stats:
                parts.append(_undo_group_index(roll.min(), df.index).add_suffix(f"__min_w{w}"))
            if "max" in stats:
                parts.append(_undo_group_index(roll.max(), df.index).add_suffix(f"__max_w{w}"))
        if "slope" in stats:
            parts.append(_rolling_slope(x0, t, keys, w, df.index).add_suffix(f"__slope_w{w}"))

    if include_diff:
        diff = x.groupby(keys.to_numpy(), sort=False).diff().fillna(0.0)
        parts.append(diff.add_suffix("__diff1"))

    if not parts:
        return pd.DataFrame(index=df.index)
    out = pd.concat(parts, axis=1)
    return out


def _rolling_slope(x: pd.DataFrame, t: pd.Series, keys: pd.Series, window: int, index: pd.Index) -> pd.DataFrame:
    """OLS slope of each column of ``x`` on ``t`` over trailing windows, per engine."""
    tx = x.mul(t, axis=0).add_suffix("__tx")
    aux = pd.DataFrame({"__n": 1.0, "__t": t, "__tt": t * t}, index=x.index)
    frame = pd.concat([x, tx, aux], axis=1)
    sums = _undo_group_index(_grouped_rolling(frame, keys, window).sum(), index)
    n = sums["__n"].to_numpy()[:, None]
    st = sums["__t"].to_numpy()[:, None]
    stt = sums["__tt"].to_numpy()[:, None]
    sx = sums[list(x.columns)].to_numpy()
    stx = sums[[f"{c}__tx" for c in x.columns]].to_numpy()
    denom = n * stt - st * st
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(denom > 1e-12, (n * stx - st * sx) / denom, 0.0)
    return pd.DataFrame(slope, index=index, columns=x.columns)
