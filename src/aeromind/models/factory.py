"""Construct estimators from configuration and expand hyperparameter grids."""

from __future__ import annotations

import itertools
from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from aeromind.models.baselines import MeanResidualLifeRegressor, MedianRULRegressor

#: Models whose input is the raw ``cycle`` column rather than the feature matrix.
CYCLE_INPUT_MODELS = frozenset({"mean_residual_life"})


def expand_grid(params: dict[str, Any], grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of ``grid`` merged over fixed ``params`` (grid wins on clashes)."""
    if not grid:
        return [dict(params)]
    keys = sorted(grid)
    return [{**params, **dict(zip(keys, combo, strict=True))} for combo in itertools.product(*(grid[k] for k in keys))]


def build_estimator(name: str, params: dict[str, Any], seed: int = 42, n_jobs: int = 4) -> Any:
    """Return an unfitted estimator.

    Seeds and thread counts are injected here so configs cannot forget them.
    HistGradientBoosting early stopping is forced off: its internal validation split is
    row-level and would mix rows of the same engine across train/holdout.
    """
    p = dict(params)
    if name == "median":
        return MedianRULRegressor(**p)
    if name == "mean_residual_life":
        return MeanResidualLifeRegressor(**p)
    if name == "ridge":
        return Ridge(**p)
    if name == "random_forest":
        p.setdefault("n_jobs", n_jobs)
        return RandomForestRegressor(random_state=seed, **p)
    if name == "hist_gradient_boosting":
        if p.get("early_stopping") not in (None, False):
            raise ValueError("early_stopping must be False (row-level internal split leaks across engines)")
        p["early_stopping"] = False
        return HistGradientBoostingRegressor(random_state=seed, **p)
    if name == "xgboost":
        from xgboost import XGBRegressor

        p.setdefault("n_jobs", n_jobs)
        p.setdefault("tree_method", "hist")
        return XGBRegressor(random_state=seed, **p)
    raise ValueError(f"Unknown model {name!r}")
