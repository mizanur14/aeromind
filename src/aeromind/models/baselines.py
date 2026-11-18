"""Reference baselines.

- :class:`MedianRULRegressor` predicts the training-target median for every row.
- :class:`MeanResidualLifeRegressor` is an *age-based* baseline: it predicts the
  expected (capped) remaining life given only that the engine has survived to cycle
  ``t``, estimated from the training engines' targets at cycle ``t``:
  ``MRL(t) = mean_i { y_i(t) : engine i observed at cycle t }``.
  It uses no sensor data. A *persistence* baseline is not meaningful for RUL because no
  past RUL values are observable at inference, so it is intentionally not provided.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, 0.5 * cum[-1])])


class MedianRULRegressor(RegressorMixin, BaseEstimator):
    """Constant predictor equal to the (optionally weighted) median training target."""

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        """Store the median of ``y``."""
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            raise ValueError("Empty target")
        if sample_weight is None:
            self.value_ = float(np.median(y))
        else:
            self.value_ = _weighted_median(y, np.asarray(sample_weight, dtype=float))
        self.n_features_in_ = np.asarray(X).shape[1] if np.asarray(X).ndim == 2 else 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the stored constant for every row."""
        if not hasattr(self, "value_"):
            raise RuntimeError("MedianRULRegressor is not fitted")
        return np.full(len(X), self.value_, dtype=float)


class MeanResidualLifeRegressor(RegressorMixin, BaseEstimator):
    """Age-only baseline. ``X`` must be a single column of raw (unscaled) cycle numbers.

    Args:
        min_engines: if fewer training engines survive to cycle ``t``, the estimate for
            the last cycle with enough survivors is decremented by the extra cycles
            (floored at 0), avoiding noisy estimates from one or two long-lived engines.
    """

    def __init__(self, min_engines: int = 5) -> None:
        self.min_engines = min_engines

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        """Build the cycle → mean target lookup table."""
        cycles = np.asarray(X, dtype=float).reshape(len(y), -1)
        if cycles.shape[1] != 1:
            raise ValueError("MeanResidualLifeRegressor expects exactly one column: cycle")
        c = cycles[:, 0].astype(np.int64)
        y = np.asarray(y, dtype=float)
        max_cycle = int(c.max())
        sums = np.bincount(c, weights=y, minlength=max_cycle + 1)
        counts = np.bincount(c, minlength=max_cycle + 1)
        table = np.zeros(max_cycle + 1)
        valid = counts >= self.min_engines
        if not valid[1:].any():
            raise ValueError("Too few engines to estimate residual life")
        table[valid] = sums[valid] / counts[valid]
        last_valid = int(np.flatnonzero(valid).max())
        tail = np.arange(last_valid + 1, max_cycle + 1)
        table[tail] = np.maximum(table[last_valid] - (tail - last_valid), 0.0)
        # Fill any interior gaps (cannot happen for contiguous trajectories) forward.
        for i in range(1, max_cycle + 1):
            if not valid[i] and i <= last_valid:
                table[i] = table[i - 1]
        self.table_ = table
        self.max_cycle_ = max_cycle
        self.n_features_in_ = 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Look up the expected remaining life at each cycle (0 beyond the longest life)."""
        if not hasattr(self, "table_"):
            raise RuntimeError("MeanResidualLifeRegressor is not fitted")
        c = np.asarray(X, dtype=float).reshape(len(X), -1)[:, 0].astype(np.int64)
        c = np.clip(c, 0, None)
        out = np.zeros(len(c))
        inside = c <= self.max_cycle_
        out[inside] = self.table_[c[inside]]
        return out
