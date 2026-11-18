"""Point-prediction metrics for RUL.

All functions take ``y_true`` and ``y_pred`` in cycles. Optional ``weights`` give
weighted means (used for engine-balanced all-window metrics).
"""

from __future__ import annotations

import numpy as np

PHM08_EARLY = 13.0  # under-prediction (pred < true) time constant
PHM08_LATE = 10.0  # over-prediction (pred > true) time constant


def _arrays(y_true, y_pred, weights=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    if t.shape != p.shape:
        raise ValueError(f"Shape mismatch: {t.shape} vs {p.shape}")
    if t.size == 0:
        raise ValueError("Empty inputs")
    if not (np.isfinite(t).all() and np.isfinite(p).all()):
        raise ValueError("Non-finite values")
    w = np.ones_like(t) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.shape != t.shape or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Invalid weights")
    return t, p, w


def mae(y_true, y_pred, weights=None) -> float:
    """Mean absolute error."""
    t, p, w = _arrays(y_true, y_pred, weights)
    return float(np.average(np.abs(p - t), weights=w))


def rmse(y_true, y_pred, weights=None) -> float:
    """Root mean squared error."""
    t, p, w = _arrays(y_true, y_pred, weights)
    return float(np.sqrt(np.average((p - t) ** 2, weights=w)))


def bias(y_true, y_pred, weights=None) -> float:
    """Mean signed error ``pred − true`` (positive = late / optimistic predictions)."""
    t, p, w = _arrays(y_true, y_pred, weights)
    return float(np.average(p - t, weights=w))


def phm08_terms(y_true, y_pred) -> np.ndarray:
    """Per-sample PHM08 penalty: ``exp(−d/13) − 1`` if ``d < 0`` else ``exp(d/10) − 1``, ``d = pred − true``."""
    t, p, _ = _arrays(y_true, y_pred)
    d = p - t
    return np.where(d < 0, np.exp(-d / PHM08_EARLY) - 1.0, np.exp(d / PHM08_LATE) - 1.0)


def phm08_score(y_true, y_pred) -> float:
    """PHM08 challenge score (sum of asymmetric penalties; lower is better)."""
    return float(phm08_terms(y_true, y_pred).sum())


def regression_metrics(y_true, y_pred, weights=None) -> dict[str, float]:
    """MAE, RMSE, bias, PHM08 sum and mean, and sample count."""
    t, p, w = _arrays(y_true, y_pred, weights)
    terms = phm08_terms(t, p)
    return {
        "n": int(t.size),
        "mae": mae(t, p, w),
        "rmse": rmse(t, p, w),
        "bias": bias(t, p, w),
        "phm08_sum": float(terms.sum()),
        "phm08_mean": float(np.average(terms, weights=w)),
    }
