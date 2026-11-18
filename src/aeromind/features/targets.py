"""RUL target formulation: uncapped true RUL and piecewise-linear capped RUL.

Convention used throughout AeroMind:

- ``rul``            uncapped true RUL (cycles until failure). Never modified.
- ``target``         the training target: ``min(rul, cap)`` if a cap is set, else ``rul``.
- ``rul_capped``     ``min(rul, evaluation.capped_truth_cap)`` for literature-style metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cap_rul(rul: np.ndarray | pd.Series, cap: int | None) -> np.ndarray:
    """Return ``min(rul, cap)`` as float (or ``rul`` unchanged if ``cap`` is ``None``)."""
    values = np.asarray(rul, dtype=float)
    if np.any(values < 0):
        raise ValueError("RUL must be non-negative")
    if cap is None:
        return values.copy()
    if cap <= 0:
        raise ValueError("cap must be positive")
    return np.minimum(values, float(cap))


def make_target(df: pd.DataFrame, cap: int | None) -> np.ndarray:
    """Training target from a frame with an uncapped ``rul`` column."""
    if "rul" not in df.columns:
        raise KeyError("Frame has no 'rul' column")
    return cap_rul(df["rul"].to_numpy(), cap)
