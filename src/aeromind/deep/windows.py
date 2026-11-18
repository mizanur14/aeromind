"""Causal sliding windows over per-engine sequences.

For row *t* of an engine, the window holds that engine's rows ``max(0, t−L+1) … t``.
Windows are **right-padded**: valid steps first (oldest → newest), zeros after, plus a
``lengths`` vector. The newest observation of every window is at index ``length − 1``.
Engines never mix, and no future row is ever included (tested).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class WindowInputs:
    """Batch of right-padded windows ``x`` (N, L, C) with valid ``lengths`` (N,)."""

    x: np.ndarray
    lengths: np.ndarray

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, idx) -> WindowInputs:
        if isinstance(idx, (int, np.integer)):
            idx = slice(int(idx), int(idx) + 1)
        return WindowInputs(self.x[idx], self.lengths[idx])

    @property
    def nbytes(self) -> int:
        """Memory held by the window tensor."""
        return int(self.x.nbytes + self.lengths.nbytes)


def build_windows(values: np.ndarray, engine_ids: np.ndarray, window: int) -> WindowInputs:
    """Build one causal window per row.

    Args:
        values: per-row channels (N, C), rows grouped by engine and sorted by cycle.
        engine_ids: engine ID per row (N,), contiguous blocks.
        window: maximum window length ``L`` (≥ 1).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    values = np.asarray(values, dtype=np.float32)
    engine_ids = np.asarray(engine_ids)
    n, c = values.shape
    if len(engine_ids) != n:
        raise ValueError("values and engine_ids differ in length")
    x = np.zeros((n, window, c), dtype=np.float32)
    lengths = np.empty(n, dtype=np.int64)
    change = np.flatnonzero(np.r_[True, engine_ids[1:] != engine_ids[:-1], True])
    if len(np.unique(engine_ids[change[:-1]])) != len(change) - 1:
        raise ValueError("Rows of each engine must be contiguous")
    for s, e in zip(change[:-1], change[1:], strict=True):
        block = values[s:e]
        t = e - s
        k = min(t, window - 1)  # partial windows
        for i in range(k):
            x[s + i, : i + 1] = block[: i + 1]
        if t >= window:
            x[s + window - 1 : e] = sliding_window_view(block, (window, c))[:, 0]
        lengths[s:e] = np.minimum(np.arange(1, t + 1), window)
    return WindowInputs(x, lengths)
