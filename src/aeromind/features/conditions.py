"""Operating-condition identification and per-condition sensor normalisation.

FD002/FD004 run under six operating conditions that shift sensor levels far more than
degradation does. Conditions are identified with k-means on the three settings, fitted on
training engines only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from aeromind.data.cmapss import SETTING_COLUMNS


class OperatingConditionModel:
    """k-means clustering of operational settings (``n_conditions == 1`` → single regime)."""

    def __init__(self, n_conditions: int = 1, seed: int = 42) -> None:
        if n_conditions < 1:
            raise ValueError("n_conditions must be >= 1")
        self.n_conditions = n_conditions
        self.seed = seed
        self.kmeans_: KMeans | None = None
        self.fitted_ = False

    def fit(self, df: pd.DataFrame) -> OperatingConditionModel:
        """Fit on training rows."""
        if self.n_conditions > 1:
            self.kmeans_ = KMeans(n_clusters=self.n_conditions, n_init=10, random_state=self.seed)
            self.kmeans_.fit(df[SETTING_COLUMNS].to_numpy(dtype=float))
        self.fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Condition index per row."""
        if not self.fitted_:
            raise RuntimeError("OperatingConditionModel is not fitted")
        if self.kmeans_ is None:
            return np.zeros(len(df), dtype=np.int64)
        return self.kmeans_.predict(df[SETTING_COLUMNS].to_numpy(dtype=float)).astype(np.int64)


class ConditionNormalizer:
    """Standardise sensors within each operating condition (statistics from training rows)."""

    def __init__(self, columns: list[str], n_conditions: int = 1, seed: int = 42) -> None:
        self.columns = list(columns)
        self.conditions = OperatingConditionModel(n_conditions, seed)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> ConditionNormalizer:
        """Fit clustering and per-condition mean/std on training rows."""
        self.conditions.fit(df)
        cond = self.conditions.predict(df)
        k = self.conditions.n_conditions
        values = df[self.columns].to_numpy(dtype=float)
        self.mean_ = np.zeros((k, len(self.columns)))
        self.std_ = np.ones((k, len(self.columns)))
        for c in range(k):
            block = values[cond == c]
            if len(block) == 0:
                raise ValueError(f"Operating condition {c} has no training rows")
            self.mean_[c] = block.mean(axis=0)
            std = block.std(axis=0)
            self.std_[c] = np.where(std > 1e-12, std, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with normalised sensor columns."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("ConditionNormalizer is not fitted")
        cond = self.conditions.predict(df)
        out = df.copy()
        values = df[self.columns].to_numpy(dtype=float)
        out[self.columns] = (values - self.mean_[cond]) / self.std_[cond]
        return out
