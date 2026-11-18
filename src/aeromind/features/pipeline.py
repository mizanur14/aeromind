"""Fitted, leakage-safe feature pipeline shared by training and inference.

``fit`` learns, from training-engine rows only:

1. which candidate sensors/settings are non-constant (optional constant removal),
2. optional per-operating-condition normalisation statistics,
3. the feature scaler.

``transform`` applies exactly the same steps to any trajectories (validation, calibration,
test, or a single engine at inference). Rolling features are causal and per-engine, so
``transform`` has no fitted state other than the three items above.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from aeromind.data.cmapss import SENSOR_COLUMNS, SETTING_COLUMNS
from aeromind.features.conditions import ConditionNormalizer
from aeromind.features.rolling import check_trajectory_order, trailing_features

if TYPE_CHECKING:
    from aeromind.config import FeatureConfig

logger = logging.getLogger(__name__)

#: Columns that must never become model inputs.
FORBIDDEN_FEATURES = frozenset({"engine_id", "rul", "target", "rul_capped", "rul_true"})


class FeaturePipeline:
    """Constant removal → optional condition normalisation → causal rolling features → scaler."""

    def __init__(self, config: FeatureConfig, seed: int = 42) -> None:
        self.config = config
        self.seed = seed
        self.candidate_columns_: list[str] = []
        self.kept_columns_: list[str] = []
        self.dropped_constant_: list[str] = []
        self.feature_names_: list[str] = []
        self.fitted_engine_ids_: list[int] = []
        self.normalizer_: ConditionNormalizer | None = None
        self.scaler_: StandardScaler | None = None
        self.fitted_ = False

    # ------------------------------------------------------------------ helpers
    def _candidates(self) -> list[str]:
        sensors = self.config.sensors
        if sensors == "all":
            cols = list(SENSOR_COLUMNS)
        else:
            unknown = [s for s in sensors if s not in SENSOR_COLUMNS]
            if unknown:
                raise ValueError(f"Unknown sensors {unknown}")
            cols = list(dict.fromkeys(sensors))
        if self.config.include_settings:
            cols = list(SETTING_COLUMNS) + cols
        return cols

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        check_trajectory_order(df)
        missing = [c for c in self.kept_columns_ + ["engine_id", "cycle"] if c not in df.columns]
        if missing:
            raise KeyError(f"Input is missing columns {missing}")
        data = df
        if self.normalizer_ is not None:
            data = self.normalizer_.transform(df)
        parts = [data[self.kept_columns_].astype(float)]
        if self.config.include_cycle:
            parts.insert(0, df[["cycle"]].astype(float))
        parts.append(
            trailing_features(
                data,
                self.kept_columns_,
                [int(w) for w in self.config.windows],
                self.config.stats,
                self.config.include_diff,
            )
        )
        feats = pd.concat(parts, axis=1)
        leaked = FORBIDDEN_FEATURES & set(feats.columns)
        if leaked:  # pragma: no cover - defensive
            raise RuntimeError(f"Forbidden columns in features: {sorted(leaked)}")
        return feats

    # ------------------------------------------------------------------ API
    def fit(self, train_df: pd.DataFrame) -> FeaturePipeline:
        """Fit on training-engine rows only."""
        if train_df.empty:
            raise ValueError("Cannot fit FeaturePipeline on an empty frame")
        self.candidate_columns_ = self._candidates()
        missing = [c for c in self.candidate_columns_ if c not in train_df.columns]
        if missing:
            raise KeyError(f"Training frame is missing columns {missing}")

        if self.config.drop_constant:
            ranges = train_df[self.candidate_columns_].max() - train_df[self.candidate_columns_].min()
            self.dropped_constant_ = [c for c in self.candidate_columns_ if ranges[c] <= self.config.constant_tol]
        else:
            self.dropped_constant_ = []
        self.kept_columns_ = [c for c in self.candidate_columns_ if c not in self.dropped_constant_]
        if not self.kept_columns_ and not self.config.include_cycle:
            raise ValueError("No usable feature columns after constant removal")
        logger.info(
            "FeaturePipeline: %d/%d columns kept; dropped constant %s",
            len(self.kept_columns_),
            len(self.candidate_columns_),
            self.dropped_constant_,
        )

        if self.config.condition_normalization:
            sensor_cols = [c for c in self.kept_columns_ if c in SENSOR_COLUMNS]
            self.normalizer_ = ConditionNormalizer(sensor_cols, self.config.n_conditions, self.seed).fit(train_df)

        feats = self._build(train_df)
        self.feature_names_ = list(feats.columns)
        values = feats.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite training features")
        if self.config.scaler == "standard":
            self.scaler_ = StandardScaler().fit(values)
        self.fitted_engine_ids_ = sorted(int(i) for i in train_df["engine_id"].unique())
        self.fitted_ = True
        return self

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return scaled features as a DataFrame aligned to ``df.index``."""
        if not self.fitted_:
            raise RuntimeError("FeaturePipeline is not fitted")
        feats = self._build(df)
        if list(feats.columns) != self.feature_names_:
            raise RuntimeError("Feature columns differ from those seen at fit time")
        values = feats.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite features")
        if self.scaler_ is not None:
            values = self.scaler_.transform(values)
        return pd.DataFrame(values, index=df.index, columns=self.feature_names_)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return the scaled feature matrix for ``df``."""
        return self.transform_frame(df).to_numpy()

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        """Fit on ``train_df`` and transform it."""
        return self.fit(train_df).transform(train_df)

    def describe(self) -> dict[str, object]:
        """JSON-serialisable summary of the fitted state."""
        return {
            "candidate_columns": self.candidate_columns_,
            "kept_columns": self.kept_columns_,
            "dropped_constant": self.dropped_constant_,
            "n_features": len(self.feature_names_),
            "feature_names": self.feature_names_,
            "fitted_engine_ids": self.fitted_engine_ids_,
            "condition_normalization": self.normalizer_ is not None,
            "scaler": self.config.scaler,
        }
