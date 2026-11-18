"""A predictor bundle couples the fitted FeaturePipeline with an estimator.

Bundles are persisted with joblib (pickle). **Only load bundles you created yourself**;
unpickling untrusted files can execute arbitrary code.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aeromind.features.pipeline import FeaturePipeline

BUNDLE_FORMAT_VERSION = 1


@dataclass
class RULPredictor:
    """Pipeline + estimator with a single ``predict(frame)`` entry point.

    ``input_kind`` is ``"features"`` (scaled pipeline output) or ``"cycle"`` (raw cycle
    column, for the age-based baseline).
    """

    name: str
    pipeline: FeaturePipeline
    estimator: Any
    input_kind: str = "features"
    target_cap: int | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def inputs(self, df: pd.DataFrame) -> np.ndarray:
        """Model inputs for trajectory rows ``df`` (sorted by engine, cycle)."""
        if self.input_kind == "cycle":
            return df[["cycle"]].to_numpy(dtype=float)
        return self.pipeline.transform(df)

    def predict_inputs(self, X: np.ndarray) -> np.ndarray:
        """Predict from precomputed inputs; clipped to ``[0, cap]``."""
        pred = np.asarray(self.estimator.predict(X), dtype=float)
        upper = float(self.target_cap) if self.target_cap is not None else None
        return np.clip(pred, 0.0, upper)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict RUL for every row of ``df`` using only that row's engine history."""
        return self.predict_inputs(self.inputs(df))


def save_bundle(predictor: Any, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    """Persist a predictor (and metadata) with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"format_version": BUNDLE_FORMAT_VERSION, "predictor": predictor, "metadata": metadata or {}},
        path,
        compress=3,
    )
    return path


def load_bundle(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load a bundle written by :func:`save_bundle` (trusted files only)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    return _check_payload(joblib.load(path), str(path))


def load_bundle_bytes(data: bytes, origin: str = "<bytes>") -> tuple[Any, dict[str, Any]]:
    """Load a bundle from bytes already read (and hash-verified) by the caller (trusted only)."""
    return _check_payload(joblib.load(io.BytesIO(data)), origin)


def _check_payload(payload: Any, origin: str) -> tuple[Any, dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError(f"{origin}: not an AeroMind bundle (format {BUNDLE_FORMAT_VERSION})")
    predictor = payload["predictor"]
    if not all(hasattr(predictor, a) for a in ("inputs", "predict_inputs", "predict")):
        raise ValueError(f"{origin}: unexpected predictor type {type(predictor).__name__}")
    return predictor, payload.get("metadata", {})
