"""Inference service shared by the API and the dashboard.

- Models are loaded lazily from the registry and **only after** the bundle's SHA-256 matches
  the registry (bundles are pickles; a mismatching file is never unpickled).
- Inputs are raw C-MAPSS cycles for one engine; the bundle's own fitted pipeline (constant
  removal, condition normalisation, causal rolling features / windows, scaler) is applied.
- Every prediction carries provenance (model id, version, source run, bundle hash, registry
  build time) and, where a calibrator exists, a conformal interval with its validated
  coverage. Nothing is synthesised: a model that cannot be loaded raises an error.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aeromind.models.bundle import load_bundle_bytes
from aeromind.serving.registry import REQUIRED_INPUT_COLUMNS, load_registry
from aeromind.utils import sha256_file

logger = logging.getLogger(__name__)

MAX_CYCLES_PER_ENGINE = 5000


class ServiceError(Exception):
    """Base error with an HTTP-like status and a stable code."""

    status = 500
    code = "internal_error"

    def __init__(self, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.message, self.details = message, details


class ModelNotFound(ServiceError):
    status, code = 404, "model_not_found"


class ModelUnavailable(ServiceError):
    status, code = 503, "model_unavailable"


class InvalidInput(ServiceError):
    status, code = 422, "invalid_input"


class IntervalUnavailable(ServiceError):
    status, code = 422, "interval_unavailable"


@dataclass
class _Loaded:
    predictor: Any
    load_seconds: float


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


class ModelService:
    """Registry-backed, thread-safe, lazily loading predictor cache."""

    def __init__(self, registry_path: str | Path, root: str | Path = ".") -> None:
        self.registry_path = Path(registry_path)
        self.root = Path(root)
        self.registry = load_registry(self.registry_path)
        self._entries = {m["id"]: m for m in self.registry["models"]}
        self._cache: dict[str, _Loaded] = {}
        self._status: dict[str, tuple[bool, str]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ catalogue
    @property
    def model_ids(self) -> list[str]:
        """All registry ids, in registry order."""
        return list(self._entries)

    def entry(self, mid: str) -> dict[str, Any]:
        """Registry entry or :class:`ModelNotFound`."""
        if mid not in self._entries:
            raise ModelNotFound(f"Unknown model id {mid!r}", {"available": self.model_ids})
        return self._entries[mid]

    def bundle_path(self, mid: str) -> Path:
        """Absolute bundle path for ``mid``."""
        return self.root / self.entry(mid)["bundle"]["path"]

    def availability(self, mid: str) -> tuple[bool, str]:
        """(available, reason). Hashes each bundle once per process."""
        if mid in self._status:
            return self._status[mid]
        e = self.entry(mid)
        path = self.bundle_path(mid)
        if not path.is_file():
            status = (False, f"bundle not found at {e['bundle']['path']} (artifacts are not distributed with the repository)")
        elif "torch" in e.get("requires", []) and not _torch_available():
            status = (False, "requires PyTorch: install with `uv sync --extra deep`")
        elif sha256_file(path) != e["bundle"]["sha256"]:
            status = (False, "bundle SHA-256 does not match the registry; refusing to load")
        else:
            status = (True, "ok")
        self._status[mid] = status
        return status

    def summary(self, mid: str) -> dict[str, Any]:
        """Compact public description of a model, with availability."""
        e = self.entry(mid)
        ok, reason = self.availability(mid)
        cv = (e["metrics"].get("cross_validation") or {}).get("values") or {}
        canon = (e["metrics"].get("canonical_validation") or {}).get("values") or {}
        return {
            "id": mid,
            "version": e["version"],
            "dataset": e["dataset"],
            "stage": e["stage"],
            "family": e["family"],
            "kind": e["kind"],
            "target_cap": e["target_cap"],
            "available": ok,
            "availability_reason": reason,
            "intervals_available": e["intervals"] is not None,
            "interval_levels": sorted(e["intervals"]["thresholds"]) if e["intervals"] else [],
            "cv_snapshot_rmse_capped": cv.get("snapshot_rmse_capped"),
            "canonical_snapshot_rmse_capped": canon.get("snapshot_rmse_capped"),
            "official_test_evaluated": e["metrics"].get("official_test") is not None,
            "source_run": e["source_run"],
        }

    # ------------------------------------------------------------------ loading
    def load(self, mid: str) -> Any:
        """Verified, cached predictor for ``mid``."""
        with self._lock:
            if mid in self._cache:
                return self._cache[mid].predictor
            ok, reason = self.availability(mid)
            if not ok:
                raise ModelUnavailable(f"Model {mid} is unavailable: {reason}")
            t0 = time.perf_counter()
            # Hash exactly the bytes that are unpickled (no check-then-reopen race).
            data = self.bundle_path(mid).read_bytes()
            if hashlib.sha256(data).hexdigest() != self.entry(mid)["bundle"]["sha256"]:
                self._status[mid] = (False, "bundle SHA-256 does not match the registry; refusing to load")
                raise ModelUnavailable(f"Model {mid} is unavailable: bundle changed on disk and no longer matches the registry")
            predictor, _ = load_bundle_bytes(data, str(self.bundle_path(mid)))
            if hasattr(predictor, "threads"):
                predictor.threads = min(int(predictor.threads), 2)
            self._cache[mid] = _Loaded(predictor, time.perf_counter() - t0)
            logger.info("Loaded %s in %.2fs", mid, self._cache[mid].load_seconds)
            return predictor

    # ------------------------------------------------------------------ inference
    @staticmethod
    def validate_frame(frame: pd.DataFrame, engine_id: Any = "engine") -> pd.DataFrame:
        """Check one engine's raw cycles; return a sorted copy with an integer ``engine_id``."""
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in frame.columns]
        if missing:
            raise InvalidInput(f"Missing required columns: {missing}", {"required": REQUIRED_INPUT_COLUMNS})
        if len(frame) == 0:
            raise InvalidInput("At least one cycle is required")
        if len(frame) > MAX_CYCLES_PER_ENGINE:
            raise InvalidInput(f"At most {MAX_CYCLES_PER_ENGINE} cycles per engine")
        df = frame[REQUIRED_INPUT_COLUMNS].copy()
        try:
            df = df.astype(float)
        except (TypeError, ValueError) as exc:
            raise InvalidInput(f"Non-numeric values: {exc}") from exc
        values = df.to_numpy()
        if not np.isfinite(values).all():
            bad = sorted({c for c in df.columns if not np.isfinite(df[c].to_numpy()).all()})
            raise InvalidInput(
                "Missing or non-finite readings are not accepted; impute causally (e.g. forward fill) first",
                {"columns": bad},
            )
        cyc = df["cycle"].to_numpy()
        if (cyc < 1).any() or (cyc != np.round(cyc)).any():
            raise InvalidInput("cycle must be a positive integer")
        df = df.sort_values("cycle", kind="mergesort").reset_index(drop=True)
        if df["cycle"].duplicated().any():
            raise InvalidInput("Duplicate cycle numbers")
        df["cycle"] = df["cycle"].astype(int)
        df.insert(0, "engine_id", 1)
        df.attrs["engine_label"] = engine_id
        return df

    def provenance(self, mid: str) -> dict[str, Any]:
        """Where a prediction comes from."""
        e = self.entry(mid)
        return {
            "model_id": mid,
            "model_version": e["version"],
            "bundle_sha256": e["bundle"]["sha256"],
            "source_run": e["source_run"],
            "report_dir": e["report_dir"],
            "trained_on": f"{e['dataset']} training-file engines ({e['training']['n_engines']} engines)",
            "registry_built_utc": self.registry["built_utc"],
            "registry_git_commit": self.registry.get("git_commit"),
            "data_kind": "NASA C-MAPSS simulation",
        }

    def top_level(self, mid: str) -> float | None:
        """Highest calibrated level with a finite threshold, or ``None``."""
        cal = self.entry(mid)["intervals"] or {}
        return max((float(k) for k, v in cal.get("thresholds", {}).items() if v is not None), default=None)

    def interval(self, mid: str, pred: np.ndarray, level: float) -> dict[str, Any]:
        """Conformal interval for predictions ``pred`` at ``level``."""
        cal = self.entry(mid)["intervals"]
        if cal is None:
            raise IntervalUnavailable(f"No calibrated intervals for {mid}")
        key = f"{level:.2f}"
        if key not in cal["thresholds"]:
            raise IntervalUnavailable(f"Level {level} not calibrated for {mid}", {"levels": sorted(cal["thresholds"])})
        q = cal["thresholds"][key]
        if q is None:
            raise IntervalUnavailable(f"Level {level} is unattainable with {cal['n_calibration_scores']} calibration engines")
        validated = ((cal.get("validated_coverage") or {}).get("levels") or {}).get(key)
        return {
            "level": float(level),
            "lower": np.maximum(pred - q, 0.0),
            "upper": pred + q,
            "half_width": float(q),
            "method": cal["method"],
            "n_calibration_engines": cal["n_calibration_scores"],
            "validated_coverage": validated,
            "guarantee": cal["guarantee"],
        }

    def predict_engine(
        self,
        mid: str,
        frame: pd.DataFrame,
        engine_id: Any = "engine",
        interval_levels: list[float] | None = None,
        include_trajectory: bool = False,
    ) -> dict[str, Any]:
        """Predict RUL at the last observed cycle (and optionally at every cycle)."""
        e = self.entry(mid)
        predictor = self.load(mid)
        df = self.validate_frame(frame, engine_id)
        t0 = time.perf_counter()
        try:
            pred = np.asarray(predictor.predict(df), dtype=float)
        except ValueError as exc:  # e.g. readings so extreme that features overflow
            raise InvalidInput(f"Inputs could not be processed by {mid}: {exc}; check units and value ranges") from exc
        if not np.isfinite(pred).all():
            raise InvalidInput(f"{mid} produced non-finite predictions for these inputs; check units and value ranges")
        ms = (time.perf_counter() - t0) * 1000
        intervals = [self.interval(mid, pred, lv) for lv in (interval_levels or [])]
        cap = e["target_cap"]
        last = float(pred[-1])
        warnings = []
        if cap is not None and last >= cap - 1e-9:
            warnings.append(f"Prediction at the cap ({cap}): the engine looks healthy; true RUL may be much larger.")
        if len(df) < (e["input"].get("sequence_window") or max(e["input"].get("rolling_windows") or [1])):
            warnings.append("Fewer cycles than the model's history window; partial-window features were used.")
        if e["dataset"] in ("FD001", "FD003") and df[["setting_1", "setting_2"]].abs().max().max() > 0.1:
            warnings.append(f"Operating settings look multi-regime; {mid} was trained on single-condition {e['dataset']}.")
        out: dict[str, Any] = {
            "model_id": mid,
            "engine_id": engine_id,
            "n_cycles": len(df),
            "last_cycle": int(df["cycle"].iloc[-1]),
            "predicted_rul": last,
            "target": f"min(RUL, {cap})" if cap is not None else "RUL (uncapped)",
            "intervals": [
                {k: (float(v[-1]) if isinstance(v, np.ndarray) else v) for k, v in iv.items()} for iv in intervals
            ],
            "inference_ms": ms,
            "warnings": warnings,
            "provenance": self.provenance(mid),
        }
        if include_trajectory:
            traj = {"cycle": df["cycle"].tolist(), "predicted_rul": pred.tolist()}
            for iv in intervals:
                traj[f"lower_{iv['level']:.2f}"] = iv["lower"].tolist()
                traj[f"upper_{iv['level']:.2f}"] = iv["upper"].tolist()
            out["trajectory"] = traj
        return out
