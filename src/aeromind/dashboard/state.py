"""Streamlit runtime state: resolved paths, registry, cached readers, model service.

The model service (``aeromind.serving.service``, which imports joblib and scikit-learn) is
imported lazily and only by research-mode pages. Public mode never constructs it, so it never
hashes, opens or unpickles a model bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from aeromind.dashboard import data as D


@dataclass(frozen=True)
class Ctx:
    """Everything a page needs to render."""

    paths: D.Paths
    mode: str
    registry: dict[str, Any] | None
    ledger: dict[str, Any] | None

    @property
    def public(self) -> bool:
        return self.mode == "public"

    def show(self, path: Path | str | None) -> str:
        """Project-relative display path."""
        return self.paths.display(path)


def _mtime(path: Path | None) -> float:
    try:
        return path.stat().st_mtime if path is not None else -1.0
    except OSError:
        return -1.0


@st.cache_data(show_spinner=False, max_entries=64)
def _read_json(path: str, mtime: float) -> dict[str, Any] | None:
    data = D.read_json_opt(Path(path))
    return data if isinstance(data, dict) else None


@st.cache_data(show_spinner=False, max_entries=64)
def _read_registry(path: str, mtime: float) -> dict[str, Any] | None:
    return D.load_registry_json(Path(path))


@st.cache_data(show_spinner=False, max_entries=128)
def _read_csv(path: str, mtime: float) -> pd.DataFrame | None:
    p = Path(path)
    return D.read_csv(p.parent, p.name)


def json_file(path: Path | None) -> dict[str, Any] | None:
    """Cached JSON reader (``None`` if absent)."""
    return None if path is None else _read_json(str(path), _mtime(path))


def csv_file(directory: Path | None, name: str) -> pd.DataFrame | None:
    """Cached ``directory/name`` CSV reader (``None`` if absent)."""
    if directory is None:
        return None
    p = directory / name
    return _read_csv(str(p), _mtime(p))


def build_context() -> Ctx:
    """Resolve paths for this run and load the registry and ledger (cached by mtime)."""
    paths = D.Paths.from_env()
    registry = _read_registry(str(paths.registry), _mtime(paths.registry))
    ledger = json_file(paths.report("test_set_ledger"))
    return Ctx(paths=paths, mode=D.dashboard_mode(), registry=registry, ledger=ledger)


# ----------------------------------------------------------------------------- research mode only
@st.cache_resource(show_spinner=False)
def get_service(registry: str, root: str) -> Any:
    """Registry-backed model service, or ``None`` (lazy import of the model stack)."""
    from aeromind.serving.service import ModelService

    try:
        return ModelService(registry, root)
    except (FileNotFoundError, ValueError, KeyError):
        return None


def service_for(ctx: Ctx) -> Any:
    """Model service for research pages; never called in public mode."""
    if ctx.public:
        raise RuntimeError("The model service is disabled in public mode")
    return get_service(str(ctx.paths.registry), str(ctx.paths.root))


def availability(ctx: Ctx) -> dict[str, tuple[bool, str]]:
    """Bundle availability per model (research mode only; hashes each bundle once per process)."""
    svc = service_for(ctx)
    return {m: svc.availability(m) for m in svc.model_ids} if svc else {}


@st.cache_data(show_spinner=False)
def get_partitioned(data_dir: str, dataset: str, manifest: str) -> pd.DataFrame | None:
    return D.load_partitioned(Path(data_dir), dataset, Path(manifest))


@st.cache_data(show_spinner="Running the registered model…", max_entries=64)
def predict_cached(registry: str, root: str, mid: str, frame: pd.DataFrame, levels: tuple[float, ...]) -> dict[str, Any]:
    svc = get_service(registry, root)
    return svc.predict_engine(mid, frame, "engine", list(levels), include_trajectory=True)


@st.cache_data(show_spinner="Predicting validation trajectories…", max_entries=16)
def fleet_predictions(registry: str, root: str, mid: str, data_dir: str, dataset: str, manifest: str, level: float | None) -> pd.DataFrame:
    svc = get_service(registry, root)
    df = get_partitioned(data_dir, dataset, manifest)
    val = df[df.partition == "validation"]
    parts = []
    for eid, g in val.groupby("engine_id"):
        res = svc.predict_engine(mid, g.drop(columns=["engine_id", "rul", "partition"]), int(eid), [level] if level else [], True)
        t = pd.DataFrame(res["trajectory"]).assign(engine_id=int(eid))
        parts.append(t.merge(g[["cycle", "rul"]], on="cycle"))
    out = pd.concat(parts, ignore_index=True).rename(columns={"predicted_rul": "pred"})
    if level:
        out = out.rename(columns={f"lower_{level:.2f}": "lower"})
    return out
