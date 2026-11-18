"""Helpers shared by the research-mode pages (need local model bundles and NASA files)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from aeromind.dashboard import state as S
from aeromind.dashboard.state import Ctx


def service(ctx: Ctx) -> Any:
    """Model service or ``None`` (research mode only)."""
    return S.service_for(ctx)


def available_models(ctx: Ctx, dataset: str) -> list[str]:
    """Registry ids for ``dataset`` whose bundles are present and hash-verified locally."""
    svc = service(ctx)
    if svc is None:
        return []
    return [m for m in svc.model_ids if svc.entry(m)["dataset"] == dataset and svc.availability(m)[0]]


def model_picker(ctx: Ctx, dataset: str, key: str, need_intervals: bool = False) -> str | None:
    """Select box over locally available models; an info message if there are none."""
    svc = service(ctx)
    ids = available_models(ctx, dataset)
    if need_intervals and svc is not None:
        ids = [i for i in ids if svc.entry(i)["intervals"] is not None]
    if not ids:
        st.info(f"No locally available model bundles for {dataset}. See *Get model artifacts* in the README.")
        return None
    default = (ctx.registry or {}).get("default_model")
    idx = ids.index(default) if default in ids else 0
    return st.selectbox("Model (registry id)", ids, index=idx, key=key, format_func=lambda i: f"{i}  ·  {svc.entry(i)['family']}")


def partitioned(ctx: Ctx, dataset: str):
    """Training-file trajectories with partitions, or ``None`` if the NASA file is absent."""
    return S.get_partitioned(str(ctx.paths.data_dir), dataset, str(ctx.paths.split_manifest(dataset)))


def missing_data(ctx: Ctx, dataset: str) -> None:
    """Message for an absent NASA training file (no local path revealed beyond the project)."""
    st.warning(f"NASA file `train_{dataset}.txt` not found in `{ctx.show(ctx.paths.data_dir)}`. See docs/DATA.md.")

