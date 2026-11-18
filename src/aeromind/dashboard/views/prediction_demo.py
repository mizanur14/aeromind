"""F. Prediction Demo: the planned interactive inference, explained but not active.

No model is loaded and no prediction, interval or example output is produced. The controls
shown are disabled previews of the planned interface.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from aeromind.dashboard import ui
from aeromind.dashboard.state import Ctx


def _schema(ctx: Ctx) -> pd.DataFrame | None:
    from aeromind.data.cmapss import SENSOR_INFO

    reg = ctx.registry or {}
    m = next((x for x in reg.get("models", []) if x["id"] == reg.get("default_model")), None) or next(iter(reg.get("models", [])), None)
    cols = ((m or {}).get("input") or {}).get("required_columns")
    if not cols:
        return None
    rows = []
    for c in cols:
        if c == "cycle":
            rows.append({"Column": c, "Meaning": "Operating cycle number (positive integer, strictly increasing)", "Unit": "cycle"})
        elif c.startswith("setting_"):
            rows.append({"Column": c, "Meaning": "Operating setting", "Unit": "—"})
        else:
            sym, desc, unit = SENSOR_INFO.get(c, ("", "", ""))
            rows.append({"Column": c, "Meaning": f"{sym}: {desc}" if sym else desc, "Unit": unit})
    return pd.DataFrame(rows)


def render(ctx: Ctx) -> None:
    ui.hero("Planned capability", "Prediction Demo", "Interactive RUL inference on an uploaded engine history. Explained here; not active on this website.")
    ui.badge("planned")
    st.warning(
        "**Interactive inference is disabled on this public website.** No model is loaded, and no prediction, interval or "
        "example output is generated. Everything below describes the planned feature."
    )

    st.subheader("How it is planned to work")
    st.markdown(
        "1. A visitor uploads the raw cycles of **one** engine (C-MAPSS columns and units, a small size limit).\n"
        "2. The input is validated: required columns, finite values, unique positive cycle numbers, a cycle limit.\n"
        "3. A registered **classical** model (no PyTorch in the cloud) applies its own saved feature pipeline and predicts capped "
        "RUL `min(RUL, 125)` at every observed cycle.\n"
        "4. Where a calibrator exists, an engine-level split-conformal interval is shown with its exact, marginal guarantee.\n"
        "5. Every result carries provenance: model id and version, the model file's SHA-256, the source run and the registry build.\n\n"
        "Uploads would have **no ground truth**: the website could not tell how accurate a given prediction is, only how the model "
        "performed on held-out validation engines (see *Model Benchmarking*)."
    )

    st.subheader("Safeguards that must hold before it is enabled")
    c = st.columns(2)
    with c[0]:
        ui.card("Verified model files only", "Model files would come from a GitHub Release asset and be installed only if their path and SHA-256 match the committed registry. Model files are never accepted from visitors.")
        st.write("")
        ui.card("No serialized uploads", "Visitors can upload numeric CSV data only. Nothing a visitor supplies is executed or deserialized.")
    with c[1]:
        ui.card("Bounded resources", "Upload size and cycle count are capped, and models are small classical regressors, to stay within a free CPU host's memory.")
        st.write("")
        ui.card("No fabricated output", "If a model file is missing or fails verification, the page shows an error. It never substitutes a fallback, sample or synthetic prediction.")

    st.subheader("Interface preview (disabled)")
    p = st.columns([2, 1, 1])
    p[0].file_uploader("One engine's cycles (CSV)", type=["csv"], disabled=True, key="demo_upload", help="Disabled on the public website.")
    p[1].selectbox("Model", ["not available"], disabled=True, key="demo_model")
    p[2].selectbox("Interval", ["not available"], disabled=True, key="demo_level")
    st.button("Predict remaining useful life", disabled=True, key="demo_go", help="Disabled on the public website.")

    st.subheader("Expected input schema")
    schema = _schema(ctx)
    if schema is None:
        ui.unavailable("The input schema (from the model registry)")
    else:
        st.dataframe(schema, hide_index=True, width="stretch")
        ui.source(ctx.show(ctx.paths.registry), "Required raw columns from the registry; sensor symbols and units follow Saxena et al. (2008), Table 2")

    st.subheader("Run inference locally today")
    st.markdown(
        "Inference already works on your own machine with the NASA files and verified model files. Model weights are not in Git: "
        "regenerate them on a CPU or import a maintainer-published archive whose hashes match the committed registry."
    )
    st.code(
        "uv sync --locked --all-extras\n"
        "uv run aeromind s3-import aeromind-models.tar.gz   # only if a release archive is available; verifies every SHA-256\n"
        "uv run aeromind dashboard                          # research mode: adds Engine Health, Sensor Intelligence, Maintenance\n"
        "uv run aeromind serve                              # local REST API on 127.0.0.1 (no authentication; keep it local)",
        language="bash",
    )
    st.markdown(f"Details: [README · Get model artifacts]({ui.REPO_URL}#2-get-model-artifacts-not-in-git) · [Limitations]({ui.doc_url('docs/LIMITATIONS.md')})")
    if not ctx.public:
        st.caption("Research mode: the *Engine Health Intelligence* page runs registered models locally when their files are present.")
