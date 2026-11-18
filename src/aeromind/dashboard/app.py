"""AeroMind research dashboard.

Run locally: ``uv run aeromind dashboard`` (research mode) or
``streamlit run streamlit_app/streamlit_app.py`` (public mode, as deployed).

Modes (``AEROMIND_DASHBOARD_MODE``):

- ``public``: only the committed reports and registry metadata. Never loads a model bundle,
  never reads NASA files, never accepts uploads.
- ``research`` (default here): adds the local pages that run registered, hash-verified
  bundles on validation engines or uploads (needs model files and NASA files).

Evidence labels used throughout: cross-validation, canonical validation, calibration,
official test (ledgered), dataset ground truth, registry metadata, live model output,
hypothetical illustration, planned capability. Engine-level predictions (research mode) are
shown only for validation engines and uploads; the official NASA test files are never read.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable

import streamlit as st

from aeromind import __version__
from aeromind.dashboard import state as S
from aeromind.dashboard import ui
from aeromind.dashboard.state import Ctx

PUBLIC_SECTIONS: dict[str, str] = {
    "Research Overview": "overview",
    "Model Benchmarking": "benchmarking",
    "Robustness & Reliability": "robustness",
    "Research Methodology": "methodology",
    "Model Registry": "model_registry",
    "Prediction Demo": "prediction_demo",
}
logger = logging.getLogger(__name__)

RESEARCH_SECTIONS: dict[str, str] = {
    "Engine Health Intelligence": "engine",
    "Sensor Intelligence": "sensors",
    "Maintenance Simulation": "maintenance",
}


def sections_for(mode: str) -> dict[str, str]:
    """Section title → view module for a dashboard mode."""
    return PUBLIC_SECTIONS | (RESEARCH_SECTIONS if mode == "research" else {})


def _view(module: str) -> Callable[[Ctx], None]:
    return importlib.import_module(f"aeromind.dashboard.views.{module}").render


def _sidebar(ctx: Ctx, sections: dict[str, str]) -> str:
    with st.sidebar:
        st.markdown("## ✈️ AeroMind")
        st.caption("Remaining-useful-life research on NASA C-MAPSS")
        section = st.radio(
            "Section",
            list(sections),
            key="section",
            label_visibility="collapsed",
            format_func=lambda s: s if s in PUBLIC_SECTIONS else f"{s} · local",
        )
        if not ctx.public:
            st.caption("Sections marked *local* need model files and NASA data on this machine.")
        st.divider()
        st.markdown(f"[GitHub repository]({ui.REPO_URL}) · [Release v{__version__}]({ui.RELEASE_URL})")
        st.markdown(f"[Limitations]({ui.doc_url('docs/LIMITATIONS.md')}) · [Report an issue]({ui.ISSUES_URL})")
        st.caption(
            "Research demonstration on **simulated** data. Not certified for operational aviation maintenance. "
            "Independent project; not endorsed by NASA."
        )
        built = f" · registry built {str(ctx.registry.get('built_utc', ''))[:10]}" if ctx.registry else ""
        st.caption(f"AeroMind v{__version__} · {'public' if ctx.public else 'research'} mode{built}")
    return section


def main() -> None:
    """Render one run of the dashboard."""
    st.set_page_config(
        page_title="AeroMind · RUL research",
        page_icon="✈️",
        layout="wide",
        menu_items={"Get help": ui.REPO_URL, "Report a bug": ui.ISSUES_URL, "About": f"AeroMind v{__version__}: RUL research on NASA C-MAPSS simulation data. {ui.REPO_URL}"},
    )
    ui.inject_css()
    ctx = S.build_context()
    sections = sections_for(ctx.mode)
    if "section" in st.session_state and st.session_state["section"] not in sections:
        del st.session_state["section"]
    section = _sidebar(ctx, sections)
    if ctx.registry is None and section != "Research Overview":
        st.warning(
            f"The model registry `{ctx.show(ctx.paths.registry)}` was not found or is not a valid registry. "
            "Report-based sections need it; build it locally with `uv run aeromind s3-registry`."
        )
    if not ctx.public:
        _view(sections[section])(ctx)  # research mode: let errors surface for the developer
        return
    try:
        _view(sections[section])(ctx)
    except Exception:  # public safety net; Streamlit's stop/rerun signals are BaseExceptions
        logger.exception("Public section %r failed to render", section)
        st.error("This section could not be displayed from the committed reports. Please try another section.")


if __name__ == "__main__":
    main()
