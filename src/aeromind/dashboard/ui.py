"""Shared presentation layer: theme, evidence badges, captions, colours and links.

Colour rules:

- Model families use one fixed categorical order (validated for the dark surface with
  adjacent-pair colour-vision-deficiency checks). Colour follows the family, never its rank,
  and models are always listed in this same order so adjacent marks keep validated pairs.
- Evidence badges carry a text label; their colour is a secondary cue only.
"""

from __future__ import annotations

import re
from typing import Any

import altair as alt
import streamlit as st

from aeromind import __version__

REPO_URL = "https://github.com/mizanur14/aeromind"
RELEASE_URL = f"{REPO_URL}/releases/tag/v{__version__}"
ISSUES_URL = f"{REPO_URL}/issues"


def doc_url(relpath: str) -> str:
    """GitHub URL of a repository file on ``main``."""
    return f"{REPO_URL}/blob/main/{relpath}"


# ----------------------------------------------------------------------------- tokens
SURFACE = "#0e1726"
TEXT, TEXT2, MUTED, GRID = "#e6edf7", "#a9b4c6", "#7d8aa1", "#24324a"

# Fixed categorical order (blue, orange, aqua, yellow, magenta, green, violet).
FAMILY_ORDER = ["hgb_reference", "xgboost", "gru", "transformer", "tcn", "ridge", "hist_gradient_boosting"]
FAMILY_COLOR = dict(zip(FAMILY_ORDER, ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9"], strict=True))
FAMILY_LABEL = {
    "hgb_reference": "HistGradientBoosting (reference)",
    "xgboost": "XGBoost",
    "gru": "GRU",
    "transformer": "Transformer",
    "tcn": "TCN",
    "ridge": "Ridge",
    "hist_gradient_boosting": "HistGradientBoosting (Stage 1)",
}
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DIVERGING = ["#e66767", "#383835", "#3987e5"]  # negative · neutral midpoint · positive

EVIDENCE: dict[str, tuple[str, str]] = {
    "cv": ("Cross-validation · engine-level folds", "#3987e5"),
    "canonical": ("Canonical validation · single held-out split", "#9085e9"),
    "calibration": ("Calibration engines · conformal intervals", "#199e70"),
    "test": ("Official NASA test · ledgered, evaluated once", "#c98500"),
    "truth": ("Dataset ground truth · NASA C-MAPSS simulation", "#0ca30c"),
    "registry": ("Registry metadata · no model is loaded", "#7d8aa1"),
    "model": ("Live model output · registered, hash-verified bundle", "#d95926"),
    "hypo": ("Hypothetical illustration · assumed inputs", "#d55181"),
    "planned": ("Planned capability · not active", "#7d8aa1"),
}

_CSS = f"""
<style>
.am-badge {{display:inline-flex;align-items:center;gap:6px;padding:2px 10px;border-radius:999px;
  font-size:0.78rem;font-weight:600;letter-spacing:.01em;border:1px solid {GRID};color:{TEXT};
  margin:0 6px 8px 0;background:rgba(255,255,255,0.02)}}
.am-dot {{width:8px;height:8px;border-radius:50%;display:inline-block}}
.am-hero h1 {{margin-bottom:0.15rem;font-size:2.1rem}}
.am-hero p {{color:{TEXT2};margin-top:0;font-size:1.05rem;max-width:62rem}}
.am-kicker {{color:{TEXT2};text-transform:uppercase;letter-spacing:.08em;font-size:.72rem;font-weight:600}}
.am-note {{color:{TEXT2};font-size:0.86rem}}
.am-card {{border:1px solid {GRID};border-radius:10px;padding:14px 16px;height:100%;background:rgba(255,255,255,0.015)}}
.am-card h4 {{margin:0 0 6px 0;font-size:1rem}}
.am-card p {{color:{TEXT2};font-size:0.9rem;margin:0}}
.am-disclaimer {{border-left:3px solid #c98500;padding:8px 12px;background:rgba(201,133,0,0.07);
  color:{TEXT};font-size:0.88rem;border-radius:0 6px 6px 0;margin:4px 0 14px 0}}
div[data-testid="stMetricValue"] {{font-variant-numeric: tabular-nums}}
</style>"""


def inject_css() -> None:
    """Page-level CSS (static constants only; never user input)."""
    st.markdown(_CSS, unsafe_allow_html=True)


@alt.theme.register("aeromind", enable=True)
def _altair_theme() -> alt.theme.ThemeConfig:
    return alt.theme.ThemeConfig(
        {
            "config": {
                "background": "transparent",
                "padding": {"top": 14, "left": 6, "right": 12, "bottom": 6},
                "view": {"stroke": "transparent"},
                "axis": {"labelColor": TEXT2, "titleColor": TEXT2, "gridColor": GRID, "domainColor": GRID, "tickColor": GRID, "labelFontSize": 11, "titleFontSize": 12, "titleFontWeight": 500, "labelLimit": 260},
                "legend": {"labelColor": TEXT, "titleColor": TEXT2, "orient": "top", "labelFontSize": 11, "labelLimit": 320},
                "title": {"color": TEXT, "fontSize": 13, "anchor": "start", "fontWeight": 600, "subtitleColor": TEXT2},
                "header": {"labelColor": TEXT, "titleColor": TEXT2},
                "axisBand": {"labelOverlap": False},
                "line": {"strokeWidth": 2},
                "point": {"size": 80, "filled": True},
            }
        }
    )


# ----------------------------------------------------------------------------- components
def badge(*kinds: str) -> None:
    """Evidence-type badges (text label first, colour as a secondary cue)."""
    html = "".join(
        f'<span class="am-badge"><span class="am-dot" style="background:{EVIDENCE[k][1]}"></span>{EVIDENCE[k][0]}</span>' for k in kinds
    )
    st.markdown(html, unsafe_allow_html=True)


def hero(kicker: str, title: str, subtitle: str) -> None:
    """Page heading block (static text only)."""
    st.markdown(f'<div class="am-hero"><div class="am-kicker">{kicker}</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def card(title: str, body: str) -> None:
    """Small bordered text card (static text only; `code` spans become <code>)."""
    body = re.sub(r"`([^`]+)`", r"<code>\1</code>", body)
    st.markdown(f'<div class="am-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def disclaimer() -> None:
    """Standard research disclaimer."""
    st.markdown(
        '<div class="am-disclaimer"><b>Research demonstration.</b> NASA C-MAPSS is <b>simulated</b> '
        "turbofan data, not flight data. Nothing here is certified or validated for operational "
        "aviation maintenance or safety decisions. AeroMind is an independent project; NASA has not "
        "reviewed or endorsed it.</div>",
        unsafe_allow_html=True,
    )


def source(display_paths: list[str] | str, note: str = "") -> None:
    """Caption naming the committed file(s) a result was copied from."""
    items = [display_paths] if isinstance(display_paths, str) else display_paths
    text = "Source: " + " · ".join(f"`{p}`" for p in items)
    st.caption(text + (f". {note}" if note else ""))


def unavailable(what: str, hint: str = "") -> None:
    """Graceful message for an optional committed file that is missing."""
    st.info(f"{what} is not available in this deployment." + (f" {hint}" if hint else ""))


def provenance_box(prov: dict[str, Any], title: str = "Provenance") -> None:
    """Expandable provenance for a displayed result."""
    with st.expander(title):
        st.json(prov, expanded=False)


# ----------------------------------------------------------------------------- colour helpers
def family_of(model_id: str) -> str:
    """Registry id → family key (``fd001-s2-xgboost`` → ``xgboost``; Stage-1 HGB → ``hist_gradient_boosting``)."""
    tail = model_id.split("-", 2)[-1]
    return "hist_gradient_boosting" if "-s1-" in model_id and tail == "hgb" else tail


def family_rank(model_id: str) -> int:
    """Position of a model's family in the fixed categorical order."""
    fam = family_of(model_id)
    return FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else len(FAMILY_ORDER)


def order_models(ids: list[str]) -> list[str]:
    """Models sorted by the fixed family order (stable), so colours and adjacency never change."""
    return sorted(dict.fromkeys(ids), key=lambda i: (family_rank(i), i))


def color_scale(ids: list[str]) -> alt.Scale:
    """Colour scale keyed by model id, coloured by family."""
    ordered = order_models(ids)
    return alt.Scale(domain=ordered, range=[FAMILY_COLOR.get(family_of(i), TEXT2) for i in ordered])


def pretty_family(model_id: str) -> str:
    """Human label for a model's family."""
    return FAMILY_LABEL.get(family_of(model_id), family_of(model_id))


def table(df: Any) -> None:
    """Small descriptive table rendered as Markdown so long text wraps (static text only)."""

    def cell(v: Any) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    head = "| " + " | ".join(cell(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    st.markdown("\n".join([head, sep, *body]))
