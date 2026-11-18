"""D. Research Methodology: splits and leakage prevention, features, models, evaluation, reproducibility.

Descriptive text summarises docs/METHODOLOGY.md, docs/STAGE2_METHODOLOGY.md and
docs/EVALUATION_PROTOCOL.md. Counts, feature dimensions and hashes are read from the
committed manifests and registry at render time.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from aeromind.dashboard import data as D
from aeromind.dashboard import state as S
from aeromind.dashboard import ui
from aeromind.dashboard.state import Ctx

PARTITION_COLOR = {"train": "#3987e5", "validation": "#d95926", "calibration": "#199e70"}  # first three categorical slots

LEAKAGE_RULES = [
    ("Engine-level splits", "Each engine belongs to exactly one of train, validation or calibration. Engine IDs are only meaningful within one subset and are never joined across subsets."),
    ("Test labels read once", "Official test labels are read only by the ledgered `aeromind test` command, never by training, tuning, feature selection, calibration, EDA, the registry build, the API or this dashboard."),
    ("Train-only fitting", "Constant-sensor removal, operating-condition clustering and scaling are fitted on training engines only and saved inside each model bundle."),
    ("Causal features", "Features at cycle t use cycles ≤ t of the same engine only: trailing windows, no centred windows, no backward fill, no cross-engine rolling."),
    ("No identifiers as inputs", "`engine_id`, `rul` and target columns are never model inputs."),
    ("Selection on validation", "Hyperparameters and model choice use validation engines (Stage 1) or engine-level CV folds (Stage 2). Calibration engines are used only to fit conformal intervals."),
]

MODELS = pd.DataFrame(
    [
        {"Family": "Ridge, XGBoost, HistGradientBoosting (reference)", "Input": "Engineered causal features: informative sensors, rolling statistics over 10 and 30 cycles, cycle", "Configuration": "Stage-1 validation-selected hyperparameters"},
        {"Family": "TCN", "Input": "Window of 30 normalised raw sensors + cycle per step", "Configuration": "3 residual causal blocks, 32 channels, kernel 3, dilations 1/2/4"},
        {"Family": "GRU", "Input": "Same causal windows (packed sequences)", "Configuration": "1 layer, 64 hidden units"},
        {"Family": "Transformer", "Input": "Same causal windows (padding masked)", "Configuration": "2 pre-norm encoder layers, d_model 32, 4 heads, feed-forward 64"},
    ]
)

EVIDENCE_TABLE = pd.DataFrame(
    [
        {"Evidence": "Cross-validation", "Engines": "85 FD001 development engines, 5 engine-level folds of 17", "Used for": "Model comparison, ablations"},
        {"Evidence": "Canonical validation", "Engines": "15 fixed validation engines (FD001); one split per subset for FD002–FD004", "Used for": "Stage-1 selection; single-split benchmarks"},
        {"Evidence": "Calibration", "Engines": "15 fixed FD001 calibration engines", "Used for": "Conformal thresholds only (plus a role-swap check)"},
        {"Evidence": "Official NASA test", "Engines": "100 FD001 test engines", "Used for": "One final evaluation of the Stage-1 selected model (ledgered)"},
    ]
)


def _splits(ctx: Ctx) -> None:
    st.subheader("Data splitting and leakage prevention")
    ui.badge("truth")
    sp = D.split_summary(ctx.paths)
    if sp.empty:
        ui.unavailable("The split manifests")
    else:
        long = sp.melt(id_vars=["subset"], value_vars=["train", "validation", "calibration"], var_name="partition", value_name="engines")
        dom = list(PARTITION_COLOR)
        chart = alt.Chart(long).mark_bar(height=22, stroke=ui.SURFACE, strokeWidth=2).encode(
            y=alt.Y("subset:N", title=None),
            x=alt.X("engines:Q", stack="zero", title="Training-file engines per partition"),
            color=alt.Color("partition:N", scale=alt.Scale(domain=dom, range=[PARTITION_COLOR[p] for p in dom]), title=None, sort=dom),
            order=alt.Order("partition_order:Q"),
            tooltip=["subset", "partition", "engines"],
        ).transform_calculate(partition_order="indexof(['train','validation','calibration'], datum.partition)")
        st.altair_chart(chart.properties(height=330, title="Seeded engine-level 70 / 15 / 15 splits of each training file"), width="stretch")
        st.dataframe(sp, hide_index=True, width="stretch")
        ui.source([r for r in sp.manifest], "Official test engines are a separate NASA file and are not part of these splits")
    cols = st.columns(3)
    for i, (title, body) in enumerate(LEAKAGE_RULES):
        with cols[i % 3]:
            ui.card(title, body)
            st.write("")
    st.caption("Leakage tests in the test suite check truncation invariance (a feature at cycle t is unchanged by later cycles) and engine-order invariance.")


def _targets(ctx: Ctx) -> None:
    st.subheader("Targets and metrics")
    st.markdown(
        "- **Training target:** piecewise-capped RUL `min(RUL, 125)`. The cap was fixed a priori from literature convention, not tuned.\n"
        "- **Snapshots (selection):** 10 seeded cut cycles per validation engine, one prediction each, so every engine counts equally; the snapshots are identical for every model in a fold.\n"
        "- **All windows:** every validation cycle, reported row-weighted and engine-weighted. **Near end of life:** cycles with true RUL ≤ 30.\n"
        "- **Truth:** every metric is reported against capped truth (`_capped`) and uncapped true RUL (`_true`).\n"
        "- **PHM08 score** with d = prediction − true RUL (late predictions are penalised more):"
    )
    st.latex(r"s(d) = \begin{cases} e^{-d/13} - 1 & d < 0 \\ e^{d/10} - 1 & d \ge 0 \end{cases}")
    st.caption("Reported as a mean per snapshot for validation and as a sum over engines for the official test, following the challenge convention.")


def _features(ctx: Ctx) -> None:
    st.subheader("Feature engineering")
    st.markdown(
        "- Raw values of the informative sensors; constant sensors are removed using training engines only.\n"
        "- `cycle` (current age).\n"
        "- For trailing windows of 10 and 30 cycles: mean, population standard deviation, least-squares slope, minimum and maximum. "
        "Windows use `min_periods=1`, so short histories use all available cycles, identically for training and test engines.\n"
        "- Operating settings are excluded on single-condition subsets (FD001, FD003). On FD002 and FD004, readings are normalised per "
        "operating condition (k-means on the settings, fitted on training engines).\n"
        "- Standard scaling is fitted on training-engine rows only. Deep models read causal windows of normalised raw sensors instead."
    )
    if ctx.registry:
        rows = []
        for m in ctx.registry["models"]:
            if m["name"] != "xgboost":
                continue
            inp = m.get("input") or {}
            rows.append({"Subset": m["dataset"], "Sensors used": len(inp.get("used_columns") or []), "Per-condition normalisation": inp.get("condition_normalization"), "Rolling windows": ", ".join(str(w) for w in inp.get("rolling_windows") or []), "Engineered features": inp.get("n_features")})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            ui.source(ctx.show(ctx.paths.registry), "Feature pipeline of each subset's classical models, as saved in the registry")


def _models_eval(ctx: Ctx) -> None:
    st.subheader("Model families")
    ui.table(MODELS)
    st.markdown(
        "Deep models are trained with AdamW, Huber loss on targets scaled by 1/125, gradient clipping and early stopping on an **inner** "
        "engine holdout of each fold's training engines, so validation engines are never seen in training. Deep architectures were fixed "
        "before results were seen and were **not tuned**; classical hyperparameters come from Stage-1 selection on the canonical validation "
        "engines, a mildly optimistic bias on folds that contain them. All predictions are clipped to `[0, 125]`."
    )
    st.subheader("Evaluation strategy")
    ui.table(EVIDENCE_TABLE)
    st.markdown(
        "- CV results are mean ± SD over folds plus paired per-fold differences against the HistGradientBoosting reference.\n"
        "- Folds share training engines, so the SD understates real uncertainty; no significance tests are claimed, and no model is called better when a difference is within fold variability.\n"
        "- Ablations use 3 folds and one seed; robustness and the FD002–FD004 benchmarks use one fold."
    )


def _reproducibility(ctx: Ctx) -> None:
    st.subheader("Reproducibility")
    reg = ctx.registry
    if reg:
        k = st.columns(3)
        k[0].metric("Registry schema", reg.get("schema_version", "—"))
        k[1].metric("Registry built (UTC)", str(reg.get("built_utc", "—"))[:10])
        k[2].metric("Test labels read during build", "No" if reg.get("official_test_labels_read") is False else "Yes")
        pk = reg.get("packages") or {}
        if pk:
            st.markdown("**Package versions recorded at registry build**")
            st.dataframe(pd.DataFrame([{"Package": k2, "Version": v} for k2, v in pk.items()]), hide_index=True, width="stretch")
        st.caption(
            "Run metadata records commit hashes from the development repository (for example "
            f"`{reg.get('git_commit', '—')}`). The public repository history starts at the v0.3.0 release, so those hashes are "
            "provenance labels rather than resolvable public commits."
        )
    prov_path = ctx.paths.report("data_manifests", "provenance")
    prov = S.json_file(prov_path)
    if prov and isinstance(prov.get("files"), dict):
        st.markdown("**NASA file provenance**: SHA-256 of every raw file every reported result used (the files are not redistributed)")
        st.dataframe(pd.DataFrame([{"File": f, "Bytes": v.get("bytes"), "SHA-256": v.get("sha256")} for f, v in prov["files"].items()]), hide_index=True, width="stretch")
        ui.source(ctx.show(prov_path), "`uv run aeromind validate --dataset all` re-hashes local copies and fails on any mismatch")
    st.markdown("**Reproduce locally** (CPU only; NASA files are downloaded separately; see the README):")
    st.code(
        "uv sync --locked --all-extras\n"
        "uv run aeromind validate --dataset all   # verify your NASA files against the committed hashes\n"
        "uv run pytest                            # real-data tests skip without the NASA files\n"
        "uv run aeromind s3-smoke                 # synthetic end-to-end check, no NASA data needed",
        language="bash",
    )
    st.markdown(
        f"Further reading: [Evaluation protocol]({ui.doc_url('docs/EVALUATION_PROTOCOL.md')}) · "
        f"[Stage-1 methodology]({ui.doc_url('docs/METHODOLOGY.md')}) · "
        f"[Stage-2 methodology]({ui.doc_url('docs/STAGE2_METHODOLOGY.md')}) · "
        f"[Reproducibility]({ui.doc_url('docs/REPRODUCIBILITY.md')}) · "
        f"[References]({ui.doc_url('docs/REFERENCES.md')})"
    )


def render(ctx: Ctx) -> None:
    ui.hero("How the numbers are produced", "Research Methodology", "Leakage-safe data handling, causal features, a fixed evaluation protocol and provenance for every reported result.")
    nav = st.tabs(["Splits & leakage", "Targets & metrics", "Features", "Models & evaluation", "Reproducibility"])
    with nav[0]:
        _splits(ctx)
    with nav[1]:
        _targets(ctx)
    with nav[2]:
        _features(ctx)
    with nav[3]:
        _models_eval(ctx)
    with nav[4]:
        _reproducibility(ctx)
