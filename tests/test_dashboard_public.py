"""Public dashboard (Streamlit Community Cloud) tests.

These run without NASA files and without model bundles, in the full research environment and
in the lightweight deployment environment built only from ``streamlit_app/requirements.txt``
(``PYTHONPATH=src``). They check that the public sections render, never load the model stack
or read data/model files, reveal no local paths, and show numbers traceable to committed
reports.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aeromind.dashboard import data as D

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "streamlit_app" / "streamlit_app.py"
REQUIREMENTS = ROOT / "streamlit_app" / "requirements.txt"
PUBLIC_SECTIONS = ["Research Overview", "Model Benchmarking", "Robustness & Reliability", "Research Methodology", "Model Registry", "Prediction Demo"]
FORBIDDEN_MODULES = ("torch", "xgboost", "joblib", "sklearn", "scipy")
FORBIDDEN_PACKAGES = ("torch", "xgboost", "scikit-learn", "joblib", "scipy", "matplotlib", "seaborn", "fastapi")
ENV_KEYS = ("AEROMIND_REGISTRY", "AEROMIND_RESULTS_INDEX", "AEROMIND_DASHBOARD_MODE", "AEROMIND_PROJECT_ROOT", "AEROMIND_ROOT", "AEROMIND_DATA_DIR")


@pytest.fixture
def no_assets(tmp_path, monkeypatch):
    """No NASA files and no model bundles, even on a machine that has them; env restored after."""
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AEROMIND_DATA_DIR", str(tmp_path / "no_nasa_data"))
    monkeypatch.setenv("AEROMIND_ROOT", str(tmp_path / "no_model_bundles"))
    return tmp_path


def _render(section: str | None = None):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ENTRY), default_timeout=120)
    at.run()
    if section is not None:
        at.radio(key="section").set_value(section).run()
    return at


def _all_text(at) -> str:
    parts: list[str] = []
    for kind in ("title", "header", "subheader", "markdown", "caption", "info", "warning", "success", "code", "latex", "json"):
        parts += [str(e.value) for e in getattr(at, kind)]
    for df in at.dataframe:
        v = df.value
        parts.append(v.to_string() if hasattr(v, "to_string") else str(v))
    return "\n".join(parts)


# ----------------------------------------------------------------------------- rendering
def test_public_navigation_lists_only_public_sections(no_assets):
    pytest.importorskip("streamlit")
    at = _render()
    assert list(at.radio(key="section").options) == PUBLIC_SECTIONS
    assert os.environ.get("AEROMIND_DASHBOARD_MODE") == "public"  # set by the entrypoint (restored by the fixture)


@pytest.mark.parametrize("section", PUBLIC_SECTIONS)
def test_public_section_renders_without_data_or_models(no_assets, section):
    pytest.importorskip("streamlit")
    at = _render(section)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    text = _all_text(at)
    assert str(ROOT) not in text, "absolute project path leaked into the page"
    for marker in ("/home/", "/run/media", "/Users/", "noreply.github.com", "@gmail"):
        assert marker not in text, marker
    assert "Research demonstration" in text or "simulat" in text.lower()


def _write_bad_inputs(root: Path) -> dict[str, dict[str, str]]:
    corrupt = root / "corrupt"
    corrupt.mkdir()
    for name in ("summary.csv", "fold_metrics.csv", "paired_vs_reference.csv", "intervals_main.csv", "coverage_by_rul_bin.csv", "ablation.csv", "robustness.csv", "validation_model_comparison.csv"):
        (corrupt / name).write_text("")
    (corrupt / "benchmark.json").write_text('{"results": 5')
    (root / "bad_registry.json").write_text('{"models": [')
    (root / "bad_ledger.json").write_text('{"entries": [')
    (root / "list_index.yaml").write_text("- a\n- b\n")
    keys = ("uq", "ablation", "robustness", "api_benchmark")
    (root / "missing.yaml").write_text("registry: reports/registry/model_registry.json\n" + "".join(f"{k}: {root}/nowhere/{k}\n" for k in keys) + f"comparison: {{FD001: {root}/nowhere/c}}\nstage1: {{FD001: {root}/nowhere/s}}\n")
    (root / "corrupt.yaml").write_text(f"registry: reports/registry/model_registry.json\ntest_set_ledger: {root}/bad_ledger.json\n" + "".join(f"{k}: {corrupt}\n" for k in keys) + f"comparison: {{FD001: {corrupt}}}\nstage1: {{FD001: {corrupt}}}\n")
    return {
        "malformed registry": {"AEROMIND_REGISTRY": str(root / "bad_registry.json")},
        "index not a mapping": {"AEROMIND_RESULTS_INDEX": str(root / "list_index.yaml")},
        "report directories missing": {"AEROMIND_RESULTS_INDEX": str(root / "missing.yaml")},
        "corrupt reports and ledger": {"AEROMIND_RESULTS_INDEX": str(root / "corrupt.yaml")},
    }


@pytest.mark.parametrize("case", ["malformed registry", "index not a mapping", "report directories missing", "corrupt reports and ledger"])
def test_public_sections_degrade_gracefully_on_missing_or_malformed_files(no_assets, monkeypatch, case):
    pytest.importorskip("streamlit")
    for key, value in _write_bad_inputs(no_assets)[case].items():
        monkeypatch.setenv(key, value)
    for section in PUBLIC_SECTIONS:
        at = _render(section)
        assert not at.exception, (section, [e.value for e in at.exception])
        assert not at.error, (section, [e.value for e in at.error])


def test_public_safety_net_hides_internal_errors(no_assets, monkeypatch):
    pytest.importorskip("streamlit")
    import aeromind.dashboard.views.methodology as view

    def boom(ctx):
        raise RuntimeError("/secret/internal/path exploded")

    monkeypatch.setattr(view, "render", boom)
    at = _render("Research Methodology")
    assert not at.exception
    assert [e.value for e in at.error] == ["This section could not be displayed from the committed reports. Please try another section."]
    assert "/secret/internal/path" not in _all_text(at)


def test_public_mode_never_imports_model_stack_or_reads_data(no_assets):
    """Fresh interpreter + audit hook: no heavy imports, no NASA/test/bundle file opened."""
    pytest.importorskip("streamlit")
    script = f"""
import json, sys
opened = []
def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        opened.append(args[0].decode() if isinstance(args[0], bytes) else args[0])
sys.addaudithook(hook)
from streamlit.testing.v1 import AppTest
for s in {PUBLIC_SECTIONS!r}:
    at = AppTest.from_file({str(ENTRY)!r}, default_timeout=120)
    at.run()
    at.radio(key="section").set_value(s).run()
    assert not at.exception, [e.value for e in at.exception]
print("RESULT" + json.dumps({{"modules": [m for m in {FORBIDDEN_MODULES!r} if m in sys.modules], "opened": opened}}))
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, env=dict(os.environ), cwd=str(no_assets))
    assert out.returncode == 0, out.stderr[-3000:]
    res = json.loads(out.stdout.split("RESULT", 1)[1])
    assert res["modules"] == [], f"public mode imported {res['modules']}"
    bad = [p for p in res["opened"] if re.search(r"(train|test|RUL)_FD00\d\.txt|\.joblib$|\.pkl$|\.pt$|/artifacts/|/data/raw/", p)]
    assert bad == [], bad


# ----------------------------------------------------------------------------- traceability
def _registry() -> dict:
    reg = D.load_registry_json(ROOT / "reports" / "registry" / "model_registry.json")
    assert reg is not None
    return reg


def test_displayed_metrics_equal_committed_reports():
    """Every CV / canonical value the dashboard charts equals the run's committed summary.csv."""
    reg = _registry()
    checked = 0
    for m in reg["models"]:
        if m["stage"] != "Stage 2":
            continue
        summary = pd.read_csv(ROOT / m["report_dir"] / "summary.csv").set_index("model").loc[m["name"]]
        can = D.metric_frame(reg, m["dataset"], "canonical_validation")
        can = can[can.model_id == m["id"]].set_index("metric")
        for metric in ("snapshot_rmse_capped", "snapshot_mae_capped", "snapshot_rmse_true"):
            assert can.loc[metric, "value"] == pytest.approx(summary[f"{metric}_canonical"]), (m["id"], metric)
            checked += 1
        cv = D.metric_frame(reg, m["dataset"], "cross_validation")
        cv = cv[cv.model_id == m["id"]].set_index("metric")
        if not cv.empty:
            assert cv.loc["snapshot_rmse_capped", "value"] == pytest.approx(summary["snapshot_rmse_capped_cv_mean"])
            assert cv.loc["snapshot_rmse_capped", "sd"] == pytest.approx(summary["snapshot_rmse_capped_cv_sd"])
            checked += 2
    assert checked > 50


def test_official_test_values_come_from_the_ledger():
    reg = _registry()
    ledger = json.loads((ROOT / "reports" / "test_set_ledger.json").read_text())
    table = D.official_test_table(reg, ledger)
    assert len(table) >= 1 and table.ledger_entry_matches.all()
    runs = {e["run"]: e for e in ledger["entries"]}
    for _, r in table.iterrows():
        src = next(m for m in reg["models"] if m["id"] == r.model_id)["source_run"]
        assert r.rmse_true == pytest.approx(runs[src]["selected_rmse_true"])
        assert r.phm08_sum_true == pytest.approx(runs[src]["selected_phm08_sum_true"])
    # A Stage-2 model must never carry an official test result.
    assert not any(m["metrics"].get("official_test") for m in reg["models"] if m["stage"] == "Stage 2")


def test_split_summary_matches_manifests(no_assets):
    sp = D.split_summary(D.Paths.from_env()).set_index("subset")
    for ds in D.DATASETS:
        j = json.loads((ROOT / "data" / "manifests" / f"{ds.lower()}_split_seed42.json").read_text())
        assert sp.loc[ds, "train"] == len(j["train_engine_ids"])
        assert sp.loc[ds, "calibration"] == len(j["calibration_engine_ids"])
        assert sp.loc[ds, "manifest"] == f"data/manifests/{ds.lower()}_split_seed42.json"


# ----------------------------------------------------------------------------- paths and modes
def test_paths_resolve_from_project_root_not_cwd(no_assets, monkeypatch):
    monkeypatch.chdir(no_assets)
    p = D.Paths.from_env()
    assert p.project_root == ROOT
    assert p.registry == ROOT / "reports" / "registry" / "model_registry.json" and p.registry.is_file()
    for key in ("uq", "ablation", "robustness", "test_set_ledger"):
        assert p.report(key) is not None and p.report(key).exists(), key
    assert p.display(p.registry) == "reports/registry/model_registry.json"
    assert p.display(Path("/somewhere/else/secret.json")) == "<external>/secret.json"


def test_unknown_mode_falls_back_to_public(monkeypatch):
    monkeypatch.setenv("AEROMIND_DASHBOARD_MODE", "Debug")
    assert D.dashboard_mode() == "public"
    monkeypatch.delenv("AEROMIND_DASHBOARD_MODE")
    assert D.dashboard_mode() == "research"


def test_public_context_refuses_model_service(no_assets):
    pytest.importorskip("streamlit")
    from aeromind.dashboard import state as S

    ctx = S.Ctx(paths=D.Paths.from_env(), mode="public", registry=_registry(), ledger=None)
    with pytest.raises(RuntimeError):
        S.service_for(ctx)


def test_metric_frame_is_empty_not_fabricated_without_cv():
    reg = _registry()
    for ds in ("FD002", "FD003", "FD004"):
        assert D.metric_frame(reg, ds, "cross_validation").empty
        assert not D.metric_frame(reg, ds, "canonical_validation").empty
    assert np.isnan(D.metric_frame(reg, "FD002", "canonical_validation").sd).all()


# ----------------------------------------------------------------------------- deployment files
def _lock_versions() -> dict[str, str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    return {p["name"].lower(): p["version"] for p in lock["package"] if "version" in p}


def test_requirements_are_pinned_to_uv_lock_and_exclude_research_stack():
    lock = _lock_versions()
    pins = {}
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, ver = line.partition("==")
            assert ver, f"unpinned requirement: {line}"
            pins[name.lower().replace("_", "-")] = ver
    for name, ver in pins.items():
        assert lock.get(name) == ver, (name, ver, lock.get(name))
    for pkg in FORBIDDEN_PACKAGES:
        assert pkg not in pins, pkg
    direct = [ln.strip().lower() for ln in (ROOT / "streamlit_app" / "requirements.in").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    assert set(direct) <= set(pins)


def test_streamlit_config_is_public_safe():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert cfg["server"]["maxUploadSize"] <= 2
    assert cfg["server"]["enableXsrfProtection"] is True
    assert cfg["client"]["toolbarMode"] == "viewer"
    assert cfg["client"]["showErrorDetails"] == "none"
    assert cfg["browser"]["gatherUsageStats"] is False
