"""Stage-3 smoke test on synthetic data: registry → calibrators → API → dashboard.

Trains toy classical + GRU models on synthetic trajectories (seconds), builds a registry with
conformal calibrators, exercises every API endpoint (if the ``api`` extra is installed) and
renders every dashboard section headlessly (if the ``dashboard`` extra is installed).
No NASA files are needed or read.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from aeromind.data.cmapss import load_train
from aeromind.data.splits import load_split
from aeromind.research.comparison import run_comparison
from aeromind.research.smoke import prepare_smoke_data, smoke_stage2_config
from aeromind.serving.registry import CalibrationSpec, RegistryConfig, SourceSpec, build_registry, write_registry

DASHBOARD_SECTIONS = [  # research mode: every public section plus the local model/data pages
    "Research Overview",
    "Model Benchmarking",
    "Robustness & Reliability",
    "Research Methodology",
    "Model Registry",
    "Prediction Demo",
    "Engine Health Intelligence",
    "Sensor Intelligence",
    "Maintenance Simulation",
]


@contextmanager
def _env(**values: str):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def build_smoke_environment(root: Path) -> dict[str, Any]:
    """Synthetic data, toy models, a registry and a results index under ``root``."""
    cfg = smoke_stage2_config(root, archs=("gru",))
    prepare_smoke_data(cfg)
    run_dir = run_comparison(cfg, root / "stage2" / "s3_smoke_comparison", isolate=False)
    reg_cfg = RegistryConfig(
        name="s3_smoke",
        output=str(root / "registry" / "model_registry.json"),
        sources=[SourceSpec("stage2_comparison", str(run_dir), str(run_dir / "report"))],
        calibration=CalibrationSpec(models=["fd001-s2-xgboost", "fd001-s2-gru"], levels=[0.8], seed=0),
        default_model="fd001-s2-xgboost",
        default_model_rationale="smoke test",
        ledger=str(root / "no_ledger.json"),
    )
    registry = build_registry(reg_cfg, root=".")
    write_registry(registry, reg_cfg.output)
    index = {
        "registry": reg_cfg.output,
        "test_set_ledger": str(root / "no_ledger.json"),
        "comparison": {"FD001": str(run_dir / "report")},
        "data_manifests": {"splits": {"FD001": str(cfg.split_manifest_path)}},
    }
    index_path = root / "results_index.yaml"
    index_path.write_text(yaml.safe_dump(index))
    return {
        "registry": reg_cfg.output,
        "index": str(index_path),
        "data_dir": cfg.paths.raw_dir,
        "manifest": str(cfg.split_manifest_path),
        "run_dir": str(run_dir),
        "model_ids": [m["id"] for m in registry["models"]],
    }


def check_api(env: dict[str, Any]) -> dict[str, Any]:
    """Call every endpoint against the smoke registry; raises AssertionError on failure."""
    from fastapi.testclient import TestClient

    from aeromind.api.app import create_app
    from aeromind.serving.service import ModelService

    pool = load_train(env["data_dir"], "FD001")
    eid = load_split(env["manifest"]).ids("validation")[0]
    cycles = pool[pool.engine_id == eid].drop(columns=["engine_id", "rul"]).to_dict(orient="records")
    app = create_app(lambda: ModelService(env["registry"], "."))
    out: dict[str, Any] = {}
    with TestClient(app) as c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["models_available"] == len(env["model_ids"]), h
        listing = c.get("/models").json()["models"]
        assert {m["id"] for m in listing} == set(env["model_ids"])
        for mid in env["model_ids"]:
            meta = c.get(f"/models/{mid}").json()
            levels = [0.8] if meta["intervals"] else []
            r = c.post(f"/models/{mid}/predict", json={"engine": {"engine_id": int(eid), "cycles": cycles}, "interval_levels": levels})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["provenance"]["bundle_sha256"] == meta["bundle"]["sha256"]
            out[mid] = body["predicted_rul"]
        r = c.post("/models/fd001-s2-xgboost/predict/batch", json={"engines": [{"engine_id": 1, "cycles": cycles[:20]}, {"engine_id": 2, "cycles": cycles}], "interval_levels": [0.8]})
        assert r.status_code == 200 and r.json()["n_engines"] == 2, r.text
        assert c.post("/models/missing/predict", json={"engine": {"cycles": cycles}}).status_code == 404
        assert c.post("/models/fd001-s2-ridge/predict", json={"engine": {"cycles": cycles}, "interval_levels": [0.8]}).status_code == 422
    return out


def check_dashboard(env: dict[str, Any]) -> dict[str, list[str]]:
    """Render every dashboard section headlessly; returns exceptions per section (empty = OK)."""
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
    problems: dict[str, list[str]] = {}
    with _env(AEROMIND_REGISTRY=env["registry"], AEROMIND_RESULTS_INDEX=env["index"], AEROMIND_DATA_DIR=env["data_dir"], AEROMIND_ROOT=".", AEROMIND_DASHBOARD_MODE="research"):
        for section in DASHBOARD_SECTIONS:
            at = AppTest.from_file(app_path, default_timeout=120)
            at.run()
            at.radio(key="section").set_value(section).run()
            problems[section] = [str(e.value) for e in at.exception]
    return problems


def run_s3_smoke(workdir: str | None = None) -> dict[str, Any]:
    """Full Stage-3 smoke; returns what was checked."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(workdir or tmp)
        env = build_smoke_environment(root)
        result: dict[str, Any] = {"models": env["model_ids"], "api": "skipped (install --extra api)", "dashboard": "skipped (install --extra dashboard)"}
        if importlib.util.find_spec("fastapi") is not None:
            preds = check_api(env)
            result["api"] = f"ok ({len(preds)} models predicted)"
        if importlib.util.find_spec("streamlit") is not None:
            problems = check_dashboard(env)
            bad = {k: v for k, v in problems.items() if v}
            if bad:
                raise RuntimeError(f"Dashboard sections raised: {bad}")
            result["dashboard"] = f"ok ({len(problems)} sections rendered)"
        return result
