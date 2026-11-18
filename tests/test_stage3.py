"""Stage 3: registry, inference service, REST API, dashboard and maintenance simulation.

Synthetic tests run everywhere (CI). ``real_data`` tests use the local NASA files and the
registered Stage-1/2 bundles and skip when those are absent.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from aeromind.api.app import create_app  # noqa: E402
from aeromind.data.cmapss import load_train  # noqa: E402
from aeromind.data.splits import load_split  # noqa: E402
from aeromind.features.targets import cap_rul  # noqa: E402
from aeromind.models.bundle import load_bundle  # noqa: E402
from aeromind.research.uq import conformal_quantile, one_per_engine  # noqa: E402
from aeromind.serving import maintenance as M  # noqa: E402
from aeromind.serving.registry import REQUIRED_INPUT_COLUMNS, load_registry  # noqa: E402
from aeromind.serving.service import InvalidInput, ModelService, ModelUnavailable  # noqa: E402
from aeromind.serving.smoke import build_smoke_environment, check_dashboard  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def smoke_env(tmp_path_factory):
    return build_smoke_environment(tmp_path_factory.mktemp("s3"))


@pytest.fixture(scope="module")
def engine(smoke_env):
    pool = load_train(smoke_env["data_dir"], "FD001")
    eid = load_split(smoke_env["manifest"]).ids("validation")[0]
    return eid, pool[pool.engine_id == eid].drop(columns=["engine_id", "rul"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def client(smoke_env):
    with TestClient(create_app(lambda: ModelService(smoke_env["registry"], "."))) as c:
        yield c


def _cycles(frame: pd.DataFrame) -> list[dict]:
    return frame.to_dict(orient="records")


# ----------------------------------------------------------------------------- registry
def test_registry_pins_bundles_and_never_reads_test_labels(smoke_env):
    reg = load_registry(smoke_env["registry"])
    assert reg["official_test_labels_read"] is False
    assert reg["default_model"] in smoke_env["model_ids"]
    svc = ModelService(smoke_env["registry"], ".")
    for m in reg["models"]:
        assert m["version"] == m["bundle"]["sha256"][:12]
        assert svc.availability(m["id"]) == (True, "ok")
        assert m["metrics"]["official_test"] is None
        assert m["metrics"]["cross_validation"]["folds"] == 3
        assert m["input"]["required_columns"] == REQUIRED_INPUT_COLUMNS
        assert any("simulation" in s for s in m["limitations"])


def test_calibrator_reproduces_split_conformal(smoke_env):
    """Thresholds equal the conformal quantile of one seeded snapshot per calibration engine."""
    reg = load_registry(smoke_env["registry"])
    entry = next(m for m in reg["models"] if m["id"] == "fd001-s2-xgboost")
    cal = entry["intervals"]
    pool = load_train(smoke_env["data_dir"], "FD001")
    ids = load_split(smoke_env["manifest"]).ids("calibration")
    assert cal["calibration_engine_ids"] == ids
    df = pool[pool.engine_id.isin(ids)].reset_index(drop=True)
    predictor, _ = load_bundle(entry["bundle"]["path"])
    idx = one_per_engine(df.engine_id.to_numpy(), np.random.default_rng(cal["seed"]))
    scores = np.abs(cap_rul(df.rul, 60)[idx] - predictor.predict(df)[idx])
    assert cal["thresholds"]["0.80"] == pytest.approx(conformal_quantile(scores, 0.2))
    assert cal["n_calibration_scores"] == len(ids)
    # Training engines of the model never overlap calibration engines.
    assert not set(entry["training"]["engine_ids"]) & set(ids)


def test_tampered_bundle_is_not_loaded(smoke_env, tmp_path):
    reg = json.loads(Path(smoke_env["registry"]).read_text())
    reg["models"][0]["bundle"]["sha256"] = "0" * 64
    path = tmp_path / "reg.json"
    path.write_text(json.dumps(reg))
    svc = ModelService(path, ".")
    mid = reg["models"][0]["id"]
    ok, reason = svc.availability(mid)
    assert not ok and "SHA-256" in reason
    with pytest.raises(ModelUnavailable):
        svc.load(mid)


def test_missing_bundle_reports_unavailable(smoke_env, tmp_path):
    reg = json.loads(Path(smoke_env["registry"]).read_text())
    reg["models"][0]["bundle"]["path"] = str(tmp_path / "absent.joblib")
    path = tmp_path / "reg.json"
    path.write_text(json.dumps(reg))
    with TestClient(create_app(lambda: ModelService(path, "."))) as c:
        r = c.post(f"/models/{reg['models'][0]['id']}/predict", json={"engine": {"cycles": [dict.fromkeys(REQUIRED_INPUT_COLUMNS, 1.0)]}})
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"
        assert c.get("/health").json()["models_available"] == len(reg["models"]) - 1


# ----------------------------------------------------------------------------- service
@pytest.mark.parametrize("mid", ["fd001-s2-xgboost", "fd001-s2-gru", "fd001-s2-ridge"])
def test_service_equals_direct_bundle_and_is_causal(smoke_env, engine, mid):
    svc = ModelService(smoke_env["registry"], ".")
    _, frame = engine
    res = svc.predict_engine(mid, frame, include_trajectory=True)
    predictor, _ = load_bundle(svc.bundle_path(mid))
    direct = predictor.predict(frame.assign(engine_id=1)[["engine_id", *frame.columns]])
    np.testing.assert_allclose(res["trajectory"]["predicted_rul"], direct, rtol=0, atol=1e-5)
    # Truncating the history at cycle t reproduces the prediction at t (no look-ahead).
    t = len(frame) // 2
    short = svc.predict_engine(mid, frame.iloc[:t])
    assert short["predicted_rul"] == pytest.approx(res["trajectory"]["predicted_rul"][t - 1], abs=1e-5)
    # Row order of the request does not matter.
    shuffled = svc.predict_engine(mid, frame.sample(frac=1, random_state=0))
    assert shuffled["predicted_rul"] == pytest.approx(res["predicted_rul"], abs=1e-6)


def test_input_validation(smoke_env, engine):
    svc = ModelService(smoke_env["registry"], ".")
    _, frame = engine
    bad = {
        "missing column": frame.drop(columns=["sensor_4"]),
        "nan": frame.assign(sensor_2=np.where(frame.index == 3, np.nan, frame.sensor_2)),
        "duplicate cycle": pd.concat([frame, frame.iloc[[0]]]),
        "zero cycle": frame.assign(cycle=frame.cycle - 1),
        "fractional cycle": frame.assign(cycle=frame.cycle + 0.5),
        "empty": frame.iloc[:0],
    }
    for name, df in bad.items():
        with pytest.raises(InvalidInput):
            svc.predict_engine("fd001-s2-xgboost", df)
            pytest.fail(f"accepted invalid input: {name}")


def test_interval_is_symmetric_threshold_clipped_at_zero(smoke_env, engine):
    svc = ModelService(smoke_env["registry"], ".")
    _, frame = engine
    res = svc.predict_engine("fd001-s2-gru", frame, interval_levels=[0.8], include_trajectory=True)
    q = next(m for m in load_registry(smoke_env["registry"])["models"] if m["id"] == "fd001-s2-gru")["intervals"]["thresholds"]["0.80"]
    pred = np.asarray(res["trajectory"]["predicted_rul"])
    np.testing.assert_allclose(res["trajectory"]["upper_0.80"], pred + q)
    np.testing.assert_allclose(res["trajectory"]["lower_0.80"], np.maximum(pred - q, 0))
    assert res["intervals"][0]["n_calibration_engines"] > 0


# ----------------------------------------------------------------------------- API
def test_health_and_models(client, smoke_env):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["models_registered"] == len(smoke_env["model_ids"])
    models = client.get("/models").json()
    assert models["default_model"] == "fd001-s2-xgboost"
    by_id = {m["id"]: m for m in models["models"]}
    assert by_id["fd001-s2-xgboost"]["intervals_available"] and not by_id["fd001-s2-ridge"]["intervals_available"]
    meta = client.get("/models/fd001-s2-gru").json()
    assert meta["kind"] == "deep" and meta["requires"] == ["torch"] and "engine_ids" not in meta["training"]


def test_predict_single_and_batch(client, engine):
    eid, frame = engine
    r = client.post("/models/fd001-s2-xgboost/predict", json={"engine": {"engine_id": eid, "cycles": _cycles(frame)}, "interval_levels": [0.8], "include_trajectory": True})
    assert r.status_code == 200
    body = r.json()
    assert body["engine_id"] == eid and body["n_cycles"] == len(frame) and body["last_cycle"] == int(frame.cycle.max())
    iv = body["intervals"][0]
    assert iv["lower"] <= body["predicted_rul"] <= iv["upper"]
    assert len(body["trajectory"]["predicted_rul"]) == len(frame)
    assert body["provenance"]["model_version"] and body["provenance"]["data_kind"] == "NASA C-MAPSS simulation"
    rb = client.post("/models/fd001-s2-xgboost/predict/batch", json={"engines": [{"engine_id": "a", "cycles": _cycles(frame)}, {"engine_id": "b", "cycles": _cycles(frame.iloc[:15])}]})
    assert rb.status_code == 200
    preds = rb.json()["predictions"]
    assert preds[0]["predicted_rul"] == pytest.approx(body["predicted_rul"])
    assert preds[1]["n_cycles"] == 15


@pytest.mark.parametrize(
    ("mutate", "status", "code"),
    [
        (lambda b: b["engine"]["cycles"][0].pop("sensor_9"), 422, "invalid_request"),
        (lambda b: b["engine"]["cycles"][0].update(extra_field=1.0), 422, "invalid_request"),
        (lambda b: b["engine"]["cycles"][0].update(sensor_2="hot"), 422, "invalid_request"),
        (lambda b: b["engine"]["cycles"][0].update(cycle=0), 422, "invalid_request"),
        (lambda b: b["engine"].update(cycles=[]), 422, "invalid_request"),
        (lambda b: b["engine"]["cycles"].append(dict(b["engine"]["cycles"][0])), 422, "invalid_input"),
        (lambda b: b.update(interval_levels=[0.95]), 422, "interval_unavailable"),
    ],
)
def test_predict_errors_are_informative(client, engine, mutate, status, code):
    _, frame = engine
    body = {"engine": {"cycles": _cycles(frame.iloc[:20])}}
    mutate(body)
    r = client.post("/models/fd001-s2-xgboost/predict", json=body)
    assert r.status_code == status, r.text
    err = r.json()["error"]
    assert err["code"] == code and err["message"]


def test_unknown_model_and_duplicate_batch_ids(client, engine):
    _, frame = engine
    r = client.post("/models/fd009-x/predict", json={"engine": {"cycles": _cycles(frame.iloc[:5])}})
    assert r.status_code == 404 and "available" in r.json()["error"]["details"]
    assert client.get("/models/fd009-x").status_code == 404
    dup = {"engines": [{"engine_id": 1, "cycles": _cycles(frame.iloc[:5])}] * 2}
    assert client.post("/models/fd001-s2-xgboost/predict/batch", json=dup).status_code == 422
    r = client.post("/models/fd001-s2-ridge/predict", json={"engine": {"cycles": _cycles(frame.iloc[:5])}, "interval_levels": [0.8]})
    assert r.json()["error"]["code"] == "interval_unavailable"


def test_openapi_schema_lists_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/models", "/models/{model_id}", "/models/{model_id}/predict", "/models/{model_id}/predict/batch"} <= set(paths)


# ----------------------------------------------------------------------------- maintenance
def _toy_traj() -> pd.DataFrame:
    rows = []
    for e, life in ((1, 50), (2, 80)):
        for c in range(1, life + 1):
            rul = life - c
            rows.append({"engine_id": e, "cycle": c, "rul": rul, "pred": rul + 5})  # 5 cycles optimistic
    return pd.DataFrame(rows)


def test_maintenance_policy_counts():
    traj = _toy_traj()
    costs = M.CostAssumptions(unscheduled_failure=10, planned_maintenance=1, per_unused_cycle=0.1)
    # Trigger at pred <= 20 -> true RUL 15 at trigger; lead 10 -> 5 unused cycles, no failure.
    out = M.simulate_policy(traj, "pred", 20, 10)
    assert not out.failure.any() and (out.unused_cycles == 5).all()
    # Lead time longer than the warning -> failure.
    assert M.simulate_policy(traj, "pred", 20, 16).failure.all()
    # Never triggers -> failure.
    assert M.simulate_policy(traj, "pred", -1, 0).failure.all()
    # Fixed age 60: engine 1 (life 50) fails first; engine 2 maintained with 20 unused.
    age = M.simulate_policy(traj, None, 0, 0, fixed_age=60).set_index("engine_id")
    assert age.loc[1, "failure"] and not age.loc[2, "failure"] and age.loc[2, "unused_cycles"] == 20
    s = M.summarise(out, costs)
    assert s["assumed_cost_total"] == pytest.approx(2 * 1 + 10 * 0.1)
    table = M.compare_policies(traj, [20], 10, costs, [60])
    oracle = table[table.policy.str.startswith("oracle")].iloc[0]
    assert oracle.unscheduled_failures == 0 and oracle.mean_unused_cycles_when_planned == 0


def test_fixed_age_candidates_use_given_lifetimes_only():
    assert M.fixed_age_candidates(np.arange(100, 200)) == [104, 109, 124]


# ----------------------------------------------------------------------------- dashboard
def test_dashboard_sections_render(smoke_env):
    pytest.importorskip("streamlit")
    problems = check_dashboard(smoke_env)
    assert problems and not any(problems.values()), problems


# ----------------------------------------------------------------------------- real artifacts
REAL_REGISTRY = Path("reports/registry/model_registry.json")


def _real_service() -> ModelService:
    if not REAL_REGISTRY.is_file() or not Path("data/raw/cmapss/train_FD001.txt").is_file():
        pytest.skip("real registry or NASA files absent")
    svc = ModelService(REAL_REGISTRY, ".")
    if not all(svc.availability(m)[0] for m in svc.model_ids if m.startswith("fd001")):
        pytest.skip("FD001 bundles not available locally")
    return svc


@pytest.mark.real_data
def test_real_registry_metrics_match_reports():
    svc = _real_service()
    for mid in svc.model_ids:
        e = svc.entry(mid)
        if e["stage"] != "Stage 2":
            continue
        summary = pd.read_csv(Path(e["report_dir"]) / "summary.csv").set_index("model")
        canon = e["metrics"]["canonical_validation"]["values"]["snapshot_rmse_capped"]
        assert canon == pytest.approx(summary.loc[e["name"], "snapshot_rmse_capped_canonical"])
    s1 = svc.entry("fd001-s1-hgb")["metrics"]["official_test"]["values"]
    ledger = json.loads(Path("reports/test_set_ledger.json").read_text())["entries"][0]
    assert s1["last_cycle_rmse_true"] == pytest.approx(ledger["selected_rmse_true"])


@pytest.mark.real_data
def test_real_api_reproduces_saved_stage2_predictions():
    """Served predictions equal the predictions saved when the Stage-2 run evaluated its bundles."""
    svc = _real_service()
    pool = load_train("data/raw/cmapss", "FD001")
    val_ids = load_split("data/manifests/fd001_split_seed42.json").ids("validation")[:3]
    with TestClient(create_app(lambda: svc)) as c:
        for mid in [m for m in svc.model_ids if m.startswith("fd001-s2-")]:
            e = svc.entry(mid)
            saved = pd.read_csv(Path(e["run_dir"]) / "predictions" / f"{e['name']}__canonical.csv")
            for eid in val_ids:
                g = pool[pool.engine_id == eid].drop(columns=["engine_id", "rul"])
                r = c.post(f"/models/{mid}/predict", json={"engine": {"engine_id": int(eid), "cycles": _cycles(g)}, "include_trajectory": True})
                assert r.status_code == 200, r.text
                ref = saved[saved.engine_id == eid].sort_values("cycle")["pred"].to_numpy()
                np.testing.assert_allclose(r.json()["trajectory"]["predicted_rul"], ref, rtol=0, atol=1e-4, err_msg=mid)


@pytest.mark.real_data
def test_real_dashboard_renders(tmp_path):
    pytest.importorskip("streamlit")
    _real_service()
    env = {"registry": str(REAL_REGISTRY), "index": "configs/stage3/results_index.yaml", "data_dir": "data/raw/cmapss"}
    problems = check_dashboard(env)
    assert not any(problems.values()), problems
    shutil.rmtree(tmp_path, ignore_errors=True)


# ----------------------------------------------------------------------------- artifact exchange
def test_export_import_roundtrip_and_tamper_rejection(smoke_env, tmp_path):
    import io
    import tarfile

    from aeromind.serving.artifacts import export_bundles, import_bundles

    reg = load_registry(smoke_env["registry"])
    archive = tmp_path / "models.tar.gz"
    res = export_bundles(smoke_env["registry"], archive)
    assert res["bundles"] == len(reg["models"]) and not res["skipped"]
    # Installing into an empty root restores byte-identical bundles under their registry paths.
    rel = json.loads(Path(smoke_env["registry"]).read_text())
    for m in rel["models"]:
        m["bundle"]["path"] = f"relbundles/{Path(m['bundle']['path']).name}"
    rel_path = tmp_path / "rel_registry.json"
    rel_path.write_text(json.dumps(rel))
    # Build an archive whose member names match the relative registry.
    good = tmp_path / "good.tar.gz"
    with tarfile.open(archive) as src, tarfile.open(good, "w:gz") as dst:
        for mem in src.getmembers():
            if mem.name.endswith(".joblib"):
                data = src.extractfile(mem).read()
                info = tarfile.TarInfo(f"relbundles/{Path(mem.name).name}")
                info.size = len(data)
                dst.addfile(info, io.BytesIO(data))
        evil = b"not a bundle"
        for name in ("../escape.joblib", "relbundles/unknown.joblib"):
            info = tarfile.TarInfo(name)
            info.size = len(evil)
            dst.addfile(info, io.BytesIO(evil))
    root = tmp_path / "root"
    out = import_bundles(good, rel_path, root)
    assert len(out["installed"]) == len(rel["models"]) and not out["missing"] and not out["rejected"]
    assert set(out["ignored"]) == {"../escape.joblib", "relbundles/unknown.joblib"}
    assert not (tmp_path / "escape.joblib").exists()
    svc = ModelService(rel_path, root)
    assert all(svc.availability(m)[0] for m in svc.model_ids)
    # A tampered bundle (hash mismatch) is rejected and never written.
    bad = tmp_path / "bad.tar.gz"
    first = rel["models"][0]["bundle"]["path"]
    with tarfile.open(bad, "w:gz") as dst:
        info = tarfile.TarInfo(first)
        info.size = len(evil)
        dst.addfile(info, io.BytesIO(evil))
    out2 = import_bundles(bad, rel_path, tmp_path / "root2")
    assert out2["rejected"] == [first] and not (tmp_path / "root2" / first).exists()


# ----------------------------------------------------------------------------- release-audit regressions
def test_bundle_swapped_after_check_is_not_unpickled(smoke_env, tmp_path):
    """The hash is computed on the exact bytes that are unpickled (no check-then-reopen race)."""
    reg = json.loads(Path(smoke_env["registry"]).read_text())
    src = Path(reg["models"][0]["bundle"]["path"])
    local = tmp_path / src.name
    shutil.copy(src, local)
    reg["models"][0]["bundle"]["path"] = str(local)
    path = tmp_path / "reg.json"
    path.write_text(json.dumps(reg))
    svc = ModelService(path, ".")
    mid = reg["models"][0]["id"]
    assert svc.availability(mid)[0]  # status cached as OK
    local.write_bytes(b"swapped after the check")
    with pytest.raises(ModelUnavailable):
        svc.load(mid)
    assert not svc.availability(mid)[0]


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda b: b["engine"]["cycles"][-1].update(sensor_2=1e308), "invalid_input"),
        (lambda b: b["engine"]["cycles"][0].update(sensor_2=True), "invalid_request"),
        (lambda b: b["engine"].update(engine_id="x" * 65), "invalid_request"),
    ],
)
def test_hostile_inputs_get_typed_errors(client, engine, mutate, code):
    _, frame = engine
    body = {"engine": {"cycles": _cycles(frame.iloc[:20])}}
    mutate(body)
    for mid in ("fd001-s2-xgboost", "fd001-s2-gru", "fd001-s2-ridge"):
        r = client.post(f"/models/{mid}/predict", json=body)
        assert r.status_code == 422, (mid, r.text)
        assert r.json()["error"]["code"] == code
