"""CPU latency benchmark of the inference service and the HTTP API.

Two paths are measured for each registered, locally available model, on one full-history
validation engine of the model's own subset:

- ``in_process``: :meth:`ModelService.predict_engine` (what the dashboard calls): input
  validation + the bundle's feature pipeline + model + interval.
- ``http``: ``POST /models/{id}/predict`` against a real ``uvicorn`` server on 127.0.0.1
  (JSON encoding, schema validation, network stack on loopback), single engine and a batch of
  all validation engines of the subset.

Latencies are medians/p95 over repeats after warm-up (the laptop throttles; single passes are
not reported). Cold model-load time is reported separately.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aeromind.data.cmapss import load_train
from aeromind.data.splits import load_split
from aeromind.serving.service import ModelService
from aeromind.utils import environment_info, peak_rss_mb, write_json

logger = logging.getLogger(__name__)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _stats(ms: list[float]) -> dict[str, float]:
    a = np.asarray(ms)
    return {"p50_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()), "n": int(a.size)}


def _engines(data_dir: str, dataset: str, manifest: str) -> list[pd.DataFrame]:
    pool = load_train(data_dir, dataset)
    ids = load_split(manifest).ids("validation")
    return [pool[pool.engine_id == e].drop(columns=["engine_id", "rul"]).reset_index(drop=True) for e in ids]


def _payload(frame: pd.DataFrame, eid: Any) -> dict[str, Any]:
    return {"engine_id": eid, "cycles": frame.to_dict(orient="records")}


def run_benchmark(
    registry: str,
    root: str = ".",
    data_dir: str = "data/raw/cmapss",
    manifest_dir: str = "data/manifests",
    models: list[str] | None = None,
    repeats: int = 30,
    http: bool = True,
    out_root: str = "reports/benchmarks",
) -> Path:
    """Benchmark ``models`` (default: every available model); returns the report directory."""
    svc = ModelService(registry, root)
    ids = [m for m in (models or svc.model_ids) if svc.availability(m)[0]]
    engines: dict[str, list[pd.DataFrame]] = {}
    rows: list[dict[str, Any]] = []
    for mid in ids:
        ds = svc.entry(mid)["dataset"]
        if ds not in engines:
            engines[ds] = _engines(data_dir, ds, str(Path(manifest_dir) / f"{ds.lower()}_split_seed42.json"))
        frame = engines[ds][0]
        t0 = time.perf_counter()
        svc.load(mid)
        load_s = time.perf_counter() - t0
        levels = [svc.top_level(mid)] if svc.top_level(mid) else []
        for _ in range(3):
            svc.predict_engine(mid, frame, "warmup", levels)
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            svc.predict_engine(mid, frame, "bench", levels)
            times.append((time.perf_counter() - t) * 1000)
        rows.append({"model_id": mid, "path": "in_process", "request": f"1 engine × {len(frame)} cycles", "load_s": load_s, **_stats(times)})
        logger.info("in-process %s p50 %.2f ms", mid, rows[-1]["p50_ms"])

    if http:
        rows += _http_benchmark(registry, root, ids, svc, engines, repeats)
    env = environment_info()
    report = {
        "measured_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "registry": registry,
        "registry_built_utc": svc.registry["built_utc"],
        "protocol": (
            f"median and p95 over {repeats} repeats after 3 warm-up calls; one full-history validation engine per "
            "subset (batch: all validation engines); CPU only; torch threads ≤ 2 per deep model"
        ),
        "environment": {**env, "cpu_count": os.cpu_count(), "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY")},
        "peak_rss_mb_benchmark_process": peak_rss_mb(),
        "results": rows,
    }
    out = Path(out_root) / f"s3_benchmark_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "benchmark.json", report)
    (out / "benchmark.md").write_text(to_markdown(report))
    latest = Path(out_root) / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(out, latest)
    return out


def _http_benchmark(registry: str, root: str, ids: list[str], svc: ModelService, engines: dict[str, list[pd.DataFrame]], repeats: int) -> list[dict[str, Any]]:
    import httpx

    port = _free_port()
    env = {**os.environ, "AEROMIND_REGISTRY": registry, "AEROMIND_ROOT": root}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "aeromind.api.app:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        env=env,
    )
    rows: list[dict[str, Any]] = []
    try:
        base = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=base, timeout=120) as client:
            for _ in range(200):
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.TransportError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("API server did not start")
            for mid in ids:
                ds = svc.entry(mid)["dataset"]
                levels = [svc.top_level(mid)] if svc.top_level(mid) else []
                single = {"engine": _payload(engines[ds][0], "e0"), "interval_levels": levels}
                batch = {"engines": [_payload(f, f"e{i}") for i, f in enumerate(engines[ds])], "interval_levels": levels}
                for _ in range(3):
                    client.post(f"/models/{mid}/predict", json=single).raise_for_status()
                times = []
                for _ in range(repeats):
                    t = time.perf_counter()
                    client.post(f"/models/{mid}/predict", json=single).raise_for_status()
                    times.append((time.perf_counter() - t) * 1000)
                rows.append({"model_id": mid, "path": "http", "request": f"1 engine × {len(engines[ds][0])} cycles", "load_s": None, **_stats(times)})
                btimes = []
                client.post(f"/models/{mid}/predict/batch", json=batch).raise_for_status()
                for _ in range(max(5, repeats // 5)):
                    t = time.perf_counter()
                    client.post(f"/models/{mid}/predict/batch", json=batch).raise_for_status()
                    btimes.append((time.perf_counter() - t) * 1000)
                n_rows = sum(len(f) for f in engines[ds])
                rows.append({"model_id": mid, "path": "http_batch", "request": f"{len(engines[ds])} engines × {n_rows} cycles total", "load_s": None, **_stats(btimes)})
                logger.info("http %s p50 %.2f ms; batch p50 %.1f ms", mid, rows[-2]["p50_ms"], rows[-1]["p50_ms"])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return rows


def to_markdown(report: dict[str, Any]) -> str:
    """Human-readable benchmark table."""
    env = report["environment"]
    lines = [
        "# AeroMind serving benchmark (CPU)",
        "",
        f"Measured {report['measured_utc']} · {env.get('processor')} · {env.get('cpu_count')} logical CPUs · "
        f"Python {env.get('python')} · registry built {report['registry_built_utc']}",
        "",
        f"Protocol: {report['protocol']}.",
        "",
        "| Model | Path | Request | p50 ms | p95 ms | Cold load s |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["results"]:
        load = f"{r['load_s']:.2f}" if r.get("load_s") is not None else "—"
        lines.append(f"| {r['model_id']} | {r['path']} | {r['request']} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {load} |")
    lines.append("")
    lines.append(f"Peak RSS of the benchmark process (all models loaded): {report['peak_rss_mb_benchmark_process']:.0f} MB.")
    return "\n".join(lines) + "\n"


def summary(path: str | Path) -> str:
    """Markdown of a saved benchmark."""
    return to_markdown(json.loads((Path(path) / "benchmark.json").read_text()))
