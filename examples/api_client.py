"""Call a running AeroMind API with one held-out FD001 validation engine.

Usage (API running via `uv run aeromind serve`):

    uv run python examples/api_client.py --engine-index 0 --model fd001-s2-xgboost

The request is built from your local NASA file (no NASA data ships with this repository).
It also writes the JSON request body to `examples/request.json` (gitignored) for use with curl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from aeromind.data.cmapss import load_train
from aeromind.data.splits import load_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="fd001-s2-xgboost")
    ap.add_argument("--engine-index", type=int, default=0, help="index into the FD001 validation engines")
    ap.add_argument("--fraction", type=float, default=0.6, help="observe only this fraction of the engine's life")
    ap.add_argument("--data-dir", default="data/raw/cmapss")
    args = ap.parse_args()

    pool = load_train(args.data_dir, "FD001")
    eid = load_split("data/manifests/fd001_split_seed42.json").ids("validation")[args.engine_index]
    engine = pool[pool.engine_id == eid]
    engine = engine.iloc[: max(1, int(len(engine) * args.fraction))]
    cycles = engine.drop(columns=["engine_id", "rul"]).to_dict(orient="records")
    body = {"engine": {"engine_id": int(eid), "cycles": cycles}, "interval_levels": [0.9]}
    Path("examples/request.json").write_text(json.dumps(body))

    r = requests.post(f"{args.url}/models/{args.model}/predict", json=body, timeout=60)
    out = r.json()
    if r.status_code != 200:
        raise SystemExit(f"HTTP {r.status_code}: {out['error']}")
    iv = out["intervals"][0] if out["intervals"] else None
    print(f"engine {eid}: {out['n_cycles']} cycles observed, predicted RUL {out['predicted_rul']:.1f} cycles")
    if iv:
        print(f"  {iv['level']:.0%} interval [{iv['lower']:.1f}, {iv['upper']:.1f}] ({iv['method']}, n={iv['n_calibration_engines']})")
    print(f"  ground truth (training file, run to failure): RUL {int(engine.rul.iloc[-1])}, capped {min(int(engine.rul.iloc[-1]), 125)}")
    print(f"  model {out['provenance']['model_id']} version {out['provenance']['model_version']} from {out['provenance']['source_run']}")


if __name__ == "__main__":
    main()
