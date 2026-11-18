"""Synthetic C-MAPSS-like trajectories for tests and the smoke path (no NASA data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aeromind.data.cmapss import COLUMNS, SENSOR_COLUMNS, add_train_rul

CONSTANT_SENSORS = ("sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19")


def make_trajectories(
    n_engines: int = 30,
    min_life: int = 40,
    max_life: int = 120,
    seed: int = 0,
    truncate: bool = False,
) -> pd.DataFrame:
    """Generate run-to-failure (or truncated, if ``truncate``) trajectories in the raw schema.

    Informative sensors drift exponentially towards failure with engine-specific offsets
    and Gaussian noise; six sensors are exactly constant like FD001.
    """
    rng = np.random.default_rng(seed)
    rows = []
    # Fixed degradation directions so train and test files (different seeds) agree.
    signs = np.random.default_rng(12345).choice([-1.0, 1.0], size=len(SENSOR_COLUMNS))
    for eid in range(1, n_engines + 1):
        life = int(rng.integers(min_life, max_life + 1))
        length = int(rng.integers(max(5, life // 4), life)) if truncate else life
        t = np.arange(1, length + 1)
        health = np.exp((t - life) / (0.25 * life))  # ~0 early, 1 at failure
        offsets = rng.normal(0, 0.3, len(SENSOR_COLUMNS))
        block = np.empty((length, len(COLUMNS)))
        block[:, 0] = eid
        block[:, 1] = t
        block[:, 2] = rng.normal(0, 0.002, length)
        block[:, 3] = rng.normal(0, 0.0003, length)
        block[:, 4] = 100.0
        for j, name in enumerate(SENSOR_COLUMNS):
            base = 100.0 + 10 * j
            if name in CONSTANT_SENSORS:
                block[:, 5 + j] = base
            else:
                block[:, 5 + j] = base + offsets[j] + signs[j] * 3.0 * health + rng.normal(0, 0.2, length)
        rows.append(block)
    df = pd.DataFrame(np.vstack(rows), columns=COLUMNS)
    df[["engine_id", "cycle"]] = df[["engine_id", "cycle"]].astype(np.int64)
    return df


def make_training_pool(n_engines: int = 30, seed: int = 0) -> pd.DataFrame:
    """Run-to-failure synthetic pool with uncapped ``rul``."""
    return add_train_rul(make_trajectories(n_engines, seed=seed))


def write_raw_files(directory, dataset: str = "FD001", n_train: int = 30, n_test: int = 10, seed: int = 0) -> None:
    """Write synthetic ``train_/test_/RUL_`` files in the NASA text format."""
    from pathlib import Path

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    train = make_trajectories(n_train, seed=seed)
    full = make_trajectories(n_test, seed=seed + 1)
    rng = np.random.default_rng(seed + 2)
    tests, ruls = [], []
    for _, g in full.groupby("engine_id"):
        cut = int(rng.integers(max(5, len(g) // 4), len(g)))
        tests.append(g.iloc[:cut])
        ruls.append(len(g) - cut)
    fmt = ["%d", "%d"] + ["%.4f"] * (len(COLUMNS) - 2)
    np.savetxt(root / f"train_{dataset}.txt", train.to_numpy(), fmt=fmt)
    np.savetxt(root / f"test_{dataset}.txt", pd.concat(tests).to_numpy(), fmt=fmt)
    np.savetxt(root / f"RUL_{dataset}.txt", np.asarray(ruls), fmt="%d")
