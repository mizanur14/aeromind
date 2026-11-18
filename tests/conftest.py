"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from aeromind.synthetic import make_training_pool, write_raw_files

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_RAW = PROJECT_ROOT / "data" / "raw" / "cmapss"


def real_data_available(dataset: str = "FD001") -> bool:
    """True if the NASA files for ``dataset`` are present locally."""
    return all((REAL_RAW / f"{p}_{dataset}.txt").is_file() for p in ("train", "test", "RUL"))


@pytest.fixture
def synthetic_raw(tmp_path: Path) -> Path:
    """Directory with synthetic FD001-format train/test/RUL files."""
    write_raw_files(tmp_path / "raw", n_train=30, n_test=10, seed=3)
    return tmp_path / "raw"


@pytest.fixture
def synthetic_pool():
    """Synthetic run-to-failure training pool with ``rul``."""
    return make_training_pool(n_engines=24, seed=7)
