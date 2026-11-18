"""Integrity tests on the real NASA files (skipped when the data are absent, e.g. in CI)."""

import json

import numpy as np
import pytest

from aeromind.data.cmapss import DATASETS, SUBSET_INFO, load_test, load_train, validate_dataset
from aeromind.data.provenance import verify_provenance
from aeromind.data.splits import load_split, make_engine_split, verify_split

from .conftest import PROJECT_ROOT, REAL_RAW, real_data_available

pytestmark = pytest.mark.real_data
MANIFESTS = PROJECT_ROOT / "data" / "manifests"


@pytest.fixture(scope="module", autouse=True)
def _require_data():
    if not real_data_available("FD001"):
        pytest.skip("NASA C-MAPSS files not present")


def test_fd001_known_facts():
    train = load_train(REAL_RAW, "FD001")
    test = load_test(REAL_RAW, "FD001")
    assert len(train) == 20631 and len(test) == 13096
    assert train.engine_id.nunique() == 100 and test.engine_id.nunique() == 100
    assert train.isna().sum().sum() == 0
    life = train.groupby("engine_id").cycle.max()
    assert (life.min(), life.max()) == (128, 362)
    constant = [c for c in train.columns if c.startswith("sensor_") and np.ptp(train[c]) == 0]
    assert constant == ["sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_all_subsets_validate(dataset):
    if not real_data_available(dataset):
        pytest.skip(f"{dataset} not present")
    report = validate_dataset(REAL_RAW, dataset)
    assert report.ok, report.errors
    assert report.warnings == []
    assert report.stats["train"]["engines"] == SUBSET_INFO[dataset]["train_engines"]


def test_existing_split_manifest_reproduced():
    train = load_train(REAL_RAW, "FD001")
    saved = load_split(MANIFESTS / "fd001_split_seed42.json")
    verify_split(saved, train.engine_id.unique())
    regenerated = make_engine_split(train.engine_id.unique(), "FD001", 42)
    for p in ("train", "validation", "calibration"):
        assert regenerated.ids(p) == saved.ids(p)


def test_provenance_hashes_match():
    manifest = MANIFESTS / "cmapss_provenance.json"
    if not manifest.is_file():
        pytest.skip("provenance manifest not written yet")
    assert verify_provenance(REAL_RAW, manifest) == []
    outer = (MANIFESTS / "nasa_cmapss_outer.sha256").read_text().split()[0]
    assert len(outer) == 64
    assert json.loads(manifest.read_text())["files"]["train_FD001.txt"]["sha256"].startswith("963b5e22")
