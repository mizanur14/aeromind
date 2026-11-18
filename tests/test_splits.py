import numpy as np
import pytest

from aeromind.data.splits import (
    SplitError,
    SplitManifest,
    load_split,
    make_engine_split,
    save_split,
    split_frame,
    verify_split,
)


def test_split_is_deterministic_and_disjoint():
    ids = np.arange(1, 101)
    a = make_engine_split(ids, "FD001", seed=42)
    b = make_engine_split(ids[::-1], "FD001", seed=42)
    assert a.to_dict() == b.to_dict()
    assert (len(a.train_engine_ids), len(a.validation_engine_ids), len(a.calibration_engine_ids)) == (70, 15, 15)
    assert not set(a.train_engine_ids) & set(a.validation_engine_ids)
    assert not set(a.train_engine_ids) & set(a.calibration_engine_ids)
    assert not set(a.validation_engine_ids) & set(a.calibration_engine_ids)
    assert make_engine_split(ids, "FD001", seed=7).train_engine_ids != a.train_engine_ids


def test_split_matches_stage0_algorithm():
    ids = np.arange(1, 101)
    ref = np.sort(ids.copy())
    np.random.default_rng(42).shuffle(ref)
    m = make_engine_split(ids, "FD001", 42)
    assert m.train_engine_ids == sorted(ref[:70].tolist())
    assert m.validation_engine_ids == sorted(ref[70:85].tolist())


@pytest.mark.parametrize("n", [260, 249, 248, 20])
def test_split_sizes_other_subsets(n):
    m = make_engine_split(np.arange(1, n + 1), "FD002", 0)
    verify_split(m, np.arange(1, n + 1))
    assert sum(len(m.ids(p)) for p in ("train", "validation", "calibration")) == n


def test_split_rejects_bad_inputs():
    with pytest.raises(SplitError):
        make_engine_split([1, 2, 2, 3], "FD001")
    with pytest.raises(SplitError):
        make_engine_split(np.arange(1, 4), "FD001")
    with pytest.raises(SplitError):
        make_engine_split(np.arange(1, 101), "FD001", fractions=(0.5, 0.5, 0.5))


def test_verify_detects_overlap_and_gaps():
    m = SplitManifest("FD001", 0, [1, 2], [3], [4])
    verify_split(m, [1, 2, 3, 4])
    with pytest.raises(SplitError, match="both"):
        verify_split(SplitManifest("FD001", 0, [1, 2], [2], [4]), [1, 2, 4])
    with pytest.raises(SplitError, match="cover"):
        verify_split(m, [1, 2, 3, 4, 5])


def test_save_refuses_silent_change(tmp_path):
    p = tmp_path / "s.json"
    ids = np.arange(1, 101)
    save_split(make_engine_split(ids, "FD001", 42), p)
    save_split(make_engine_split(ids, "FD001", 42), p)  # identical IDs: fine
    with pytest.raises(SplitError, match="Refusing"):
        save_split(make_engine_split(ids, "FD001", 1), p)
    save_split(make_engine_split(ids, "FD001", 1), p, force=True)
    assert load_split(p).seed == 1


def test_split_frame_partitions_rows(synthetic_pool):
    m = make_engine_split(synthetic_pool.engine_id.unique(), "FD001", 0)
    parts = [split_frame(synthetic_pool, m, p) for p in ("train", "validation", "calibration")]
    assert sum(len(p) for p in parts) == len(synthetic_pool)
    for i, a in enumerate(parts):
        for b in parts[i + 1 :]:
            assert not set(a.engine_id) & set(b.engine_id)
