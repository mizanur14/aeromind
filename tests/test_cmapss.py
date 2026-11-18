import pandas as pd

from aeromind.data.cmapss import load_cmapss


def test_rul_logic(tmp_path):
    # Construct a tiny artificial C-MAPSS dataset.
    def make_row(engine, cycle):
        return [engine, cycle] + [0.0] * 24

    train_rows = [
        make_row(1, 1),
        make_row(1, 2),
        make_row(1, 3),
        make_row(2, 1),
        make_row(2, 2),
    ]

    test_rows = [
        make_row(1, 1),
        make_row(1, 2),
        make_row(2, 1),
    ]

    pd.DataFrame(train_rows).to_csv(
        tmp_path / "train_FD001.txt",
        sep=" ",
        index=False,
        header=False,
    )

    pd.DataFrame(test_rows).to_csv(
        tmp_path / "test_FD001.txt",
        sep=" ",
        index=False,
        header=False,
    )

    # Test engine 1 has 5 remaining cycles.
    # Test engine 2 has 9 remaining cycles.
    pd.DataFrame([5, 9]).to_csv(
        tmp_path / "RUL_FD001.txt",
        index=False,
        header=False,
    )

    train, test = load_cmapss(tmp_path, "FD001")

    assert train["rul"].tolist() == [2, 1, 0, 1, 0]
    assert test["rul"].tolist() == [6, 5, 9]

    assert len(train.columns) == 27
    assert len(test.columns) == 27


# --------------------------------------------------------------------------- Stage 1
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from aeromind.data.cmapss import (  # noqa: E402
    COLUMNS,
    SENSOR_COLUMNS,
    SENSOR_INFO,
    DatasetError,
    add_test_rul,
    load_test,
    load_train,
    read_test_rul,
    read_trajectory,
    validate_dataset,
)


def _write_rows(path, rows):
    path.write_text("\n".join(" ".join(str(v) for v in r) for r in rows) + "\n")


def _row(engine, cycle, n_values=24, fill=1.0):
    return [engine, cycle] + [fill] * n_values


def test_schema_constants():
    assert len(COLUMNS) == 26
    assert set(SENSOR_INFO) == set(SENSOR_COLUMNS)
    assert SENSOR_INFO["sensor_11"][0] == "Ps30"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="never downloads"):
        read_trajectory(tmp_path / "train_FD001.txt")


def test_empty_file_raises(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("")
    with pytest.raises(DatasetError, match="empty"):
        read_trajectory(p)


def test_wrong_column_count_raises(tmp_path):
    p = tmp_path / "x.txt"
    _write_rows(p, [_row(1, 1, 23), _row(1, 2, 23)])
    with pytest.raises(DatasetError, match="26-column"):
        read_trajectory(p)


def test_ragged_row_raises(tmp_path):
    p = tmp_path / "x.txt"
    _write_rows(p, [_row(1, 1), _row(1, 2, 25)])
    with pytest.raises(DatasetError):
        read_trajectory(p)


def test_short_row_detected_as_missing(tmp_path):
    p = tmp_path / "x.txt"
    _write_rows(p, [_row(1, 1), _row(1, 2, 23)])
    with pytest.raises(DatasetError, match="missing"):
        read_trajectory(p)


def test_non_numeric_raises(tmp_path):
    p = tmp_path / "x.txt"
    rows = [_row(1, 1), _row(1, 2)]
    rows[1][5] = "abc"
    _write_rows(p, rows)
    with pytest.raises(DatasetError, match="non-numeric"):
        read_trajectory(p)


def test_nonfinite_raises(tmp_path):
    p = tmp_path / "x.txt"
    rows = [_row(1, 1), _row(1, 2)]
    rows[1][7] = "inf"
    _write_rows(p, rows)
    with pytest.raises(DatasetError, match="non-finite"):
        read_trajectory(p)


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([_row(1, 1), _row(1, 1)], "duplicate"),
        ([_row(1, 1), _row(1, 3)], "cycle sequence"),
        ([_row(1, 2), _row(1, 3)], "cycle sequence"),
        ([_row(0, 1)], "positive"),
        ([_row(1, 1.5)], "non-integer"),
    ],
)
def test_trajectory_integrity(tmp_path, rows, match):
    p = tmp_path / "x.txt"
    _write_rows(p, rows)
    with pytest.raises(DatasetError, match=match):
        read_trajectory(p)


def test_unsorted_rows_are_sorted(tmp_path):
    p = tmp_path / "x.txt"
    _write_rows(p, [_row(2, 1), _row(1, 2), _row(1, 1)])
    df = read_trajectory(p)
    assert df[["engine_id", "cycle"]].values.tolist() == [[1, 1], [1, 2], [2, 1]]


@pytest.mark.parametrize("content", ["5\n-1\n", "5\n2.5\n", "5 6\n7 8\n", "5\nx\n"])
def test_invalid_rul_file(tmp_path, content):
    p = tmp_path / "RUL.txt"
    p.write_text(content)
    with pytest.raises(DatasetError):
        read_test_rul(p)


def test_rul_count_mismatch(tmp_path):
    p = tmp_path / "t.txt"
    _write_rows(p, [_row(1, 1), _row(2, 1)])
    test = read_trajectory(p)
    with pytest.raises(DatasetError, match="do not match"):
        add_test_rul(test, read_test_rul_from([4]))


def read_test_rul_from(values):
    import pandas as pd

    return pd.Series(values, index=np.arange(1, len(values) + 1))


def test_load_train_does_not_read_rul(synthetic_raw):
    (synthetic_raw / "RUL_FD001.txt").unlink()
    train = load_train(synthetic_raw, "FD001")
    assert (train.groupby("engine_id")["rul"].min() == 0).all()
    with pytest.raises(FileNotFoundError):
        load_test(synthetic_raw, "FD001")


def test_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_train("data/raw/cmapss", "FD005")


def test_validate_dataset_synthetic(synthetic_raw):
    report = validate_dataset(synthetic_raw, "FD001")
    assert report.ok, report.errors
    assert report.stats["train"]["engines"] == 30
    assert any("expected 100" in w for w in report.warnings)


def test_validate_dataset_reports_errors(tmp_path):
    report = validate_dataset(tmp_path, "FD002")
    assert not report.ok
    assert len(report.errors) == 2
