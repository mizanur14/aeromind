"""NASA C-MAPSS turbofan dataset: parsing, validation and RUL labelling.

Public API (stable since the Stage-0 checkpoint):

- :func:`read_trajectory`, :func:`read_test_rul`, :func:`load_cmapss`

Stage-1 additions:

- :func:`load_train` (never touches official test labels)
- :func:`load_test` (the only path that reads ``RUL_FD00x.txt``)
- :func:`validate_dataset` (integrity report used by ``aeromind validate``)
- Sensor metadata (:data:`SENSOR_INFO`) after Saxena et al. (2008), Table 2.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

logger = logging.getLogger(__name__)

SETTING_COLUMNS: list[str] = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS: list[str] = [f"sensor_{i}" for i in range(1, 22)]
ID_COLUMNS: list[str] = ["engine_id", "cycle"]
COLUMNS: list[str] = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS
N_COLUMNS = len(COLUMNS)  # 26

VALID_DATASETS = {"FD001", "FD002", "FD003", "FD004"}
DATASETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")

#: Symbol, description and unit for each sensor (Saxena et al., PHM08, Table 2).
SENSOR_INFO: dict[str, tuple[str, str, str]] = {
    "sensor_1": ("T2", "Total temperature at fan inlet", "°R"),
    "sensor_2": ("T24", "Total temperature at LPC outlet", "°R"),
    "sensor_3": ("T30", "Total temperature at HPC outlet", "°R"),
    "sensor_4": ("T50", "Total temperature at LPT outlet", "°R"),
    "sensor_5": ("P2", "Pressure at fan inlet", "psia"),
    "sensor_6": ("P15", "Total pressure in bypass-duct", "psia"),
    "sensor_7": ("P30", "Total pressure at HPC outlet", "psia"),
    "sensor_8": ("Nf", "Physical fan speed", "rpm"),
    "sensor_9": ("Nc", "Physical core speed", "rpm"),
    "sensor_10": ("epr", "Engine pressure ratio (P50/P2)", "-"),
    "sensor_11": ("Ps30", "Static pressure at HPC outlet", "psia"),
    "sensor_12": ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    "sensor_13": ("NRf", "Corrected fan speed", "rpm"),
    "sensor_14": ("NRc", "Corrected core speed", "rpm"),
    "sensor_15": ("BPR", "Bypass ratio", "-"),
    "sensor_16": ("farB", "Burner fuel-air ratio", "-"),
    "sensor_17": ("htBleed", "Bleed enthalpy", "-"),
    "sensor_18": ("Nf_dmd", "Demanded fan speed", "rpm"),
    "sensor_19": ("PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    "sensor_20": ("W31", "HPT coolant bleed", "lbm/s"),
    "sensor_21": ("W32", "LPT coolant bleed", "lbm/s"),
}

#: Documented properties per subset (readme.txt). Engine counts are the counts
#: *observed in the files*; readme.txt swaps the FD004 train/test counts.
SUBSET_INFO: dict[str, dict[str, object]] = {
    "FD001": {"conditions": 1, "fault_modes": ["HPC"], "train_engines": 100, "test_engines": 100},
    "FD002": {"conditions": 6, "fault_modes": ["HPC"], "train_engines": 260, "test_engines": 259},
    "FD003": {
        "conditions": 1,
        "fault_modes": ["HPC", "Fan"],
        "train_engines": 100,
        "test_engines": 100,
    },
    "FD004": {
        "conditions": 6,
        "fault_modes": ["HPC", "Fan"],
        "train_engines": 249,
        "test_engines": 248,
    },
}


class DatasetError(ValueError):
    """Raised when a C-MAPSS file is missing, malformed or internally inconsistent."""


def check_dataset_name(dataset: str) -> str:
    """Return ``dataset`` if it is a valid subset name, else raise :class:`DatasetError`."""
    if dataset not in VALID_DATASETS:
        raise DatasetError(f"Unknown dataset {dataset!r}; expected one of {sorted(VALID_DATASETS)}")
    return dataset


def dataset_paths(data_dir: str | Path, dataset: str) -> dict[str, Path]:
    """Return the expected train/test/RUL file paths of a subset."""
    check_dataset_name(dataset)
    root = Path(data_dir)
    return {
        "train": root / f"train_{dataset}.txt",
        "test": root / f"test_{dataset}.txt",
        "rul": root / f"RUL_{dataset}.txt",
    }


def _missing_file_message(path: Path) -> str:
    return (
        f"Missing dataset file: {path}. Place the NASA C-MAPSS text files in "
        f"{path.parent} (see docs/DATA.md). AeroMind never downloads data automatically."
    )


def read_trajectory(path: Path) -> pd.DataFrame:
    """Read and validate one C-MAPSS trajectory file.

    Checks: exactly 26 numeric whitespace-separated fields per row, no missing or
    non-finite values, positive integer engine IDs and cycles, no duplicate
    (engine, cycle) pairs, and each engine's cycles are ``1..n`` without gaps.

    Returns:
        DataFrame with :data:`COLUMNS`, sorted by ``engine_id, cycle``.

    Raises:
        FileNotFoundError: file absent.
        DatasetError: any schema or integrity violation (subclass of ``ValueError``).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(_missing_file_message(path))

    try:
        df = pd.read_csv(path, sep=r"\s+", header=None)
    except EmptyDataError as exc:
        raise DatasetError(f"{path.name}: file is empty") from exc
    except ParserError as exc:
        raise DatasetError(f"{path.name}: inconsistent number of fields ({exc})") from exc

    if df.empty or df.shape[1] != N_COLUMNS:
        raise DatasetError(f"{path.name}: expected a nonempty {N_COLUMNS}-column dataset, got shape {df.shape}")
    df.columns = COLUMNS

    if df.isna().any().any():
        bad_rows = df.index[df.isna().any(axis=1)][:5].tolist()
        raise DatasetError(f"{path.name}: unexpected missing values (rows {bad_rows})")

    non_numeric = [c for c in COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise DatasetError(f"{path.name}: non-numeric values in columns {non_numeric}")

    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise DatasetError(f"{path.name}: found non-finite measurements")

    for col in ID_COLUMNS:
        values = df[col].to_numpy()
        if not np.all(values == np.round(values)):
            raise DatasetError(f"{path.name}: non-integer {col}")
        df[col] = values.astype(np.int64)

    if (df[ID_COLUMNS] <= 0).any().any():
        raise DatasetError(f"{path.name}: engine IDs and cycles must be positive")

    if df.duplicated(ID_COLUMNS).any():
        raise DatasetError(f"{path.name}: duplicate engine/cycle observations")

    df = df.sort_values(ID_COLUMNS).reset_index(drop=True)

    # Every engine trajectory must start at cycle 1 and be consecutive.
    expected = df.groupby("engine_id", sort=False).cumcount() + 1
    if not (df["cycle"].to_numpy() == expected.to_numpy()).all():
        bad = df.loc[df["cycle"] != expected, "engine_id"].unique()[:5].tolist()
        raise DatasetError(f"{path.name}: invalid cycle sequence for engine(s) {bad}")

    return df


def read_test_rul(path: Path) -> pd.Series:
    """Read the official final-cycle RUL labels (one row per test engine, in ID order)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(_missing_file_message(path))

    try:
        values = pd.read_csv(path, sep=r"\s+", header=None)
    except (EmptyDataError, ParserError) as exc:
        raise DatasetError(f"{path.name}: invalid test RUL file ({exc})") from exc

    if values.shape[1] != 1 or values.isna().any().any():
        raise DatasetError(f"{path.name}: invalid test RUL file (shape {values.shape})")
    if not pd.api.types.is_numeric_dtype(values.iloc[:, 0]):
        raise DatasetError(f"{path.name}: non-numeric test RUL labels")

    rul = values.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(rul).all():
        raise DatasetError(f"{path.name}: non-finite test RUL labels")
    if (rul < 0).any() or not np.all(rul == np.round(rul)):
        raise DatasetError(f"{path.name}: invalid test RUL values (negative or non-integer)")

    return pd.Series(rul.astype(np.int64), index=np.arange(1, len(rul) + 1), name="rul")


def add_train_rul(train: pd.DataFrame) -> pd.DataFrame:
    """Add uncapped ``rul = failure_cycle − cycle`` to run-to-failure trajectories.

    The last observed cycle of each training engine is its failure cycle (RUL 0).
    """
    out = train.copy()
    failure = out.groupby("engine_id")["cycle"].transform("max")
    out["rul"] = (failure - out["cycle"]).astype(np.int64)
    return out


def add_test_rul(test: pd.DataFrame, test_rul: pd.Series) -> pd.DataFrame:
    """Add uncapped RUL to truncated test trajectories using the official labels.

    ``rul(t) = last_observed_cycle − t + official_rul``.
    """
    test_ids = np.sort(test["engine_id"].unique())
    if not np.array_equal(test_ids, np.arange(1, len(test_rul) + 1)):
        raise DatasetError(f"Test engine IDs (n={len(test_ids)}) do not match RUL row ordering (n={len(test_rul)})")
    out = test.copy()
    last = out.groupby("engine_id")["cycle"].transform("max")
    official = out["engine_id"].map(test_rul)
    if official.isna().any():
        raise DatasetError("Test RUL mapping failed")
    out["rul"] = (last - out["cycle"] + official).astype(np.int64)
    return out


def load_train(data_dir: str | Path, dataset: str = "FD001") -> pd.DataFrame:
    """Load the training pool with uncapped RUL. Never reads official test labels."""
    paths = dataset_paths(data_dir, dataset)
    train = add_train_rul(read_trajectory(paths["train"]))
    if (train["rul"] < 0).any():  # pragma: no cover - guaranteed by construction
        raise DatasetError("Negative training RUL")
    logger.debug("Loaded %s train: %d rows", dataset, len(train))
    return train


def load_test(data_dir: str | Path, dataset: str = "FD001") -> pd.DataFrame:
    """Load test trajectories with RUL from the official labels.

    This is the **only** function that reads ``RUL_FD00x.txt``. Use it exclusively
    for final evaluation (``aeromind test``).
    """
    paths = dataset_paths(data_dir, dataset)
    test = read_trajectory(paths["test"])
    labelled = add_test_rul(test, read_test_rul(paths["rul"]))
    logger.info("Loaded %s test trajectories with official RUL labels", dataset)
    return labelled


def load_test_features_only(data_dir: str | Path, dataset: str = "FD001") -> pd.DataFrame:
    """Load test trajectories *without* labels (for inference only)."""
    return read_trajectory(dataset_paths(data_dir, dataset)["test"])


def load_cmapss(
    data_dir: str | Path,
    dataset: str = "FD001",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test trajectories with correct uncapped RUL labels.

    Kept for backward compatibility. Note that it reads the official test labels; model
    development code should use :func:`load_train` instead.
    """
    check_dataset_name(dataset)
    return load_train(data_dir, dataset), load_test(data_dir, dataset)


@dataclass
class ValidationReport:
    """Integrity summary of one subset."""

    dataset: str
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict."""
        return asdict(self)


def _trajectory_stats(df: pd.DataFrame) -> dict[str, object]:
    lengths = df.groupby("engine_id")["cycle"].max()
    return {
        "rows": int(len(df)),
        "engines": int(df["engine_id"].nunique()),
        "engine_id_min": int(df["engine_id"].min()),
        "engine_id_max": int(df["engine_id"].max()),
        "length_min": int(lengths.min()),
        "length_median": float(lengths.median()),
        "length_max": int(lengths.max()),
        "missing_values": int(df.isna().sum().sum()),
    }


def validate_dataset(data_dir: str | Path, dataset: str) -> ValidationReport:
    """Run all integrity checks on one subset and return a report (never raises on data errors).

    Checks schema and trajectories (via :func:`read_trajectory`), contiguous engine IDs,
    test/RUL alignment and training RUL ending at 0. Engine counts are compared against
    :data:`SUBSET_INFO` as warnings.
    """
    check_dataset_name(dataset)
    report = ValidationReport(dataset=dataset)
    paths = dataset_paths(data_dir, dataset)

    try:
        train = load_train(data_dir, dataset)
        report.stats["train"] = _trajectory_stats(train)
        ids = np.sort(train["engine_id"].unique())
        if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
            report.errors.append("train engine IDs are not contiguous 1..n")
        last_rul = train.groupby("engine_id")["rul"].min()
        if not (last_rul == 0).all():
            report.errors.append("train RUL does not reach 0 for every engine")
        constant = [c for c in SETTING_COLUMNS + SENSOR_COLUMNS if train[c].max() - train[c].min() < 1e-12]
        report.stats["train_constant_columns"] = constant
    except (FileNotFoundError, DatasetError) as exc:
        report.errors.append(f"train: {exc}")

    try:
        test = load_test(data_dir, dataset)
        report.stats["test"] = _trajectory_stats(test)
        # Labels are checked for integrity only; their distribution is deliberately not
        # reported, so validation output never exposes test-label statistics.
        report.stats["test_labels_valid"] = bool((test["rul"] >= 0).all())
    except (FileNotFoundError, DatasetError) as exc:
        report.errors.append(f"test: {exc}")

    expected = SUBSET_INFO[dataset]
    for part in ("train", "test"):
        got = report.stats.get(part, {})
        if isinstance(got, dict) and got.get("engines") != expected[f"{part}_engines"]:
            report.warnings.append(f"{part} engines {got.get('engines')} != expected {expected[f'{part}_engines']}")
    report.stats["files"] = {k: str(v) for k, v in paths.items()}
    report.ok = not report.errors
    return report
