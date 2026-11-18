"""C-MAPSS data loading, validation, provenance and engine-level splitting."""

from aeromind.data.cmapss import (
    COLUMNS,
    DATASETS,
    SENSOR_COLUMNS,
    SENSOR_INFO,
    SETTING_COLUMNS,
    DatasetError,
    load_cmapss,
    load_test,
    load_train,
    read_test_rul,
    read_trajectory,
    validate_dataset,
)

__all__ = [
    "COLUMNS",
    "DATASETS",
    "SENSOR_COLUMNS",
    "SENSOR_INFO",
    "SETTING_COLUMNS",
    "DatasetError",
    "load_cmapss",
    "load_test",
    "load_train",
    "read_test_rul",
    "read_trajectory",
    "validate_dataset",
]
