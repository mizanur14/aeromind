from pathlib import Path

import pytest

from aeromind.config import ConfigError, ExperimentConfig, dump_config, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_load(path, tmp_path):
    cfg = load_config(path)
    out = dump_config(cfg, tmp_path / "c.yaml")
    assert load_config(out).to_dict() == cfg.to_dict()


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "x", "models": {"ridge": {}}, "typo": 1},
        {"name": "x", "models": {"ridge": {}}, "features": {"windowz": [3]}},
        {"name": "x", "models": {"lstm": {}}},
        {"name": "x", "models": {}},
        {"name": "x", "dataset": "FD009", "models": {"ridge": {}}},
        {"name": "x", "models": {"ridge": {}}, "features": {"windows": [1]}},
        {"name": "x", "models": {"ridge": {}}, "target": {"cap": 0}},
        {"name": "x", "models": {"ridge": {"grid": {"alpha": []}}}},
    ],
)
def test_invalid_configs(payload):
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(payload)


def test_config_error_is_value_error():
    assert issubclass(ConfigError, ValueError)
