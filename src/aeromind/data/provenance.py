"""Dataset provenance: SHA-256 manifests of the raw NASA files and verification."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aeromind.data.cmapss import DATASETS, dataset_paths
from aeromind.utils import read_json, sha256_file, write_json

logger = logging.getLogger(__name__)

SOURCE = {
    "name": "NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set",
    "publisher": "NASA Ames Prognostics Center of Excellence (PCoE) Data Repository",
    "reference": (
        "A. Saxena, K. Goebel, D. Simon, N. Eklund, 'Damage Propagation Modeling for "
        "Aircraft Engine Run-to-Failure Simulation', Proc. 1st Int. Conf. on Prognostics "
        "and Health Management (PHM08), Denver, CO, 2008."
    ),
}


def build_provenance(data_dir: str | Path, datasets: tuple[str, ...] = DATASETS) -> dict[str, Any]:
    """Hash every present raw file of the given subsets.

    Missing files are recorded with ``sha256: null`` rather than raising, so a partial
    local copy can still be described.
    """
    root = Path(data_dir)
    files: dict[str, Any] = {}
    for ds in datasets:
        for path in dataset_paths(root, ds).values():
            rel = path.name
            if path.is_file():
                files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            else:
                files[rel] = {"sha256": None, "bytes": None}
    return {
        "source": SOURCE,
        "data_dir": str(root),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
    }


def write_provenance(data_dir: str | Path, output: str | Path) -> Path:
    """Build and write a provenance manifest."""
    return write_json(output, build_provenance(data_dir))


def verify_provenance(
    data_dir: str | Path, manifest_path: str | Path, datasets: tuple[str, ...] = DATASETS
) -> list[str]:
    """Compare current raw files with a manifest; return a list of problems (empty = OK)."""
    manifest = read_json(manifest_path)
    expected = manifest.get("files", {})
    problems: list[str] = []
    root = Path(data_dir)
    for ds in datasets:
        for path in dataset_paths(root, ds).values():
            entry = expected.get(path.name)
            if entry is None or entry.get("sha256") is None:
                problems.append(f"{path.name}: not recorded in manifest")
                continue
            if not path.is_file():
                problems.append(f"{path.name}: missing on disk")
                continue
            actual = sha256_file(path)
            if actual != entry["sha256"]:
                problems.append(f"{path.name}: sha256 mismatch ({actual[:12]} != {entry['sha256'][:12]})")
    return problems


def dataset_hashes(data_dir: str | Path, dataset: str, include_test_labels: bool = False) -> dict[str, str]:
    """SHA-256 of the files a run actually used (test labels only when requested)."""
    paths = dataset_paths(data_dir, dataset)
    keys = ["train", "test"] + (["rul"] if include_test_labels else [])
    return {paths[k].name: sha256_file(paths[k]) for k in keys if paths[k].is_file()}
