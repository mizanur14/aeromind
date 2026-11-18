"""Small shared helpers: logging, hashing, JSON, environment metadata, resources."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

try:  # POSIX only
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]

TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "scipy", "joblib", "pyyaml", "torch")


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once with a compact format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def write_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as indented JSON (numpy-aware), creating parent dirs."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    return out


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    return json.loads(Path(path).read_text())


def git_commit(repo: str | Path = ".") -> str | None:
    """Return the current git commit (with ``-dirty`` suffix) or ``None``."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_info() -> dict[str, Any]:
    """Collect interpreter, platform and package versions for run metadata."""
    versions: dict[str, str | None] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "packages": versions,
        "git_commit": git_commit(),
    }


def peak_rss_mb() -> float | None:
    """Peak resident set size of this process in MiB (Linux ``ru_maxrss`` is KiB)."""
    if resource is None:  # pragma: no cover
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    return rss / scale


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Context manager yielding a dict whose ``seconds`` key is set on exit."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
