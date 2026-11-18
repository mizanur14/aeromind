"""Package and install model bundles outside Git (e.g. as a GitHub Release asset).

Bundles are pickles, so an archive is only as trustworthy as its origin. ``import_bundles``
therefore trusts the **committed** registry, not the archive: a member is written only if its
path is a bundle path listed in the local registry and its SHA-256 equals the registry's.
Nothing is unpickled during import; unknown members and path-traversal names are ignored.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from typing import Any

from aeromind.serving.registry import load_registry
from aeromind.utils import sha256_file


def export_bundles(registry_path: str | Path, out: str | Path, root: str | Path = ".") -> dict[str, Any]:
    """Write ``out`` (.tar.gz) with every registry bundle plus the registry and SHA256SUMS."""
    reg = load_registry(registry_path)
    root = Path(root)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sums, missing = [], []
    with tarfile.open(out, "w:gz") as tar:
        for m in reg["models"]:
            src = root / m["bundle"]["path"]
            if not src.is_file() or sha256_file(src) != m["bundle"]["sha256"]:
                missing.append(m["id"])
                continue
            tar.add(src, arcname=m["bundle"]["path"])
            sums.append(f"{m['bundle']['sha256']}  {m['bundle']['path']}")
        for name, data in (("SHA256SUMS", "\n".join(sums) + "\n"), ("model_registry.json", Path(registry_path).read_text())):
            info = tarfile.TarInfo(name)
            payload = data.encode()
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return {"archive": str(out), "bundles": len(sums), "skipped": missing, "sha256": sha256_file(out)}


def import_bundles(archive: str | Path, registry_path: str | Path, root: str | Path = ".") -> dict[str, Any]:
    """Install bundles from ``archive`` whose path and hash match the local registry."""
    reg = load_registry(registry_path)
    expected = {m["bundle"]["path"]: m["bundle"]["sha256"] for m in reg["models"]}
    root = Path(root)
    installed, rejected, ignored = [], [], []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = member.name
            if name not in expected:
                if name not in ("SHA256SUMS", "model_registry.json"):
                    ignored.append(name)
                continue
            if not member.isfile() or name.startswith("/") or ".." in Path(name).parts:
                rejected.append(name)
                continue
            data = tar.extractfile(member).read()  # type: ignore[union-attr]
            if hashlib.sha256(data).hexdigest() != expected[name]:
                rejected.append(name)
                continue
            dest = root / name
            if dest.is_file() and sha256_file(dest) == expected[name]:
                installed.append(name)
                continue
            if dest.exists():
                rejected.append(f"{name} (exists with a different hash; not overwritten)")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            installed.append(name)
    return {"installed": installed, "rejected": rejected, "ignored": ignored, "missing": sorted(set(expected) - set(installed))}
