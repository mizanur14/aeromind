"""RETIRED Stage-0 split script — now a thin wrapper around ``aeromind split``.

The original version rewrote ``data/manifests/fd001_split_seed42.json`` unconditionally
in the Stage-0 format, which changed the manifest hash recorded by training runs. This
wrapper delegates to the package implementation, which:

- produces the same engine IDs (same algorithm, seed 42),
- writes byte-identical output when the IDs are unchanged,
- refuses to change an existing manifest's IDs unless ``--force`` is given.

Usage (equivalent): ``uv run python scripts/create_splits.py [aeromind split options]``
Preferred:          ``uv run aeromind split --dataset FD001 --seed 42``
"""

import sys

from aeromind.cli import main

if __name__ == "__main__":
    args = sys.argv[1:] or ["--dataset", "FD001", "--seed", "42"]
    sys.exit(main(["split", *args]))
