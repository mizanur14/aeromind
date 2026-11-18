"""Streamlit Community Cloud entrypoint for the public AeroMind dashboard.

Deploy with main file path ``streamlit_app/streamlit_app.py``. Community Cloud installs
``streamlit_app/requirements.txt`` (a dependency file next to the entrypoint takes precedence
over the repository-root ``uv.lock``) and runs from the repository root, where
``.streamlit/config.toml`` lives.

This file only makes ``src/`` importable and selects public mode: the dashboard then shows the
committed reports and registry metadata, and never loads model bundles, reads NASA files or
accepts uploads. The full research stack (scikit-learn, XGBoost, PyTorch) is not needed.

Local run: ``streamlit run streamlit_app/streamlit_app.py`` from the repository root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("AEROMIND_PROJECT_ROOT", str(ROOT))
os.environ["AEROMIND_DASHBOARD_MODE"] = "public"

from aeromind.dashboard.app import main  # noqa: E402

main()
