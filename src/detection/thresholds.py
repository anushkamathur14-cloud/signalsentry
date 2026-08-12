"""Threshold loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.paths import THRESHOLDS_PATH


def load_thresholds(path: Path | None = None) -> dict[str, Any]:
    config_path = path or THRESHOLDS_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
