"""Shared path helpers for SignalSentry."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
OUTPUTS_DIR = DATA_DIR / "outputs"
SHOWCASE_DIR = OUTPUTS_DIR / "showcase"
THRESHOLDS_PATH = CONFIG_DIR / "thresholds.yaml"


def ensure_data_dirs() -> None:
    for path in (GENERATED_DIR, GROUND_TRUTH_DIR, OUTPUTS_DIR, SHOWCASE_DIR):
        path.mkdir(parents=True, exist_ok=True)
