"""Shared detection helpers."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.models.schemas import CandidateAlert, Severity


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(3, window // 2)).mean()
    std = series.rolling(window, min_periods=max(3, window // 2)).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def pct_change_vs_baseline(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if current == 0 else 1.0
    return (current - baseline) / abs(baseline)


def linear_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    if np.allclose(values, values[0]):
        return 0.0
    return float(np.polyfit(x, values, 1)[0])


def severity_from_magnitude(magnitude: float, medium: float, high: float) -> Severity:
    if magnitude >= high:
        return Severity.HIGH
    if magnitude >= medium:
        return Severity.MEDIUM
    return Severity.LOW


def make_alert(
    *,
    entity_id: str,
    start: date,
    end: date,
    alert_type: str,
    severity: Severity,
    metrics: list[str],
    current: float,
    expected: float,
    calcs: dict[str, Any],
    domain: str,
) -> CandidateAlert:
    return CandidateAlert(
        entity_id=entity_id,
        start_date=start,
        end_date=end,
        alert_type=alert_type,
        severity=severity,
        metrics_involved=metrics,
        current_value=float(current),
        expected_value=float(expected),
        supporting_calculations=calcs,
        domain=domain,
    )


def to_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()


def historical_baseline(frame: pd.DataFrame, column: str, recent_n: int) -> pd.Series:
    """
    Baseline from early/mid history, excluding the most recent window.

    Persistent anomalies make "prior window" look like "recent"; use an older
    slice (roughly the middle 50% before the last 2 windows) instead.
    """
    if len(frame) <= recent_n + 2:
        return frame[column].iloc[:-1] if len(frame) > 1 else frame[column]

    excluded = max(recent_n * 2, recent_n + 1)
    hist = frame.iloc[: max(1, len(frame) - excluded)]
    # Prefer the latter half of history (stable pre-anomaly period for late injections)
    start = len(hist) // 4
    return hist.iloc[start:][column]
