"""Deterministic churn early-warning detectors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.detection.utils import (
    historical_baseline,
    linear_slope,
    make_alert,
    pct_change_vs_baseline,
    severity_from_magnitude,
    to_date,
)
from src.models.schemas import CandidateAlert, Severity


def detect_churn_alerts(df: pd.DataFrame, thresholds: dict[str, Any]) -> list[CandidateAlert]:
    cfg = thresholds["churn"]
    window = int(cfg["rolling_window"])
    alerts: list[CandidateAlert] = []

    for account_id, group in df.sort_values("date").groupby("account_id"):
        g = group.reset_index(drop=True)
        if len(g) < window + 1:
            continue

        latest = g.iloc[-1]
        recent = g.iloc[-window:]
        baseline_active = historical_baseline(g, "active_users", window)
        latest_date = to_date(latest["date"])
        start_date = to_date(recent.iloc[0]["date"])

        # Skip obvious low-volume noise
        if float(recent["active_users"].mean()) < float(cfg["min_active_users"]) and float(
            baseline_active.mean()
        ) < float(cfg["min_active_users"]):
            continue

        month = latest_date.month
        seasonal = bool(cfg.get("suppress_benign_seasonal", True)) and month in set(cfg.get("seasonal_months", []))

        def baseline_mean(col: str) -> float:
            series = historical_baseline(g, col, window)
            return float(series.mean()) if len(series) else float(recent[col].mean())

        # Usage decline (sessions / active users)
        cur_sessions = float(recent["weekly_sessions"].mean())
        exp_sessions = baseline_mean("weekly_sessions")
        drop = -pct_change_vs_baseline(cur_sessions, exp_sessions)
        # Longer slope to distinguish gradual vs sudden
        long_tail = g.iloc[-max(window * 2, 8) :]
        sessions_slope = linear_slope(long_tail["weekly_sessions"].to_numpy(dtype=float))
        # Sudden: large drop vs baseline with steep recent slope
        recent_slope = linear_slope(recent["weekly_sessions"].to_numpy(dtype=float))

        if drop >= float(cfg["usage_drop_pct"]) and (not seasonal or drop >= float(cfg["sudden_collapse_pct"]) * 0.85):
            if drop >= float(cfg["sudden_collapse_pct"]) or recent_slope < -max(5.0, abs(sessions_slope) * 2):
                alert_type = "sudden_usage_collapse"
                sev = Severity.CRITICAL if drop >= float(cfg["sudden_collapse_pct"]) else Severity.HIGH
            else:
                alert_type = "gradual_usage_decline"
                sev = severity_from_magnitude(drop, float(cfg["usage_drop_pct"]), float(cfg["sudden_collapse_pct"]))
            alerts.append(
                make_alert(
                    entity_id=str(account_id),
                    start=start_date,
                    end=latest_date,
                    alert_type=alert_type,
                    severity=sev,
                    metrics=["weekly_sessions", "active_users"],
                    current=cur_sessions,
                    expected=exp_sessions,
                    calcs={
                        "pct_drop": round(drop, 4),
                        "sessions_slope": round(sessions_slope, 4),
                        "recent_slope": round(recent_slope, 4),
                        "window": window,
                        "seasonal_guard_applied": seasonal,
                    },
                    domain="churn",
                )
            )

        # Seat utilization
        util = float(latest["active_users"]) / max(float(latest["licensed_users"]), 1.0)
        if util <= float(cfg["seat_util_threshold"]):
            alerts.append(
                make_alert(
                    entity_id=str(account_id),
                    start=start_date,
                    end=latest_date,
                    alert_type="low_seat_utilization",
                    severity=Severity.HIGH if util < 0.2 else Severity.MEDIUM,
                    metrics=["active_users", "licensed_users"],
                    current=util,
                    expected=float(cfg["seat_util_threshold"]),
                    calcs={"seat_utilization": round(util, 4)},
                    domain="churn",
                )
            )

        # Admin inactivity
        admin_days = float(latest["days_since_admin_login"])
        if admin_days >= float(cfg["admin_inactivity_days"]):
            alerts.append(
                make_alert(
                    entity_id=str(account_id),
                    start=start_date,
                    end=latest_date,
                    alert_type="administrator_inactivity",
                    severity=Severity.HIGH if admin_days >= 60 else Severity.MEDIUM,
                    metrics=["days_since_admin_login"],
                    current=admin_days,
                    expected=float(cfg["admin_inactivity_days"]),
                    calcs={"days_since_admin_login": admin_days},
                    domain="churn",
                )
            )

        # Feature adoption
        cur_adopt = float(recent["key_feature_adoption"].mean())
        exp_adopt = baseline_mean("key_feature_adoption")
        adopt_drop = -pct_change_vs_baseline(cur_adopt, exp_adopt)
        adopt_slope = linear_slope(recent["key_feature_adoption"].to_numpy(dtype=float))
        if adopt_drop >= float(cfg["feature_adoption_drop_pct"]) or adopt_slope <= float(
            cfg["feature_slope_threshold"]
        ):
            alerts.append(
                make_alert(
                    entity_id=str(account_id),
                    start=start_date,
                    end=latest_date,
                    alert_type="falling_feature_adoption",
                    severity=severity_from_magnitude(adopt_drop, 0.25, 0.45),
                    metrics=["key_feature_adoption"],
                    current=cur_adopt,
                    expected=exp_adopt,
                    calcs={"pct_drop": round(adopt_drop, 4), "slope": round(adopt_slope, 4)},
                    domain="churn",
                )
            )

        # Support escalations
        high_sev_sum = float(recent["high_severity_ticket_count"].sum())
        if high_sev_sum >= float(cfg["high_severity_ticket_min"]):
            alerts.append(
                make_alert(
                    entity_id=str(account_id),
                    start=start_date,
                    end=latest_date,
                    alert_type="repeated_support_escalations",
                    severity=Severity.HIGH if high_sev_sum >= 5 else Severity.MEDIUM,
                    metrics=["high_severity_ticket_count", "avg_resolution_hours"],
                    current=high_sev_sum,
                    expected=float(cfg["high_severity_ticket_min"]),
                    calcs={
                        "high_severity_sum": high_sev_sum,
                        "avg_resolution_hours": float(recent["avg_resolution_hours"].mean()),
                    },
                    domain="churn",
                )
            )

        # Sentiment / NPS
        cur_nps = float(recent["nps_score"].mean())
        exp_nps = baseline_mean("nps_score")
        nps_drop = exp_nps - cur_nps
        if nps_drop >= float(cfg["nps_drop"]) or cur_nps <= float(cfg["nps_absolute_low"]):
            alerts.append(
                make_alert(
                    entity_id=str(account_id),
                    start=start_date,
                    end=latest_date,
                    alert_type="negative_sentiment",
                    severity=Severity.HIGH if cur_nps < 25 else Severity.MEDIUM,
                    metrics=["nps_score"],
                    current=cur_nps,
                    expected=exp_nps,
                    calcs={"nps_drop": round(nps_drop, 2)},
                    domain="churn",
                )
            )

    return _dedupe_alerts(alerts)


def _dedupe_alerts(alerts: list[CandidateAlert]) -> list[CandidateAlert]:
    """Keep highest severity per (entity, alert_type)."""
    rank = {Severity.LOW: 1, Severity.MEDIUM: 2, Severity.HIGH: 3, Severity.CRITICAL: 4}
    best: dict[tuple[str, str], CandidateAlert] = {}
    for alert in alerts:
        key = (alert.entity_id, alert.alert_type)
        if key not in best or rank[alert.severity] > rank[best[key].severity]:
            best[key] = alert
    return list(best.values())
