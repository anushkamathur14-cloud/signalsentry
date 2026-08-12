"""Deterministic paid-media anomaly detectors."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.detection.utils import (
    historical_baseline,
    make_alert,
    pct_change_vs_baseline,
    rolling_zscore,
    severity_from_magnitude,
    to_date,
)
from src.models.schemas import CandidateAlert, Severity


def detect_media_alerts(df: pd.DataFrame, thresholds: dict[str, Any]) -> list[CandidateAlert]:
    cfg = thresholds["media"]
    window = int(cfg["rolling_window"])
    alerts: list[CandidateAlert] = []

    for campaign_id, group in df.sort_values("date").groupby("campaign_id"):
        g = group.reset_index(drop=True)
        if len(g) < window + 2:
            continue

        work = g.copy()
        if cfg.get("suppress_weekends", True):
            weekdays = work[pd.to_datetime(work["date"]).dt.weekday < 5]
            if len(weekdays) >= window + 2:
                work = weekdays.reset_index(drop=True)

        latest = work.iloc[-1]
        recent = work.iloc[-window:]
        latest_date = to_date(latest["date"])
        start_date = to_date(recent.iloc[0]["date"])

        if float(recent["spend"].mean()) < float(cfg["min_spend"]) and float(recent["impressions"].mean()) < float(
            cfg["min_impressions"]
        ):
            continue

        month = latest_date.month
        seasonal = bool(cfg.get("suppress_benign_seasonal", True)) and month in set(cfg.get("seasonal_months", []))

        def baseline_mean(col: str) -> float:
            series = historical_baseline(work, col, window)
            return float(series.mean()) if len(series) else float(recent[col].mean())

        cur_spend = float(recent["spend"].mean())
        exp_spend = baseline_mean("spend")
        cur_conv = float(recent["conversions"].mean())
        exp_conv = baseline_mean("conversions")
        spend_lift = pct_change_vs_baseline(cur_spend, exp_spend)
        conv_lift = pct_change_vs_baseline(cur_conv, exp_conv)

        # Spend spike without conversion growth
        # Seasonal months: still flag when conversions fail to track spend (efficiency break).
        efficiency_break = spend_lift >= float(cfg["spend_spike_pct"]) and conv_lift < spend_lift * 0.35
        if efficiency_break and (not seasonal or conv_lift < 0.15):
            alerts.append(
                make_alert(
                    entity_id=str(campaign_id),
                    start=start_date,
                    end=latest_date,
                    alert_type="spend_spike_no_conversion_growth",
                    severity=severity_from_magnitude(spend_lift, 0.4, 0.8),
                    metrics=["spend", "conversions"],
                    current=cur_spend,
                    expected=exp_spend,
                    calcs={
                        "spend_lift": round(spend_lift, 4),
                        "conversion_lift": round(conv_lift, 4),
                    },
                    domain="media",
                )
            )

        for metric, key, alert_type in [
            ("cpc", "cpc_increase_pct", "cpc_increase"),
            ("cpm", "cpm_increase_pct", "cpm_increase"),
        ]:
            cur = float(recent[metric].mean())
            exp = baseline_mean(metric)
            lift = pct_change_vs_baseline(cur, exp)
            if lift >= float(cfg[key]) and float(recent["clicks"].mean()) >= float(cfg["min_clicks"]) / 2:
                alerts.append(
                    make_alert(
                        entity_id=str(campaign_id),
                        start=start_date,
                        end=latest_date,
                        alert_type=alert_type,
                        severity=severity_from_magnitude(lift, float(cfg[key]), float(cfg[key]) * 1.8),
                        metrics=[metric, "spend", "clicks"],
                        current=cur,
                        expected=exp,
                        calcs={"pct_increase": round(lift, 4)},
                        domain="media",
                    )
                )

        cur_cvr = float(recent["conversion_rate"].mean())
        exp_cvr = baseline_mean("conversion_rate")
        cvr_drop = -pct_change_vs_baseline(cur_cvr, exp_cvr)
        if cvr_drop >= float(cfg["cvr_collapse_pct"]) and float(recent["clicks"].mean()) >= float(cfg["min_clicks"]) / 2:
            alerts.append(
                make_alert(
                    entity_id=str(campaign_id),
                    start=start_date,
                    end=latest_date,
                    alert_type="conversion_rate_collapse",
                    severity=severity_from_magnitude(cvr_drop, 0.4, 0.7),
                    metrics=["conversion_rate", "clicks", "conversions"],
                    current=cur_cvr,
                    expected=exp_cvr,
                    calcs={"pct_drop": round(cvr_drop, 4)},
                    domain="media",
                )
            )

        track = float(recent["tracking_event_count"].sum())
        conv = float(recent["conversions"].sum())
        if conv > 1:
            ratio = track / conv if track > 0 else 0.0
            if ratio < (1 / float(cfg["tracking_discrepancy_ratio"])) or (
                track > 0 and conv / max(track, 1e-9) >= float(cfg["tracking_discrepancy_ratio"])
            ):
                alerts.append(
                    make_alert(
                        entity_id=str(campaign_id),
                        start=start_date,
                        end=latest_date,
                        alert_type="tracking_discrepancy",
                        severity=Severity.CRITICAL if ratio < 0.5 else Severity.HIGH,
                        metrics=["tracking_event_count", "conversions"],
                        current=track,
                        expected=conv,
                        calcs={"tracking_to_conversion_ratio": round(ratio, 4)},
                        domain="media",
                    )
                )

        budget = float(recent["budget"].mean())
        spend_ratio = cur_spend / budget if budget else 1.0
        if spend_ratio <= float(cfg["budget_underspend_ratio"]):
            alerts.append(
                make_alert(
                    entity_id=str(campaign_id),
                    start=start_date,
                    end=latest_date,
                    alert_type="budget_underspend",
                    severity=Severity.MEDIUM if spend_ratio > 0.3 else Severity.HIGH,
                    metrics=["spend", "budget"],
                    current=cur_spend,
                    expected=budget,
                    calcs={"spend_to_budget_ratio": round(spend_ratio, 4)},
                    domain="media",
                )
            )

        cur_freq = float(recent["frequency"].mean())
        cur_ctr = float(recent["ctr"].mean())
        exp_ctr = baseline_mean("ctr")
        ctr_drop = -pct_change_vs_baseline(cur_ctr, exp_ctr)
        if cur_freq >= float(cfg["frequency_fatigue"]) and ctr_drop >= 0.2:
            alerts.append(
                make_alert(
                    entity_id=str(campaign_id),
                    start=start_date,
                    end=latest_date,
                    alert_type="creative_fatigue",
                    severity=Severity.HIGH if cur_freq >= 6 else Severity.MEDIUM,
                    metrics=["frequency", "ctr"],
                    current=cur_freq,
                    expected=float(cfg["frequency_fatigue"]),
                    calcs={"frequency": round(cur_freq, 4), "ctr_drop": round(ctr_drop, 4)},
                    domain="media",
                )
            )

        cur_cpa = float(recent["cpa"].mean())
        exp_cpa = baseline_mean("cpa")
        cpa_lift = pct_change_vs_baseline(cur_cpa, exp_cpa)
        if cpa_lift >= 0.35 and ctr_drop >= 0.15:
            alerts.append(
                make_alert(
                    entity_id=str(campaign_id),
                    start=start_date,
                    end=latest_date,
                    alert_type="geographic_performance_shift",
                    severity=severity_from_magnitude(cpa_lift, 0.35, 0.7),
                    metrics=["cpa", "ctr", "conversion_rate"],
                    current=cur_cpa,
                    expected=exp_cpa,
                    calcs={"cpa_lift": round(cpa_lift, 4), "ctr_drop": round(ctr_drop, 4)},
                    domain="media",
                )
            )

        conv_lift_only = pct_change_vs_baseline(cur_conv, exp_conv)
        click_lift = pct_change_vs_baseline(float(recent["clicks"].mean()), baseline_mean("clicks"))
        if conv_lift_only >= float(cfg["duplicate_conversion_spike_pct"]) and conv_lift_only > click_lift + 0.35:
            alerts.append(
                make_alert(
                    entity_id=str(campaign_id),
                    start=start_date,
                    end=latest_date,
                    alert_type="duplicate_conversion_events",
                    severity=Severity.HIGH,
                    metrics=["conversions", "clicks", "tracking_event_count"],
                    current=cur_conv,
                    expected=exp_conv,
                    calcs={
                        "conversion_lift": round(conv_lift_only, 4),
                        "click_lift": round(click_lift, 4),
                    },
                    domain="media",
                )
            )

        z = rolling_zscore(work["spend"], window).iloc[-1]
        if pd.notna(z) and abs(float(z)) >= float(cfg["severity"]["high_zscore"]) and spend_lift > 0.2:
            if not any(
                a.entity_id == campaign_id and a.alert_type == "spend_spike_no_conversion_growth" for a in alerts
            ):
                alerts.append(
                    make_alert(
                        entity_id=str(campaign_id),
                        start=start_date,
                        end=latest_date,
                        alert_type="spend_spike_no_conversion_growth",
                        severity=Severity.MEDIUM,
                        metrics=["spend"],
                        current=cur_spend,
                        expected=exp_spend,
                        calcs={"spend_zscore": round(float(z), 4)},
                        domain="media",
                    )
                )

    return _dedupe_alerts(alerts)


def _dedupe_alerts(alerts: list[CandidateAlert]) -> list[CandidateAlert]:
    rank = {Severity.LOW: 1, Severity.MEDIUM: 2, Severity.HIGH: 3, Severity.CRITICAL: 4}
    best: dict[tuple[str, str], CandidateAlert] = {}
    for alert in alerts:
        key = (alert.entity_id, alert.alert_type)
        if key not in best or rank[alert.severity] > rank[best[key].severity]:
            best[key] = alert
    return list(best.values())
