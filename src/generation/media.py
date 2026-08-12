"""Synthetic paid-media campaign data generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

PLATFORMS = ["google_ads", "meta_ads", "linkedin_ads"]
OBJECTIVES = ["awareness", "traffic", "leads", "conversions"]


@dataclass(frozen=True)
class CampaignInjection:
    campaign_id: str
    pattern: str
    start_date: date
    end_date: date
    severity: str
    notes: str


def generate_media_dataset(
    seed: int = 42,
    n_campaigns: int = 40,
    n_days: int = 183,
    start_date: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (daily campaign metrics, ground-truth injection labels)."""
    rng = np.random.default_rng(seed + 7)
    start = start_date or date(2024, 7, 1)
    days = [start + timedelta(days=i) for i in range(n_days)]

    # Pattern catalog — assigned to random campaigns per seed so each regen is a new world.
    pattern_specs: list[tuple[str, str]] = [
        ("spend_spike_no_conversion_growth", "high"),
        ("cpc_increase", "medium"),
        ("cpm_increase", "medium"),
        ("conversion_rate_collapse", "high"),
        ("tracking_discrepancy", "critical"),
        ("budget_underspend", "medium"),
        ("creative_fatigue", "high"),
        ("geographic_performance_shift", "medium"),
        ("duplicate_conversion_events", "high"),
        ("benign_weekend_variation", "low"),
        ("benign_seasonal_variation", "low"),
        ("spend_spike_no_conversion_growth", "medium"),
        ("conversion_rate_collapse", "medium"),
        ("creative_fatigue", "medium"),
        ("tracking_discrepancy", "high"),
    ]
    n_inject = min(len(pattern_specs), n_campaigns)
    target_campaigns = rng.choice(n_campaigns, size=n_inject, replace=False)
    injection_by_idx = {
        int(c_id): pattern_specs[i] for i, c_id in enumerate(target_campaigns)
    }

    rows: list[dict[str, Any]] = []
    labels: list[CampaignInjection] = []

    for c_i in range(n_campaigns):
        campaign_id = f"CMP-{c_i:03d}"
        platform = PLATFORMS[c_i % len(PLATFORMS)]
        objective = OBJECTIVES[c_i % len(OBJECTIVES)]
        daily_budget = float(rng.choice([80, 120, 200, 350, 500]))
        base_cpc = {"google_ads": 1.8, "meta_ads": 1.2, "linkedin_ads": 4.5}[platform]
        base_cvr = {"awareness": 0.005, "traffic": 0.01, "leads": 0.03, "conversions": 0.04}[objective]
        base_ctr = float(rng.uniform(0.008, 0.035))
        frequency = float(rng.uniform(1.2, 2.5))

        pattern = None
        severity = "low"
        if c_i in injection_by_idx:
            pattern, severity = injection_by_idx[c_i]

        inject_start = 90
        if pattern == "benign_weekend_variation":
            inject_start = 0
        if pattern == "benign_seasonal_variation":
            inject_start = 120  # into Nov/Dec depending on start
        if pattern in {"sudden", "spend_spike_no_conversion_growth"}:
            inject_start = 100

        for d_i, day in enumerate(days):
            weekend = day.weekday() >= 5
            season = 1.15 if day.month in (11, 12) else (0.92 if day.month in (8,) else 1.0)
            weekend_factor = 0.75 if weekend else 1.0

            spend = daily_budget * float(rng.uniform(0.85, 1.05)) * weekend_factor * season
            cpc = base_cpc * float(rng.uniform(0.9, 1.1)) * (1.05 if weekend else 1.0)
            clicks = max(1.0, spend / max(cpc, 0.01))
            impressions = max(clicks / max(base_ctr, 1e-4), clicks)
            cpm = (spend / impressions) * 1000 if impressions else 0.0
            cvr = base_cvr * float(rng.uniform(0.85, 1.15))
            conversions = max(0.0, clicks * cvr)
            revenue = conversions * float(rng.uniform(40, 120))
            tracking_events = conversions * float(rng.uniform(0.95, 1.05))
            ctr = clicks / impressions if impressions else 0.0
            cpa = spend / conversions if conversions > 0 else spend
            freq = frequency * float(rng.uniform(0.95, 1.05))

            # Mild healthy weekend dips — not labeled unless pattern is benign_weekend
            if weekend and pattern != "benign_weekend_variation":
                spend *= 0.9
                conversions *= 0.9

            if pattern and d_i >= inject_start:
                progress = (d_i - inject_start) / max(1, n_days - inject_start)
                if pattern == "spend_spike_no_conversion_growth":
                    spend *= 1.0 + 0.9 * min(1.0, progress * 2)
                    # conversions intentionally flat-ish
                    conversions *= 1.0 + 0.05 * progress
                elif pattern == "cpc_increase":
                    cpc *= 1.0 + 0.8 * progress
                    clicks = max(1.0, spend / cpc)
                    impressions = max(clicks / max(base_ctr, 1e-4), clicks)
                elif pattern == "cpm_increase":
                    impressions *= max(0.4, 1.0 - 0.5 * progress)
                    cpm = (spend / impressions) * 1000
                elif pattern == "conversion_rate_collapse":
                    cvr *= max(0.2, 1.0 - 0.7 * progress)
                    conversions = clicks * cvr
                elif pattern == "tracking_discrepancy":
                    tracking_events = conversions * (0.4 - 0.2 * progress)
                elif pattern == "budget_underspend":
                    spend = daily_budget * max(0.2, 0.45 - 0.2 * progress)
                elif pattern == "creative_fatigue":
                    freq = 3.5 + 3.0 * progress
                    ctr = base_ctr * max(0.35, 1.0 - 0.55 * progress)
                    clicks = impressions * ctr
                    conversions = clicks * cvr
                elif pattern == "geographic_performance_shift":
                    # Proxy: CPA rises while CTR falls (geo mix shift)
                    cvr *= max(0.35, 1.0 - 0.5 * progress)
                    conversions = clicks * cvr
                    cpa = spend / conversions if conversions > 0 else spend * 2
                elif pattern == "duplicate_conversion_events":
                    conversions = clicks * cvr * (1.0 + 1.5 * progress)
                    tracking_events = conversions * 1.1
                elif pattern == "benign_weekend_variation":
                    if weekend:
                        spend *= 0.7
                        clicks *= 0.7
                        conversions *= 0.7
                elif pattern == "benign_seasonal_variation":
                    if day.month in (11, 12):
                        spend *= 1.25
                        conversions *= 1.22

            # Recompute derived metrics after mutations
            impressions = max(impressions, 1.0)
            clicks = max(clicks, 0.0)
            ctr = clicks / impressions if impressions else 0.0
            cpm = (spend / impressions) * 1000 if impressions else 0.0
            cpc = spend / clicks if clicks > 0 else cpc
            cvr = conversions / clicks if clicks > 0 else 0.0
            cpa = spend / conversions if conversions > 0 else spend
            revenue = conversions * float(rng.uniform(40, 120)) if pattern != "duplicate_conversion_events" else revenue

            rows.append(
                {
                    "campaign_id": campaign_id,
                    "date": day,
                    "platform": platform,
                    "objective": objective,
                    "budget": float(round(daily_budget, 2)),
                    "spend": float(round(spend, 2)),
                    "impressions": float(round(impressions, 2)),
                    "clicks": float(round(clicks, 2)),
                    "conversions": float(round(conversions, 4)),
                    "revenue": float(round(revenue, 2)),
                    "cpm": float(round(cpm, 4)),
                    "cpc": float(round(cpc, 4)),
                    "ctr": float(round(ctr, 6)),
                    "cpa": float(round(cpa, 4)),
                    "conversion_rate": float(round(cvr, 6)),
                    "frequency": float(round(freq, 4)),
                    "tracking_event_count": float(round(tracking_events, 4)),
                }
            )

        if pattern:
            labels.append(
                CampaignInjection(
                    campaign_id=campaign_id,
                    pattern=pattern,
                    start_date=days[inject_start],
                    end_date=days[-1],
                    severity=severity,
                    notes=f"Injected {pattern} starting day index {inject_start}",
                )
            )

    metrics = pd.DataFrame(rows)
    truth = pd.DataFrame([label.__dict__ for label in labels])
    return metrics, truth
