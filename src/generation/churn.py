"""Synthetic customer churn data generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

PLAN_TIERS = ["starter", "growth", "enterprise"]
TIER_BASE = {
    "starter": {"licensed": 10, "sessions": 40, "adoption": 0.45, "nps": 55},
    "growth": {"licensed": 40, "sessions": 160, "adoption": 0.58, "nps": 48},
    "enterprise": {"licensed": 200, "sessions": 700, "adoption": 0.72, "nps": 42},
}


@dataclass(frozen=True)
class ChurnInjection:
    account_id: str
    pattern: str
    start_date: date
    end_date: date
    severity: str
    notes: str


def _week_starts(start: date, n_weeks: int) -> list[date]:
    return [start + timedelta(weeks=i) for i in range(n_weeks)]


def _seasonal_factor(d: date) -> float:
    # Mild summer / late-year dip for SaaS usage.
    if d.month in (6, 7):
        return 0.88
    if d.month == 12:
        return 0.90
    if d.month in (1, 2):
        return 0.95
    return 1.0


def generate_churn_dataset(
    seed: int = 42,
    n_accounts: int = 100,
    n_weeks: int = 52,
    start_date: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (weekly account metrics, ground-truth injection labels)."""
    rng = np.random.default_rng(seed)
    start = start_date or date(2024, 1, 1)
    weeks = _week_starts(start, n_weeks)
    renewal_offsets = rng.integers(30, 360, size=n_accounts)

    # Assign injection patterns to specific accounts (last few are controls / benign).
    injection_plan: list[tuple[int, str, str]] = [
        (0, "gradual_usage_decline", "high"),
        (1, "low_seat_utilization", "medium"),
        (2, "administrator_inactivity", "high"),
        (3, "falling_feature_adoption", "medium"),
        (4, "repeated_support_escalations", "high"),
        (5, "negative_sentiment", "medium"),
        (6, "sudden_usage_collapse", "critical"),
        (7, "benign_seasonal_decline", "low"),
        (8, "gradual_usage_decline", "medium"),
        (9, "repeated_support_escalations", "medium"),
        (10, "low_seat_utilization", "low"),
        (11, "sudden_usage_collapse", "high"),
        (12, "falling_feature_adoption", "high"),
        (13, "administrator_inactivity", "medium"),
        (14, "benign_seasonal_decline", "low"),
    ]

    rows: list[dict[str, Any]] = []
    labels: list[ChurnInjection] = []

    for acct_idx in range(n_accounts):
        account_id = f"ACC-{acct_idx:03d}"
        tier = PLAN_TIERS[acct_idx % len(PLAN_TIERS)]
        base = TIER_BASE[tier]
        licensed = int(base["licensed"] * float(rng.uniform(0.8, 1.25)))
        noise_scale = float(rng.uniform(0.05, 0.12))
        pattern = None
        severity = "low"
        for idx, pat, sev in injection_plan:
            if idx == acct_idx:
                pattern = pat
                severity = sev
                break

        inject_start_week = 28 if pattern else None
        if pattern == "benign_seasonal_decline":
            inject_start_week = 22  # June-ish
        if pattern == "sudden_usage_collapse":
            inject_start_week = 40

        admin_days = float(rng.integers(1, 14))
        nps = float(base["nps"] + rng.normal(0, 5))
        adoption = float(base["adoption"])
        active_ratio = float(rng.uniform(0.55, 0.85))

        for w_i, week in enumerate(weeks):
            season = _seasonal_factor(week)
            # Healthy baseline trajectory
            active = max(1, int(licensed * active_ratio * season * (1 + rng.normal(0, noise_scale))))
            sessions = max(1.0, base["sessions"] * (active / max(licensed, 1)) * season * (1 + rng.normal(0, noise_scale)))
            tickets = max(0, int(rng.poisson(0.4 if tier != "enterprise" else 0.8)))
            high_sev = 1 if tickets and rng.random() < 0.15 else 0
            resolution = float(rng.uniform(4, 24))
            renewal_proximity = int(max(0, renewal_offsets[acct_idx] - w_i * 7))

            # Drift admin login outward slowly for realism
            admin_days = max(0.0, admin_days + float(rng.normal(0.2, 0.8)))
            adoption = float(np.clip(adoption + rng.normal(0, 0.01), 0.05, 0.95))
            nps = float(np.clip(nps + rng.normal(0, 1.2), 0, 100))

            if pattern and inject_start_week is not None and w_i >= inject_start_week:
                progress = (w_i - inject_start_week) / max(1, n_weeks - inject_start_week)
                if pattern == "gradual_usage_decline":
                    factor = 1.0 - 0.55 * progress
                    active = max(1, int(active * factor))
                    sessions *= factor
                    adoption *= 1.0 - 0.35 * progress
                elif pattern == "low_seat_utilization":
                    active = max(1, int(licensed * 0.18 * (1 - 0.2 * progress)))
                    sessions *= 0.35
                elif pattern == "administrator_inactivity":
                    admin_days = 20 + progress * 70
                elif pattern == "falling_feature_adoption":
                    adoption = max(0.05, adoption * (1.0 - 0.6 * progress))
                elif pattern == "repeated_support_escalations":
                    tickets = int(2 + progress * 6 + rng.integers(0, 2))
                    high_sev = int(1 + progress * 3)
                    resolution = 30 + progress * 40
                elif pattern == "negative_sentiment":
                    nps = max(5.0, 40 - progress * 35 + rng.normal(0, 2))
                elif pattern == "sudden_usage_collapse":
                    if w_i >= inject_start_week:
                        active = max(1, int(active * 0.25))
                        sessions *= 0.2
                        adoption *= 0.5
                elif pattern == "benign_seasonal_decline":
                    # Extra summer dip only — should not be treated as churn by detectors with seasonal guards
                    if week.month in (6, 7):
                        active = max(1, int(active * 0.82))
                        sessions *= 0.82

            rows.append(
                {
                    "account_id": account_id,
                    "date": week,
                    "plan_tier": tier,
                    "licensed_users": licensed,
                    "active_users": int(active),
                    "weekly_sessions": float(round(sessions, 2)),
                    "key_feature_adoption": float(round(adoption, 4)),
                    "days_since_admin_login": float(round(admin_days, 2)),
                    "support_ticket_count": int(tickets),
                    "high_severity_ticket_count": int(high_sev),
                    "avg_resolution_hours": float(round(resolution, 2)),
                    "nps_score": float(round(nps, 2)),
                    "renewal_proximity_days": renewal_proximity,
                }
            )

        if pattern and inject_start_week is not None:
            labels.append(
                ChurnInjection(
                    account_id=account_id,
                    pattern=pattern,
                    start_date=weeks[inject_start_week],
                    end_date=weeks[-1],
                    severity=severity,
                    notes=f"Injected {pattern} starting week index {inject_start_week}",
                )
            )

    metrics = pd.DataFrame(rows)
    truth = pd.DataFrame([label.__dict__ for label in labels])
    return metrics, truth
