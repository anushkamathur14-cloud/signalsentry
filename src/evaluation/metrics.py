"""Evaluate detections against injected ground truth."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd

from src.models.schemas import CandidateAlert, EvaluationMetrics, Severity


# Map detector alert types to ground-truth pattern names (identity for most).
CHURN_TYPE_ALIASES = {
    "gradual_usage_decline": {"gradual_usage_decline"},
    "sudden_usage_collapse": {"sudden_usage_collapse"},
    "low_seat_utilization": {"low_seat_utilization"},
    "administrator_inactivity": {"administrator_inactivity"},
    "falling_feature_adoption": {"falling_feature_adoption"},
    "repeated_support_escalations": {"repeated_support_escalations"},
    "negative_sentiment": {"negative_sentiment"},
}

MEDIA_TYPE_ALIASES = {
    "spend_spike_no_conversion_growth": {"spend_spike_no_conversion_growth"},
    "cpc_increase": {"cpc_increase"},
    "cpm_increase": {"cpm_increase"},
    "conversion_rate_collapse": {"conversion_rate_collapse"},
    "tracking_discrepancy": {"tracking_discrepancy"},
    "budget_underspend": {"budget_underspend"},
    "creative_fatigue": {"creative_fatigue"},
    "geographic_performance_shift": {"geographic_performance_shift"},
    "duplicate_conversion_events": {"duplicate_conversion_events"},
}

BENIGN_PATTERNS = {
    "benign_seasonal_decline",
    "benign_weekend_variation",
    "benign_seasonal_variation",
}


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["start_date"] = pd.to_datetime(out["start_date"]).dt.date
    out["end_date"] = pd.to_datetime(out["end_date"]).dt.date
    return out


def _match(
    alert: CandidateAlert,
    truth_row: pd.Series,
    aliases: dict[str, set[str]],
    tolerance_days: int,
) -> bool:
    entity_col = "account_id" if "account_id" in truth_row.index else "campaign_id"
    if alert.entity_id != truth_row[entity_col]:
        return False
    patterns = aliases.get(alert.alert_type, {alert.alert_type})
    if truth_row["pattern"] not in patterns:
        return False
    # Overlapping windows with tolerance
    a_start = alert.start_date - timedelta(days=tolerance_days)
    a_end = alert.end_date + timedelta(days=tolerance_days)
    t_start = truth_row["start_date"]
    t_end = truth_row["end_date"]
    return a_start <= t_end and a_end >= t_start


def evaluate_detections(
    alerts: list[CandidateAlert],
    ground_truth: pd.DataFrame,
    *,
    domain: str,
    tolerance_days: int = 14,
) -> EvaluationMetrics:
    truth = _parse_dates(ground_truth)
    aliases = CHURN_TYPE_ALIASES if domain == "churn" else MEDIA_TYPE_ALIASES

    # Positive ground-truth rows exclude benign patterns (should not be flagged).
    positives = truth[~truth["pattern"].isin(BENIGN_PATTERNS)].reset_index(drop=True)
    benign = truth[truth["pattern"].isin(BENIGN_PATTERNS)].reset_index(drop=True)

    matched_truth: set[int] = set()
    matched_alerts: set[int] = set()
    fp_indices: list[int] = []

    for a_i, alert in enumerate(alerts):
        found = False
        for t_i, row in positives.iterrows():
            if t_i in matched_truth:
                continue
            if _match(alert, row, aliases, tolerance_days):
                matched_truth.add(int(t_i))
                matched_alerts.add(a_i)
                found = True
                break
        if not found:
            # Alerts on benign-labeled entities for seasonal types count as FP specially
            fp_indices.append(a_i)

    tp = len(matched_truth)
    fn = len(positives) - tp
    fp = len(fp_indices)
    # True negatives: benign entities that were not alerted for their benign pattern window
    benign_entities = set(benign["account_id" if "account_id" in benign.columns else "campaign_id"])
    alerted_entities = {alerts[i].entity_id for i in fp_indices}
    # Approximate TN as benign entities with no alert at all
    tn = len(benign_entities - {a.entity_id for a in alerts})

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    by_type: dict[str, dict[str, float]] = {}
    for pattern, group in positives.groupby("pattern"):
        idxs = list(group.index)
        pattern_tp = sum(1 for i in idxs if i in matched_truth)
        pattern_fn = len(idxs) - pattern_tp
        # FPs approximated as alerts of matching alias type that didn't match truth
        alias_types = {k for k, v in aliases.items() if pattern in v}
        pattern_fp = sum(
            1
            for i in fp_indices
            if alerts[i].alert_type in alias_types
        )
        p = pattern_tp / (pattern_tp + pattern_fp) if (pattern_tp + pattern_fp) else 0.0
        r = pattern_tp / (pattern_tp + pattern_fn) if (pattern_tp + pattern_fn) else 0.0
        by_type[str(pattern)] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round((2 * p * r / (p + r)) if (p + r) else 0.0, 4),
            "support": float(len(idxs)),
        }

    by_severity: dict[str, dict[str, float]] = {}
    for sev in Severity:
        sev_alerts = [(i, a) for i, a in enumerate(alerts) if a.severity == sev]
        if not sev_alerts:
            continue
        sev_tp = sum(1 for i, _ in sev_alerts if i in matched_alerts)
        sev_fp = sum(1 for i, _ in sev_alerts if i not in matched_alerts)
        p = sev_tp / (sev_tp + sev_fp) if (sev_tp + sev_fp) else 0.0
        by_severity[sev.value] = {
            "precision": round(p, 4),
            "alert_count": float(len(sev_alerts)),
            "true_positives": float(sev_tp),
            "false_positives": float(sev_fp),
        }

    return EvaluationMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        false_positive_rate=round(fpr, 4),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        by_anomaly_type=by_type,
        by_severity=by_severity,
    )


def combine_evaluations(parts: dict[str, EvaluationMetrics]) -> dict[str, Any]:
    return {name: metrics.model_dump(mode="json") for name, metrics in parts.items()}
