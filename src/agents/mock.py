"""Mock investigators — deterministic, offline, facts-only from alert payload."""

from __future__ import annotations

from src.models.schemas import (
    CampaignInvestigation,
    CandidateAlert,
    ChurnInvestigation,
    RiskLevel,
    Severity,
)


SEVERITY_TO_RISK = {
    Severity.LOW: (35.0, RiskLevel.LOW),
    Severity.MEDIUM: (58.0, RiskLevel.MEDIUM),
    Severity.HIGH: (78.0, RiskLevel.HIGH),
    Severity.CRITICAL: (92.0, RiskLevel.CRITICAL),
}


CHURN_ACTIONS = {
    "gradual_usage_decline": "Schedule a value-review call; confirm champion status and upcoming renewal risks.",
    "sudden_usage_collapse": "Trigger same-day CSM outreach; verify outage, admin change, or competitor displacement.",
    "low_seat_utilization": "Run a seat-optimization workshop and share adoption playbooks for idle licenses.",
    "administrator_inactivity": "Contact the named admin and secondary contacts; refresh enablement materials.",
    "falling_feature_adoption": "Offer targeted feature enablement on underused capabilities tied to ROI.",
    "repeated_support_escalations": "Open an executive support bridge and review open P1/P2 tickets with CS + Support.",
    "negative_sentiment": "Conduct a listening session; capture product gaps and set a recovery plan with dates.",
}


MEDIA_ACTIONS = {
    "spend_spike_no_conversion_growth": "Pause scale-up; audit bid strategy, audience overlap, and landing-page health before increasing budget.",
    "cpc_increase": "Review auction competition and query/audience mix; tighten targeting or refresh creatives.",
    "cpm_increase": "Check placement mix and frequency caps; test broader or alternate inventory.",
    "conversion_rate_collapse": "Validate tracking and landing pages; halt aggressive scaling until CVR stabilizes.",
    "tracking_discrepancy": "Immediately audit pixels/UTMs/CAPI; do not optimize to conversion KPIs until reconciled.",
    "budget_underspend": "Inspect delivery constraints, bids, and audience size; adjust caps only after diagnosis.",
    "creative_fatigue": "Rotate creatives, reset frequency, and refresh hooks before further spend.",
    "geographic_performance_shift": "Break out geo performance and reallocate budget away from degraded regions.",
    "duplicate_conversion_events": "Deduplicate conversion firing paths and restate performance before optimization.",
}


def investigate_churn_mock(alert: CandidateAlert, metric_context: dict | None = None) -> ChurnInvestigation:
    score, level = SEVERITY_TO_RISK[alert.severity]
    calcs = alert.supporting_calculations
    evidence = [
        f"Detector `{alert.alert_type}` fired for {alert.entity_id} between {alert.start_date} and {alert.end_date}.",
        f"Metrics involved: {', '.join(alert.metrics_involved)}.",
        f"Current value={alert.current_value:.4f} vs expected={alert.expected_value:.4f}.",
    ]
    if calcs:
        evidence.append(f"Supporting calculations: {calcs}.")

    hypotheses = [
        f"Hypothesis (not verified beyond supplied metrics): `{alert.alert_type}` pattern is consistent with the detected metric movement.",
    ]
    if metric_context:
        hypotheses.append("Additional supplied metric window was considered; no external data was fetched.")

    limitations = [
        "Mock investigator used — no LLM inference was performed.",
        "Claims are limited to detector outputs and optional metric summaries supplied in-process.",
        "Recommendations are advisory and require human review.",
    ]
    return ChurnInvestigation(
        account_id=alert.entity_id,
        risk_score=score,
        risk_level=level,
        evidence=evidence,
        likely_causes=hypotheses,
        recommended_csm_action=CHURN_ACTIONS.get(
            alert.alert_type,
            "Review account health metrics with the CSM and confirm next-best action manually.",
        ),
        confidence=0.62 if alert.severity in {Severity.LOW, Severity.MEDIUM} else 0.74,
        data_limitations=limitations,
    )


def investigate_campaign_mock(alert: CandidateAlert, metric_context: dict | None = None) -> CampaignInvestigation:
    calcs = alert.supporting_calculations
    evidence = [
        f"Detector `{alert.alert_type}` fired for {alert.entity_id} between {alert.start_date} and {alert.end_date}.",
        f"Metrics involved: {', '.join(alert.metrics_involved)}.",
        f"Current value={alert.current_value:.4f} vs expected={alert.expected_value:.4f}.",
    ]
    if calcs:
        evidence.append(f"Supporting calculations: {calcs}.")

    causes = [
        f"Hypothesis: observed `{alert.alert_type}` may explain the metric deviation in the supplied window.",
    ]
    if metric_context:
        causes.append("Hypothesis constrained to the provided campaign metric context only.")

    immediate = alert.severity in {Severity.HIGH, Severity.CRITICAL} or alert.alert_type in {
        "tracking_discrepancy",
        "duplicate_conversion_events",
        "conversion_rate_collapse",
    }

    return CampaignInvestigation(
        campaign_id=alert.entity_id,
        severity=alert.severity,
        anomaly_summary=(
            f"{alert.alert_type.replace('_', ' ').title()} detected "
            f"(current={alert.current_value:.4f}, expected={alert.expected_value:.4f})."
        ),
        evidence=evidence,
        likely_causes=causes,
        recommended_action=MEDIA_ACTIONS.get(
            alert.alert_type,
            "Investigate the flagged metrics and confirm remediation with a media buyer before changing delivery.",
        ),
        requires_immediate_human_review=immediate,
        confidence=0.6 if alert.severity == Severity.LOW else 0.72,
        data_limitations=[
            "Mock investigator used — no LLM inference was performed.",
            "No live ad-platform APIs were queried.",
            "Recommendations are advisory; the app does not change campaigns automatically.",
        ],
    )
