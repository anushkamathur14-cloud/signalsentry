"""Plain-language briefing cards for dashboard charts (CSM-style narrative)."""

from __future__ import annotations

from typing import Any

# Pattern library: short, non-technical copy for charts + investigation panels.
# Structure mirrors a CSM monthly review brief: insight → opportunity → action →
# next steps → expected impact → success metrics (platform-agnostic).

CHURN_COPY: dict[str, dict[str, Any]] = {
    "gradual_usage_decline": {
        "headline": "Usage is quietly sliding week over week",
        "insight": "Active use and sessions are trending down versus this account’s own baseline — a classic early churn signal before renewal.",
        "opportunity": "Re-engage the champion while there is still product habit to rebuild, instead of waiting for a cancel request.",
        "next_steps": [
            "Confirm who the day-to-day champion and exec sponsor are.",
            "Book a value-review call tied to their renewal window.",
            "Share 1–2 workflows that recreate the usage they lost.",
        ],
        "expected_impact": [
            "Stabilize weekly active users within 2–4 weeks if outreach lands.",
            "Reduce renewal risk by restoring a clear product habit.",
        ],
        "success_metrics": ["Weekly sessions", "Active users", "Feature adoption"],
        "guardrails": ["No cold outbound beyond named contacts", "Advisory only — human CSM decides"],
    },
    "sudden_usage_collapse": {
        "headline": "Usage dropped sharply — treat as urgent",
        "insight": "Activity fell far below the recent baseline in a short window. That usually means an outage, admin change, or competitive displacement — not normal seasonality.",
        "opportunity": "Same-day outreach can catch a fixable issue before the account goes dark.",
        "next_steps": [
            "Check for known incidents or admin/permission changes.",
            "Call the admin and secondary contact the same day.",
            "If competitor risk, escalate to a recovery plan with dates.",
        ],
        "expected_impact": [
            "Faster diagnosis of technical vs commercial root cause.",
            "Higher chance of recovering usage before renewal talks harden.",
        ],
        "success_metrics": ["Active users", "Weekly sessions", "Days since admin login"],
        "guardrails": ["Do not auto-message customers", "Confirm facts before promising fixes"],
    },
    "low_seat_utilization": {
        "headline": "Paid seats are mostly idle",
        "insight": "Licensed seats far exceed people actually logging in. Budget owners notice idle spend; expansion stalls when utilization stays low.",
        "opportunity": "Turn unused licenses into adoption wins — or right-size before a tough renewal conversation.",
        "next_steps": [
            "Map who has seats vs who is active.",
            "Run a short seat-optimization / enablement workshop.",
            "Agree a 30-day adoption target with the champion.",
        ],
        "expected_impact": [
            "Higher active/licensed ratio.",
            "Clearer renewal narrative (value realized vs shelfware).",
        ],
        "success_metrics": ["Active / licensed ratio", "Weekly sessions"],
        "guardrails": ["Avoid surprising the customer with a downsell first", "Lead with adoption"],
    },
    "administrator_inactivity": {
        "headline": "The admin has gone quiet",
        "insight": "Days since admin login are elevated. When the admin disengages, renewals and escalations lose an internal owner.",
        "opportunity": "Re-establish a live owner before the account becomes orphaned.",
        "next_steps": [
            "Contact the named admin and a backup stakeholder.",
            "Refresh enablement / SSO / access if login friction is the issue.",
            "Confirm an exec sponsor if the admin has left.",
        ],
        "expected_impact": [
            "Restored account ownership.",
            "Fewer surprise renewals with no internal advocate.",
        ],
        "success_metrics": ["Days since admin login", "Active users"],
        "guardrails": ["Verify contact data before outreach"],
    },
    "falling_feature_adoption": {
        "headline": "Core features are being used less",
        "insight": "Key feature adoption is declining versus baseline. Accounts that stop using differentiating features often churn even if login still happens.",
        "opportunity": "Tie underused features to a concrete ROI story for this account.",
        "next_steps": [
            "Identify which feature(s) dropped.",
            "Offer a targeted enablement session linked to their use case.",
            "Set a simple adoption checkpoint in 2 weeks.",
        ],
        "expected_impact": [
            "Recovery in feature adoption rate.",
            "Stronger stickiness ahead of renewal.",
        ],
        "success_metrics": ["Key feature adoption", "Weekly sessions"],
        "guardrails": ["Don’t push features that don’t match their plan tier"],
    },
    "repeated_support_escalations": {
        "headline": "Support load is spiking",
        "insight": "Ticket volume and severity are elevated. Repeated escalations erode trust and often precede churn or contraction.",
        "opportunity": "An executive support bridge can stop the bleed before the customer shops alternatives.",
        "next_steps": [
            "Review open high-severity tickets with Support.",
            "Open a time-boxed recovery bridge with clear owners.",
            "Share status cadence with the customer until volume normalizes.",
        ],
        "expected_impact": [
            "Lower high-severity ticket count.",
            "Improved sentiment and renewal confidence.",
        ],
        "success_metrics": ["Support ticket count", "High-severity tickets", "Avg resolution hours"],
        "guardrails": ["No automated ticket replies from this app"],
    },
    "negative_sentiment": {
        "headline": "Sentiment / NPS is sliding",
        "insight": "NPS (or equivalent sentiment) moved down versus baseline. Soft signals often lead hard cancel decisions.",
        "opportunity": "A structured listening session can convert frustration into a dated recovery plan.",
        "next_steps": [
            "Schedule a listening call; capture themes without defending.",
            "Translate themes into a written recovery plan with owners/dates.",
            "Re-check sentiment after the first delivery milestone.",
        ],
        "expected_impact": [
            "Clearer product/process gaps.",
            "Path to stabilize NPS before renewal.",
        ],
        "success_metrics": ["NPS score", "Support ticket count"],
        "guardrails": ["Advisory recommendations only"],
    },
    "benign_seasonal_decline": {
        "headline": "Seasonal dip — probably not churn",
        "insight": "Usage dipped in a period that often shows seasonal softness. Detectors still surface it so a human can confirm it’s benign.",
        "opportunity": "Avoid over-reacting; confirm seasonality and keep a light touch.",
        "next_steps": [
            "Compare to the same period last year if available.",
            "Send a light check-in only if other risk signals appear.",
            "Keep monitoring the next 2–3 weeks.",
        ],
        "expected_impact": [
            "Avoid false-alarm CSM load.",
            "Still catch true risk if the dip persists off-season.",
        ],
        "success_metrics": ["Weekly sessions", "Active users"],
        "guardrails": ["Do not treat as critical without corroboration"],
    },
}

MEDIA_COPY: dict[str, dict[str, Any]] = {
    "spend_spike_no_conversion_growth": {
        "headline": "Spend rose, conversions didn’t",
        "insight": "Budget delivery increased without a matching conversion lift. You’re paying more for roughly the same outcomes — efficiency is slipping.",
        "opportunity": "Pause blind scale-up and put spend back behind validated conversion quality.",
        "next_steps": [
            "Freeze further budget increases until diagnosis is done.",
            "Check bid strategy, audience overlap, and landing-page health.",
            "Reallocate only after conversions respond in a controlled test.",
        ],
        "expected_impact": [
            "Lower wasted spend.",
            "Restore cost-per-conversion toward baseline.",
        ],
        "success_metrics": ["Spend", "Conversions", "Cost per conversion", "Conversion rate"],
        "guardrails": ["No automatic bid/budget changes from this app", "Human media buyer approves changes"],
    },
    "cpc_increase": {
        "headline": "Each click is getting more expensive",
        "insight": "CPC moved above the campaign’s own baseline. Auction pressure, mix shift, or creative weakness usually sits underneath.",
        "opportunity": "Tighten who you compete for — or refresh creative — before CPC erodes the whole funnel.",
        "next_steps": [
            "Review auction competition and audience/query mix.",
            "Test tighter targeting or creative refresh.",
            "Watch CPC and downstream CVR together (cheap clicks that don’t convert aren’t a win).",
        ],
        "expected_impact": [
            "CPC closer to historical range.",
            "More stable cost per outcome.",
        ],
        "success_metrics": ["CPC", "CTR", "Conversion rate"],
        "guardrails": ["Keep frequency and CPM in view as guardrails"],
    },
    "cpm_increase": {
        "headline": "Impressions are costing more",
        "insight": "CPM is elevated versus baseline. Placement mix, seasonality, or frequency can all inflate impression cost.",
        "opportunity": "Fix delivery mix before pouring more budget into expensive inventory.",
        "next_steps": [
            "Inspect placement mix and frequency caps.",
            "Test alternate inventory or broader efficient audiences.",
            "Re-check CPM after one learning cycle.",
        ],
        "expected_impact": [
            "More efficient reach for the same spend.",
        ],
        "success_metrics": ["CPM", "Spend", "Frequency", "Impressions"],
        "guardrails": ["Don’t chase CPM alone if conversions are healthy"],
    },
    "conversion_rate_collapse": {
        "headline": "Traffic stopped converting",
        "insight": "Conversion rate fell hard versus expected. That can be tracking, landing page, offer, or audience quality — treat it as a funnel break.",
        "opportunity": "Stop scaling into a broken funnel; fix measurement and experience first.",
        "next_steps": [
            "Validate tracking and landing-page health.",
            "Halt aggressive scaling until CVR stabilizes.",
            "Compare creative/audience cohorts to find what still converts.",
        ],
        "expected_impact": [
            "CVR recovery toward baseline.",
            "Avoid burning budget on non-converting traffic.",
        ],
        "success_metrics": ["Conversion rate", "Conversions", "CPC"],
        "guardrails": ["Immediate human review recommended", "No auto-optimize to broken signals"],
    },
    "tracking_discrepancy": {
        "headline": "The numbers may not be trustworthy",
        "insight": "Platform vs expected conversion signals disagree. Optimizing on bad measurement makes every decision worse.",
        "opportunity": "Reconcile pixels / UTMs / server events before any performance optimization.",
        "next_steps": [
            "Audit pixel, UTM, and server-side event paths.",
            "Pause KPI-based optimization until totals reconcile.",
            "Document which number is source-of-truth going forward.",
        ],
        "expected_impact": [
            "Reliable reporting for buyers and stakeholders.",
            "Safer budget decisions once signals match.",
        ],
        "success_metrics": ["Platform conversions vs expected", "Spend delivery"],
        "guardrails": ["Do not scale while discrepancy is open"],
    },
    "budget_underspend": {
        "headline": "The campaign isn’t spending its budget",
        "insight": "Delivery is below budget pace. Caps, bids, or audience size are often constraining reach.",
        "opportunity": "Unblock delivery only after you know which constraint is real — don’t just raise budgets blindly.",
        "next_steps": [
            "Inspect bids, caps, and audience size.",
            "Fix the binding constraint first.",
            "Raise budget only after delivery recovers healthily.",
        ],
        "expected_impact": [
            "Spend closer to plan.",
            "More stable learning for the algorithm.",
        ],
        "success_metrics": ["Spend vs budget", "Impressions", "CPC"],
        "guardrails": ["Human approves cap/bid edits"],
    },
    "creative_fatigue": {
        "headline": "Creative is wearing out",
        "insight": "Frequency is up while engagement/efficiency softens — a classic fatigue pattern. More spend on tired assets usually hurts.",
        "opportunity": "Rotate hooks and proofs before further investment.",
        "next_steps": [
            "Rotate in fresh creatives (new hook / proof / CTA).",
            "Reset frequency where the platform allows.",
            "Keep a control asset to measure lift.",
        ],
        "expected_impact": [
            "CTR / CVR recovery on refreshed assets.",
            "Healthier frequency at similar spend.",
        ],
        "success_metrics": ["Frequency", "CTR", "Conversion rate", "CPC"],
        "guardrails": ["Test a small set of variants; don’t flood the account"],
    },
    "geographic_performance_shift": {
        "headline": "Some regions are dragging results",
        "insight": "Aggregate campaign averages are hiding geo winners and losers. Blended KPIs can look fine while weak regions waste spend.",
        "opportunity": "Break out geo performance and move budget toward efficient regions.",
        "next_steps": [
            "Export/review geo split for the alert window.",
            "Pause or throttle degraded regions.",
            "Reallocate toward regions still hitting target CPA/ROAS.",
        ],
        "expected_impact": [
            "Better blended efficiency without total budget cuts.",
        ],
        "success_metrics": ["CPA/ROAS by geo", "Spend by geo", "Conversion rate"],
        "guardrails": ["Confirm sample size before cutting a region"],
    },
    "duplicate_conversion_events": {
        "headline": "Conversions may be double-counted",
        "insight": "Event volume looks inflated versus spend/click reality. Duplicate firing makes performance look better than it is.",
        "opportunity": "Deduplicate event paths and restate results before optimizing.",
        "next_steps": [
            "Trace conversion firing paths (browser + server).",
            "Deduplicate and restate the reporting window.",
            "Only then resume optimization against clean events.",
        ],
        "expected_impact": [
            "Honest conversion totals.",
            "Budgets steered by real outcomes.",
        ],
        "success_metrics": ["Conversions", "Conversion rate", "Cost per conversion"],
        "guardrails": ["Immediate human review — do not scale on inflated events"],
    },
    "benign_weekend_variation": {
        "headline": "Weekend pattern — likely normal",
        "insight": "Metrics moved in a way that often matches weekend behavior. Flagged so you can confirm it’s not a real break.",
        "opportunity": "Avoid major structural changes on weekend noise alone.",
        "next_steps": [
            "Compare weekday vs weekend baselines.",
            "Act only if the pattern persists into the business week.",
        ],
        "expected_impact": [
            "Fewer false-alarm optimizations.",
        ],
        "success_metrics": ["Spend", "CTR", "Conversions"],
        "guardrails": ["Low urgency unless paired with another alert"],
    },
    "benign_seasonal_variation": {
        "headline": "Seasonal shift — confirm before reacting",
        "insight": "Performance moved in a period that often shows seasonal swings. Confirm before rewriting the media plan.",
        "opportunity": "Separate seasonality from structural problems.",
        "next_steps": [
            "Compare to prior-year / prior-season if available.",
            "Keep guardrails on CPA and delivery while you watch.",
        ],
        "expected_impact": [
            "Avoid over-correcting seasonal noise.",
        ],
        "success_metrics": ["Spend", "CPA", "Conversion rate"],
        "guardrails": ["Directional only without year-over-year context"],
    },
}


def _delta_phrase(current: float, expected: float) -> str:
    if expected == 0:
        return f"current={current:.2f} (no baseline)"
    pct = ((current - expected) / abs(expected)) * 100
    direction = "above" if pct > 0 else "below"
    return f"{abs(pct):.0f}% {direction} expected ({current:.2f} vs {expected:.2f})"


def build_churn_briefing(alert: dict[str, Any], investigation: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble a CSM-style card from detector alert + optional investigation."""
    inv = investigation or {}
    alert_type = alert.get("alert_type") or ""
    copy = CHURN_COPY.get(alert_type, {
        "headline": alert_type.replace("_", " ").title() or "Account risk signal",
        "insight": "A detector flagged unusual account behavior versus its own baseline.",
        "opportunity": "Review with the CSM before the risk compounds.",
        "next_steps": ["Inspect the metric chart", "Validate with the customer context", "Choose a human-approved action"],
        "expected_impact": ["Clearer next action with measured follow-up"],
        "success_metrics": alert.get("metrics_involved") or [],
        "guardrails": ["Advisory only"],
    })
    agent_action = inv.get("recommended_csm_action")
    return {
        "domain": "churn",
        "entity_id": alert.get("entity_id"),
        "alert_type": alert_type,
        "severity": alert.get("severity") or inv.get("risk_level"),
        "headline": copy["headline"],
        "insight": copy["insight"],
        "opportunity": copy["opportunity"],
        "what_changed": _delta_phrase(float(alert.get("current_value") or 0), float(alert.get("expected_value") or 0)),
        "metrics_involved": alert.get("metrics_involved") or copy.get("success_metrics") or [],
        "diagnosis": (inv.get("likely_causes") or ["Hypothesis pending investigation."])[:3],
        "evidence": inv.get("evidence") or [],
        "recommended_action": agent_action or (
            copy["next_steps"][0] if copy.get("next_steps") else "Review with CSM."
        ),
        "next_steps": copy["next_steps"],
        "expected_impact": copy["expected_impact"],
        "success_metrics": copy.get("success_metrics") or alert.get("metrics_involved") or [],
        "guardrails": copy.get("guardrails") or [],
        "confidence": inv.get("confidence"),
        "risk_score": inv.get("risk_score"),
        "data_limitations": inv.get("data_limitations") or [
            "Synthetic demo data — directional expectations only."
        ],
        "window": f"{alert.get('start_date')} → {alert.get('end_date')}",
    }


def build_media_briefing(alert: dict[str, Any], investigation: dict[str, Any] | None = None) -> dict[str, Any]:
    inv = investigation or {}
    alert_type = alert.get("alert_type") or ""
    copy = MEDIA_COPY.get(alert_type, {
        "headline": alert_type.replace("_", " ").title() or "Campaign anomaly",
        "insight": "A detector flagged unusual campaign behavior versus its own baseline.",
        "opportunity": "Diagnose before changing bids or budgets.",
        "next_steps": ["Inspect the time series", "Validate tracking", "Propose a human-approved change"],
        "expected_impact": ["Safer media decisions once the signal is understood"],
        "success_metrics": alert.get("metrics_involved") or [],
        "guardrails": ["No automatic campaign changes"],
    })
    return {
        "domain": "media",
        "entity_id": alert.get("entity_id"),
        "alert_type": alert_type,
        "severity": alert.get("severity") or inv.get("severity"),
        "headline": copy["headline"],
        "insight": inv.get("anomaly_summary") or copy["insight"],
        "opportunity": copy["opportunity"],
        "what_changed": _delta_phrase(float(alert.get("current_value") or 0), float(alert.get("expected_value") or 0)),
        "metrics_involved": alert.get("metrics_involved") or copy.get("success_metrics") or [],
        "diagnosis": (inv.get("likely_causes") or ["Hypothesis pending investigation."])[:3],
        "evidence": inv.get("evidence") or [],
        "recommended_action": inv.get("recommended_action") or copy["next_steps"][0],
        "next_steps": copy["next_steps"],
        "expected_impact": copy["expected_impact"],
        "success_metrics": copy.get("success_metrics") or alert.get("metrics_involved") or [],
        "guardrails": copy.get("guardrails") or [],
        "confidence": inv.get("confidence"),
        "immediate_review": inv.get("requires_immediate_human_review"),
        "data_limitations": inv.get("data_limitations") or [
            "Synthetic demo data — directional expectations only."
        ],
        "window": f"{alert.get('start_date')} → {alert.get('end_date')}",
    }
