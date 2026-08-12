"""Core tests for SignalSentry."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.agents.mock import investigate_campaign_mock, investigate_churn_mock
from src.detection import detect_churn_alerts, detect_media_alerts, load_thresholds
from src.generation.churn import generate_churn_dataset
from src.generation.media import generate_media_dataset
from src.models.schemas import CampaignInvestigation, CandidateAlert, ChurnInvestigation, Severity
from src.paths import GENERATED_DIR, GROUND_TRUTH_DIR, OUTPUTS_DIR
from src.run_analysis import run_analysis


@pytest.fixture(scope="module")
def churn_data():
    return generate_churn_dataset(seed=42)


@pytest.fixture(scope="module")
def media_data():
    return generate_media_dataset(seed=42)


def test_churn_reproducible_with_seed():
    a, _ = generate_churn_dataset(seed=42)
    b, _ = generate_churn_dataset(seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_media_reproducible_with_seed():
    a, _ = generate_media_dataset(seed=42)
    b, _ = generate_media_dataset(seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_produce_different_worlds():
    a, a_truth = generate_churn_dataset(seed=42)
    b, b_truth = generate_churn_dataset(seed=99)
    assert not a.equals(b)
    # Same anomaly types, different account assignment
    assert set(a_truth["pattern"]) == set(b_truth["pattern"])
    assert set(a_truth["account_id"]) != set(b_truth["account_id"])


def test_ground_truth_contains_injected_patterns(churn_data, media_data):
    _, churn_truth = churn_data
    _, media_truth = media_data
    churn_patterns = set(churn_truth["pattern"])
    media_patterns = set(media_truth["pattern"])
    for required in {
        "gradual_usage_decline",
        "low_seat_utilization",
        "administrator_inactivity",
        "falling_feature_adoption",
        "repeated_support_escalations",
        "negative_sentiment",
        "sudden_usage_collapse",
        "benign_seasonal_decline",
    }:
        assert required in churn_patterns
    for required in {
        "spend_spike_no_conversion_growth",
        "cpc_increase",
        "cpm_increase",
        "conversion_rate_collapse",
        "tracking_discrepancy",
        "budget_underspend",
        "creative_fatigue",
        "geographic_performance_shift",
        "duplicate_conversion_events",
        "benign_weekend_variation",
        "benign_seasonal_variation",
    }:
        assert required in media_patterns


def test_detectors_find_representative_anomalies(churn_data, media_data):
    churn_df, churn_truth = churn_data
    media_df, media_truth = media_data
    thresholds = load_thresholds()
    churn_alerts = detect_churn_alerts(churn_df, thresholds)
    media_alerts = detect_media_alerts(media_df, thresholds)

    churn_entities = {a.entity_id for a in churn_alerts}
    media_entities = {a.entity_id for a in media_alerts}

    # Injected non-benign accounts/campaigns should largely be surfaced
    positive_accounts = set(
        churn_truth.loc[~churn_truth["pattern"].str.startswith("benign"), "account_id"]
    )
    positive_campaigns = set(
        media_truth.loc[~media_truth["pattern"].str.startswith("benign"), "campaign_id"]
    )
    assert len(positive_accounts & churn_entities) >= 5
    assert len(positive_campaigns & media_entities) >= 5


def test_benign_seasonality_not_obvious_false_positive(churn_data, media_data):
    churn_df, churn_truth = churn_data
    media_df, media_truth = media_data
    thresholds = load_thresholds()
    churn_alerts = detect_churn_alerts(churn_df, thresholds)
    media_alerts = detect_media_alerts(media_df, thresholds)

    benign_accounts = set(
        churn_truth.loc[churn_truth["pattern"] == "benign_seasonal_decline", "account_id"]
    )
    # Benign seasonal accounts should not get sudden_usage_collapse / gradual as the only signal flood
    benign_churn_hits = [
        a for a in churn_alerts if a.entity_id in benign_accounts and a.alert_type in {"gradual_usage_decline", "sudden_usage_collapse"}
    ]
    assert len(benign_churn_hits) <= 1

    benign_campaigns = set(
        media_truth.loc[media_truth["pattern"].str.startswith("benign"), "campaign_id"]
    )
    benign_media_hits = [a for a in media_alerts if a.entity_id in benign_campaigns]
    # Allow at most a small number of residual FPs
    assert len(benign_media_hits) <= 3


def test_pydantic_outputs_validate():
    alert = CandidateAlert(
        entity_id="ACC-000",
        start_date="2024-08-01",
        end_date="2024-08-28",
        alert_type="gradual_usage_decline",
        severity=Severity.HIGH,
        metrics_involved=["weekly_sessions"],
        current_value=10.0,
        expected_value=40.0,
        supporting_calculations={"pct_drop": 0.75},
        domain="churn",
    )
    churn = investigate_churn_mock(alert)
    assert isinstance(churn, ChurnInvestigation)
    ChurnInvestigation.model_validate(churn.model_dump())

    media_alert = alert.model_copy(
        update={"entity_id": "CMP-000", "alert_type": "tracking_discrepancy", "domain": "media"}
    )
    camp = investigate_campaign_mock(media_alert)
    assert isinstance(camp, CampaignInvestigation)
    CampaignInvestigation.model_validate(camp.model_dump())


def test_mock_mode_works_without_network(monkeypatch, tmp_path):
    monkeypatch.setenv("USE_MOCK_MODEL", "true")
    monkeypatch.setenv("MODEL_API_KEY", "")
    # Use in-memory generation written to real project paths if present; otherwise skip integration.
    from src.generation.generate_all import generate_all

    generate_all(seed=42)
    summary = run_analysis(max_investigations=3)
    assert summary["model_mode"] == "mock"
    assert summary["source_datasets_unmodified"] is True
    assert (OUTPUTS_DIR / "evaluation.json").exists()


def test_source_dataset_not_modified_during_investigation():
    churn_path = GENERATED_DIR / "churn_metrics.parquet"
    media_path = GENERATED_DIR / "media_metrics.parquet"
    if not churn_path.exists():
        from src.generation.generate_all import generate_all

        generate_all(seed=42)

    before_c = churn_path.read_bytes()
    before_m = media_path.read_bytes()
    run_analysis(max_investigations=2)
    assert churn_path.read_bytes() == before_c
    assert media_path.read_bytes() == before_m
