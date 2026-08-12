"""Pydantic schemas for alerts and investigations."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CandidateAlert(BaseModel):
    entity_id: str
    start_date: date
    end_date: date
    alert_type: str
    severity: Severity
    metrics_involved: list[str]
    current_value: float
    expected_value: float
    supporting_calculations: dict[str, Any] = Field(default_factory=dict)
    domain: str  # "churn" | "media"

    def to_context_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ChurnInvestigation(BaseModel):
    account_id: str
    risk_score: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    evidence: list[str]
    likely_causes: list[str]
    recommended_csm_action: str
    confidence: float = Field(ge=0, le=1)
    data_limitations: list[str]

    @field_validator("evidence", "likely_causes", "data_limitations")
    @classmethod
    def non_empty_lists(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("list must not be empty")
        return value


class CampaignInvestigation(BaseModel):
    campaign_id: str
    severity: Severity
    anomaly_summary: str
    evidence: list[str]
    likely_causes: list[str]
    recommended_action: str
    requires_immediate_human_review: bool
    confidence: float = Field(ge=0, le=1)
    data_limitations: list[str]

    @field_validator("evidence", "likely_causes", "data_limitations")
    @classmethod
    def non_empty_lists(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("list must not be empty")
        return value


class EvaluationMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    by_anomaly_type: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_severity: dict[str, dict[str, float]] = Field(default_factory=dict)
