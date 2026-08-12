"""LangChain structured-output investigators (OpenAI-compatible / NemoClaw)."""

from __future__ import annotations

import json
from typing import Any

from src.agents.mock import investigate_campaign_mock, investigate_churn_mock
from src.models.llm import ModelConfig, build_chat_model, load_model_config
from src.models.schemas import CampaignInvestigation, CandidateAlert, ChurnInvestigation
from src.privacy.audit import append_investigation_log, build_inference_payload_preview

CHURN_SYSTEM = """You are a customer-success investigation assistant for SignalSentry.
You receive detector alerts and optional metric summaries derived from SYNTHETIC data.
Rules:
- Only make claims supported by the supplied JSON.
- Clearly separate facts (from data) from hypotheses.
- Never invent metrics, tickets, people, or events that are not present.
- Recommendations are advisory for human CSM review only.
- If information is missing, list it under data_limitations.
Return a structured investigation object.
"""

MEDIA_SYSTEM = """You are a paid-media anomaly investigation assistant for SignalSentry.
You receive detector alerts and optional metric summaries derived from SYNTHETIC data.
Rules:
- Only make claims supported by the supplied JSON.
- Clearly separate facts (from data) from hypotheses.
- Never invent platform UI states, creative assets, or geo tables that are not present.
- Do not claim that campaigns were changed; recommendations require human review.
- If information is missing, list it under data_limitations.
Return a structured investigation object.
"""


def _user_prompt(alert: CandidateAlert, metric_context: dict[str, Any] | None) -> str:
    payload = {
        "alert": alert.to_context_dict(),
        "metric_context": metric_context or {},
        "instructions": "Investigate using only this payload. Label hypotheses explicitly.",
    }
    return json.dumps(payload, indent=2)


def investigate_churn(
    alert: CandidateAlert,
    metric_context: dict[str, Any] | None = None,
    config: ModelConfig | None = None,
) -> tuple[ChurnInvestigation, dict[str, Any]]:
    cfg = config or load_model_config()
    user_prompt = _user_prompt(alert, metric_context)
    preview = build_inference_payload_preview(
        system_prompt=CHURN_SYSTEM,
        user_prompt=user_prompt,
        schema_name="ChurnInvestigation",
        config=cfg,
    )

    if cfg.use_mock:
        result = investigate_churn_mock(alert, metric_context)
    else:
        llm = build_chat_model(cfg)
        structured = llm.with_structured_output(ChurnInvestigation)
        result = structured.invoke(
            [
                {"role": "system", "content": CHURN_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
        )

    append_investigation_log(
        {
            "domain": "churn",
            "entity_id": alert.entity_id,
            "alert_type": alert.alert_type,
            "mode": "mock" if cfg.use_mock else "live",
            "destination": cfg.destination_label,
            "payload_preview": preview,
            "result": result.model_dump(mode="json"),
        }
    )
    return result, preview


def investigate_campaign(
    alert: CandidateAlert,
    metric_context: dict[str, Any] | None = None,
    config: ModelConfig | None = None,
) -> tuple[CampaignInvestigation, dict[str, Any]]:
    cfg = config or load_model_config()
    user_prompt = _user_prompt(alert, metric_context)
    preview = build_inference_payload_preview(
        system_prompt=MEDIA_SYSTEM,
        user_prompt=user_prompt,
        schema_name="CampaignInvestigation",
        config=cfg,
    )

    if cfg.use_mock:
        result = investigate_campaign_mock(alert, metric_context)
    else:
        llm = build_chat_model(cfg)
        structured = llm.with_structured_output(CampaignInvestigation)
        result = structured.invoke(
            [
                {"role": "system", "content": MEDIA_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
        )

    append_investigation_log(
        {
            "domain": "media",
            "entity_id": alert.entity_id,
            "alert_type": alert.alert_type,
            "mode": "mock" if cfg.use_mock else "live",
            "destination": cfg.destination_label,
            "payload_preview": preview,
            "result": result.model_dump(mode="json"),
        }
    )
    return result, preview
