"""Verify live LangChain ↔ NemoClaw / OpenAI-compatible inference connectivity."""

from __future__ import annotations

import json
import sys

from src.models.llm import build_chat_model, load_model_config
from src.models.schemas import CandidateAlert, Severity
from src.agents.investigators import investigate_churn


def main() -> int:
    cfg = load_model_config()
    print("SignalSentry inference check")
    print(f"  mode:        {'MOCK' if cfg.use_mock else 'LIVE LangChain'}")
    print(f"  base_url:    {cfg.base_url}")
    print(f"  model:       {cfg.model_name}")
    print(f"  nemoclaw:    {cfg.is_nemoclaw_route}")
    print(f"  api_key_set: {bool(cfg.api_key)}")

    if cfg.use_mock:
        print(
            "\nUSE_MOCK_MODEL=true — not calling a model.\n"
            "For live NemoClaw + LangChain set in .env:\n"
            "  USE_MOCK_MODEL=false\n"
            "  MODEL_BASE_URL=https://inference.local/v1\n"
            "  MODEL_API_KEY=nemoclaw-local-placeholder\n"
            "  MODEL_NAME=nvidia/nemotron-mini\n"
            "Then re-run: python -m src.verify_inference"
        )
        return 0

    print("\n1) Chat Completions ping...")
    llm = build_chat_model(cfg)
    ping = llm.invoke("Reply with exactly: ok")
    print(f"   response: {getattr(ping, 'content', ping)!r}")

    print("\n2) Structured investigation via LangChain...")
    alert = CandidateAlert(
        entity_id="ACC-000",
        start_date="2024-08-01",
        end_date="2024-08-28",
        alert_type="gradual_usage_decline",
        severity=Severity.HIGH,
        metrics_involved=["weekly_sessions", "active_users"],
        current_value=12.0,
        expected_value=40.0,
        supporting_calculations={"pct_drop": 0.7},
        domain="churn",
    )
    result, preview = investigate_churn(
        alert,
        metric_context={"note": "verify_inference synthetic probe"},
        config=cfg,
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    print("\nPayload destination:", preview.get("destination"))
    print("Live LangChain path OK.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED: {exc}", file=sys.stderr)
        print(
            "\nTroubleshooting:\n"
            "- Run inside a NemoClaw sandbox (or where inference.local resolves).\n"
            "- Confirm host onboarding selected an OpenAI-compatible / Nemotron route.\n"
            "- App must call https://inference.local/v1 — not a public provider domain.\n"
            "- Try STRUCTURED_OUTPUT_METHOD=json_mode if json_schema is unsupported.",
            file=sys.stderr,
        )
        raise SystemExit(1)
