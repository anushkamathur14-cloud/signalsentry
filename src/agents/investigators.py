"""LangChain structured-output investigators (OpenAI-compatible / NemoClaw)."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from src.agents.mock import investigate_campaign_mock, investigate_churn_mock
from src.agents.tracing import RunTrace, timed_step
from src.models.llm import ModelConfig, build_chat_model, load_model_config
from src.models.schemas import CampaignInvestigation, CandidateAlert, ChurnInvestigation
from src.privacy.audit import append_investigation_log, build_inference_payload_preview

T = TypeVar("T", bound=BaseModel)

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

ASSISTANT_SYSTEM = """You are the SignalSentry portfolio demo assistant.
Explain how the backend works: synthetic data → deterministic detectors → LangChain
structured investigators over an OpenAI-compatible endpoint (NemoClaw locally at
inference.local, or NVIDIA public endpoints via BYOK on Streamlit Cloud).
OpenClaw is the local agent/chat gateway; it is not embedded in the public Cloud demo.
Keep answers concise. Never claim live customer data is used. Never ask for secrets
to be pasted into chat — API keys belong only in the sidebar BYOK field.
"""


def _user_prompt(alert: CandidateAlert, metric_context: dict[str, Any] | None) -> str:
    payload = {
        "alert": alert.to_context_dict(),
        "metric_context": metric_context or {},
        "instructions": "Investigate using only this payload. Label hypotheses explicitly.",
    }
    return json.dumps(payload, indent=2)


def _invoke_structured(
    *,
    schema: Type[T],
    system_prompt: str,
    user_prompt: str,
    config: ModelConfig,
    trace: RunTrace | None = None,
) -> T:
    """
    Live LangChain path via NemoClaw / OpenAI-compatible chat completions.

    Tries json_schema structured output first, then json_mode for backends that
    do not fully support tool/json_schema calling (common on some local Nemotron routes).
    """
    llm = build_chat_model(config)
    if trace is not None:
        trace.add(
            "langchain.ChatOpenAI",
            f"Client ready · base_url={config.base_url} · model={config.model_name}",
            data={"path_label": config.path_label},
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    methods = [config.structured_output_method]
    for fallback in ("json_schema", "json_mode"):
        if fallback not in methods:
            methods.append(fallback)

    last_error: Exception | None = None
    for method in methods:
        try:
            t0 = time.perf_counter()
            structured = llm.with_structured_output(schema, method=method)
            result = structured.invoke(messages)
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            if result is None:
                raise RuntimeError(f"structured output returned None (method={method})")
            if trace is not None:
                trace.add(
                    "with_structured_output.invoke",
                    f"method={method} schema={schema.__name__}",
                    data={"method": method, "schema": schema.__name__},
                    duration_ms=duration_ms,
                )
            return result  # type: ignore[return-value]
        except Exception as exc:  # noqa: BLE001 - try next compatibility mode
            last_error = exc
            if trace is not None:
                trace.add(
                    "structured_output_retry",
                    f"method={method} failed: {exc}",
                    status="error",
                    data={"method": method},
                )
            continue
    raise RuntimeError(
        f"Live LangChain investigation failed against {config.base_url} "
        f"(model={config.model_name}). Last error: {last_error}"
    )


def _new_trace(kind: str, config: ModelConfig) -> RunTrace:
    return RunTrace(run_id=str(uuid.uuid4())[:8], kind=kind, path_label=config.path_label)


def investigate_churn(
    alert: CandidateAlert,
    metric_context: dict[str, Any] | None = None,
    config: ModelConfig | None = None,
) -> tuple[ChurnInvestigation, dict[str, Any]]:
    cfg = config or load_model_config()
    trace = _new_trace("churn_investigation", cfg)
    trace.add(
        "detector_alert_received",
        f"{alert.entity_id} · {alert.alert_type} · severity={alert.severity.value}",
        data={"alert": alert.to_context_dict()},
    )

    user_prompt = _user_prompt(alert, metric_context)
    preview = build_inference_payload_preview(
        system_prompt=CHURN_SYSTEM,
        user_prompt=user_prompt,
        schema_name="ChurnInvestigation",
        config=cfg,
    )
    trace.add(
        "build_langchain_payload",
        "System + user JSON assembled for structured ChurnInvestigation",
        data={"schema": "ChurnInvestigation", "use_mock": cfg.use_mock},
    )

    if cfg.use_mock:
        with timed_step(trace, "mock_investigator", "Offline template investigator (no LLM call)") as step:
            result = investigate_churn_mock(alert, metric_context)
            step.status = "mock"
    else:
        result = _invoke_structured(
            schema=ChurnInvestigation,
            system_prompt=CHURN_SYSTEM,
            user_prompt=user_prompt,
            config=cfg,
            trace=trace,
        )

    trace.add(
        "structured_result",
        f"risk_level={result.risk_level.value} confidence={result.confidence}",
        data={"result": result.model_dump(mode="json")},
    )
    trace.finish()

    append_investigation_log(
        {
            "domain": "churn",
            "entity_id": alert.entity_id,
            "alert_type": alert.alert_type,
            "mode": "mock" if cfg.use_mock else "live-langchain",
            "destination": cfg.destination_label,
            "nemoclaw_route": cfg.is_nemoclaw_route,
            "path_label": cfg.path_label,
            "payload_preview": preview,
            "result": result.model_dump(mode="json"),
            "langchain_trace": trace.to_dict(),
        }
    )
    return result, preview


def investigate_campaign(
    alert: CandidateAlert,
    metric_context: dict[str, Any] | None = None,
    config: ModelConfig | None = None,
) -> tuple[CampaignInvestigation, dict[str, Any]]:
    cfg = config or load_model_config()
    trace = _new_trace("media_investigation", cfg)
    trace.add(
        "detector_alert_received",
        f"{alert.entity_id} · {alert.alert_type} · severity={alert.severity.value}",
        data={"alert": alert.to_context_dict()},
    )

    user_prompt = _user_prompt(alert, metric_context)
    preview = build_inference_payload_preview(
        system_prompt=MEDIA_SYSTEM,
        user_prompt=user_prompt,
        schema_name="CampaignInvestigation",
        config=cfg,
    )
    trace.add(
        "build_langchain_payload",
        "System + user JSON assembled for structured CampaignInvestigation",
        data={"schema": "CampaignInvestigation", "use_mock": cfg.use_mock},
    )

    if cfg.use_mock:
        with timed_step(trace, "mock_investigator", "Offline template investigator (no LLM call)") as step:
            result = investigate_campaign_mock(alert, metric_context)
            step.status = "mock"
    else:
        result = _invoke_structured(
            schema=CampaignInvestigation,
            system_prompt=MEDIA_SYSTEM,
            user_prompt=user_prompt,
            config=cfg,
            trace=trace,
        )

    trace.add(
        "structured_result",
        f"severity={result.severity.value} confidence={result.confidence}",
        data={"result": result.model_dump(mode="json")},
    )
    trace.finish()

    append_investigation_log(
        {
            "domain": "media",
            "entity_id": alert.entity_id,
            "alert_type": alert.alert_type,
            "mode": "mock" if cfg.use_mock else "live-langchain",
            "destination": cfg.destination_label,
            "nemoclaw_route": cfg.is_nemoclaw_route,
            "path_label": cfg.path_label,
            "payload_preview": preview,
            "result": result.model_dump(mode="json"),
            "langchain_trace": trace.to_dict(),
        }
    )
    return result, preview


def ask_assistant(
    question: str,
    *,
    config: ModelConfig | None = None,
    context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Portfolio Q&A over the architecture. Live when config is non-mock; otherwise
    returns a fixed explanation so Cloud demos never hang.
    """
    cfg = config or load_model_config()
    trace = _new_trace("assistant_chat", cfg)
    trace.add("user_question", question[:500], data={"context_keys": list((context or {}).keys())})

    if cfg.use_mock:
        answer = (
            "SignalSentry's backend is: synthetic generators → YAML-threshold detectors → "
            "LangChain `ChatOpenAI.with_structured_output` investigators. "
            "Locally, LangChain talks to NemoClaw at `https://inference.local/v1` (OpenClaw "
            "is the sandbox chat/gateway). On this hosted demo, investigations default to "
            "mock templates. Paste your own NVIDIA API key in the sidebar (BYOK), pick a "
            "public model, and ask again — or open Structure for the full system map."
        )
        trace.add("mock_assistant_reply", "Returned architecture explanation without LLM", status="mock")
        trace.finish()
        payload = {"answer": answer, "langchain_trace": trace.to_dict(), "mode": "mock"}
        append_investigation_log(
            {
                "domain": "assistant",
                "entity_id": "portfolio-chat",
                "alert_type": "architecture_qa",
                "mode": "mock",
                "destination": cfg.destination_label,
                "path_label": cfg.path_label,
                "result": {"answer": answer},
                "langchain_trace": trace.to_dict(),
            }
        )
        return answer, payload

    llm = build_chat_model(cfg)
    messages = [
        {"role": "system", "content": ASSISTANT_SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                {"question": question, "demo_context": context or {}, "instructions": "Answer briefly."},
                indent=2,
            ),
        },
    ]
    try:
        with timed_step(trace, "ChatOpenAI.invoke", f"base_url={cfg.base_url} model={cfg.model_name}") as step:
            response = llm.invoke(messages)
            answer = getattr(response, "content", None) or str(response)
            step.status = "ok"
    except Exception as exc:  # noqa: BLE001 - portfolio UX must not crash on bad model/key
        err = str(exc)
        trace.add("ChatOpenAI.invoke", f"failed: {err}", status="error")
        trace.finish()
        answer = (
            f"Live LangChain call failed against `{cfg.base_url}` with model `{cfg.model_name}`.\n\n"
            f"**Error:** {err}\n\n"
            "Try another model in the sidebar (BYOK → Model), confirm the key is an `nvapi-…` "
            "NVIDIA key from build.nvidia.com, then ask again. Until then, use Structure for "
            "the system map — mock answers still work without a key."
        )
        payload = {
            "answer": answer,
            "langchain_trace": trace.to_dict(),
            "mode": "live-error",
            "error": err,
        }
        append_investigation_log(
            {
                "domain": "assistant",
                "entity_id": "portfolio-chat",
                "alert_type": "architecture_qa",
                "mode": "live-error",
                "destination": cfg.destination_label,
                "path_label": cfg.path_label,
                "result": {"answer": answer, "error": err},
                "langchain_trace": trace.to_dict(),
            }
        )
        return answer, payload

    trace.finish()
    payload = {"answer": answer, "langchain_trace": trace.to_dict(), "mode": "live-langchain"}
    append_investigation_log(
        {
            "domain": "assistant",
            "entity_id": "portfolio-chat",
            "alert_type": "architecture_qa",
            "mode": "live-langchain",
            "destination": cfg.destination_label,
            "path_label": cfg.path_label,
            "nemoclaw_route": cfg.is_nemoclaw_route,
            "result": {"answer": answer},
            "langchain_trace": trace.to_dict(),
        }
    )
    return answer, payload
