"""LLM client factory for OpenAI-compatible endpoints (including NemoClaw)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

NEMOCLAW_DEFAULT_BASE_URL = "https://inference.local/v1"
NEMOCLAW_PLACEHOLDER_KEY = "nemoclaw-local-placeholder"
DEFAULT_MODEL_NAME = "nvidia/nemotron-mini"


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    api_key: str
    model_name: str
    use_mock: bool
    structured_output_method: str = "json_schema"

    @property
    def destination_label(self) -> str:
        return "mock (offline)" if self.use_mock else self.base_url

    @property
    def is_nemoclaw_route(self) -> bool:
        return "inference.local" in self.base_url


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_model_config() -> ModelConfig:
    """
    Load inference settings.

    Live NemoClaw/LangChain is the primary path:
      USE_MOCK_MODEL=false
      MODEL_BASE_URL=https://inference.local/v1
      MODEL_API_KEY=nemoclaw-local-placeholder

    Mock is an offline fallback for CI / Streamlit Community Cloud (no inference.local).
    """
    load_dotenv()

    # Streamlit Cloud cannot reach NemoClaw's inference.local — force mock there unless overridden.
    on_streamlit_cloud = bool(os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("STREAMLIT_RUNTIME_ENV"))
    default_mock = on_streamlit_cloud or _env_bool("SIGNAL_SENTRY_FORCE_MOCK", False)

    use_mock = _env_bool("USE_MOCK_MODEL", default_mock)
    base_url = os.getenv("MODEL_BASE_URL", NEMOCLAW_DEFAULT_BASE_URL).strip()
    api_key = os.getenv("MODEL_API_KEY", NEMOCLAW_PLACEHOLDER_KEY).strip() or NEMOCLAW_PLACEHOLDER_KEY
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip()
    method = os.getenv("STRUCTURED_OUTPUT_METHOD", "json_schema").strip() or "json_schema"

    return ModelConfig(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        use_mock=use_mock,
        structured_output_method=method,
    )


def build_chat_model(config: ModelConfig | None = None):
    """Return a LangChain ChatOpenAI client for an OpenAI-compatible / NemoClaw route."""
    from langchain_openai import ChatOpenAI

    cfg = config or load_model_config()
    if cfg.use_mock:
        raise RuntimeError("build_chat_model called while mock mode is enabled")

    return ChatOpenAI(
        model=cfg.model_name,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=0,
        # Chat Completions path — NemoClaw's default OpenAI-compatible route.
        timeout=120,
        max_retries=2,
    )
