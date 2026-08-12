"""LLM client factory for OpenAI-compatible endpoints (including NemoClaw)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    api_key: str
    model_name: str
    use_mock: bool

    @property
    def destination_label(self) -> str:
        return "mock (offline)" if self.use_mock else self.base_url


def load_model_config() -> ModelConfig:
    load_dotenv()
    use_mock_raw = os.getenv("USE_MOCK_MODEL", "true").strip().lower()
    use_mock = use_mock_raw in {"1", "true", "yes", "on"}
    api_key = os.getenv("MODEL_API_KEY", "").strip()
    base_url = os.getenv("MODEL_BASE_URL", "https://inference.local/v1").strip()
    model_name = os.getenv("MODEL_NAME", "nvidia/nemotron-mini").strip()

    # Prefer mock when explicitly requested or when no key is configured.
    if use_mock or not api_key:
        use_mock = True

    return ModelConfig(
        base_url=base_url,
        api_key=api_key or "nemoclaw-local-placeholder",
        model_name=model_name,
        use_mock=use_mock,
    )


def build_chat_model(config: ModelConfig | None = None):
    """Return a LangChain ChatOpenAI client pointed at an OpenAI-compatible base URL."""
    from langchain_openai import ChatOpenAI

    cfg = config or load_model_config()
    if cfg.use_mock:
        raise RuntimeError("build_chat_model called while USE_MOCK_MODEL is enabled")

    return ChatOpenAI(
        model=cfg.model_name,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=0,
    )
