"""LLM client factory for OpenAI-compatible endpoints (including NemoClaw)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

NEMOCLAW_DEFAULT_BASE_URL = "https://inference.local/v1"
# Public NVIDIA Endpoints (same OpenAI-compatible surface NemoClaw fronts locally).
NVIDIA_PUBLIC_BASE_URL = "https://integrate.api.nvidia.com/v1"
NEMOCLAW_PLACEHOLDER_KEY = "nemoclaw-local-placeholder"
DEFAULT_MODEL_NAME = "nvidia/nemotron-mini"
# Widely available on build.nvidia.com / integrate.api (avoid obscure IDs that 404).
DEFAULT_PUBLIC_MODEL_NAME = "meta/llama-3.1-8b-instruct"
PUBLIC_MODEL_CHOICES = (
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.3-70b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/nemotron-4-340b-instruct",
    "google/gemma-2-9b-it",
    "mistralai/mistral-small-24b-instruct",
)


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    api_key: str
    model_name: str
    use_mock: bool
    structured_output_method: str = "json_schema"
    path_label: str = "mock"

    @property
    def destination_label(self) -> str:
        return "mock (offline)" if self.use_mock else self.base_url

    @property
    def is_nemoclaw_route(self) -> bool:
        return "inference.local" in self.base_url

    @property
    def is_nvidia_public_route(self) -> bool:
        return "integrate.api.nvidia.com" in self.base_url or "api.nvcf.nvidia.com" in self.base_url


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def is_hosted_demo_environment() -> bool:
    """
    True on Streamlit Community Cloud (and similar hosts) where inference.local
    is unreachable. Detection must not rely on a single env var — Cloud mounts
    the app under /mount/src/<repo>.
    """
    for key in (
        "STREAMLIT_SHARING_MODE",
        "STREAMLIT_RUNTIME_ENV",
        "STREAMLIT_CLOUD",
    ):
        if os.getenv(key):
            return True
    try:
        cwd = str(Path.cwd()).replace("\\", "/")
    except OSError:
        cwd = ""
    # Streamlit Cloud mounts the repo at /mount/src/<app>
    if cwd.startswith("/mount/src"):
        return True
    if Path("/mount/src").is_dir():
        return True
    return False


def force_mock_if_hosted_demo() -> bool:
    """
    Default hosted demos to mock (no owner API spend / no inference.local).

    Visitors can still enable live LangChain via BYOK (see resolve_model_config).
    """
    if not is_hosted_demo_environment():
        return False
    if os.getenv("SIGNAL_SENTRY_BYOK_ACTIVE") == "1":
        return False
    os.environ["USE_MOCK_MODEL"] = "true"
    return True


def load_model_config() -> ModelConfig:
    """
    Load inference settings from the environment.

    Local primary path (inside NemoClaw):
      USE_MOCK_MODEL=false
      MODEL_BASE_URL=https://inference.local/v1

    Streamlit Community Cloud defaults to mock unless BYOK is active.
    """
    load_dotenv()
    force_mock_if_hosted_demo()

    force_offline = _env_bool("SIGNAL_SENTRY_FORCE_MOCK", False)
    if force_offline or (
        is_hosted_demo_environment() and os.getenv("SIGNAL_SENTRY_BYOK_ACTIVE") != "1"
    ):
        use_mock = True
        path_label = "mock-hosted" if is_hosted_demo_environment() else "mock"
    else:
        use_mock = _env_bool("USE_MOCK_MODEL", False)
        path_label = "mock" if use_mock else (
            "nemoclaw-local" if "inference.local" in os.getenv("MODEL_BASE_URL", NEMOCLAW_DEFAULT_BASE_URL)
            else "openai-compatible"
        )

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
        path_label=path_label,
    )


def resolve_model_config(
    *,
    visitor_api_key: str | None = None,
    visitor_model: str | None = None,
) -> ModelConfig:
    """
    Resolve the effective model config for the dashboard.

    Portfolio policy:
    - Default on Streamlit Cloud: mock (your key is NOT used; visitors don't pay either).
    - Optional BYOK: visitor pastes their own NVIDIA API key → live LangChain against
      NVIDIA's public OpenAI-compatible endpoint (same client shape as NemoClaw).
    - Local NemoClaw: uses inference.local from .env when USE_MOCK_MODEL=false.
    """
    key = (visitor_api_key or "").strip()
    if key:
        os.environ["SIGNAL_SENTRY_BYOK_ACTIVE"] = "1"
        os.environ["USE_MOCK_MODEL"] = "false"
        os.environ.pop("SIGNAL_SENTRY_FORCE_MOCK", None)
        chosen = (visitor_model or "").strip()
        model_name = (
            chosen
            or os.getenv("MODEL_NAME_PUBLIC")
            or DEFAULT_PUBLIC_MODEL_NAME
        )
        base_url = os.getenv("MODEL_BASE_URL_PUBLIC", NVIDIA_PUBLIC_BASE_URL).strip()
        method = os.getenv("STRUCTURED_OUTPUT_METHOD", "json_mode").strip() or "json_mode"
        return ModelConfig(
            base_url=base_url,
            api_key=key,
            model_name=model_name.strip(),
            use_mock=False,
            structured_output_method=method,
            path_label="byok-nvidia-public",
        )

    os.environ.pop("SIGNAL_SENTRY_BYOK_ACTIVE", None)
    return load_model_config()


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
