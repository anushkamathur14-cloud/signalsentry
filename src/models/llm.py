"""LLM client factory for OpenAI-compatible endpoints (including NemoClaw)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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


def is_hosted_demo_environment() -> bool:
    """
    True on Streamlit Community Cloud (and similar hosts) where inference.local
    is unreachable. Detection must not rely on a single env var — Cloud mounts
    the app under /mount/src/<repo>.
    """
    if _env_bool("SIGNAL_SENTRY_FORCE_MOCK", False):
        return True
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
    """Force mock investigators on hosted demos. Returns True if forced."""
    if not is_hosted_demo_environment():
        return False
    os.environ["USE_MOCK_MODEL"] = "true"
    os.environ["SIGNAL_SENTRY_FORCE_MOCK"] = "true"
    return True


def load_model_config() -> ModelConfig:
    """
    Load inference settings.

    Live NemoClaw/LangChain is the primary path locally:
      USE_MOCK_MODEL=false
      MODEL_BASE_URL=https://inference.local/v1
      MODEL_API_KEY=nemoclaw-local-placeholder

    Streamlit Community Cloud / CI always use mock (no inference.local).
    """
    load_dotenv()
    force_mock_if_hosted_demo()

    # Hosted portfolio demos cannot reach NemoClaw — always mock there.
    if is_hosted_demo_environment():
        use_mock = True
    else:
        use_mock = _env_bool("USE_MOCK_MODEL", False)

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
