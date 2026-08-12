"""Privacy, audit log, and inference payload preview helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.llm import ModelConfig, load_model_config
from src.paths import DATA_DIR, GENERATED_DIR, GROUND_TRUTH_DIR, OUTPUTS_DIR, ROOT, THRESHOLDS_PATH


def list_readable_files() -> list[dict[str, str]]:
    """Files the application may read during analysis / dashboard rendering."""
    candidates = [
        THRESHOLDS_PATH,
        GENERATED_DIR / "churn_metrics.parquet",
        GENERATED_DIR / "media_metrics.parquet",
        GROUND_TRUTH_DIR / "churn_labels.parquet",
        GROUND_TRUTH_DIR / "media_labels.parquet",
        OUTPUTS_DIR / "churn_alerts.json",
        OUTPUTS_DIR / "media_alerts.json",
        OUTPUTS_DIR / "churn_investigations.json",
        OUTPUTS_DIR / "media_investigations.json",
        OUTPUTS_DIR / "evaluation.json",
        OUTPUTS_DIR / "investigation_audit.jsonl",
    ]
    rows: list[dict[str, str]] = []
    for path in candidates:
        rows.append(
            {
                "path": str(path.relative_to(ROOT)) if path.exists() else str(path.relative_to(ROOT)),
                "exists": str(path.exists()),
                "kind": path.suffix.lstrip(".") or "dir",
            }
        )
    return rows


def build_inference_payload_preview(
    *,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    config: ModelConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_model_config()
    return {
        "destination": cfg.destination_label,
        "model": cfg.model_name,
        "use_mock": cfg.use_mock,
        "api": "openai-compatible /v1/chat/completions (structured output)",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_schema", "schema_name": schema_name},
        "note": "Advisory only — no campaign changes or customer outreach are executed.",
    }


def append_investigation_log(
    entry: dict[str, Any],
    path: Path | None = None,
) -> Path:
    log_path = path or (OUTPUTS_DIR / "investigation_audit.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return log_path


def read_investigation_log(path: Path | None = None, limit: int = 200) -> list[dict[str, Any]]:
    log_path = path or (OUTPUTS_DIR / "investigation_audit.jsonl")
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return rows[-limit:]


def synthetic_data_confirmation() -> dict[str, Any]:
    return {
        "synthetic_only": True,
        "message": "SignalSentry uses synthetically generated account and campaign data only. No live customer or ad-platform data is loaded.",
        "data_root": str(DATA_DIR.relative_to(ROOT)),
    }
