"""Run detection, investigation, evaluation, and showcase export."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.investigators import investigate_campaign, investigate_churn
from src.detection import detect_churn_alerts, detect_media_alerts, load_thresholds
from src.evaluation import combine_evaluations, evaluate_detections
from src.models.llm import load_model_config
from src.models.schemas import CandidateAlert
from src.paths import GENERATED_DIR, GROUND_TRUTH_DIR, OUTPUTS_DIR, SHOWCASE_DIR, ensure_data_dirs


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_context(df: pd.DataFrame, entity_col: str, entity_id: str, n: int = 8) -> dict[str, Any]:
    subset = df[df[entity_col] == entity_id].sort_values("date").tail(n)
    records = subset.copy()
    records["date"] = pd.to_datetime(records["date"]).dt.strftime("%Y-%m-%d")
    return {"recent_rows": records.to_dict(orient="records")}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_analysis(*, max_investigations: int | None = None) -> dict[str, Any]:
    ensure_data_dirs()
    cfg = load_model_config()
    thresholds = load_thresholds()

    churn_path = GENERATED_DIR / "churn_metrics.parquet"
    media_path = GENERATED_DIR / "media_metrics.parquet"
    churn_truth_path = GROUND_TRUTH_DIR / "churn_labels.parquet"
    media_truth_path = GROUND_TRUTH_DIR / "media_labels.parquet"

    for required in (churn_path, media_path, churn_truth_path, media_truth_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing {required}. Run `python -m src.generation.generate_all` first."
            )

    before_hashes = {
        str(churn_path): _file_fingerprint(churn_path),
        str(media_path): _file_fingerprint(media_path),
    }

    churn_df = pd.read_parquet(churn_path)
    media_df = pd.read_parquet(media_path)
    churn_truth = pd.read_parquet(churn_truth_path)
    media_truth = pd.read_parquet(media_truth_path)

    churn_alerts = detect_churn_alerts(churn_df, thresholds)
    media_alerts = detect_media_alerts(media_df, thresholds)

    # Rank by severity for investigation budget
    sev_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    churn_alerts_sorted = sorted(
        churn_alerts, key=lambda a: sev_rank[a.severity.value], reverse=True
    )
    media_alerts_sorted = sorted(
        media_alerts, key=lambda a: sev_rank[a.severity.value], reverse=True
    )

    if max_investigations is not None:
        churn_to_invest = churn_alerts_sorted[:max_investigations]
        media_to_invest = media_alerts_sorted[:max_investigations]
    else:
        churn_to_invest = churn_alerts_sorted
        media_to_invest = media_alerts_sorted

    # Clear audit log for this run
    audit_path = OUTPUTS_DIR / "investigation_audit.jsonl"
    if audit_path.exists():
        audit_path.unlink()

    churn_investigations = []
    media_investigations = []
    sample_payload = None

    for alert in churn_to_invest:
        ctx = _metric_context(churn_df, "account_id", alert.entity_id)
        result, preview = investigate_churn(alert, ctx, config=cfg)
        churn_investigations.append(
            {
                "alert": alert.model_dump(mode="json"),
                "investigation": result.model_dump(mode="json"),
            }
        )
        sample_payload = sample_payload or preview

    for alert in media_to_invest:
        ctx = _metric_context(media_df, "campaign_id", alert.entity_id)
        result, preview = investigate_campaign(alert, ctx, config=cfg)
        media_investigations.append(
            {
                "alert": alert.model_dump(mode="json"),
                "investigation": result.model_dump(mode="json"),
            }
        )
        sample_payload = sample_payload or preview

    tol = int(thresholds.get("evaluation", {}).get("date_tolerance_days", 14))
    churn_eval = evaluate_detections(churn_alerts, churn_truth, domain="churn", tolerance_days=tol)
    media_eval = evaluate_detections(media_alerts, media_truth, domain="media", tolerance_days=tol)
    evaluation = combine_evaluations({"churn": churn_eval, "media": media_eval})

    # Persist outputs — never overwrite source datasets
    _write_json(OUTPUTS_DIR / "churn_alerts.json", [a.model_dump(mode="json") for a in churn_alerts])
    _write_json(OUTPUTS_DIR / "media_alerts.json", [a.model_dump(mode="json") for a in media_alerts])
    _write_json(OUTPUTS_DIR / "churn_investigations.json", churn_investigations)
    _write_json(OUTPUTS_DIR / "media_investigations.json", media_investigations)
    _write_json(OUTPUTS_DIR / "evaluation.json", evaluation)

    # Showcase bundle for eventual static/Vercel demo
    if SHOWCASE_DIR.exists():
        shutil.rmtree(SHOWCASE_DIR)
    SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)

    overview = {
        "active_alerts": len(churn_alerts) + len(media_alerts),
        "churn_alerts": len(churn_alerts),
        "media_alerts": len(media_alerts),
        "model_mode": "mock" if cfg.use_mock else "live",
        "model_destination": cfg.destination_label,
        "evaluation": evaluation,
        "synthetic_only": True,
    }
    _write_json(SHOWCASE_DIR / "overview.json", overview)
    _write_json(SHOWCASE_DIR / "churn_alerts.json", [a.model_dump(mode="json") for a in churn_alerts])
    _write_json(SHOWCASE_DIR / "campaign_alerts.json", [a.model_dump(mode="json") for a in media_alerts])
    _write_json(
        SHOWCASE_DIR / "investigations.json",
        {"churn": churn_investigations, "media": media_investigations},
    )
    _write_json(SHOWCASE_DIR / "evaluation.json", evaluation)
    if sample_payload:
        _write_json(SHOWCASE_DIR / "sample_inference_payload.json", sample_payload)

    # Slim series for charts (last 16 points per top entities)
    top_accounts = [a.entity_id for a in churn_alerts_sorted[:15]]
    top_campaigns = [a.entity_id for a in media_alerts_sorted[:15]]
    churn_series = churn_df[churn_df["account_id"].isin(top_accounts)].copy()
    media_series = media_df[media_df["campaign_id"].isin(top_campaigns)].copy()
    churn_series["date"] = pd.to_datetime(churn_series["date"]).dt.strftime("%Y-%m-%d")
    media_series["date"] = pd.to_datetime(media_series["date"]).dt.strftime("%Y-%m-%d")
    _write_json(SHOWCASE_DIR / "churn_series.json", churn_series.to_dict(orient="records"))
    _write_json(SHOWCASE_DIR / "media_series.json", media_series.to_dict(orient="records"))
    churn_truth_out = churn_truth.copy()
    media_truth_out = media_truth.copy()
    for frame in (churn_truth_out, media_truth_out):
        for col in ("start_date", "end_date"):
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col]).dt.strftime("%Y-%m-%d")
    _write_json(SHOWCASE_DIR / "churn_ground_truth.json", churn_truth_out.to_dict(orient="records"))
    _write_json(SHOWCASE_DIR / "media_ground_truth.json", media_truth_out.to_dict(orient="records"))

    after_hashes = {
        str(churn_path): _file_fingerprint(churn_path),
        str(media_path): _file_fingerprint(media_path),
    }
    if before_hashes != after_hashes:
        raise RuntimeError("Source datasets were modified during analysis — aborting.")

    summary = {
        "churn_alerts": len(churn_alerts),
        "media_alerts": len(media_alerts),
        "churn_investigations": len(churn_investigations),
        "media_investigations": len(media_investigations),
        "model_mode": "mock" if cfg.use_mock else "live",
        "evaluation": evaluation,
        "outputs_dir": str(OUTPUTS_DIR),
        "showcase_dir": str(SHOWCASE_DIR),
        "source_datasets_unmodified": True,
    }
    _write_json(OUTPUTS_DIR / "run_summary.json", summary)
    return summary


def main() -> None:
    summary = run_analysis()
    print("Analysis complete:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
