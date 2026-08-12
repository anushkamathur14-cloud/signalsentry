"""Generate all synthetic datasets and write them to disk."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

from src.generation.churn import generate_churn_dataset
from src.generation.media import generate_media_dataset
from src.paths import GENERATED_DIR, GROUND_TRUTH_DIR, ensure_data_dirs

META_PATH = GENERATED_DIR / "generation_meta.json"


def generate_all(
    seed: int | None = None,
    *,
    n_accounts: int = 100,
    n_campaigns: int = 40,
    n_weeks: int = 52,
    n_days: int = 183,
) -> dict[str, str]:
    """
    Build a fresh synthetic world (metrics + ground-truth labels).

    Changing ``seed`` (or sizes) yields a different demo dataset while keeping
    the same anomaly *types* for evaluation.
    """
    load_dotenv()
    seed = int(seed if seed is not None else os.getenv("SEED", "42"))
    ensure_data_dirs()

    churn_df, churn_truth = generate_churn_dataset(
        seed=seed, n_accounts=n_accounts, n_weeks=n_weeks
    )
    media_df, media_truth = generate_media_dataset(
        seed=seed, n_campaigns=n_campaigns, n_days=n_days
    )

    paths = {
        "churn_metrics": str(GENERATED_DIR / "churn_metrics.parquet"),
        "media_metrics": str(GENERATED_DIR / "media_metrics.parquet"),
        "churn_ground_truth": str(GROUND_TRUTH_DIR / "churn_labels.parquet"),
        "media_ground_truth": str(GROUND_TRUTH_DIR / "media_labels.parquet"),
    }
    # Also write CSV copies for easy inspection.
    csv_paths = {
        "churn_metrics_csv": str(GENERATED_DIR / "churn_metrics.csv"),
        "media_metrics_csv": str(GENERATED_DIR / "media_metrics.csv"),
        "churn_ground_truth_csv": str(GROUND_TRUTH_DIR / "churn_labels.csv"),
        "media_ground_truth_csv": str(GROUND_TRUTH_DIR / "media_labels.csv"),
    }

    churn_df.to_parquet(paths["churn_metrics"], index=False)
    media_df.to_parquet(paths["media_metrics"], index=False)
    churn_truth.to_parquet(paths["churn_ground_truth"], index=False)
    media_truth.to_parquet(paths["media_ground_truth"], index=False)

    churn_df.to_csv(csv_paths["churn_metrics_csv"], index=False)
    media_df.to_csv(csv_paths["media_metrics_csv"], index=False)
    churn_truth.to_csv(csv_paths["churn_ground_truth_csv"], index=False)
    media_truth.to_csv(csv_paths["media_ground_truth_csv"], index=False)

    meta: dict[str, Any] = {
        "seed": seed,
        "n_accounts": n_accounts,
        "n_campaigns": n_campaigns,
        "n_weeks": n_weeks,
        "n_days": n_days,
        "churn_rows": int(len(churn_df)),
        "media_rows": int(len(media_df)),
        "churn_injections": int(len(churn_truth)),
        "media_injections": int(len(media_truth)),
        "synthetic_only": True,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {**paths, **csv_paths, "seed": str(seed), "meta": str(META_PATH)}


def main() -> None:
    written = generate_all()
    print("Generated synthetic datasets:")
    for key, value in written.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
