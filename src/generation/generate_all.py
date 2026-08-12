"""Generate all synthetic datasets and write them to disk."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from src.generation.churn import generate_churn_dataset
from src.generation.media import generate_media_dataset
from src.paths import GENERATED_DIR, GROUND_TRUTH_DIR, ensure_data_dirs


def generate_all(seed: int | None = None) -> dict[str, str]:
    load_dotenv()
    seed = int(seed if seed is not None else os.getenv("SEED", "42"))
    ensure_data_dirs()

    churn_df, churn_truth = generate_churn_dataset(seed=seed)
    media_df, media_truth = generate_media_dataset(seed=seed)

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

    return {**paths, **csv_paths, "seed": str(seed)}


def main() -> None:
    written = generate_all()
    print("Generated synthetic datasets:")
    for key, value in written.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
