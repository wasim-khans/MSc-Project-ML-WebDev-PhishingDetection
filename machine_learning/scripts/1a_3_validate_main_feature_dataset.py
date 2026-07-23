"""Validate that the processed main feature dataset matches the source URLs.

This script regenerates the URL-only features from the cleaned URL sidecar and
checks that the saved processed CSV and labels are exact matches.
"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.scripts.helpers.project_paths import (
    MAIN_CLEANED_URL_LABELS_PATH,
    MAIN_PROCESSED_DATASET_PATH,
    MAIN_RAW_DATASET_PATH,
    PROJECT_ROOT,
)

from machine_learning.scripts.core.url_feature_extractor import (
    FEATURE_NAMES,
    extract_features,
)


RAW_DATASET_PATH = MAIN_RAW_DATASET_PATH
PROCESSED_DATASET_PATH = MAIN_PROCESSED_DATASET_PATH
CLEANED_URLS_PATH = MAIN_CLEANED_URL_LABELS_PATH


def main() -> None:
    print("Step 3: Validate main processed feature dataset")
    print(f"Raw source: {RAW_DATASET_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Processed source: {PROCESSED_DATASET_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Cleaned URL source: {CLEANED_URLS_PATH.relative_to(PROJECT_ROOT)}")
    raw_df = pd.read_csv(RAW_DATASET_PATH, usecols=["URL", "label"])
    processed_df = pd.read_csv(PROCESSED_DATASET_PATH)
    cleaned_urls_df = pd.read_csv(CLEANED_URLS_PATH)

    expected_columns = FEATURE_NAMES + ["label"]
    if list(processed_df.columns) != expected_columns:
        raise AssertionError(
            "Processed dataset columns do not match expected feature order.\n"
            f"Expected: {expected_columns}\n"
            f"Actual: {list(processed_df.columns)}"
        )

    if len(cleaned_urls_df) != len(processed_df):
        raise AssertionError(
            "Row count mismatch after cleaning: "
            f"cleaned_urls={len(cleaned_urls_df)}, processed={len(processed_df)}"
        )

    if cleaned_urls_df["url_normalized"].duplicated().any():
        raise AssertionError("Cleaned URL sidecar still contains duplicate URLs.")

    if not cleaned_urls_df["label"].astype(int).equals(
        processed_df["label"].astype(int)
    ):
        raise AssertionError(
            "Label column does not match between cleaned URLs and processed data."
        )

    regenerated_features = pd.DataFrame(
        [extract_features(url) for url in cleaned_urls_df["url_normalized"]],
        columns=FEATURE_NAMES,
    )

    if not regenerated_features.equals(processed_df[FEATURE_NAMES]):
        differences = regenerated_features.compare(processed_df[FEATURE_NAMES])
        raise AssertionError(
            "Processed features do not match regenerated features from raw URLs.\n"
            f"First differences:\n{differences.head(10)}"
        )

    print("Processed feature validation passed.")
    print(f"Raw rows checked for source availability: {len(raw_df):,}")
    print(f"Rows checked: {len(processed_df):,}")
    print("Duplicate URL rows in processed data: 0")
    print(f"Feature columns checked: {len(FEATURE_NAMES)}")
    print("Labels checked: yes")


if __name__ == "__main__":
    main()
