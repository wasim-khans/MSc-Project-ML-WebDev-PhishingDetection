"""Clean the raw main dataset and generate the URL-only feature CSV.

This script removes safe exact duplicates, regenerates the 15 lexical URL
features, and writes the cleaned label sidecar used by later steps.
"""

from pathlib import Path
import json
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.scripts.helpers.project_paths import (
    EXPERIMENT_1A_DIR,
    MAIN_CLEANED_URL_LABELS_PATH,
    MAIN_PROCESSED_DATASET_PATH,
    MAIN_RAW_DATASET_PATH,
    PROJECT_ROOT,
)

from machine_learning.scripts.core.url_feature_extractor import (
    FEATURE_NAMES,
    extract_features,
)
from machine_learning.scripts.core.dataset_cleaning import (
    deduplicate_url_rows,
)


RAW_DATASET_PATH = MAIN_RAW_DATASET_PATH
PROCESSED_DATASET_PATH = MAIN_PROCESSED_DATASET_PATH
CLEANED_URLS_PATH = MAIN_CLEANED_URL_LABELS_PATH
CLEANING_REPORT_PATH = EXPERIMENT_1A_DIR / "main_dataset_cleaning_report.json"


def main() -> None:
    print("Step 2: Build main URL-only feature dataset")
    print(f"Raw source: {RAW_DATASET_PATH.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(RAW_DATASET_PATH, usecols=["URL", "label"])
    df = df.reset_index(names="raw_row_id")
    cleaned_df, cleaning_report = deduplicate_url_rows(
        df,
        url_column="URL",
        label_column="label",
        dataset_name="PhiUSIIL main",
    )
    cleaned_df.insert(0, "source_row_id", range(len(cleaned_df)))
    label_counts = cleaned_df["label"].value_counts().sort_index().to_dict()
    print(f"Rows loaded: {len(df):,}")
    print(f"Rows after URL deduplication: {len(cleaned_df):,}")
    print(f"Duplicate URL rows removed: {cleaning_report['duplicate_rows_removed']:,}")
    print(f"Label counts after cleaning: {label_counts} (0=phishing, 1=legitimate)")
    print(f"Generating {len(FEATURE_NAMES)} URL-only features...")

    feature_rows = [extract_features(url) for url in cleaned_df["url_normalized"]]
    feature_df = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    feature_df["label"] = cleaned_df["label"].astype(int)

    PROCESSED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(PROCESSED_DATASET_PATH, index=False)
    cleaned_url_labels = cleaned_df[
        [
            "source_row_id",
            "raw_row_id",
            "URL",
            "url_normalized",
            "label",
            "duplicate_count",
        ]
    ].rename(columns={"URL": "url"})
    cleaned_url_labels.to_csv(CLEANED_URLS_PATH, index=False)
    CLEANING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEANING_REPORT_PATH.write_text(
        json.dumps(cleaning_report, indent=2),
        encoding="utf-8",
    )

    print(f"Processed file: {PROCESSED_DATASET_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Cleaned URL sidecar: {CLEANED_URLS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Cleaning report: {CLEANING_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Rows written: {feature_df.shape[0]:,}")
    print(f"Columns written: {feature_df.shape[1]}")


if __name__ == "__main__":
    main()
