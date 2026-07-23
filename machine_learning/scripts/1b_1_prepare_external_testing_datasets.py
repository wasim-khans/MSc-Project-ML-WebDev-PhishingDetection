"""Prepare external datasets for cross-dataset evaluation.

This script cleans LegitPhish and PhishStorm, standardises the label meaning to
0=phishing and 1=legitimate, and regenerates the same 15 URL-only features.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.scripts.helpers.project_paths import (
    EXTERNAL_DATASETS_ROOT as EXTERNAL_DATA_DIR,
    LEGITPHISH_RAW_DATASET_PATH,
    PHISHSTORM_RAW_DATASET_PATH,
    PROJECT_ROOT,
)

from machine_learning.scripts.core.url_feature_extractor import (
    FEATURE_NAMES,
    extract_features,
)
from machine_learning.scripts.core.dataset_cleaning import (
    deduplicate_url_rows,
)
FEATURES_FILENAME = "external_url_features_without_labels.csv"
LABELS_FILENAME = "external_url_labels_for_comparison.csv"
METADATA_FILENAME = "external_dataset_metadata.json"

DATASET_CHOICES = ["phishstorm", "legitphish"]


def main():
    parser = argparse.ArgumentParser(
        description="Prepare external datasets for model generalisation testing."
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Dataset to prepare: all, phishstorm, or legitphish.",
    )
    args = parser.parse_args()

    selected_datasets = normalise_dataset_selection(args.datasets)
    print("Step 5: Prepare external testing datasets")
    print(f"Selected datasets: {', '.join(selected_datasets)}")
    print(f"External data root: {EXTERNAL_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print()
    for dataset_name in selected_datasets:
        if dataset_name == "phishstorm":
            prepare_phishstorm_dataset()
        elif dataset_name == "legitphish":
            prepare_legitphish_dataset()


def normalise_dataset_selection(selection):
    """Convert a CLI value into one or more known dataset names."""
    cleaned = selection.strip().lower()
    if cleaned == "all":
        return DATASET_CHOICES
    if cleaned not in DATASET_CHOICES:
        raise ValueError(
            f"Unknown dataset '{selection}'. Choose from: all, "
            + ", ".join(DATASET_CHOICES)
        )
    return [cleaned]


def prepare_phishstorm_dataset():
    """Prepare PhishStorm from its raw URL column and original label column."""
    raw_path = PHISHSTORM_RAW_DATASET_PATH
    processed_dir = EXTERNAL_DATA_DIR / "phishstorm" / "processed"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing PhishStorm raw file: {raw_path}")

    print("Preparing PhishStorm...")
    print(f"  raw file: {raw_path.relative_to(PROJECT_ROOT)}")
    raw_df = pd.read_csv(
        raw_path,
        encoding="latin1",
        engine="python",
        on_bad_lines="skip",
    )
    features_df, labels_df, cleaning_report = build_phishstorm_processed_tables(
        raw_df.to_dict(orient="records")
    )
    metadata = {
        "dataset": "PhishStorm",
        "raw_file": str(raw_path.relative_to(PROJECT_ROOT)),
        "status": "prepared",
        "raw_rows_read": int(len(raw_df)),
        "usable_rows": int(len(features_df)),
        "dropped_rows": int(len(raw_df) - len(features_df)),
        "cleaning": cleaning_report,
        "url_column": "domain",
        "original_label_column": "label",
        "original_label_mapping": {
            "0": "legitimate",
            "1": "phishing",
        },
        "project_label_mapping": {
            "0": "phishing",
            "1": "legitimate",
        },
        "important_note": (
            "This local CSV contains malformed/missing-label rows. The preparation "
            "step skips malformed rows and drops rows without a usable label."
        ),
    }
    write_processed_dataset(processed_dir, features_df, labels_df, metadata)
    print_prepared_summary("PhishStorm", processed_dir, features_df, labels_df)


def build_phishstorm_processed_tables(records):
    """Build feature and label tables while keeping labels away from model input."""
    raw_df = pd.DataFrame(records)
    required_columns = {"domain", "label"}
    missing_columns = required_columns.difference(raw_df.columns)
    if missing_columns:
        raise ValueError(f"PhishStorm is missing columns: {sorted(missing_columns)}")

    working_df = raw_df[["domain", "label"]].copy()
    working_df["url"] = working_df["domain"].astype(str).str.strip()
    working_df["original_label"] = pd.to_numeric(working_df["label"], errors="coerce")
    working_df = working_df.dropna(subset=["url", "original_label"])
    working_df = working_df[working_df["url"] != ""].copy()
    working_df["original_label"] = working_df["original_label"].astype(int)
    working_df["project_label"] = working_df["original_label"].map(
        map_phishstorm_label_to_project_label
    )
    working_df = working_df.dropna(subset=["project_label"]).copy()
    working_df["project_label"] = working_df["project_label"].astype(int)
    working_df = working_df.reset_index(drop=True)
    working_df["row_id"] = working_df.index + 1

    working_df, cleaning_report = deduplicate_url_rows(
        working_df,
        url_column="url",
        label_column="project_label",
        dataset_name="PhishStorm",
    )
    working_df = working_df.reset_index(drop=True)
    working_df["row_id"] = working_df.index + 1

    features_df, labels_df = build_processed_tables_from_url_and_labels(
        working_df[["row_id", "url", "original_label", "project_label"]]
    )
    return features_df, labels_df, cleaning_report


def map_phishstorm_label_to_project_label(label):
    """PhishStorm uses 1=phishing and 0=legitimate; this project uses the reverse."""
    label = int(label)
    if label == 1:
        return 0
    if label == 0:
        return 1
    raise ValueError(f"Unexpected PhishStorm label: {label}")


def prepare_legitphish_dataset():
    """Prepare LegitPhish from its URL and ClassLabel columns."""
    raw_path = LEGITPHISH_RAW_DATASET_PATH
    processed_dir = EXTERNAL_DATA_DIR / "legitphish" / "processed"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing LegitPhish raw file: {raw_path}")

    print("Preparing LegitPhish...")
    print(f"  raw file: {raw_path.relative_to(PROJECT_ROOT)}")
    raw_df = pd.read_csv(raw_path, low_memory=False)
    features_df, labels_df, cleaning_report = build_legitphish_processed_tables(raw_df)
    metadata = {
        "dataset": "LegitPhish",
        "raw_file": str(raw_path.relative_to(PROJECT_ROOT)),
        "status": "prepared",
        "raw_rows_read": int(len(raw_df)),
        "usable_rows": int(len(features_df)),
        "dropped_rows": int(len(raw_df) - len(features_df)),
        "cleaning": cleaning_report,
        "url_column": "URL",
        "original_label_column": "ClassLabel",
        "original_label_mapping": {
            "0": "phishing",
            "1": "legitimate",
        },
        "project_label_mapping": {
            "0": "phishing",
            "1": "legitimate",
        },
        "important_note": (
            "The downloaded file includes its own extracted features, but this "
            "project ignores them and regenerates the same URL-only features used "
            "for the main PhiUSIIL experiment."
        ),
    }
    write_processed_dataset(processed_dir, features_df, labels_df, metadata)
    print_prepared_summary("LegitPhish", processed_dir, features_df, labels_df)


def build_legitphish_processed_tables(raw_df):
    """Build LegitPhish feature and label tables from URL/ClassLabel only."""
    required_columns = {"URL", "ClassLabel"}
    missing_columns = required_columns.difference(raw_df.columns)
    if missing_columns:
        raise ValueError(f"LegitPhish is missing columns: {sorted(missing_columns)}")

    working_df = raw_df[["URL", "ClassLabel"]].copy()
    working_df["url"] = working_df["URL"].astype(str).str.strip()
    working_df["original_label"] = pd.to_numeric(
        working_df["ClassLabel"], errors="coerce"
    )
    working_df = working_df.dropna(subset=["url", "original_label"])
    working_df = working_df[working_df["url"] != ""].copy()
    working_df["original_label"] = working_df["original_label"].astype(int)
    working_df["project_label"] = working_df["original_label"].map(
        map_legitphish_label_to_project_label
    )
    working_df = working_df.dropna(subset=["project_label"]).copy()
    working_df["project_label"] = working_df["project_label"].astype(int)
    working_df = working_df.reset_index(drop=True)
    working_df["row_id"] = working_df.index + 1

    working_df, cleaning_report = deduplicate_url_rows(
        working_df,
        url_column="url",
        label_column="project_label",
        dataset_name="LegitPhish",
    )
    working_df = working_df.reset_index(drop=True)
    working_df["row_id"] = working_df.index + 1

    features_df, labels_df = build_processed_tables_from_url_and_labels(
        working_df[["row_id", "url", "original_label", "project_label"]]
    )
    return features_df, labels_df, cleaning_report


def map_legitphish_label_to_project_label(label):
    """LegitPhish already matches this project: 0=phishing and 1=legitimate."""
    label = int(label)
    if label in (0, 1):
        return label
    raise ValueError(f"Unexpected LegitPhish label: {label}")


def build_processed_tables_from_url_and_labels(label_df):
    """Generate project features and keep labels in a separate comparison table."""
    feature_rows = [extract_features(url) for url in label_df["url"]]
    features_df = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    features_df.insert(1, "url_normalized", label_df["url"].astype(str).str.strip().to_list())
    features_df.insert(0, "url", label_df["url"].to_list())
    features_df.insert(0, "row_id", label_df["row_id"].to_list())

    labels_df = label_df.copy()
    labels_df["url_normalized"] = labels_df["url"].astype(str).str.strip()
    labels_df = labels_df[
        ["row_id", "url", "url_normalized", "original_label", "project_label"]
    ].reset_index(drop=True)
    return features_df, labels_df


def write_processed_dataset(processed_dir, features_df, labels_df, metadata):
    """Write the feature, label, and metadata files for one external dataset."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(processed_dir / FEATURES_FILENAME, index=False)
    labels_df.to_csv(processed_dir / LABELS_FILENAME, index=False)
    (processed_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def print_prepared_summary(dataset_name, processed_dir, features_df, labels_df):
    """Print a concise terminal summary for the user running the script."""
    print(f"  status: prepared")
    print(f"  usable rows: {len(features_df):,}")
    print(f"  URL-only features: {len(FEATURE_NAMES)}")
    print(
        "  project label counts: "
        f"{labels_df['project_label'].value_counts().sort_index().to_dict()}"
    )
    print(f"  feature file: {(processed_dir / FEATURES_FILENAME).relative_to(PROJECT_ROOT)}")
    print(f"  label file: {(processed_dir / LABELS_FILENAME).relative_to(PROJECT_ROOT)}")
    print(f"  metadata: {(processed_dir / METADATA_FILENAME).relative_to(PROJECT_ROOT)}")
    print()


if __name__ == "__main__":
    main()
