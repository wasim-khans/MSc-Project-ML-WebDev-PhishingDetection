import json

import pandas as pd
from sklearn.model_selection import train_test_split

from machine_learning.scripts.helpers import cross_dataset_config as config
from machine_learning.scripts.core.dataset_cleaning import (
    remove_cross_source_duplicate_urls,
)


def validate_feature_columns(frame, dataset_name):
    """Ensure a dataset has the exact URL-only feature set and label column."""
    missing = [
        column
        for column in [*config.FEATURE_COLUMNS, config.LABEL_COLUMN]
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def merge_external_features_and_labels(features, labels, dataset_name):
    """Join prepared external URL features to held-back labels by row_id."""
    required_feature_columns = ["row_id", "url", *config.FEATURE_COLUMNS]
    missing_features = [
        column for column in required_feature_columns if column not in features
    ]
    missing_labels = [column for column in ["row_id", "project_label"] if column not in labels]
    if missing_features or missing_labels:
        raise ValueError(
            f"{dataset_name} missing feature columns {missing_features} "
            f"or label columns {missing_labels}"
        )

    merged = features.merge(
        labels[["row_id", "project_label"]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    if merged["project_label"].isna().any():
        raise ValueError(f"{dataset_name} has feature rows without comparison labels")

    if "url_normalized" not in merged.columns:
        merged["url_normalized"] = merged["url"].astype(str).str.strip()

    output = merged[
        [*config.FEATURE_COLUMNS, "project_label", "row_id", "url_normalized"]
    ].copy()
    output = output.rename(
        columns={
            "project_label": config.LABEL_COLUMN,
            "row_id": "source_row_id",
        }
    )
    output.insert(0, "source_dataset", dataset_name)
    output[config.LABEL_COLUMN] = output[config.LABEL_COLUMN].astype(int)
    return output


def load_main_dataset():
    """Load the main PhiUSIIL processed URL-only feature table."""
    frame = pd.read_csv(config.DATASETS["main"]["features_path"])
    url_labels = pd.read_csv(config.DATASETS["main"]["url_labels_path"])
    if len(frame) != len(url_labels):
        raise ValueError(
            "Main feature table and cleaned URL sidecar have different row counts: "
            f"features={len(frame)}, urls={len(url_labels)}"
        )
    frame = frame[[*config.FEATURE_COLUMNS, config.LABEL_COLUMN]].copy()
    frame.insert(0, "url_normalized", url_labels["url_normalized"].to_list())
    frame.insert(0, "source_row_id", url_labels["source_row_id"].to_list())
    frame.insert(0, "source_dataset", "main")
    frame[config.LABEL_COLUMN] = frame[config.LABEL_COLUMN].astype(int)
    validate_feature_columns(frame, "main")
    return frame


def load_external_dataset(dataset_name):
    """Load one prepared external dataset and attach project labels."""
    dataset = config.DATASETS[dataset_name]
    features = pd.read_csv(dataset["features_path"])
    labels = pd.read_csv(dataset["labels_path"])
    frame = merge_external_features_and_labels(features, labels, dataset_name)
    validate_feature_columns(frame, dataset_name)
    return frame


def load_all_datasets():
    """Load all three datasets used in the cross-dataset experiment."""
    return {
        "main": load_main_dataset(),
        "legitphish": load_external_dataset("legitphish"),
        "phishstorm": load_external_dataset("phishstorm"),
    }


def split_dataset(frame, dataset_name):
    """Create a repeatable stratified 80/20 split for one dataset."""
    run_config = config.load_cross_dataset_config()
    if "url_normalized" in frame.columns:
        groups = (
            frame[["url_normalized", config.LABEL_COLUMN]]
            .drop_duplicates(subset=["url_normalized"])
            .reset_index(drop=True)
        )
        train_groups, test_groups = train_test_split(
            groups,
            test_size=float(run_config["test_size"]),
            random_state=int(run_config["random_state"]),
            stratify=groups[config.LABEL_COLUMN],
        )
        train_urls = set(train_groups["url_normalized"])
        test_urls = set(test_groups["url_normalized"])
        train = frame[frame["url_normalized"].isin(train_urls)]
        test = frame[frame["url_normalized"].isin(test_urls)]
        return train.sort_index().reset_index(drop=True), test.sort_index().reset_index(
            drop=True
        )

    train, test = train_test_split(
        frame,
        test_size=float(run_config["test_size"]),
        random_state=int(run_config["random_state"]),
        stratify=frame[config.LABEL_COLUMN],
    )
    return train.sort_index().reset_index(drop=True), test.sort_index().reset_index(
        drop=True
    )


def split_metadata_entry(dataset_name, source_frame, train_frame, test_frame):
    """Summarise split sizes and label counts for documentation."""
    return {
        "dataset": dataset_name,
        "display_name": config.DATASETS[dataset_name]["display_name"],
        "source_rows": int(len(source_frame)),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "source_label_counts": label_counts(source_frame),
        "train_label_counts": label_counts(train_frame),
        "test_label_counts": label_counts(test_frame),
        "source_unique_urls": unique_url_count(source_frame),
        "train_unique_urls": unique_url_count(train_frame),
        "test_unique_urls": unique_url_count(test_frame),
        "train_test_url_overlap": url_overlap_count(train_frame, test_frame),
    }


def label_counts(frame):
    counts = frame[config.LABEL_COLUMN].value_counts().sort_index().to_dict()
    return {str(label): int(counts.get(label, 0)) for label in config.LABEL_ORDER}


def build_combined_split_frames(train_splits, test_splits):
    """Create saved combined views from the already-created split frames."""
    source_names = config.TRAINING_SCENARIOS["combined_dataset"]
    raw_combined_train = pd.concat(
        [train_splits[dataset_name] for dataset_name in source_names],
        ignore_index=True,
    )
    raw_combined_test = pd.concat(
        [test_splits[dataset_name] for dataset_name in source_names],
        ignore_index=True,
    )
    combined_train, _ = remove_cross_source_duplicate_urls(
        raw_combined_train,
        source_priority=source_names,
    )
    combined_test, _ = remove_cross_source_duplicate_urls(
        raw_combined_test,
        source_priority=source_names,
    )
    complete_combined = pd.concat(
        [combined_train, combined_test],
        ignore_index=True,
    )
    complete_combined, _ = remove_cross_source_duplicate_urls(
        complete_combined,
        source_priority=source_names,
    )
    return {
        "combined_dataset_train": combined_train,
        "combined_test": combined_test,
        "complete_combined_dataset": complete_combined,
    }


def combined_metadata_entry(split_name, frame):
    """Summarise a combined split file for repeatable documentation."""
    return {
        "file": config.COMBINED_SPLIT_FILES[split_name]["filename"],
        "description": config.COMBINED_SPLIT_FILES[split_name]["description"],
        "rows": int(len(frame)),
        "source_datasets": sorted(frame["source_dataset"].unique().tolist()),
        "unique_urls": unique_url_count(frame),
        "label_counts": label_counts(frame),
    }


def unique_url_count(frame):
    if "url_normalized" not in frame.columns:
        return None
    return int(frame["url_normalized"].nunique())


def url_overlap_count(left_frame, right_frame):
    if "url_normalized" not in left_frame.columns or "url_normalized" not in right_frame.columns:
        return None
    return int(
        len(set(left_frame["url_normalized"]) & set(right_frame["url_normalized"]))
    )


def build_cross_dataset_splits():
    """Build and save all cross-dataset train/test split files."""
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    run_config = config.load_cross_dataset_config()
    datasets = load_all_datasets()
    train_splits = {}
    test_splits = {}
    metadata = {
        "random_state": int(run_config["random_state"]),
        "test_size": float(run_config["test_size"]),
        "run_config": str(config.CONFIG_PATH.relative_to(config.PROJECT_ROOT)),
        "label_mapping": {"0": "phishing", "1": "legitimate"},
        "feature_columns": config.FEATURE_COLUMNS,
        "datasets": {},
        "combined_split_files": {},
    }

    print("Cross-dataset experiment step 1: build dataset splits")
    for dataset_name, frame in datasets.items():
        train, test = split_dataset(frame, dataset_name)
        train_splits[dataset_name] = train
        test_splits[dataset_name] = test
        train_path = config.SPLITS_DIR / config.train_split_filename(dataset_name)
        test_path = config.SPLITS_DIR / config.test_split_filename(dataset_name)
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        metadata["datasets"][dataset_name] = split_metadata_entry(
            dataset_name,
            frame,
            train,
            test,
        )
        print(
            f"{dataset_name}: {len(train):,} train rows, {len(test):,} test rows, "
            f"labels {label_counts(frame)}"
        )

    combined_frames = build_combined_split_frames(train_splits, test_splits)
    for split_name, frame in combined_frames.items():
        combined_path = (
            config.SPLITS_DIR / config.COMBINED_SPLIT_FILES[split_name]["filename"]
        )
        frame.to_csv(combined_path, index=False)
        metadata["combined_split_files"][split_name] = combined_metadata_entry(
            split_name, frame
        )
        print(
            f"{split_name}: {len(frame):,} rows, labels {label_counts(frame)}, "
            f"saved={combined_path.relative_to(config.PROJECT_ROOT)}"
        )

    metadata_path = config.SPLITS_DIR / "split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Metadata: {metadata_path.relative_to(config.PROJECT_ROOT)}")
    return metadata
