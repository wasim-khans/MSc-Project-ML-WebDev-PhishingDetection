import json
import re

from machine_learning.scripts.helpers.project_paths import (
    CROSS_DATASET_CONFIG_PATH,
    CROSS_DATASET_MODELS_DIR,
    EXPERIMENT_1C_DIR,
    PROJECT_ROOT,
    SPLITS_ROOT,
    EXTERNAL_DATASETS_ROOT,
    MAIN_CLEANED_URL_LABELS_PATH,
    MAIN_PROCESSED_DATASET_PATH,
)

CONFIG_PATH = CROSS_DATASET_CONFIG_PATH
SPLITS_DIR = SPLITS_ROOT
MODELS_DIR = CROSS_DATASET_MODELS_DIR
REPORTS_DIR = EXPERIMENT_1C_DIR

RANDOM_STATE = 42
TEST_SIZE = 0.2
LABEL_COLUMN = "label"
LABEL_ORDER = [0, 1]
LABEL_NAMES = {
    0: "phishing",
    1: "legitimate",
}

FEATURE_COLUMNS = [
    "url_length",
    "domain_length",
    "path_length",
    "dot_count",
    "hyphen_count",
    "digit_count",
    "special_char_count",
    "has_https",
    "has_ip_address",
    "has_at_symbol",
    "subdomain_count",
    "query_param_count",
    "suspicious_word_count",
    "tld_length",
    "has_url_shortener",
]

DATASETS = {
    "main": {
        "display_name": "PhiUSIIL main",
        "split_slug": "phiusiil_main",
        "features_path": MAIN_PROCESSED_DATASET_PATH,
        "url_labels_path": MAIN_CLEANED_URL_LABELS_PATH,
        "labels_path": None,
    },
    "legitphish": {
        "display_name": "LegitPhish",
        "split_slug": "legitphish",
        "features_path": EXTERNAL_DATASETS_ROOT
        / "legitphish"
        / "processed"
        / "external_url_features_without_labels.csv",
        "labels_path": EXTERNAL_DATASETS_ROOT
        / "legitphish"
        / "processed"
        / "external_url_labels_for_comparison.csv",
    },
    "phishstorm": {
        "display_name": "PhishStorm",
        "split_slug": "phishstorm",
        "features_path": EXTERNAL_DATASETS_ROOT
        / "phishstorm"
        / "processed"
        / "external_url_features_without_labels.csv",
        "labels_path": EXTERNAL_DATASETS_ROOT
        / "phishstorm"
        / "processed"
        / "external_url_labels_for_comparison.csv",
    },
}

TRAINING_SCENARIOS = {
    "phiusiil_main": ["main"],
    "legitphish": ["legitphish"],
    "phishstorm": ["phishstorm"],
    "combined_dataset": ["main", "legitphish", "phishstorm"],
}

TEST_SETS = {
    "phiusiil_main_test": ["main"],
    "legitphish_test": ["legitphish"],
    "phishstorm_test": ["phishstorm"],
    "combined_test": ["main", "legitphish", "phishstorm"],
}

COMPLETE_TEST_SETS = {
    "complete_combined_dataset": ["main", "legitphish", "phishstorm"],
}

COMBINED_SPLIT_FILES = {
    "combined_dataset_train": {
        "filename": "combined_dataset_80_train_dataset.csv",
        "description": "Combined 80 percent training split for all cross-dataset experiment datasets.",
    },
    "combined_test": {
        "filename": "combined_dataset_20_test_dataset.csv",
        "description": "Combined untouched 20 percent test split for all cross-dataset experiment datasets.",
    },
    "complete_combined_dataset": {
        "filename": "combined_dataset_complete_diagnostic_dataset.csv",
        "description": "Full combined diagnostic dataset, including train and test rows.",
    },
}
DEFAULT_CROSS_DATASET_CONFIG = {
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "models": "all",
    "training_scenarios": list(TRAINING_SCENARIOS.keys()),
    "test_sets": list(TEST_SETS.keys()),
    "complete_test_sets": list(COMPLETE_TEST_SETS.keys()),
}


def load_cross_dataset_config(config_path=CONFIG_PATH):
    """Load cross-dataset experiment settings from JSON, with stable defaults."""
    run_config = DEFAULT_CROSS_DATASET_CONFIG.copy()
    if config_path.exists():
        run_config.update(json.loads(config_path.read_text(encoding="utf-8")))
    return run_config


def normalise_selection_text(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def select_names(available_names, selection):
    """Select names from a configured all/string/list value."""
    available_names = list(available_names)
    if selection == "all":
        return available_names
    requested = [selection] if isinstance(selection, str) else list(selection)
    if not requested:
        return []
    requested_slugs = {normalise_selection_text(item) for item in requested}
    selected = [
        name
        for name in available_names
        if normalise_selection_text(name) in requested_slugs
    ]
    if not selected:
        raise ValueError(f"No matching names for selection: {selection}")
    return selected


def train_split_filename(dataset_name):
    split_slug = DATASETS[dataset_name]["split_slug"]
    return f"{split_slug}_80_train_dataset.csv"


def test_split_filename(dataset_name):
    split_slug = DATASETS[dataset_name]["split_slug"]
    return f"{split_slug}_20_test_dataset.csv"
