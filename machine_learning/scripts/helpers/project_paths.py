from __future__ import annotations

from pathlib import Path


MACHINE_LEARNING_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MACHINE_LEARNING_ROOT.parent
DOCS_ROOT = PROJECT_ROOT / "docs"

DATASETS_ROOT = MACHINE_LEARNING_ROOT / "datasets"
TRAINED_MODELS_ROOT = MACHINE_LEARNING_ROOT / "trained_models"
EXPERIMENTS_ROOT = MACHINE_LEARNING_ROOT / "experiments"
CONFIGS_ROOT = MACHINE_LEARNING_ROOT / "configs"
SCRIPTS_ROOT = MACHINE_LEARNING_ROOT / "scripts"

MAIN_DATASET_ROOT = DATASETS_ROOT / "main"
EXTERNAL_DATASETS_ROOT = DATASETS_ROOT / "external_testing"
SPLITS_ROOT = DATASETS_ROOT / "splits"

MAIN_RAW_DATASET_PATH = MAIN_DATASET_ROOT / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MAIN_RAW_ZIP_PATH = MAIN_DATASET_ROOT / "raw" / "phiusiil_phishing_url_dataset.zip"
MAIN_PROCESSED_DATASET_PATH = (
    MAIN_DATASET_ROOT / "processed" / "phishing_url_features.csv"
)
MAIN_CLEANED_URL_LABELS_PATH = (
    MAIN_DATASET_ROOT / "processed" / "cleaned_url_labels.csv"
)

LEGITPHISH_ROOT = EXTERNAL_DATASETS_ROOT / "legitphish"
PHISHSTORM_ROOT = EXTERNAL_DATASETS_ROOT / "phishstorm"

LEGITPHISH_RAW_DATASET_PATH = (
    LEGITPHISH_ROOT / "raw" / "LegitPhish_dataset_url_features_extracted1.csv"
)
LEGITPHISH_PROCESSED_FEATURES_PATH = (
    LEGITPHISH_ROOT / "processed" / "external_url_features_without_labels.csv"
)
LEGITPHISH_PROCESSED_LABELS_PATH = (
    LEGITPHISH_ROOT / "processed" / "external_url_labels_for_comparison.csv"
)
LEGITPHISH_METADATA_PATH = (
    LEGITPHISH_ROOT / "processed" / "external_dataset_metadata.json"
)

PHISHSTORM_RAW_DATASET_PATH = PHISHSTORM_ROOT / "raw" / "PhishStorm_urlset.csv"
PHISHSTORM_PROCESSED_FEATURES_PATH = (
    PHISHSTORM_ROOT / "processed" / "external_url_features_without_labels.csv"
)
PHISHSTORM_PROCESSED_LABELS_PATH = (
    PHISHSTORM_ROOT / "processed" / "external_url_labels_for_comparison.csv"
)
PHISHSTORM_METADATA_PATH = (
    PHISHSTORM_ROOT / "processed" / "external_dataset_metadata.json"
)

MAIN_MODELS_DIR = TRAINED_MODELS_ROOT / "1a_train_on_main_test_on_main"
CROSS_DATASET_MODELS_DIR = TRAINED_MODELS_ROOT / "2_cross_dataset_generalisation"
INTERACTIVE_MODELS_DIR = TRAINED_MODELS_ROOT / "interactive"

EXPERIMENT_1A_DIR = EXPERIMENTS_ROOT / "1a_train_on_main_test_on_main"
EXPERIMENT_1B_DIR = EXPERIMENTS_ROOT / "1b_train_on_main_test_on_others"
EXPERIMENT_1C_DIR = EXPERIMENTS_ROOT / "1c_train_on_other_datasets_test_on_other_datasets"
EXPERIMENT_2A_DIR = EXPERIMENTS_ROOT / "2a_train_on_each_dataset_test_on_combined"
EXPERIMENT_2B_DIR = EXPERIMENTS_ROOT / "2b_train_on_combined_test_on_combined"
EXPERIMENT_3_DIR = EXPERIMENTS_ROOT / "3_real_world_sanity_probe"
EXPERIMENT_4_DIR = EXPERIMENTS_ROOT / "4_google_slash_brittleness_check"
INTERACTIVE_RUNS_DIR = EXPERIMENTS_ROOT / "interactive_runs"

MAIN_TRAINING_CONFIG_PATH = CONFIGS_ROOT / "main_training_config.json"
EXTERNAL_TESTING_CONFIG_PATH = CONFIGS_ROOT / "external_testing_config.json"
CROSS_DATASET_CONFIG_PATH = CONFIGS_ROOT / "cross_dataset_config.json"

ALL_EXPERIMENTS_COMBINED_REPORT_HTML = (
    MACHINE_LEARNING_ROOT / "all_experiments_combined_report.html"
)
ALL_EXPERIMENTS_COMBINED_REPORT_JSON = (
    MACHINE_LEARNING_ROOT / "all_experiments_combined_report.json"
)
