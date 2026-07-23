import json
from time import perf_counter

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    XGBClassifier = None

from machine_learning.scripts.helpers import cross_dataset_config as config
from machine_learning.scripts.core.model_artifacts import (
    model_artifact_filename as standard_model_artifact_filename,
)


def model_artifact_filename(training_scenario, model_name):
    return standard_model_artifact_filename(
        model_name,
        training_scenario,
        config.load_cross_dataset_config()["test_size"],
    )


def load_train_splits():
    """Read dataset train splits, including the saved combined train file."""
    splits = {}
    for dataset_name in config.DATASETS:
        path = config.SPLITS_DIR / config.train_split_filename(dataset_name)
        if not path.exists():
            raise FileNotFoundError(f"Missing train split: {path}")
        splits[dataset_name] = pd.read_csv(path)
    combined_path = (
        config.SPLITS_DIR
        / config.COMBINED_SPLIT_FILES["combined_dataset_train"]["filename"]
    )
    if combined_path.exists():
        splits["combined_dataset"] = pd.read_csv(combined_path)
    return splits


def build_training_scenarios(train_splits):
    """Build single-source and combined-source training frames."""
    scenarios = {}
    for scenario_name, dataset_names in config.TRAINING_SCENARIOS.items():
        if scenario_name in train_splits:
            scenarios[scenario_name] = train_splits[scenario_name].copy()
        else:
            scenarios[scenario_name] = pd.concat(
                [train_splits[dataset_name] for dataset_name in dataset_names],
                ignore_index=True,
            )
    return scenarios


def build_model_registry():
    """Create the five model types used across the dissertation experiments."""
    if XGBClassifier is None:
        raise RuntimeError(
            "XGBoost is not installed. Run: python -m pip install -r "
            "machine_learning/requirements.txt"
        )
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "Linear SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearSVC(
                        class_weight="balanced",
                        dual=False,
                        max_iter=5000,
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_one_model(model, train_frame):
    """Fit one model and return the local training duration."""
    x_train = train_frame[config.FEATURE_COLUMNS]
    y_train = train_frame[config.LABEL_COLUMN]
    start = perf_counter()
    model.fit(x_train, y_train)
    return perf_counter() - start


def train_cross_dataset_models():
    """Train all five models under all four cross-dataset scenarios."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_config = config.load_cross_dataset_config()
    train_splits = load_train_splits()
    scenarios = build_training_scenarios(train_splits)
    selected_scenarios = config.select_names(
        scenarios.keys(), run_config["training_scenarios"]
    )
    model_registry = build_model_registry()
    selected_models = config.select_names(model_registry.keys(), run_config["models"])
    metadata = {
        "random_state": int(run_config["random_state"]),
        "test_size": float(run_config["test_size"]),
        "run_config": str(config.CONFIG_PATH.relative_to(config.PROJECT_ROOT)),
        "feature_columns": config.FEATURE_COLUMNS,
        "label_mapping": {"0": "phishing", "1": "legitimate"},
        "models": [],
    }

    print("Cross-dataset experiment step 2: train models")
    for scenario_name in selected_scenarios:
        train_frame = scenarios[scenario_name]
        print(f"Training scenario {scenario_name}: {len(train_frame):,} rows")
        for model_name in selected_models:
            model = model_registry[model_name]
            print(f"  Training {model_name}...")
            train_seconds = train_one_model(model, train_frame)
            model_path = config.MODELS_DIR / model_artifact_filename(
                scenario_name, model_name
            )
            joblib.dump(model, model_path)
            metadata["models"].append(
                {
                    "training_scenario": scenario_name,
                    "model": model_name,
                    "model_file": str(model_path.relative_to(config.PROJECT_ROOT)),
                    "train_rows": int(len(train_frame)),
                    "source_datasets": config.TRAINING_SCENARIOS[scenario_name],
                    "train_seconds": train_seconds,
                }
            )
            print(
                f"    train={train_seconds:.4f}s "
                f"saved={model_path.relative_to(config.PROJECT_ROOT)}"
            )

    metadata_path = config.MODELS_DIR / "cross_dataset_model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Metadata: {metadata_path.relative_to(config.PROJECT_ROOT)}")
    return metadata
