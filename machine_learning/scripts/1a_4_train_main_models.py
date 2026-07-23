"""Train all main-dataset baseline models and save the outputs.

This script uses the cleaned PhiUSIIL feature dataset, performs the fixed
80/20 split, evaluates each model, and writes reports plus saved joblib files.
"""

import json
import sys
from time import perf_counter
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - covered by the helpful runtime error.
    XGBClassifier = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.scripts.helpers.project_paths import (
    EXPERIMENT_1A_DIR,
    MAIN_MODELS_DIR,
    MAIN_PROCESSED_DATASET_PATH,
    MAIN_TRAINING_CONFIG_PATH,
    PROJECT_ROOT,
)

from machine_learning.scripts.core.model_evaluation import (
    evaluate_predictions,
    select_best_model,
)
from machine_learning.scripts.core.model_artifacts import (
    best_model_artifact_filename as standard_best_model_artifact_filename,
    model_artifact_filename as standard_model_artifact_filename,
    slugify,
)


# Project paths are defined once so the script works no matter where it is run from.
DATASET_PATH = MAIN_PROCESSED_DATASET_PATH
REPORTS_DIR = EXPERIMENT_1A_DIR
MODELS_DIR = MAIN_MODELS_DIR
CONFIG_PATH = MAIN_TRAINING_CONFIG_PATH

# Fixed experiment settings keep the model comparison repeatable.
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATASET_NAME = "phiusiil_main"
LABEL_COLUMN = "label"
LABEL_ORDER = [0, 1]
LABEL_NAMES = {
    0: "phishing",
    1: "legitimate",
}
DEFAULT_TRAINING_CONFIG = {
    "dataset_name": DATASET_NAME,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "models": "all",
}


def main():
    """Train, test, compare, report, and save the project models."""
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    training_config = load_training_config()
    dataset_name = training_config["dataset_name"]
    test_size = float(training_config["test_size"])
    random_state = int(training_config["random_state"])

    print("Step 4: Train and evaluate models")
    print(f"Dataset: {DATASET_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Config: {CONFIG_PATH.relative_to(PROJECT_ROOT)}")

    # Load the processed URL-only feature table created from the raw URL column.
    dataset = pd.read_csv(DATASET_PATH)
    feature_columns = [column for column in dataset.columns if column != LABEL_COLUMN]

    x = dataset[feature_columns]
    y = dataset[LABEL_COLUMN]
    label_counts = y.value_counts().sort_index().to_dict()
    print(f"Rows loaded: {len(dataset):,}")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Label counts: {label_counts} (0=phishing, 1=legitimate)")

    # Use one fixed split so every model is evaluated on the same test rows.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(
        f"Train/test split: {len(x_train):,} train rows, {len(x_test):,} test rows "
        f"({int((1 - test_size) * 100)}/{int(test_size * 100)})"
    )
    print()

    models = select_models(build_model_registry(), training_config["models"])

    metric_rows = []  # One dictionary per model, later written to test_summary.csv.
    confusion_matrices = {}  # Raw actual-vs-predicted counts for each model.
    saved_model_files = {}  # Tracks where each trained model was saved.

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        train_start = perf_counter()
        model.fit(x_train, y_train)
        train_seconds = perf_counter() - train_start

        # Testing happens here: the trained model predicts the held-out test rows.
        predict_start = perf_counter()
        predictions = model.predict(x_test)
        predict_seconds = perf_counter() - predict_start

        metrics = evaluate_predictions(y_test, predictions)
        metric_rows.append(
            {
                "model": model_name,
                "accuracy": float(metrics["accuracy"]),
                "phishing_precision": float(metrics["phishing_precision"]),
                "phishing_recall": float(metrics["phishing_recall"]),
                "phishing_f1": float(metrics["phishing_f1"]),
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
            }
        )

        # The confusion matrix records true/false positives and negatives.
        confusion_matrices[model_name] = {
            "label_order": LABEL_ORDER,
            "label_names": LABEL_NAMES,
            "matrix": confusion_matrix(
                y_test, predictions, labels=LABEL_ORDER
            ).tolist(),
        }

        # Save every trained model, not just the winner, for traceability.
        model_path = MODELS_DIR / model_artifact_filename(
            model_name,
            dataset_name=dataset_name,
            test_size=test_size,
        )
        joblib.dump(model, model_path)
        saved_model_files[model_name] = str(model_path.relative_to(PROJECT_ROOT))
        print(
            "  "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"phishing_recall={metrics['phishing_recall']:.4f}, "
            f"phishing_f1={metrics['phishing_f1']:.4f}, "
            f"train={train_seconds:.4f}s, predict={predict_seconds:.4f}s"
        )
        print(f"  saved: {model_path.relative_to(PROJECT_ROOT)}")

    # Pick the best model after all models have been evaluated on the same test set.
    best_metrics = select_best_model(metric_rows)
    best_model_name = best_metrics["model"]
    best_model = models[best_model_name]
    best_model_file = MODELS_DIR / best_model_artifact_filename(
        best_model_name,
        dataset_name=dataset_name,
        test_size=test_size,
    )

    # Sort the table so the strongest model appears first in the report.
    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        by=["phishing_f1", "phishing_recall", "accuracy"],
        ascending=False,
    )
    metrics_frame.to_csv(REPORTS_DIR / "test_summary.csv", index=False)

    # Save machine-readable outputs as JSON/CSV and human-readable output as Markdown.
    with (REPORTS_DIR / "confusion_matrices.json").open("w", encoding="utf-8") as file:
        json.dump(confusion_matrices, file, indent=2)

    metadata = {
        "best_model": best_model_name,
        "selection_rule": "Highest phishing_f1, then phishing_recall, then accuracy.",
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "dataset_name": dataset_name,
        "dataset_rows": int(len(dataset)),
        "feature_columns": feature_columns,
        "label_column": LABEL_COLUMN,
        "label_mapping": {
            "0": "phishing",
            "1": "legitimate",
        },
        "test_size": test_size,
        "random_state": random_state,
        "training_config": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
        "best_model_file": str(best_model_file.relative_to(PROJECT_ROOT)),
        "saved_model_files": saved_model_files,
        "svm_note": "LinearSVC is used as the SVM variant because full kernel SVM is not practical for this dataset size.",
        "xgboost_note": "XGBoost is included as a stronger gradient-boosted tree baseline.",
    }

    # Save a named copy of the selected model so the chosen artefact is obvious.
    joblib.dump(best_model, best_model_file)

    with (MODELS_DIR / "model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    with (MODELS_DIR / "feature_columns.json").open("w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2)

    report = build_markdown_report(
        metrics_frame=metrics_frame,
        confusion_matrices=confusion_matrices,
        metadata=metadata,
    )
    (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")

    print()
    print("Model training complete.")
    print(f"Best model: {best_model_name}")
    print(f"Best model file: {best_model_file.relative_to(PROJECT_ROOT)}")
    print(f"Metrics: {(REPORTS_DIR / 'test_summary.csv').relative_to(PROJECT_ROOT)}")
    print(f"Report: {(REPORTS_DIR / 'report.md').relative_to(PROJECT_ROOT)}")
    print(f"Metadata: {(MODELS_DIR / 'model_metadata.json').relative_to(PROJECT_ROOT)}")


def load_training_config(config_path=CONFIG_PATH):
    """Load main-training settings from JSON, with stable defaults."""
    config = DEFAULT_TRAINING_CONFIG.copy()
    if config_path.exists():
        config.update(json.loads(config_path.read_text(encoding="utf-8")))
    return config


def select_models(model_registry, selection):
    """Return the configured model subset while preserving registry order."""
    if selection == "all":
        return model_registry
    if isinstance(selection, str):
        requested = [selection]
    else:
        requested = list(selection)

    requested_slugs = {slugify(name) for name in requested}
    selected = {
        model_name: model
        for model_name, model in model_registry.items()
        if slugify(model_name) in requested_slugs
    }
    if not selected:
        raise ValueError(f"No matching models for selection: {selection}")
    return selected


def model_artifact_filename(model_name, dataset_name=DATASET_NAME, test_size=TEST_SIZE):
    """Create the standard trained-model filename for the main experiment."""
    return standard_model_artifact_filename(model_name, dataset_name, test_size)


def build_model_registry():
    """Create the models compared in the dissertation experiment."""
    if XGBClassifier is None:
        raise RuntimeError(
            "XGBoost is not installed. Run: python -m pip install -r "
            "machine_learning/requirements.txt"
        )

    # Pipelines include scaling only for models that benefit from scaled numeric input.
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
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
                        random_state=RANDOM_STATE,
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
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def best_model_artifact_filename(
    model_name,
    dataset_name=DATASET_NAME,
    test_size=TEST_SIZE,
):
    """Create the standard best-model alias filename."""
    return standard_best_model_artifact_filename(model_name, dataset_name, test_size)


def build_markdown_report(metrics_frame, confusion_matrices, metadata):
    """Build the Markdown report used as dissertation evidence."""
    dataset_rows = metadata.get("dataset_rows")
    svm_size_text = f"{dataset_rows:,} records" if dataset_rows else "this dataset size"
    lines = [
        "# Model Evaluation Report",
        "",
        "This report was generated by `machine_learning/scripts/1a_4_train_main_models.py`.",
        "",
        "## Experiment Setup",
        "",
        f"- Dataset: `{metadata['dataset']}`",
        f"- Train/test split: {int((1 - metadata['test_size']) * 100)}/{int(metadata['test_size'] * 100)}",
        f"- Random state: `{metadata['random_state']}`",
        "- Positive class for precision, recall, and F1-score: `0` = phishing",
        "- Label `1` means legitimate",
        f"- Linear SVM is used because full kernel SVM is not practical for {svm_size_text}.",
        "- XGBoost is included as a gradient-boosted tree baseline.",
        "- Training and prediction times are local machine measurements and may vary on another computer.",
        "",
        "## Testing Phase",
        "",
        "Testing happens immediately after each model is trained. The model is fitted only on the 80 percent training split, then it predicts labels for the untouched 20 percent test split. Those predictions are compared with the real `label` values to calculate accuracy, phishing precision, phishing recall, phishing F1-score, and the confusion matrix.",
        "",
        "## Metrics",
        "",
        "| Model | Accuracy | Phishing Precision | Phishing Recall | Phishing F1 | Training Time | Prediction Time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in metrics_frame.to_dict(orient="records"):
        lines.append(
            "| {model} | {accuracy:.4f} | {phishing_precision:.4f} | "
            "{phishing_recall:.4f} | {phishing_f1:.4f} | "
            "{train_seconds:.4f}s | {predict_seconds:.4f}s |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Best Model",
            "",
            f"Best model selected: **{metadata['best_model']}**.",
            "",
            "Selection rule:",
            "",
            f"- {metadata['selection_rule']}",
            "",
            "This project prioritises phishing F1-score and phishing recall because a false negative means a phishing URL is incorrectly treated as legitimate.",
            "",
            "## Confusion Matrices",
            "",
            "Rows are actual labels. Columns are predicted labels.",
            "",
            "Label order:",
            "",
            "- `0` = phishing",
            "- `1` = legitimate",
            "",
        ]
    )

    for model_name, payload in confusion_matrices.items():
        matrix = payload["matrix"]
        lines.extend(
            [
                f"### {model_name}",
                "",
                "| Actual / Predicted | 0 phishing | 1 legitimate |",
                "|---|---:|---:|",
                f"| 0 phishing | {matrix[0][0]} | {matrix[0][1]} |",
                f"| 1 legitimate | {matrix[1][0]} | {matrix[1][1]} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Saved Artefacts",
            "",
            "- `machine_learning/experiments/1a_train_on_main_test_on_main/test_summary.csv`",
            "- `machine_learning/experiments/1a_train_on_main_test_on_main/confusion_matrices.json`",
            "- `machine_learning/experiments/1a_train_on_main_test_on_main/report.md`",
            "- `machine_learning/trained_models/1a_train_on_main_test_on_main/model_metadata.json`",
            "- `machine_learning/trained_models/1a_train_on_main_test_on_main/feature_columns.json`",
            "",
            "Saved model files:",
            "",
        ]
    )

    for model_path in metadata["saved_model_files"].values():
        lines.append(f"- `{model_path}`")

    lines.extend(
        [
            f"- `{metadata['best_model_file']}`",
            "",
            f"`{metadata['best_model_file']}` names the best model explicitly.",
            "",
            "`machine_learning/trained_models/1a_train_on_main_test_on_main/*.joblib` files are generated local artefacts and are ignored by Git.",
            "",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
