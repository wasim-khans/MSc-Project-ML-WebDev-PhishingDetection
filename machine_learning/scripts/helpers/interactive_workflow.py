import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
except ImportError:  # pragma: no cover - depends on local environment.
    XGBClassifier = None

from machine_learning.scripts.helpers.project_paths import (
    CROSS_DATASET_MODELS_DIR,
    EXTERNAL_DATASETS_ROOT,
    INTERACTIVE_MODELS_DIR,
    INTERACTIVE_RUNS_DIR,
    MAIN_MODELS_DIR,
    MAIN_PROCESSED_DATASET_PATH,
    PHISHSTORM_PROCESSED_FEATURES_PATH,
    PHISHSTORM_PROCESSED_LABELS_PATH,
    LEGITPHISH_PROCESSED_FEATURES_PATH,
    LEGITPHISH_PROCESSED_LABELS_PATH,
    PROJECT_ROOT,
    SPLITS_ROOT,
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.scripts.core.model_artifacts import slugify
from machine_learning.scripts.core.model_evaluation import evaluate_predictions
from machine_learning.scripts.core.url_feature_extractor import extract_features
from machine_learning.scripts.helpers import cross_dataset_config as config

INTERACTIVE_REPORTS_DIR = INTERACTIVE_RUNS_DIR
LABEL_COLUMN = "label"
LABEL_ALIASES = ["label", "project_label", "Label", "target", "class"]
URL_COLUMN_ALIASES = ["url", "URL", "Url"]
RANDOM_STATE = config.RANDOM_STATE


@dataclass(frozen=True)
class DatasetOption:
    key: str
    display_name: str
    features_path: Path
    labels_path: Path | None = None
    label_column: str | None = None


@dataclass(frozen=True)
class ModelFileOption:
    key: str
    display_name: str
    model_path: Path


def build_model_registry(random_state=RANDOM_STATE):
    """Return fresh, untrained model builders for interactive training."""
    registry = {
        "Logistic Regression": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Decision Tree": lambda: DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Linear SVM": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearSVC(
                        class_weight="balanced",
                        dual=False,
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    if XGBClassifier is not None:
        registry["XGBoost"] = lambda: XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    return registry


def default_training_datasets():
    splits_dir = SPLITS_ROOT
    return [
        DatasetOption(
            key="phiusiil_full",
            display_name="PhiUSIIL main processed full dataset",
            features_path=MAIN_PROCESSED_DATASET_PATH,
        ),
        DatasetOption(
            key="phiusiil_train_split",
            display_name="PhiUSIIL main 80 percent train split",
            features_path=splits_dir / "phiusiil_main_80_train_dataset.csv",
        ),
        DatasetOption(
            key="legitphish_train_split",
            display_name="LegitPhish 80 percent train split",
            features_path=splits_dir / "legitphish_80_train_dataset.csv",
        ),
        DatasetOption(
            key="phishstorm_train_split",
            display_name="PhishStorm 80 percent train split",
            features_path=splits_dir / "phishstorm_80_train_dataset.csv",
        ),
        DatasetOption(
            key="combined_train_split",
            display_name="Combined 80 percent train split",
            features_path=splits_dir / "combined_dataset_80_train_dataset.csv",
        ),
        DatasetOption(
            key="legitphish_full",
            display_name="LegitPhish full prepared dataset",
            features_path=LEGITPHISH_PROCESSED_FEATURES_PATH,
            labels_path=LEGITPHISH_PROCESSED_LABELS_PATH,
            label_column="project_label",
        ),
        DatasetOption(
            key="phishstorm_full",
            display_name="PhishStorm full prepared dataset",
            features_path=PHISHSTORM_PROCESSED_FEATURES_PATH,
            labels_path=PHISHSTORM_PROCESSED_LABELS_PATH,
            label_column="project_label",
        ),
    ]


def default_testing_datasets():
    splits_dir = SPLITS_ROOT
    return [
        DatasetOption(
            key="phiusiil_test_split",
            display_name="PhiUSIIL held-out test split",
            features_path=splits_dir / "phiusiil_main_20_test_dataset.csv",
        ),
        DatasetOption(
            key="legitphish_test_split",
            display_name="LegitPhish held-out test split",
            features_path=splits_dir / "legitphish_20_test_dataset.csv",
        ),
        DatasetOption(
            key="phishstorm_test_split",
            display_name="PhishStorm held-out test split",
            features_path=splits_dir / "phishstorm_20_test_dataset.csv",
        ),
        DatasetOption(
            key="combined_test_split",
            display_name="Combined held-out test split",
            features_path=splits_dir / "combined_dataset_20_test_dataset.csv",
        ),
        DatasetOption(
            key="complete_combined_dataset",
            display_name="Complete combined dataset diagnostic set",
            features_path=splits_dir / "combined_dataset_complete_diagnostic_dataset.csv",
        ),
        *default_training_datasets(),
    ]


def discover_default_model_files():
    model_roots = [
        INTERACTIVE_MODELS_DIR,
        CROSS_DATASET_MODELS_DIR,
        MAIN_MODELS_DIR,
    ]
    options = []
    seen = set()
    for root in model_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.joblib")):
            if path.name.startswith("best_model_"):
                continue
            key = str(path.relative_to(PROJECT_ROOT))
            if key in seen:
                continue
            seen.add(key)
            options.append(
                ModelFileOption(
                    key=key,
                    display_name=path.stem,
                    model_path=path,
                )
            )
    return options


def load_dataset(option_or_path, labels_path=None, label_column=None):
    """Load either feature CSVs or raw URL+label CSVs into X/y frames."""
    path = Path(option_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    frame = pd.read_csv(path)
    label_series = _read_labels(frame, labels_path=labels_path, label_column=label_column)
    feature_frame = _read_features(frame)

    if len(feature_frame) != len(label_series):
        raise ValueError(
            "Feature row count and label row count do not match: "
            f"{len(feature_frame)} != {len(label_series)}"
        )

    return feature_frame, label_series.astype(int)


def _read_labels(frame, labels_path=None, label_column=None):
    label_column = label_column or _find_first_column(frame, LABEL_ALIASES)
    if label_column:
        return frame[label_column]

    if labels_path:
        labels_frame = pd.read_csv(labels_path)
        labels_column = _find_first_column(labels_frame, LABEL_ALIASES)
        if not labels_column:
            raise ValueError(
                f"Could not find a label column in labels file: {labels_path}"
            )
        return labels_frame[labels_column]

    raise ValueError(
        "Could not find a label column. Expected one of: "
        f"{', '.join(LABEL_ALIASES)}. For separate labels, provide a labels path."
    )


def _read_features(frame):
    if all(column in frame.columns for column in config.FEATURE_COLUMNS):
        return frame[config.FEATURE_COLUMNS].copy()

    url_column = _find_first_column(frame, URL_COLUMN_ALIASES)
    if not url_column:
        missing = [column for column in config.FEATURE_COLUMNS if column not in frame.columns]
        raise ValueError(
            "Dataset is missing URL-only feature columns and no URL column was found. "
            f"Missing feature examples: {missing[:5]}"
        )

    feature_rows = [extract_features(url) for url in frame[url_column].astype(str)]
    return pd.DataFrame(feature_rows)[config.FEATURE_COLUMNS]


def _find_first_column(frame, candidates):
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def train_and_save_model(
    model_name,
    dataset_path,
    output_name,
    labels_path=None,
    label_column=None,
    dataset_name=None,
    output_dir=INTERACTIVE_MODELS_DIR,
):
    registry = build_model_registry()
    if model_name not in registry:
        raise ValueError(f"Unknown model: {model_name}")

    features, labels = load_dataset(
        dataset_path,
        labels_path=labels_path,
        label_column=label_column,
    )
    model = registry[model_name]()
    start = perf_counter()
    model.fit(features, labels)
    train_seconds = perf_counter() - start

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / normalise_joblib_filename(output_name)
    joblib.dump(model, model_path)

    metadata_path = model_path.with_suffix(".metadata.json")
    metadata = {
        "model_name": model_name,
        "dataset_name": dataset_name or Path(dataset_path).stem,
        "dataset_path": display_path(Path(dataset_path)),
        "labels_path": display_path(Path(labels_path)) if labels_path else None,
        "model_file": display_path(model_path),
        "rows_trained": int(len(features)),
        "feature_columns": config.FEATURE_COLUMNS,
        "label_mapping": {"0": "phishing", "1": "legitimate"},
        "train_seconds": train_seconds,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model_name": model_name,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "rows_trained": int(len(features)),
        "label_counts": labels.value_counts().sort_index().to_dict(),
        "train_seconds": train_seconds,
    }


def evaluate_model_file(model_path, dataset_path, labels_path=None, label_column=None):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    features, labels = load_dataset(
        dataset_path,
        labels_path=labels_path,
        label_column=label_column,
    )
    model = joblib.load(model_path)
    start = perf_counter()
    predictions = model.predict(features)
    predict_seconds = perf_counter() - start
    metrics = evaluate_predictions(labels, predictions)

    return {
        "model_path": model_path,
        "dataset_path": Path(dataset_path),
        "rows_tested": int(len(features)),
        "label_counts": labels.value_counts().sort_index().to_dict(),
        "accuracy": float(metrics["accuracy"]),
        "phishing_precision": float(metrics["phishing_precision"]),
        "phishing_recall": float(metrics["phishing_recall"]),
        "phishing_f1": float(metrics["phishing_f1"]),
        "predict_seconds": predict_seconds,
    }


def write_test_report(result):
    INTERACTIVE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = INTERACTIVE_REPORTS_DIR / f"interactive_test_{timestamp}.json"
    payload = {
        **result,
        "model_path": display_path(result["model_path"]),
        "dataset_path": display_path(result["dataset_path"]),
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def normalise_joblib_filename(value):
    value = str(value).strip()
    if not value:
        raise ValueError("Model filename must not be blank.")
    path = Path(value)
    if path.suffix != ".joblib":
        value = f"{value}.joblib"
    return Path(value).name


def default_output_name(model_name, dataset_name):
    return f"{slugify(model_name)}_T_ON_{slugify(dataset_name)}_interactive.joblib"


def prompt_choice(title, options, custom_label=None):
    print()
    print(title)
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option.display_name}")
    if custom_label:
        print(f"  {len(options) + 1}. {custom_label}")

    while True:
        answer = input("Choose a number: ").strip()
        if answer.isdigit():
            selected = int(answer)
            if 1 <= selected <= len(options):
                return options[selected - 1]
            if custom_label and selected == len(options) + 1:
                return None
        print("Please choose one of the listed numbers.")


def prompt_text(message, default=None):
    suffix = f" [{default}]" if default else ""
    answer = input(f"{message}{suffix}: ").strip()
    return answer or default


def dataset_from_key_or_prompt(dataset_key, dataset_path, labels_path, label_column, options):
    if dataset_path:
        return DatasetOption(
            key="custom",
            display_name=Path(dataset_path).stem,
            features_path=Path(dataset_path),
            labels_path=Path(labels_path) if labels_path else None,
            label_column=label_column,
        )
    if dataset_key:
        for option in options:
            if option.key == dataset_key:
                return option
        raise ValueError(f"Unknown dataset key: {dataset_key}")
    selected = prompt_choice("Choose dataset", options, custom_label="Use custom CSV path")
    if selected:
        return selected
    custom_path = prompt_text("Dataset CSV path")
    custom_labels_path = prompt_text("Separate labels CSV path, if needed", default="")
    custom_label_column = prompt_text("Label column name, if not label/project_label", default="")
    return DatasetOption(
        key="custom",
        display_name=Path(custom_path).stem,
        features_path=Path(custom_path),
        labels_path=Path(custom_labels_path) if custom_labels_path else None,
        label_column=custom_label_column or None,
    )


def model_name_from_key_or_prompt(model_name):
    registry = build_model_registry()
    if model_name:
        for available_name in registry:
            if slugify(available_name) == slugify(model_name):
                return available_name
        raise ValueError(f"Unknown model: {model_name}")

    options = [
        DatasetOption(
            key=slugify(name),
            display_name=name,
            features_path=Path(""),
        )
        for name in registry
    ]
    selected = prompt_choice("Choose model to train", options)
    return selected.display_name


def model_file_from_key_or_prompt(model_key, model_path):
    if model_path:
        return Path(model_path)
    options = discover_default_model_files()
    if model_key:
        for option in options:
            if option.key == model_key or slugify(option.display_name) == slugify(model_key):
                return option.model_path
        raise ValueError(f"Unknown model file choice: {model_key}")
    selected = prompt_choice("Choose trained model file", options, custom_label="Use custom .joblib path")
    if selected:
        return selected.model_path
    return Path(prompt_text("Trained model .joblib path"))


def print_training_summary(result):
    print()
    print("Training complete.")
    print(f"Model: {result['model_name']}")
    print(f"Rows trained: {result['rows_trained']:,}")
    print(f"Label counts: {result['label_counts']} (0=phishing, 1=legitimate)")
    print(f"Training time: {result['train_seconds']:.4f}s")
    print(f"Joblib file: {display_path(result['model_path'])}")
    print(f"Metadata file: {display_path(result['metadata_path'])}")


def print_test_summary(result):
    rows = [
        ("Rows tested", f"{result['rows_tested']:,}"),
        ("Accuracy", f"{result['accuracy']:.4f}"),
        ("Phishing precision", f"{result['phishing_precision']:.4f}"),
        ("Phishing recall", f"{result['phishing_recall']:.4f}"),
        ("Phishing F1", f"{result['phishing_f1']:.4f}"),
        ("Prediction time", f"{result['predict_seconds']:.4f}s"),
    ]
    width = max(len(label) for label, _ in rows)
    print()
    print("Testing summary")
    print("-" * (width + 18))
    for label, value in rows:
        print(f"{label:<{width}}  {value}")
    print("-" * (width + 18))
    print(f"Model file: {display_path(result['model_path'])}")
    print(f"Dataset: {display_path(result['dataset_path'])}")
    print(f"Label counts: {result['label_counts']} (0=phishing, 1=legitimate)")


def display_path(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def add_common_dataset_args(parser):
    parser.add_argument(
        "--dataset",
        default=None,
        help="Default dataset key to use instead of prompting.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Custom dataset CSV path. CSV may contain URL+label or feature columns+label.",
    )
    parser.add_argument(
        "--labels-path",
        default=None,
        help="Optional labels CSV path when feature CSV does not contain labels.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional label column name. Defaults to label/project_label detection.",
    )


def build_train_parser():
    parser = argparse.ArgumentParser(
        description="Interactively train one fresh model on one selected dataset."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name or slug. Example: xgboost, random_forest, linear_svm.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output .joblib filename. Saved under machine_learning/experiments/interactive/models/.",
    )
    add_common_dataset_args(parser)
    return parser


def build_test_parser():
    parser = argparse.ArgumentParser(
        description="Interactively test one saved model against one selected dataset."
    )
    parser.add_argument(
        "--model-choice",
        default=None,
        help="Default discovered model key or slug to use instead of prompting.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Custom trained .joblib model path.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write the JSON report file.",
    )
    add_common_dataset_args(parser)
    return parser


def train_interactive_main(argv=None):
    args = build_train_parser().parse_args(argv)
    model_name = model_name_from_key_or_prompt(args.model)
    dataset = dataset_from_key_or_prompt(
        dataset_key=args.dataset,
        dataset_path=args.dataset_path,
        labels_path=args.labels_path,
        label_column=args.label_column,
        options=default_training_datasets(),
    )
    default_name = default_output_name(model_name, dataset.key)
    output_name = args.output_name or prompt_text("Trained model filename", default_name)
    result = train_and_save_model(
        model_name=model_name,
        dataset_path=dataset.features_path,
        labels_path=dataset.labels_path,
        label_column=dataset.label_column,
        output_name=output_name,
        dataset_name=dataset.key,
    )
    print_training_summary(result)
    return result


def test_interactive_main(argv=None):
    args = build_test_parser().parse_args(argv)
    model_path = model_file_from_key_or_prompt(
        model_key=args.model_choice,
        model_path=args.model_path,
    )
    dataset = dataset_from_key_or_prompt(
        dataset_key=args.dataset,
        dataset_path=args.dataset_path,
        labels_path=args.labels_path,
        label_column=args.label_column,
        options=default_testing_datasets(),
    )
    result = evaluate_model_file(
        model_path=model_path,
        dataset_path=dataset.features_path,
        labels_path=dataset.labels_path,
        label_column=dataset.label_column,
    )
    print_test_summary(result)
    if not args.no_report:
        report_path = write_test_report(result)
        print(f"JSON report: {display_path(report_path)}")
    return result
