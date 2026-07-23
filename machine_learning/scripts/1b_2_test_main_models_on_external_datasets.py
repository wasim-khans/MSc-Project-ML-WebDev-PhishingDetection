"""Test the main trained models against prepared external datasets.

This script loads the saved PhiUSIIL-trained models, scores them on the
prepared external datasets, and writes summary CSV, Markdown, and HTML reports.
"""

import argparse
from datetime import datetime
import html
import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.scripts.helpers.project_paths import (
    EXPERIMENT_1A_DIR,
    EXPERIMENT_1B_DIR,
    EXTERNAL_DATASETS_ROOT as EXTERNAL_DATA_DIR,
    EXTERNAL_TESTING_CONFIG_PATH as CONFIG_PATH,
    MAIN_MODELS_DIR as MODELS_DIR,
    PROJECT_ROOT,
)

from machine_learning.scripts.core.model_evaluation import evaluate_predictions


REPORTS_DIR = EXPERIMENT_1B_DIR
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
FEATURES_FILENAME = "external_url_features_without_labels.csv"
LABELS_FILENAME = "external_url_labels_for_comparison.csv"
HTML_REPORT_FILENAME = "report.html"
MAIN_METRICS_FILENAME = "test_summary.csv"
MAIN_CONFUSION_MATRICES_FILENAME = "confusion_matrices.json"
MAIN_DATASET_NAME = "PhiUSIIL held-out test split"
MAIN_TRAINING_SCENARIO = "phiusiil_main"
LABEL_ORDER = [0, 1]
LABEL_NAMES = {
    0: "phishing",
    1: "legitimate",
}
DEFAULT_EXTERNAL_TESTING_CONFIG = {
    "models": "all",
    "datasets": "all",
}
METRIC_GUIDE = {
    "Dataset": "The dataset whose URLs are being scored.",
    "Type": "Whether the row is the original main held-out test or an external dataset.",
    "Model": "The machine-learning algorithm used for prediction.",
    "Trained On": "The dataset or training scenario used to fit the model before testing.",
    "Tested On": "The dataset used as model input during this evaluation.",
    "Rows": "Number of URL rows scored in this result.",
    "Model File": "The saved .joblib file that was loaded to make these predictions.",
    "Accuracy": "Accuracy: overall correctness. Example: if 90 out of 100 URLs are classified correctly, accuracy is 90%.",
    "Precision": "Precision: when the model says phishing, how often it is right. Example: if it flags 10 URLs and 8 are really phishing, precision is 80%.",
    "Recall": "Recall: how many real phishing URLs the model catches. Example: if there are 100 phishing URLs and it catches 90, recall is 90%.",
    "F1": "F1-score: combines precision and recall into one score. It is useful when false alarms and missed phishing URLs both matter.",
    "Confusion matrix": "Confusion matrix: table of actual labels versus predicted labels. It shows correct phishing catches, missed phishing URLs, false alarms, and correct legitimate predictions.",
    "Robustness Score": "Robustness score: average phishing F1 minus the F1 range across selected datasets. Higher means strong and steadier performance.",
    "F1 Drop": "F1 drop: how much phishing F1 falls from the main held-out test to selected external datasets. A large drop suggests weak generalisation.",
    "Worst F1": "The lowest phishing F1 score for this model across the selected datasets.",
    "F1 Range": "The gap between the best and worst phishing F1 scores. Smaller means steadier performance.",
    "False Positive": "False positive: a legitimate URL wrongly predicted as phishing.",
    "False Negative": "False negative: a phishing URL wrongly predicted as legitimate. This is especially risky in phishing detection.",
}
MODEL_GUIDE = [
    {
        "name": "Logistic Regression",
        "role": "Simple baseline",
        "description": "A linear model that learns weighted evidence from the URL-only features. It is useful as a simple baseline because it is fast and easy to compare against stronger models.",
    },
    {
        "name": "Decision Tree",
        "role": "Interpretable tree",
        "description": "A rule-based model that splits data using feature thresholds. It was chosen because its decisions are easier to inspect than more complex ensembles.",
    },
    {
        "name": "Random Forest",
        "role": "Tree ensemble",
        "description": "A group of decision trees whose votes are combined. It was chosen to test whether many trees generalise better than one tree.",
    },
    {
        "name": "Linear SVM",
        "role": "Margin-based classifier",
        "description": "A support vector machine that finds a separating boundary between classes. The linear version was chosen because it is practical for this dataset size.",
    },
    {
        "name": "XGBoost",
        "role": "Boosted-tree model",
        "description": "A gradient-boosted tree model that builds trees sequentially to fix earlier mistakes. It was chosen because it is often strong on structured tabular features.",
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved models against prepared external testing datasets."
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help="JSON config file describing model and dataset selection.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Override config model selection: all, a display name, a slug, or one model.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Override config dataset selection: all or one prepared external dataset folder name.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask which model and dataset to test before running.",
    )
    args = parser.parse_args()

    print("Step 6: Test saved models against external datasets")
    run_config = load_external_testing_config(Path(args.config))
    available_models = discover_available_models()
    available_datasets = discover_prepared_external_datasets()
    print(f"Models found: {', '.join(available_models)}")
    print(f"Prepared datasets found: {', '.join(available_datasets)}")
    print(f"Config: {display_path(Path(args.config))}")
    model_selection = args.models if args.models is not None else run_config["models"]
    dataset_selection = (
        args.datasets if args.datasets is not None else run_config["datasets"]
    )

    if args.interactive:
        model_selection = prompt_for_selection("Model", available_models.keys())
        dataset_selection = prompt_for_selection("Dataset", available_datasets.keys())

    selected_models = normalise_model_selection(model_selection, available_models)
    selected_datasets = normalise_dataset_selection(dataset_selection, available_datasets)
    print(f"Selected models: {', '.join(selected_models)}")
    print(f"Selected datasets: {', '.join(selected_datasets)}")
    print()

    metric_rows = []
    confusion_matrices = {}
    for dataset_name in selected_datasets:
        features_df, labels_df = load_external_dataset(available_datasets[dataset_name])
        label_counts = labels_df["project_label"].value_counts().sort_index().to_dict()
        print(
            f"Dataset {dataset_name}: {len(features_df):,} rows, "
            f"label counts {label_counts}"
        )
        for model_name in selected_models:
            print(f"  Testing {model_name}...")
            row, matrix_payload = evaluate_model_on_external_dataset(
                model_name=model_name,
                model_path=available_models[model_name],
                dataset_name=dataset_name,
                features_df=features_df,
                labels_df=labels_df,
            )
            metric_rows.append(row)
            confusion_matrices[f"{dataset_name}::{model_name}"] = matrix_payload
            print(
                "    "
                f"accuracy={row['accuracy']:.4f}, "
                f"phishing_recall={row['phishing_recall']:.4f}, "
                f"phishing_f1={row['phishing_f1']:.4f}"
            )
        print()

    write_external_evaluation_reports(metric_rows, confusion_matrices)
    print("External model testing complete.")
    print(
        "Metrics: "
        f"{(REPORTS_DIR / 'test_summary.csv').relative_to(PROJECT_ROOT)}"
    )
    print(
        "Confusion matrices: "
        f"{(REPORTS_DIR / 'confusion_matrices.json').relative_to(PROJECT_ROOT)}"
    )
    print(
        "Report: "
        f"{(REPORTS_DIR / 'report.md').relative_to(PROJECT_ROOT)}"
    )
    print(
        "HTML report: "
        f"{(REPORTS_DIR / HTML_REPORT_FILENAME).relative_to(PROJECT_ROOT)}"
    )


def load_external_testing_config(config_path=CONFIG_PATH):
    """Load external-testing model/dataset selection from JSON."""
    config = DEFAULT_EXTERNAL_TESTING_CONFIG.copy()
    if config_path.exists():
        config.update(json.loads(config_path.read_text(encoding="utf-8")))
    return config


def display_path(path):
    """Show project-local paths neatly while still accepting absolute paths."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def discover_available_models():
    """Find saved trained model files, excluding best-model alias copies."""
    model_files = sorted(MODELS_DIR.glob("*.joblib"))
    available_models = {}
    for path in model_files:
        if path.stem.startswith("best_model"):
            continue
        model_name = display_name_from_model_slug(path.stem)
        if model_name not in available_models or "_T_ON_" in path.stem:
            available_models[model_name] = path
    if not available_models:
        raise FileNotFoundError(
            f"No saved model files found in {MODELS_DIR}. Run the main baseline training script first."
        )
    return available_models


def display_name_from_model_slug(slug):
    """Turn random_forest into Random Forest for reports and CLI selection."""
    slug = slug.split("_T_ON_", 1)[0]
    special_names = {
        "linear_svm": "Linear SVM",
        "xgboost": "XGBoost",
    }
    if slug in special_names:
        return special_names[slug]
    return slug.replace("_", " ").title()


def discover_prepared_external_datasets():
    """Find external dataset folders containing both features and comparison labels."""
    datasets = {}
    for processed_dir in sorted(EXTERNAL_DATA_DIR.glob("*/processed")):
        feature_path = processed_dir / FEATURES_FILENAME
        label_path = processed_dir / LABELS_FILENAME
        if feature_path.exists() and label_path.exists():
            datasets[processed_dir.parent.name] = processed_dir
    if not datasets:
        raise FileNotFoundError(
            "No prepared external datasets found. Run "
            "machine_learning/scripts/1b_1_prepare_external_testing_datasets.py first."
        )
    return datasets


def normalise_model_selection(selection, available_models):
    """Map a user selection such as all or xgboost to display model names."""
    if isinstance(selection, (list, tuple)):
        selected = []
        for item in selection:
            for model_name in normalise_model_selection(item, available_models):
                if model_name not in selected:
                    selected.append(model_name)
        return selected

    cleaned = normalise_selection_text(selection)
    if cleaned == "all":
        return list(available_models.keys())

    for model_name in available_models:
        if cleaned in {
            normalise_selection_text(model_name),
            normalise_selection_text(Path(available_models[model_name]).stem),
        }:
            return [model_name]

    raise ValueError(
        f"Unknown model '{selection}'. Choose from: all, "
        + ", ".join(available_models.keys())
    )


def normalise_dataset_selection(selection, available_datasets):
    """Map a user selection such as all or phishstorm to dataset folder names."""
    if isinstance(selection, (list, tuple)):
        selected = []
        for item in selection:
            for dataset_name in normalise_dataset_selection(item, available_datasets):
                if dataset_name not in selected:
                    selected.append(dataset_name)
        return selected

    cleaned = normalise_selection_text(selection)
    if cleaned == "all":
        return list(available_datasets.keys())

    for dataset_name in available_datasets:
        if cleaned == normalise_selection_text(dataset_name):
            return [dataset_name]

    raise ValueError(
        f"Unknown dataset '{selection}'. Choose from: all, "
        + ", ".join(available_datasets.keys())
    )


def normalise_selection_text(value):
    """Make CLI matching forgiving about spaces, hyphens, and case."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def prompt_for_selection(label, options):
    """Tiny interactive helper for users who prefer choosing at runtime."""
    option_list = ["all", *list(options)]
    print(f"Available {label.lower()} choices:")
    for index, option in enumerate(option_list, start=1):
        print(f"{index}. {option}")
    answer = input(f"Choose {label.lower()} [all]: ").strip()
    if not answer:
        return "all"
    if answer.isdigit() and 1 <= int(answer) <= len(option_list):
        return option_list[int(answer) - 1]
    return answer


def load_external_dataset(processed_dir):
    """Read one prepared external dataset and align feature rows with labels."""
    features_df = pd.read_csv(processed_dir / FEATURES_FILENAME)
    labels_df = pd.read_csv(processed_dir / LABELS_FILENAME)
    if len(features_df) != len(labels_df):
        raise ValueError(f"Feature/label row mismatch in {processed_dir}")
    if not features_df["row_id"].equals(labels_df["row_id"]):
        raise ValueError(f"Feature/label row_id mismatch in {processed_dir}")
    return features_df, labels_df


def evaluate_model_on_external_dataset(
    model_name,
    model_path,
    dataset_name,
    features_df,
    labels_df,
):
    """Load one saved model, predict external rows, and calculate metrics."""
    model = joblib.load(model_path)
    feature_columns = load_feature_columns()
    missing_columns = [column for column in feature_columns if column not in features_df]
    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing model feature columns: {missing_columns}"
        )

    x_external = features_df[feature_columns]
    y_true = labels_df["project_label"].astype(int)
    predictions = model.predict(x_external)
    metrics = evaluate_predictions(y_true, predictions)
    row = {
        "dataset": dataset_name,
        "trained_on": MAIN_TRAINING_SCENARIO,
        "model": model_name,
        "rows_tested": int(len(features_df)),
        "accuracy": float(metrics["accuracy"]),
        "phishing_precision": float(metrics["phishing_precision"]),
        "phishing_recall": float(metrics["phishing_recall"]),
        "phishing_f1": float(metrics["phishing_f1"]),
        "model_file": str(model_path.relative_to(PROJECT_ROOT)),
    }
    matrix_payload = {
        "label_order": LABEL_ORDER,
        "label_names": LABEL_NAMES,
        "matrix": confusion_matrix(y_true, predictions, labels=LABEL_ORDER).tolist(),
    }
    return row, matrix_payload


def load_feature_columns():
    """Read the feature order saved during training."""
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature column file: {FEATURE_COLUMNS_PATH}. Run the main baseline training script first."
        )
    return json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))


def load_main_evaluation_results():
    """Read the main PhiUSIIL held-out test results generated by 4_train_models.py."""
    metrics_path = EXPERIMENT_1A_DIR / MAIN_METRICS_FILENAME
    matrices_path = EXPERIMENT_1A_DIR / MAIN_CONFUSION_MATRICES_FILENAME
    metadata_path = MODELS_DIR / "model_metadata.json"
    if not metrics_path.exists() or not matrices_path.exists():
        return pd.DataFrame(), {}

    metrics_frame = pd.read_csv(metrics_path)
    confusion_matrices = json.loads(matrices_path.read_text(encoding="utf-8"))
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_model_files = metadata.get("saved_model_files", {})
    rows = []
    for row in metrics_frame.to_dict(orient="records"):
        model_name = str(row["model"])
        matrix = confusion_matrices.get(model_name, {}).get("matrix", [])
        rows_tested = sum(sum(matrix_row) for matrix_row in matrix)
        model_file = saved_model_files.get(
            model_name,
            str(
                (MODELS_DIR / f"{normalise_selection_text(model_name)}.joblib")
                .relative_to(PROJECT_ROOT)
            ),
        )
        rows.append(
            {
                "model": model_name,
                "trained_on": MAIN_TRAINING_SCENARIO,
                "rows_tested": int(rows_tested),
                "accuracy": float(row["accuracy"]),
                "phishing_precision": float(row["phishing_precision"]),
                "phishing_recall": float(row["phishing_recall"]),
                "phishing_f1": float(row["phishing_f1"]),
                "train_seconds": float(row.get("train_seconds", 0)),
                "predict_seconds": float(row.get("predict_seconds", 0)),
                "model_file": str(model_file),
            }
        )
    return pd.DataFrame(rows), confusion_matrices


def write_external_evaluation_reports(metric_rows, confusion_matrices):
    """Write CSV, JSON, and Markdown evidence for external testing."""
    REPORTS_DIR.mkdir(exist_ok=True)
    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        by=["dataset", "phishing_f1", "phishing_recall", "accuracy"],
        ascending=[True, False, False, False],
    )
    metrics_frame.to_csv(REPORTS_DIR / "test_summary.csv", index=False)
    (REPORTS_DIR / "confusion_matrices.json").write_text(
        json.dumps(confusion_matrices, indent=2),
        encoding="utf-8",
    )
    (REPORTS_DIR / "report.md").write_text(
        build_external_markdown_report(metrics_frame, confusion_matrices),
        encoding="utf-8",
    )
    main_metrics_frame, main_confusion_matrices = load_main_evaluation_results()
    (REPORTS_DIR / HTML_REPORT_FILENAME).write_text(
        build_external_html_report(
            metrics_frame=metrics_frame,
            confusion_matrices=confusion_matrices,
            main_metrics_frame=main_metrics_frame,
            main_confusion_matrices=main_confusion_matrices,
        ),
        encoding="utf-8",
    )


def build_external_markdown_report(metrics_frame, confusion_matrices):
    """Build a readable report explaining external-only model testing."""
    lines = [
        "# External Testing Model Evaluation",
        "",
        "This report was generated by `machine_learning/scripts/1b_2_test_main_models_on_external_datasets.py`.",
        "",
        "External datasets are used only for testing generalisation. They are not used to train or tune the models.",
        "",
        "## Metrics",
        "",
        "| Trained On | Tested On | Model | Rows Tested | Accuracy | Phishing Precision | Phishing Recall | Phishing F1 | Model File |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in metrics_frame.to_dict(orient="records"):
        lines.append(
            "| {trained_on} | {dataset} | {model} | {rows_tested:,} | {accuracy:.4f} | "
            "{phishing_precision:.4f} | {phishing_recall:.4f} | "
            "{phishing_f1:.4f} | `{model_file}` |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Confusion Matrices",
            "",
            "Rows are actual labels. Columns are predicted labels.",
            "",
            "- `0` = phishing",
            "- `1` = legitimate",
            "",
        ]
    )
    for key, payload in confusion_matrices.items():
        dataset_name, model_name = key.split("::", 1)
        matrix = payload["matrix"]
        lines.extend(
            [
                f"### {dataset_name} / {model_name}",
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
            "## Interpretation Note",
            "",
            "These results should be discussed separately from the main 80/20 PhiUSIIL test split. Strong performance on the original dataset does not automatically prove generalisation to datasets collected in different ways.",
            "",
        ]
    )
    return "\n".join(lines)


def build_model_summary(metrics_frame):
    """Summarise each model across all external datasets."""
    rows = []
    for model_name, group in metrics_frame.groupby("model", sort=True):
        f1_range = group["phishing_f1"].max() - group["phishing_f1"].min()
        worst_row = group.sort_values(
            by=["phishing_f1", "phishing_recall", "accuracy"],
            ascending=True,
        ).iloc[0]
        rows.append(
            {
                "model": model_name,
                "datasets_tested": int(group["dataset"].nunique()),
                "evaluations": int(len(group)),
                "average_accuracy": float(group["accuracy"].mean()),
                "average_phishing_precision": float(
                    group["phishing_precision"].mean()
                ),
                "average_phishing_recall": float(group["phishing_recall"].mean()),
                "average_phishing_f1": float(group["phishing_f1"].mean()),
                "worst_phishing_f1": float(group["phishing_f1"].min()),
                "f1_range": float(f1_range),
                "robustness_score": float(group["phishing_f1"].mean() - f1_range),
                "worst_dataset": str(worst_row["dataset"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["robustness_score", "average_phishing_f1", "average_phishing_recall"],
        ascending=False,
    )


def build_external_html_report(
    metrics_frame,
    confusion_matrices,
    main_metrics_frame=None,
    main_confusion_matrices=None,
    generated_at=None,
):
    """Build a static HTML report for dissertation-friendly model comparison."""
    if generated_at is None:
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    report_data = build_html_report_data(
        metrics_frame=metrics_frame,
        confusion_matrices=confusion_matrices,
        generated_at=generated_at,
        main_metrics_frame=main_metrics_frame,
        main_confusion_matrices=main_confusion_matrices,
    )
    all_metrics_frame = pd.DataFrame(report_data["metrics"])
    model_count = all_metrics_frame["model"].nunique()
    external_dataset_count = len(report_data["external_datasets"])
    evaluation_count = len(all_metrics_frame)
    prediction_rows = int(all_metrics_frame["rows_tested"].sum())

    sections = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>External Testing Report</title>",
        _html_styles(),
        "</head>",
        "<body>",
        "<main>",
        "<header>",
        "<p class=\"eyebrow\">Phishing URL Detection</p>",
        "<h1>External Testing Report</h1>",
        (
            "<p class=\"lede\">Dissertation report for URL-only phishing detection "
            "models, covering the main PhiUSIIL held-out test split and external "
            "generalisation testing.</p>"
        ),
        "</header>",
        "<section>",
        "<h2>Run Summary</h2>",
        "<div class=\"metric-grid\">",
        _summary_card(
            "Models",
            _plural_count(model_count, "model tested", "models tested"),
        ),
        _summary_card("Main Test Dataset", report_data["main_dataset"]),
        _summary_card(
            "External Datasets",
            _plural_count(
                external_dataset_count,
                "external dataset tested",
                "external datasets tested",
            ),
        ),
        _summary_card(
            "Evaluations",
            _plural_count(
                evaluation_count,
                "model-dataset evaluation",
                "model-dataset evaluations",
            ),
        ),
        _summary_card("Predictions", f"{prediction_rows:,} prediction rows scored"),
        _summary_card("Generated", generated_at),
        _summary_card("Feature Set", "15 URL-only features"),
        "</div>",
        "</section>",
        "<section>",
        "<h2>How to Read the Metrics</h2>",
        (
            "<p>Hover or focus the question marks beside metric names for short "
            "examples. These explanations are included so the report can stand alone "
            "in dissertation review.</p>"
        ),
        '<div class="guide-grid">',
        _metric_guide_cards(),
        "</div>",
        "</section>",
        "<section>",
        "<h2>Models Used</h2>",
        (
            "<p>Each model is included for a specific comparison purpose, from a "
            "simple baseline through stronger tree-based methods.</p>"
        ),
        '<div class="guide-grid">',
        _model_explanation_cards(),
        "</div>",
        "</section>",
        "<section>",
        "<h2>Summary View</h2>",
        (
            "<p>This is the dissertation-facing view. It always shows the main "
            "PhiUSIIL held-out test results, then compares all models against the "
            "selected external datasets only.</p>"
        ),
        "<fieldset>",
        "<legend>Choose External Datasets</legend>",
        '<div id="summaryExternalDatasetFilters" class="choice-grid">',
        _checkboxes("summaryExternalDatasetFilter", report_data["external_datasets"]),
        "</div>",
        '<div class="button-row">',
        '<button type="button" data-filter-action="select-all-summary-external-datasets">Select all</button>',
        '<button type="button" data-filter-action="clear-summary-external-datasets">Clear</button>',
        "</div>",
        "</fieldset>",
        '<div id="summaryCards" class="metric-grid"></div>',
        "<h3>Main PhiUSIIL Held-Out Test Results</h3>",
        '<div id="summaryMainChart"></div>',
        '<div id="summaryMainResults"></div>',
        "<h3>External Generalisation Results</h3>",
        '<div id="summaryExternalChart"></div>',
        '<div id="summaryExternalResults"></div>',
        "<h3>Generalisation Drop Chart</h3>",
        '<div id="summaryDropChart"></div>',
        "<h3>External Dataset Leaderboards</h3>",
        '<div id="summaryExternalLeaderboards"></div>',
        "<h3>Dissertation Summary</h3>",
        '<div id="dissertationSummary"></div>',
        "</section>",
        "<section>",
        "<h2>Detailed Explorer</h2>",
        (
            "<p>Use this view for focused inspection. It has separate filters from "
            "the summary view, so you can compare one model, many models, one dataset, "
            "or many datasets without changing the dissertation summary.</p>"
        ),
        "<div class=\"filter-layout\">",
        "<fieldset>",
        "<legend>Choose Models</legend>",
        '<div id="explorerModelFilters" class="choice-grid">',
        _checkboxes("explorerModelFilter", report_data["models"]),
        "</div>",
        '<div class="button-row">',
        '<button type="button" data-filter-action="select-all-explorer-models">Select all</button>',
        '<button type="button" data-filter-action="clear-explorer-models">Clear</button>',
        "</div>",
        "</fieldset>",
        "<fieldset>",
        "<legend>Choose Datasets</legend>",
        '<div id="explorerDatasetFilters" class="choice-grid">',
        _checkboxes("explorerDatasetFilter", report_data["datasets"]),
        "</div>",
        '<div class="button-row">',
        '<button type="button" data-filter-action="select-all-explorer-datasets">Select all</button>',
        '<button type="button" data-filter-action="clear-explorer-datasets">Clear</button>',
        "</div>",
        "</fieldset>",
        "</div>",
        '<div id="explorerSelectionSummary" class="metric-grid"></div>',
        "<h3>Selected Overall Model Summary</h3>",
        "<p>The robustness score rewards high average phishing F1 and penalises large swings between selected datasets.</p>",
        '<div id="explorerSelectedChart"></div>',
        '<div id="explorerOverallSummary"></div>',
        "<h3>Per Dataset Leaderboard</h3>",
        '<div id="explorerDatasetLeaderboards"></div>',
        "<h3>Per Model Detail</h3>",
        '<div id="explorerModelDetails"></div>',
        "<h3>Confusion Matrices</h3>",
        "<p>Rows are actual labels. Columns are predicted labels. Label 0 is phishing; label 1 is legitimate.</p>",
        '<div id="explorerConfusionMatrices"></div>',
        "<h3>Selected Results Summary</h3>",
        '<div id="explorerInterpretation"></div>',
        "</section>",
        "<section>",
        "<h2>Files Used</h2>",
        "<ul>",
        "<li><code>machine_learning/experiments/1a_train_on_main_test_on_main/test_summary.csv</code></li>",
        "<li><code>machine_learning/experiments/1a_train_on_main_test_on_main/confusion_matrices.json</code></li>",
        "<li><code>machine_learning/trained_models/1a_train_on_main_test_on_main/feature_columns.json</code></li>",
        "<li><code>machine_learning/experiments/1b_train_on_main_test_on_others/test_summary.csv</code></li>",
        "<li><code>machine_learning/experiments/1b_train_on_main_test_on_others/confusion_matrices.json</code></li>",
        "<li><code>machine_learning/experiments/1b_train_on_main_test_on_others/report.md</code></li>",
        f"<li><code>machine_learning/experiments/1b_train_on_main_test_on_others/{HTML_REPORT_FILENAME}</code></li>",
        "</ul>",
        "</section>",
        _json_script("reportData", report_data),
        _interactive_report_script(),
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(sections)


def build_html_report_data(
    metrics_frame,
    confusion_matrices,
    generated_at,
    main_metrics_frame=None,
    main_confusion_matrices=None,
):
    """Prepare JSON-safe data used by the browser-side report filters."""
    records = []
    unified_confusion_matrices = {}
    main_metrics_frame = (
        pd.DataFrame() if main_metrics_frame is None else main_metrics_frame
    )
    main_confusion_matrices = (
        {} if main_confusion_matrices is None else main_confusion_matrices
    )

    for row in main_metrics_frame.to_dict(orient="records"):
        model_name = str(row["model"])
        matrix = main_confusion_matrices.get(model_name, {}).get("matrix", [])
        rows_tested = int(row.get("rows_tested") or sum(sum(item) for item in matrix))
        model_file = row.get(
            "model_file",
            str(
                (MODELS_DIR / f"{normalise_selection_text(model_name)}.joblib")
                .relative_to(PROJECT_ROOT)
            ),
        )
        records.append(
            {
                "dataset": MAIN_DATASET_NAME,
                "dataset_type": "main",
                "model": model_name,
                "trained_on": MAIN_TRAINING_SCENARIO,
                "rows_tested": rows_tested,
                "accuracy": float(row["accuracy"]),
                "phishing_precision": float(row["phishing_precision"]),
                "phishing_recall": float(row["phishing_recall"]),
                "phishing_f1": float(row["phishing_f1"]),
                "model_file": str(model_file),
            }
        )
        unified_confusion_matrices[f"{MAIN_DATASET_NAME}::{model_name}"] = (
            main_confusion_matrices.get(model_name, {})
        )

    for row in metrics_frame.to_dict(orient="records"):
        dataset_name = str(row["dataset"])
        model_name = str(row["model"])
        records.append(
            {
                "dataset": dataset_name,
                "dataset_type": "external",
                "model": model_name,
                "trained_on": str(row.get("trained_on", MAIN_TRAINING_SCENARIO)),
                "rows_tested": int(row["rows_tested"]),
                "accuracy": float(row["accuracy"]),
                "phishing_precision": float(row["phishing_precision"]),
                "phishing_recall": float(row["phishing_recall"]),
                "phishing_f1": float(row["phishing_f1"]),
                "model_file": str(row["model_file"]),
            }
        )
        unified_confusion_matrices[f"{dataset_name}::{model_name}"] = (
            confusion_matrices.get(f"{dataset_name}::{model_name}", {})
        )

    models = sorted({record["model"] for record in records})
    external_datasets = sorted(metrics_frame["dataset"].unique().tolist())
    datasets = [MAIN_DATASET_NAME, *external_datasets]

    return {
        "generated_at": generated_at,
        "models": models,
        "main_dataset": MAIN_DATASET_NAME,
        "external_datasets": external_datasets,
        "datasets": datasets,
        "metrics": records,
        "confusion_matrices": unified_confusion_matrices,
    }


def _checkboxes(input_name, values):
    controls = []
    for value in values:
        element_id = f"{input_name}-{normalise_selection_text(value)}"
        controls.append(
            '<label class="choice">'
            f'<input id="{_escape(element_id)}" name="{_escape(input_name)}" '
            f'type="checkbox" value="{_escape(value)}" checked>'
            f"<span>{_escape(value)}</span>"
            "</label>"
        )
    return "\n".join(controls)


def _json_script(element_id, payload):
    json_payload = json.dumps(payload, ensure_ascii=True).replace("</", "<\\/")
    return (
        f'<script id="{_escape(element_id)}" type="application/json">'
        f"{json_payload}"
        "</script>"
    )


def _interactive_report_script():
    return """
<script>
const reportData = JSON.parse(document.getElementById("reportData").textContent);
const MAIN_DATASET = reportData.main_dataset;
const metricGuide = {
  Dataset: "The dataset whose URLs are being scored.",
  Type: "Whether the row is the original main held-out test or an external dataset.",
  Model: "The machine-learning algorithm used for prediction.",
  "Trained On": "The dataset or training scenario used to fit the model before testing.",
  "Tested On": "The dataset used as model input during this evaluation.",
  Rows: "Number of URL rows scored in this result.",
  "Model File": "The saved .joblib file that was loaded to make these predictions.",
  Accuracy: "Accuracy: overall correctness. Example: 90 correct predictions out of 100 means 90% accuracy.",
  Precision: "Precision: when the model says phishing, how often it is right.",
  Recall: "Recall: how many real phishing URLs the model catches.",
  F1: "F1-score: one score combining precision and recall.",
  "Confusion matrix": "Confusion matrix: actual labels by predicted labels, showing correct predictions and mistakes.",
  "Robustness Score": "Robustness score: average phishing F1 minus the F1 range across selected datasets.",
  "F1 Drop": "F1 drop: how much phishing F1 falls from main held-out testing to external testing.",
  "Worst F1": "The lowest phishing F1 score for this model across the selected datasets.",
  "F1 Range": "The gap between the best and worst phishing F1 scores. Smaller means steadier performance.",
};

document
  .querySelectorAll('input[name="summaryExternalDatasetFilter"]')
  .forEach((input) => input.addEventListener("change", renderSummaryView));

document
  .querySelectorAll('input[name="explorerModelFilter"], input[name="explorerDatasetFilter"]')
  .forEach((input) => input.addEventListener("change", renderExplorerView));

document.querySelectorAll("[data-filter-action]").forEach((button) => {
  button.addEventListener("click", () => {
    const action = button.dataset.filterAction;
    if (action === "select-all-summary-external-datasets") {
      setAll("summaryExternalDatasetFilter", true);
      renderSummaryView();
    }
    if (action === "clear-summary-external-datasets") {
      setAll("summaryExternalDatasetFilter", false);
      renderSummaryView();
    }
    if (action === "select-all-explorer-models") {
      setAll("explorerModelFilter", true);
      renderExplorerView();
    }
    if (action === "clear-explorer-models") {
      setAll("explorerModelFilter", false);
      renderExplorerView();
    }
    if (action === "select-all-explorer-datasets") {
      setAll("explorerDatasetFilter", true);
      renderExplorerView();
    }
    if (action === "clear-explorer-datasets") {
      setAll("explorerDatasetFilter", false);
      renderExplorerView();
    }
  });
});

renderSummaryView();
renderExplorerView();

function setAll(name, checked) {
  document
    .querySelectorAll(`input[name="${name}"]`)
    .forEach((input) => {
      input.checked = checked;
    });
}

function checkedValues(name) {
  return Array.from(document.querySelectorAll(`input[name="${name}"]:checked`)).map(
    (input) => input.value
  );
}

function summaryExternalRows() {
  const selectedDatasets = new Set(checkedValues("summaryExternalDatasetFilter"));
  return reportData.metrics.filter(
    (row) => row.dataset_type === "external" && selectedDatasets.has(row.dataset)
  );
}

function explorerRows() {
  const selectedModels = new Set(checkedValues("explorerModelFilter"));
  const selectedDatasets = new Set(checkedValues("explorerDatasetFilter"));
  return reportData.metrics.filter(
    (row) => selectedModels.has(row.model) && selectedDatasets.has(row.dataset)
  );
}

function mainRows() {
  return reportData.metrics.filter((row) => row.dataset_type === "main");
}

function renderSummaryView() {
  const mainResultRows = mainRows().sort(sortByPhishingF1);
  const externalRows = summaryExternalRows();
  const selectedExternalDatasets = checkedValues("summaryExternalDatasetFilter");

  document.getElementById("summaryCards").innerHTML = summaryViewCards(
    mainResultRows,
    externalRows,
    selectedExternalDatasets
  );
  document.getElementById("summaryMainResults").innerHTML = mainResultRows.length
    ? metricsTable(mainResultRows)
    : '<p class="empty-state">Main PhiUSIIL held-out test results are not available. Run script 4 first.</p>';
  renderSummaryCharts(mainResultRows, externalRows);

  if (externalRows.length === 0) {
    const emptyMessage = '<p class="empty-state">Select at least one external dataset to show external generalisation results.</p>';
    document.getElementById("summaryExternalResults").innerHTML = emptyMessage;
    document.getElementById("summaryExternalLeaderboards").innerHTML = emptyMessage;
    document.getElementById("dissertationSummary").innerHTML = dissertationSummary(
      mainResultRows,
      []
    );
    return;
  }

  document.getElementById("summaryExternalResults").innerHTML =
    modelSummaryTable(buildModelSummary(externalRows));
  document.getElementById("summaryExternalLeaderboards").innerHTML =
    datasetLeaderboards(externalRows);
  document.getElementById("dissertationSummary").innerHTML = dissertationSummary(
    mainResultRows,
    externalRows
  );
}

function renderExplorerView() {
  const rows = explorerRows();
  const selectedModels = checkedValues("explorerModelFilter");
  const selectedDatasets = checkedValues("explorerDatasetFilter");
  renderExplorerSelectionSummary(rows, selectedModels, selectedDatasets);

  if (rows.length === 0) {
    const emptyMessage = '<p class="empty-state">Select at least one model and one dataset to show results.</p>';
    document.getElementById("explorerOverallSummary").innerHTML = emptyMessage;
    document.getElementById("explorerSelectedChart").innerHTML = emptyMessage;
    document.getElementById("explorerDatasetLeaderboards").innerHTML = emptyMessage;
    document.getElementById("explorerModelDetails").innerHTML = emptyMessage;
    document.getElementById("explorerConfusionMatrices").innerHTML = emptyMessage;
    document.getElementById("explorerInterpretation").innerHTML = emptyMessage;
    return;
  }

  renderExplorerChart(rows);
  document.getElementById("explorerOverallSummary").innerHTML =
    modelSummaryTable(buildModelSummary(rows));
  document.getElementById("explorerDatasetLeaderboards").innerHTML =
    datasetLeaderboards(rows);
  document.getElementById("explorerModelDetails").innerHTML = modelDetails(rows);
  document.getElementById("explorerConfusionMatrices").innerHTML =
    confusionMatrixSections(rows);
  document.getElementById("explorerInterpretation").innerHTML =
    explorerInterpretation(rows);
}

function renderExplorerSelectionSummary(rows, models, datasets) {
  const predictionRows = rows.reduce((total, row) => total + row.rows_tested, 0);
  document.getElementById("explorerSelectionSummary").innerHTML = [
    summaryCard("Selected Models", pluralCount(models.length, "model", "models")),
    summaryCard("Selected Datasets", pluralCount(datasets.length, "dataset", "datasets")),
    summaryCard("Visible Evaluations", pluralCount(rows.length, "evaluation", "evaluations")),
    summaryCard("Prediction Rows", formatInteger(predictionRows)),
  ].join("");
}

function renderSummaryCharts(mainResultRows, externalRows) {
  document.getElementById("summaryMainChart").innerHTML = mainResultRows.length
    ? barChart(
        "Main Held-Out Phishing F1 Chart",
        mainResultRows.map((row) => ({
          label: row.model,
          value: row.phishing_f1,
          badge: row === bestByF1(mainResultRows) ? "Main winner" : "",
        })),
        "F1"
      )
    : "";

  if (!externalRows.length) {
    const emptyMessage =
      '<p class="empty-state">Select at least one external dataset to show external chart results.</p>';
    document.getElementById("summaryExternalChart").innerHTML = emptyMessage;
    document.getElementById("summaryDropChart").innerHTML = emptyMessage;
    return;
  }

  document.getElementById("summaryExternalChart").innerHTML = barChart(
    "External Average Phishing F1 Chart",
    buildModelSummary(externalRows).map((row, index) => ({
      label: row.model,
      value: row.average_phishing_f1,
      badge: index === 0 ? "External winner" : "",
    })),
    "F1"
  );

  document.getElementById("summaryDropChart").innerHTML = barChart(
    "Generalisation Drop Chart",
    generalisationDropRows(mainResultRows, externalRows).map((row, index) => ({
      label: row.model,
      value: row.drop,
      badge: index === 0 ? "Biggest drop" : "",
    })),
    "F1 Drop"
  );
}

function renderExplorerChart(rows) {
  document.getElementById("explorerSelectedChart").innerHTML = barChart(
    "Selected Results Phishing F1 Chart",
    rows
      .slice()
      .sort(sortByPhishingF1)
      .map((row, index) => ({
        label: `${row.model} / ${row.dataset}`,
        value: row.phishing_f1,
        badge: index === 0 ? "Winner" : "",
      })),
    "F1"
  );
}

function summaryViewCards(mainResultRows, externalRows, selectedExternalDatasets) {
  const bestMain = bestByF1(mainResultRows);
  const bestExternal = bestModelSummary(externalRows);
  const drop = largestGeneralisationDrop(mainResultRows, externalRows);
  return [
    summaryCard(
      "Best Main Model",
      bestMain ? `${bestMain.model} F1 ${formatMetric(bestMain.phishing_f1)}` : "Not available"
    ),
    summaryCard(
      "Best External Robustness",
      bestExternal
        ? `${bestExternal.model} score ${formatMetric(bestExternal.robustness_score)}`
        : "Not available"
    ),
    summaryCard(
      "Largest F1 Drop",
      drop ? `${drop.model} drop ${formatMetric(drop.drop)}` : "Not available"
    ),
    summaryCard(
      "External Datasets Selected",
      pluralCount(selectedExternalDatasets.length, "dataset", "datasets")
    ),
  ].join("");
}

function buildModelSummary(rows) {
  return Object.entries(groupBy(rows, "model"))
    .map(([model, group]) => {
      const f1Values = group.map((row) => row.phishing_f1);
      const worstRow = group
        .slice()
        .sort(
          (a, b) =>
            a.phishing_f1 - b.phishing_f1 ||
            a.phishing_recall - b.phishing_recall ||
            a.accuracy - b.accuracy
        )[0];
      const averageF1 = average(f1Values);
      const f1Range = Math.max(...f1Values) - Math.min(...f1Values);
      return {
        model,
        datasets_tested: unique(group.map((row) => row.dataset)).length,
        average_accuracy: average(group.map((row) => row.accuracy)),
        average_phishing_recall: average(group.map((row) => row.phishing_recall)),
        average_phishing_f1: averageF1,
        worst_phishing_f1: Math.min(...f1Values),
        f1_range: f1Range,
        robustness_score: averageF1 - f1Range,
        worst_dataset: worstRow.dataset,
      };
    })
    .sort(
      (a, b) =>
        b.robustness_score - a.robustness_score ||
        b.average_phishing_f1 - a.average_phishing_f1 ||
        b.average_phishing_recall - a.average_phishing_recall
    );
}

function bestModelSummary(rows) {
  if (!rows.length) return null;
  return buildModelSummary(rows)[0];
}

function bestByF1(rows) {
  if (!rows.length) return null;
  return rows.slice().sort(sortByPhishingF1)[0];
}

function largestGeneralisationDrop(mainResultRows, externalRows) {
  const dropRows = generalisationDropRows(mainResultRows, externalRows);
  return dropRows.length ? dropRows[0] : null;
}

function generalisationDropRows(mainResultRows, externalRows) {
  if (!mainResultRows.length || !externalRows.length) return [];
  const mainByModel = groupBy(mainResultRows, "model");
  const externalByModel = groupBy(externalRows, "model");
  return Object.keys(mainByModel)
    .filter((model) => externalByModel[model])
    .map((model) => {
      const mainF1 = mainByModel[model][0].phishing_f1;
      const externalAverageF1 = average(
        externalByModel[model].map((row) => row.phishing_f1)
      );
      return {
        model,
        main_f1: mainF1,
        external_average_f1: externalAverageF1,
        drop: mainF1 - externalAverageF1,
      };
    })
    .sort((a, b) => b.drop - a.drop);
}

function modelSummaryTable(rows) {
  return `
    <table>
      <thead>
        <tr>
          ${metricHeader("Model", "Model")}
          ${metricHeader("Datasets", "Tested On", "num")}
          ${metricHeader("Avg Accuracy", "Accuracy", "num")}
          ${metricHeader("Avg Phishing Precision", "Precision", "num")}
          ${metricHeader("Avg Phishing Recall", "Recall", "num")}
          ${metricHeader("Avg Phishing F1", "F1", "num")}
          ${metricHeader("Worst F1", "Worst F1", "num")}
          ${metricHeader("F1 Range", "F1 Range", "num")}
          ${metricHeader("Robustness Score", "Robustness Score", "num")}
          ${metricHeader("Worst Dataset", "Tested On")}
        </tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (row, index) => `
              <tr>
                <td>${escapeHtml(row.model)} ${index === 0 ? winnerBadge("Winner") : ""}</td>
                <td class="num">${row.datasets_tested}</td>
                <td class="num">${formatMetric(row.average_accuracy)}</td>
                <td class="num">${formatMetric(row.average_phishing_precision)}</td>
                <td class="num">${formatMetric(row.average_phishing_recall)}</td>
                <td class="num">${formatMetric(row.average_phishing_f1)}</td>
                <td class="num">${formatMetric(row.worst_phishing_f1)}</td>
                <td class="num">${formatMetric(row.f1_range)}</td>
                <td class="num">${formatMetric(row.robustness_score)}</td>
                <td>${escapeHtml(row.worst_dataset)}</td>
              </tr>
            `
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function datasetLeaderboards(rows) {
  return unique(rows.map((row) => row.dataset))
    .sort()
    .map((dataset) => {
      const datasetRows = rows
        .filter((row) => row.dataset === dataset)
        .sort(sortByPhishingF1);
      return `<h3>${escapeHtml(dataset)}</h3>${metricsTable(datasetRows)}`;
    })
    .join("");
}

function modelDetails(rows) {
  return unique(rows.map((row) => row.model))
    .sort()
    .map((model) => {
      const modelRows = rows
        .filter((row) => row.model === model)
        .sort((a, b) => a.dataset.localeCompare(b.dataset));
      return `<h3>${escapeHtml(model)}</h3>${metricsTable(modelRows)}`;
    })
    .join("");
}

function metricsTable(rows) {
  const bestRow = bestByF1(rows);
  return `
    <table>
      <thead>
        <tr>
          ${metricHeader("Trained On", "Trained On")}
          ${metricHeader("Tested On", "Tested On")}
          ${metricHeader("Type", "Type")}
          ${metricHeader("Model", "Model")}
          ${metricHeader("Rows", "Rows", "num")}
          ${metricHeader("Accuracy", "Accuracy", "num")}
          ${metricHeader("Precision", "Precision", "num")}
          ${metricHeader("Recall", "Recall", "num")}
          ${metricHeader("F1", "F1", "num")}
          ${metricHeader("Model File", "Model File")}
        </tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (row) => `
              <tr>
                <td>${escapeHtml(row.trained_on || "phiusiil_main")}</td>
                <td>${escapeHtml(row.dataset)}</td>
                <td>${escapeHtml(row.dataset_type)}</td>
                <td>${escapeHtml(row.model)} ${row === bestRow ? winnerBadge("Winner") : ""}</td>
                <td class="num">${formatInteger(row.rows_tested)}</td>
                <td class="num">${formatMetric(row.accuracy)}</td>
                <td class="num">${formatMetric(row.phishing_precision)}</td>
                <td class="num">${formatMetric(row.phishing_recall)}</td>
                <td class="num">${formatMetric(row.phishing_f1)}</td>
                <td><code>${escapeHtml(row.model_file)}</code></td>
              </tr>
            `
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function confusionMatrixSections(rows) {
  return rows
    .slice()
    .sort((a, b) => a.dataset.localeCompare(b.dataset) || a.model.localeCompare(b.model))
    .map((row) => {
      const key = `${row.dataset}::${row.model}`;
      const payload = reportData.confusion_matrices[key];
      if (!payload) return "";
      const matrix = payload.matrix;
      return `
        <h3>${escapeHtml(row.dataset)} / ${escapeHtml(row.model)}</h3>
        <table>
          <thead>
            <tr>
              ${metricHeader("Actual / Predicted", "Confusion matrix")}
              <th class="num">0 phishing</th>
              <th class="num">1 legitimate</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0 phishing</td>
              <td class="num">${formatInteger(matrix[0][0])}</td>
              <td class="num">${formatInteger(matrix[0][1])}</td>
            </tr>
            <tr>
              <td>1 legitimate</td>
              <td class="num">${formatInteger(matrix[1][0])}</td>
              <td class="num">${formatInteger(matrix[1][1])}</td>
            </tr>
          </tbody>
        </table>
      `;
    })
    .join("");
}

function dissertationSummary(mainResultRows, externalRows) {
  const bestMain = bestByF1(mainResultRows);
  const bestExternal = bestModelSummary(externalRows);
  const drop = largestGeneralisationDrop(mainResultRows, externalRows);
  const selectedExternalDatasets = unique(externalRows.map((row) => row.dataset));

  if (!bestMain) {
    return '<p class="empty-state">Main held-out test results are not available, so the dissertation summary cannot compare main and external performance yet.</p>';
  }

  const externalSentence = bestExternal
    ? `Across the selected external datasets (${selectedExternalDatasets.join(", ")}), ${escapeHtml(bestExternal.model)} has the highest external robustness score of ${formatMetric(bestExternal.robustness_score)}.`
    : "No external dataset is selected, so external robustness is not calculated.";
  const dropSentence = drop
    ? `${escapeHtml(drop.model)} shows the largest phishing F1 drop from the main held-out split to the selected external datasets: ${formatMetric(drop.main_f1)} to ${formatMetric(drop.external_average_f1)}.`
    : "A generalisation drop cannot be calculated until at least one external dataset is selected.";

  return `
    <div class="interpretation-box">
      <p><strong>Main held-out result:</strong> ${escapeHtml(bestMain.model)} is strongest on the PhiUSIIL held-out test split with phishing F1 ${formatMetric(bestMain.phishing_f1)}.</p>
      <p><strong>External generalisation:</strong> ${externalSentence}</p>
      <p><strong>Generalisation gap:</strong> ${dropSentence}</p>
      <p><strong>Dissertation interpretation:</strong> The main split measures performance on the same dataset distribution used for training, while the external datasets test whether the URL-only features transfer to different data sources. Large drops should be discussed as dataset dependency rather than hidden in the headline accuracy.</p>
    </div>
  `;
}

function explorerInterpretation(rows) {
  const best = bestByF1(rows);
  const selectedDatasetTypes = unique(rows.map((row) => row.dataset_type));
  const selectedDatasets = unique(rows.map((row) => row.dataset));
  if (!best) return "";

  const scope =
    selectedDatasetTypes.includes("main") && selectedDatasetTypes.includes("external")
      ? "main and external datasets"
      : selectedDatasetTypes.includes("external")
        ? "external datasets only"
        : "the main held-out test split only";

  return `
    <div class="interpretation-box">
      <p><strong>Selection scope:</strong> ${escapeHtml(selectedDatasets.join(", "))} (${escapeHtml(scope)}).</p>
      <p><strong>Best selected result:</strong> ${escapeHtml(best.model)} on ${escapeHtml(best.dataset)} with phishing F1 ${formatMetric(best.phishing_f1)}.</p>
      <p><strong>Reading note:</strong> Use this explorer for inspection, but use the Summary View for the main dissertation narrative because it keeps main held-out testing separate from external generalisation testing.</p>
    </div>
  `;
}

function barChart(title, rows, metricName) {
  if (!rows.length) {
    return '<p class="empty-state">No chart data is available for this selection.</p>';
  }
  const maxValue = Math.max(...rows.map((row) => row.value), 1);
  return `
    <div class="chart-panel">
      <h4>${escapeHtml(title)} ${tooltip(metricName)}</h4>
      <div class="bar-chart">
        ${rows
          .map((row) => {
            const width = Math.max((row.value / maxValue) * 100, 2);
            return `
              <div class="bar-row">
                <div class="bar-label">${escapeHtml(row.label)} ${row.badge ? winnerBadge(row.badge) : ""}</div>
                <div class="bar-track">
                  <div class="bar-fill" style="width: ${width.toFixed(2)}%"></div>
                </div>
                <div class="bar-value">${formatMetric(row.value)}</div>
              </div>
            `;
          })
          .join("")}
      </div>
    </div>
  `;
}

function metricHeader(label, metricName, className = "") {
  const classAttribute = className ? ` class="${className}"` : "";
  return `<th${classAttribute}>${escapeHtml(label)} ${tooltip(metricName)}</th>`;
}

function tooltip(metricName) {
  const text = metricGuide[metricName] || metricName;
  return `<span class="tooltip" tabindex="0" data-tooltip="${escapeHtml(text)}">?</span>`;
}

function winnerBadge(label) {
  return `<span class="winner-badge">${escapeHtml(label)}</span>`;
}

function summaryCard(label, value) {
  return `<div class="metric-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function sortByPhishingF1(a, b) {
  return (
    b.phishing_f1 - a.phishing_f1 ||
    b.phishing_recall - a.phishing_recall ||
    b.accuracy - a.accuracy
  );
}

function groupBy(rows, key) {
  return rows.reduce((groups, row) => {
    const groupKey = row[key];
    groups[groupKey] = groups[groupKey] || [];
    groups[groupKey].push(row);
    return groups;
  }, {});
}

function unique(values) {
  return Array.from(new Set(values));
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function pluralCount(count, singular, plural) {
  return `${count} ${count === 1 ? singular : plural}`;
}

function formatMetric(value) {
  return Number(value).toFixed(4);
}

function formatInteger(value) {
  return Number(value).toLocaleString("en-GB");
}

function escapeHtml(value) {
  const replacements = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };
  return String(value).replace(/[&<>"']/g, (character) => replacements[character]);
}
</script>
""".strip()


def _html_styles():
    return """
<style>
:root {
  --text: #17202a;
  --muted: #52616f;
  --line: #d9e1e8;
  --surface: #f7f9fb;
  --accent: #0f766e;
  --accent-soft: #e6f4f1;
}
body {
  margin: 0;
  font-family: Arial, Helvetica, sans-serif;
  color: var(--text);
  background: white;
  line-height: 1.5;
}
main {
  max-width: 1180px;
  margin: 0 auto;
  padding: 32px 20px 56px;
}
header {
  border-bottom: 1px solid var(--line);
  margin-bottom: 28px;
  padding-bottom: 20px;
}
h1, h2, h3 {
  line-height: 1.2;
}
h1 {
  font-size: 34px;
  margin: 0 0 8px;
}
h2 {
  font-size: 22px;
  margin-top: 32px;
}
h3 {
  font-size: 17px;
  margin-top: 24px;
}
.eyebrow {
  color: var(--accent);
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  margin: 0 0 8px;
}
.lede, p {
  color: var(--muted);
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}
.guide-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 12px;
}
.guide-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.guide-card h3 {
  margin-top: 0;
}
.filter-layout {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
  margin: 16px 0;
}
fieldset {
  border: 1px solid var(--line);
  border-radius: 8px;
  margin: 0;
  padding: 14px;
}
legend {
  color: var(--text);
  font-weight: 700;
  padding: 0 6px;
}
.choice-grid {
  display: grid;
  gap: 8px;
  margin-top: 8px;
}
.choice {
  align-items: center;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
  display: flex;
  gap: 8px;
  padding: 8px 10px;
}
.choice input {
  margin: 0;
}
.button-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}
button {
  background: var(--accent);
  border: 1px solid var(--accent);
  border-radius: 6px;
  color: white;
  cursor: pointer;
  font: inherit;
  font-size: 13px;
  padding: 7px 10px;
}
button:hover {
  background: #115e59;
}
.metric-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.metric-card span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 4px;
  text-transform: uppercase;
}
.metric-card strong {
  font-size: 18px;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 12px 0 20px;
  font-size: 14px;
}
th, td {
  border: 1px solid var(--line);
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
}
th {
  background: var(--accent-soft);
}
td.num, th.num {
  text-align: right;
}
code {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 1px 4px;
}
.tooltip {
  align-items: center;
  background: var(--accent);
  border-radius: 999px;
  color: white;
  cursor: help;
  display: inline-flex;
  font-size: 11px;
  font-weight: 700;
  height: 18px;
  justify-content: center;
  margin-left: 4px;
  position: relative;
  width: 18px;
}
.tooltip::after {
  background: #17202a;
  border-radius: 6px;
  bottom: calc(100% + 8px);
  color: white;
  content: attr(data-tooltip);
  display: none;
  font-size: 12px;
  font-weight: 400;
  left: 50%;
  line-height: 1.35;
  max-width: 280px;
  min-width: 220px;
  padding: 8px 10px;
  position: absolute;
  text-align: left;
  transform: translateX(-50%);
  white-space: normal;
  z-index: 10;
}
.tooltip:hover::after,
.tooltip:focus::after {
  display: block;
}
.winner-badge {
  background: #fef3c7;
  border: 1px solid #f59e0b;
  border-radius: 999px;
  color: #78350f;
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  margin-left: 6px;
  padding: 1px 7px;
  vertical-align: middle;
}
.chart-panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin: 12px 0 20px;
  padding: 14px;
}
.chart-panel h4 {
  font-size: 15px;
  margin: 0 0 12px;
}
.bar-chart {
  display: grid;
  gap: 9px;
}
.bar-row {
  align-items: center;
  display: grid;
  gap: 10px;
  grid-template-columns: minmax(180px, 1.4fr) minmax(160px, 3fr) 72px;
}
.bar-label {
  font-size: 13px;
}
.bar-track {
  background: white;
  border: 1px solid var(--line);
  border-radius: 999px;
  height: 14px;
  overflow: hidden;
}
.bar-fill {
  background: var(--accent);
  height: 100%;
}
.bar-value {
  font-variant-numeric: tabular-nums;
  text-align: right;
}
.empty-state {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin: 12px 0 20px;
  padding: 14px;
}
.interpretation-box {
  background: var(--accent-soft);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin: 12px 0 20px;
  padding: 14px;
}
.interpretation-box p:last-child {
  margin-bottom: 0;
}
@media (max-width: 720px) {
  .bar-row {
    grid-template-columns: 1fr;
  }
  .bar-value {
    text-align: left;
  }
}
</style>
""".strip()


def _summary_card(label, value):
    return (
        '<div class="metric-card">'
        f"<span>{_escape(label)}</span>"
        f"<strong>{_escape(value)}</strong>"
        "</div>"
    )


def _metric_guide_cards():
    cards = []
    for name in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Confusion matrix",
        "Robustness Score",
        "F1 Drop",
        "False Positive",
        "False Negative",
    ]:
        explanation = METRIC_GUIDE[name]
        cards.append(
            '<article class="guide-card">'
            f"<h3>{_escape(name)} {_tooltip(name)}</h3>"
            f"<p>{_escape(explanation)}</p>"
            "</article>"
        )
    return "\n".join(cards)


def _model_explanation_cards():
    cards = []
    for model in MODEL_GUIDE:
        cards.append(
            '<article class="guide-card">'
            f"<h3>{_escape(model['name'])}</h3>"
            f"<p><strong>{_escape(model['role'])}.</strong> "
            f"{_escape(model['description'])}</p>"
            "</article>"
        )
    return "\n".join(cards)


def _tooltip(metric_name):
    return (
        '<span class="tooltip" tabindex="0" '
        f'data-tooltip="{_escape(METRIC_GUIDE[metric_name])}">?</span>'
    )


def _model_summary_table(model_summary):
    rows = [
        "<table>",
        "<thead><tr>",
        _th("Model", METRIC_GUIDE["Model"]),
        _th("Datasets", METRIC_GUIDE["Tested On"], "num"),
        _th("Avg Accuracy", METRIC_GUIDE["Accuracy"], "num"),
        _th("Avg Phishing Precision", METRIC_GUIDE["Precision"], "num"),
        _th("Avg Phishing Recall", METRIC_GUIDE["Recall"], "num"),
        _th("Avg Phishing F1", METRIC_GUIDE["F1"], "num"),
        _th("Worst F1", METRIC_GUIDE["Worst F1"], "num"),
        _th("F1 Range", METRIC_GUIDE["F1 Range"], "num"),
        _th("Robustness Score", METRIC_GUIDE["Robustness Score"], "num"),
        _th("Worst Dataset", METRIC_GUIDE["Tested On"]),
        "</tr></thead>",
        "<tbody>",
    ]
    for row in model_summary.to_dict(orient="records"):
        rows.append(
            "<tr>"
            f"<td>{_escape(row['model'])}</td>"
            f"<td class=\"num\">{row['datasets_tested']}</td>"
            f"<td class=\"num\">{_fmt(row['average_accuracy'])}</td>"
            f"<td class=\"num\">{_fmt(row['average_phishing_precision'])}</td>"
            f"<td class=\"num\">{_fmt(row['average_phishing_recall'])}</td>"
            f"<td class=\"num\">{_fmt(row['average_phishing_f1'])}</td>"
            f"<td class=\"num\">{_fmt(row['worst_phishing_f1'])}</td>"
            f"<td class=\"num\">{_fmt(row['f1_range'])}</td>"
            f"<td class=\"num\">{_fmt(row['robustness_score'])}</td>"
            f"<td>{_escape(row['worst_dataset'])}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def _dataset_leaderboards(metrics_frame):
    sections = []
    for dataset_name in sorted(metrics_frame["dataset"].unique()):
        dataset_rows = metrics_frame[metrics_frame["dataset"] == dataset_name].sort_values(
            by=["phishing_f1", "phishing_recall", "accuracy"],
            ascending=False,
        )
        sections.extend(
            [
                f"<h3>{_escape(dataset_name)}</h3>",
                _metrics_table(dataset_rows),
            ]
        )
    return "\n".join(sections)


def _per_model_details(metrics_frame):
    sections = []
    for model_name in sorted(metrics_frame["model"].unique()):
        model_rows = metrics_frame[metrics_frame["model"] == model_name].sort_values(
            by=["dataset"]
        )
        sections.extend(
            [
                f"<h3>{_escape(model_name)}</h3>",
                _metrics_table(model_rows),
            ]
        )
    return "\n".join(sections)


def _metrics_table(rows_frame):
    rows = [
        "<table>",
        "<thead><tr>",
        _th("Trained On", "The dataset or scenario used to train this saved model."),
        _th("Tested On", "The dataset used as model input during this evaluation."),
        _th("Model", "The machine-learning algorithm used for prediction."),
        _th("Rows", "Number of URL rows scored in this result.", "num"),
        _th("Accuracy", METRIC_GUIDE["Accuracy"], "num"),
        _th("Precision", METRIC_GUIDE["Precision"], "num"),
        _th("Recall", METRIC_GUIDE["Recall"], "num"),
        _th("F1", METRIC_GUIDE["F1"], "num"),
        _th("Model File", METRIC_GUIDE["Model File"]),
        "</tr></thead>",
        "<tbody>",
    ]
    for row in rows_frame.to_dict(orient="records"):
        rows.append(
            "<tr>"
            f"<td>{_escape(row.get('trained_on', MAIN_TRAINING_SCENARIO))}</td>"
            f"<td>{_escape(row['dataset'])}</td>"
            f"<td>{_escape(row['model'])}</td>"
            f"<td class=\"num\">{int(row['rows_tested']):,}</td>"
            f"<td class=\"num\">{_fmt(row['accuracy'])}</td>"
            f"<td class=\"num\">{_fmt(row['phishing_precision'])}</td>"
            f"<td class=\"num\">{_fmt(row['phishing_recall'])}</td>"
            f"<td class=\"num\">{_fmt(row['phishing_f1'])}</td>"
            f"<td><code>{_escape(row['model_file'])}</code></td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def _th(label, tooltip_text, class_name=""):
    class_attribute = f' class="{class_name}"' if class_name else ""
    return (
        f"<th{class_attribute}>{_escape(label)} "
        f'<span class="tooltip" tabindex="0" data-tooltip="{_escape(tooltip_text)}">?</span></th>'
    )


def _confusion_matrix_sections(confusion_matrices):
    sections = []
    for key in sorted(confusion_matrices):
        dataset_name, model_name = key.split("::", 1)
        matrix = confusion_matrices[key]["matrix"]
        sections.extend(
            [
                f"<h3>{_escape(dataset_name)} / {_escape(model_name)}</h3>",
                "<table>",
                "<thead><tr><th>Actual / Predicted</th><th class=\"num\">0 phishing</th><th class=\"num\">1 legitimate</th></tr></thead>",
                "<tbody>",
                f"<tr><td>0 phishing</td><td class=\"num\">{matrix[0][0]:,}</td><td class=\"num\">{matrix[0][1]:,}</td></tr>",
                f"<tr><td>1 legitimate</td><td class=\"num\">{matrix[1][0]:,}</td><td class=\"num\">{matrix[1][1]:,}</td></tr>",
                "</tbody>",
                "</table>",
            ]
        )
    return "\n".join(sections)


def _fmt(value):
    return f"{float(value):.4f}"


def _plural_count(count, singular_phrase, plural_phrase):
    if count == 1:
        return f"{count} {singular_phrase}"
    return f"{count} {plural_phrase}"


def _escape(value):
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
