from datetime import datetime
import html
import json
from time import perf_counter

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

from machine_learning.scripts.helpers import (
    cross_dataset_config as config,
    cross_dataset_model_training as model_training,
)
from machine_learning.scripts.core.model_evaluation import evaluate_predictions


METRIC_GUIDE = {
    "Accuracy": "Overall correctness across phishing and legitimate URLs.",
    "Phishing Precision": "When the model predicts phishing, how often it is right.",
    "Phishing Recall": "How many real phishing URLs the model catches.",
    "Phishing F1": "A combined phishing precision and recall score.",
}

MODEL_GUIDE = {
    "Logistic Regression": (
        "Simple linear baseline. Useful because it is fast, explainable, and shows "
        "whether the URL-only features already separate the classes."
    ),
    "Decision Tree": (
        "Rule-based model. Useful because its splits are easy to understand, but it "
        "can overfit one dataset."
    ),
    "Random Forest": (
        "Many decision trees combined. Useful because it is stronger than one tree "
        "and handles non-linear feature patterns."
    ),
    "Linear SVM": (
        "Margin-based linear classifier. Useful as a strong classical baseline for "
        "numeric feature vectors."
    ),
    "XGBoost": (
        "Gradient-boosted trees from the separate XGBoost library. Useful because it "
        "often performs strongly on tabular data."
    ),
}

TRAINING_SCENARIO_ORDER = [
    "phiusiil_main",
    "legitphish",
    "phishstorm",
    "combined_dataset",
]

TEST_SCENARIO_ORDER = [
    "phiusiil_main_test",
    "legitphish_test",
    "phishstorm_test",
    "combined_test",
    "complete_combined_dataset",
]

SCENARIO_LABELS = {
    "phiusiil_main": "PhiUSIIL",
    "legitphish": "LegitPhish",
    "phishstorm": "PhishStorm",
    "combined_dataset": "Combined Dataset",
    "phiusiil_main_test": "PhiUSIIL Test",
    "legitphish_test": "LegitPhish Test",
    "phishstorm_test": "PhishStorm Test",
    "combined_test": "Combined Test",
    "complete_combined_dataset": "Complete Combined Dataset",
}

COLUMN_GUIDE = {
    "Scope": "Whether this is clean holdout evidence or a diagnostic complete-dataset check.",
    "Rank": "Position after sorting by the selected score. Rank 1 is the current winner.",
    "Trained On": "The dataset source used to train the model.",
    "Training Scenario": "The dataset source used to train the model.",
    "Model": "The machine-learning algorithm that was trained and tested.",
    "Winning Model": "The highest-scoring model for this training/testing situation.",
    "Tested On": "The dataset used to test the trained model.",
    "Test Dataset": "The dataset used to test the trained model.",
    "Rows": "Number of URL rows scored in this result.",
    "Rows Tested": "Number of URL rows scored in this result.",
    "URL Overlap Removed": "Rows removed before scoring because the same URL appeared in the training data.",
    "Training Rows Included?": "Whether the test contains rows that were also used during training.",
    "Evidence Type": "Whether the row should be treated as clean holdout evidence or diagnostic evidence.",
    "Chart": "A small visual bar for the score in this row.",
    "Difference": "Combined-training F1 minus the best single-source F1 for the same test dataset.",
}


def load_model_metadata():
    path = config.MODELS_DIR / "cross_dataset_model_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing cross-dataset model metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_test_splits():
    """Read untouched test splits, plus saved combined evaluation files."""
    test_splits = {}
    for dataset_name in config.DATASETS:
        path = config.SPLITS_DIR / config.test_split_filename(dataset_name)
        if not path.exists():
            raise FileNotFoundError(f"Missing test split: {path}")
        test_splits[dataset_name] = pd.read_csv(path)
    for split_name in ["combined_test", "complete_combined_dataset"]:
        path = config.SPLITS_DIR / config.COMBINED_SPLIT_FILES[split_name]["filename"]
        if path.exists():
            test_splits[split_name] = pd.read_csv(path)
    return test_splits


def load_train_splits():
    """Read dataset train splits, including the saved combined train file."""
    train_splits = {}
    for dataset_name in config.DATASETS:
        path = config.SPLITS_DIR / config.train_split_filename(dataset_name)
        if not path.exists():
            raise FileNotFoundError(f"Missing train split: {path}")
        train_splits[dataset_name] = pd.read_csv(path)
    combined_path = (
        config.SPLITS_DIR
        / config.COMBINED_SPLIT_FILES["combined_dataset_train"]["filename"]
    )
    if combined_path.exists():
        train_splits["combined_dataset"] = pd.read_csv(combined_path)
    return train_splits


def build_test_sets(test_splits):
    """Build individual and combined test sets from untouched split files."""
    test_sets = {}
    for test_set_name, dataset_names in config.TEST_SETS.items():
        if test_set_name in test_splits:
            test_sets[test_set_name] = test_splits[test_set_name].copy()
        else:
            test_sets[test_set_name] = pd.concat(
                [test_splits[dataset_name] for dataset_name in dataset_names],
                ignore_index=True,
            )
    return test_sets


def build_complete_test_sets(train_splits, test_splits):
    """Build diagnostic full-data test sets that include train and test rows."""
    complete_sets = {}
    for test_set_name, dataset_names in config.COMPLETE_TEST_SETS.items():
        if test_set_name in test_splits:
            complete_sets[test_set_name] = test_splits[test_set_name].copy()
        else:
            frames = []
            for dataset_name in dataset_names:
                frames.extend([train_splits[dataset_name], test_splits[dataset_name]])
            complete_sets[test_set_name] = pd.concat(frames, ignore_index=True)
    return complete_sets


def build_evaluation_sets(train_splits, test_splits):
    """Build all clean holdout and complete-dataset evaluation sets."""
    evaluation_sets = {}
    for test_set_name, frame in build_test_sets(test_splits).items():
        evaluation_sets[test_set_name] = {
            "frame": frame,
            "evaluation_scope": "clean_holdout",
            "contains_training_rows": False,
            "row_source_note": (
                "Untouched 20 percent test split rows only; these rows were not "
                "used for training the matching source split."
            ),
        }
    for test_set_name, frame in build_complete_test_sets(
        train_splits, test_splits
    ).items():
        evaluation_sets[test_set_name] = {
            "frame": frame,
            "evaluation_scope": "complete_dataset",
            "contains_training_rows": True,
            "row_source_note": (
                "Diagnostic full combined dataset check; includes both training "
                "rows and untouched test rows."
            ),
        }
    return evaluation_sets


def filter_evaluation_set_for_training_urls(evaluation_set, training_frame):
    """Remove clean-holdout rows whose URL was already seen in training."""
    output = dict(evaluation_set)
    output["training_url_overlap_removed"] = 0
    if output["contains_training_rows"]:
        return output
    if "url_normalized" not in output["frame"].columns:
        return output
    if "url_normalized" not in training_frame.columns:
        return output

    training_urls = set(training_frame["url_normalized"])
    keep_mask = ~output["frame"]["url_normalized"].isin(training_urls)
    removed = int((~keep_mask).sum())
    output["frame"] = output["frame"][keep_mask].reset_index(drop=True)
    output["training_url_overlap_removed"] = removed
    if removed:
        output["row_source_note"] = (
            f"{output['row_source_note']} {removed:,} row(s) with URLs already "
            "seen in this training scenario were removed before scoring."
        )
    return output


def build_metric_row(
    training_scenario,
    model_name,
    test_dataset,
    rows_tested,
    metrics,
    model_file,
    train_seconds=0,
    predict_seconds=0,
    evaluation_scope="clean_holdout",
    contains_training_rows=False,
    training_url_overlap_removed=0,
    row_source_note="Untouched test split rows only.",
):
    return {
        "training_scenario": training_scenario,
        "model": model_name,
        "test_dataset": test_dataset,
        "evaluation_scope": evaluation_scope,
        "contains_training_rows": bool(contains_training_rows),
        "training_url_overlap_removed": int(training_url_overlap_removed),
        "row_source_note": row_source_note,
        "rows_tested": int(rows_tested),
        "accuracy": float(metrics["accuracy"]),
        "phishing_precision": float(metrics["phishing_precision"]),
        "phishing_recall": float(metrics["phishing_recall"]),
        "phishing_f1": float(metrics["phishing_f1"]),
        "model_file": model_file,
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
    }


def evaluate_one_model(model_payload, test_dataset_name, evaluation_set, training_frame):
    """Load one cross-dataset model and score it against one test set."""
    model_path = config.PROJECT_ROOT / model_payload["model_file"]
    model = joblib.load(model_path)
    evaluation_set = filter_evaluation_set_for_training_urls(
        evaluation_set,
        training_frame,
    )
    test_frame = evaluation_set["frame"]
    x_test = test_frame[config.FEATURE_COLUMNS]
    y_true = test_frame[config.LABEL_COLUMN].astype(int)
    start = perf_counter()
    predictions = model.predict(x_test)
    predict_seconds = perf_counter() - start
    metrics = evaluate_predictions(y_true, predictions)
    row = build_metric_row(
        training_scenario=model_payload["training_scenario"],
        model_name=model_payload["model"],
        test_dataset=test_dataset_name,
        rows_tested=len(test_frame),
        metrics=metrics,
        model_file=model_payload["model_file"],
        train_seconds=model_payload.get("train_seconds", 0),
        predict_seconds=predict_seconds,
        evaluation_scope=evaluation_set["evaluation_scope"],
        contains_training_rows=evaluation_set["contains_training_rows"],
        training_url_overlap_removed=evaluation_set.get(
            "training_url_overlap_removed", 0
        ),
        row_source_note=evaluation_set["row_source_note"],
    )
    matrix_payload = {
        "label_order": config.LABEL_ORDER,
        "label_names": config.LABEL_NAMES,
        "matrix": confusion_matrix(
            y_true, predictions, labels=config.LABEL_ORDER
        ).tolist(),
    }
    return row, matrix_payload


def evaluate_cross_dataset_models():
    """Evaluate all cross-dataset models against all cross-dataset test sets."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_config = config.load_cross_dataset_config()
    metadata = load_model_metadata()
    train_splits = load_train_splits()
    test_splits = load_test_splits()
    training_scenarios = model_training.build_training_scenarios(train_splits)
    all_evaluation_sets = build_evaluation_sets(train_splits, test_splits)
    selected_test_set_names = config.select_names(
        config.TEST_SETS.keys(), run_config["test_sets"]
    )
    selected_complete_set_names = config.select_names(
        config.COMPLETE_TEST_SETS.keys(), run_config.get("complete_test_sets", [])
    )
    selected_model_names = set(
        config.select_names(
            sorted({model["model"] for model in metadata["models"]}),
            run_config["models"],
        )
    )
    selected_scenario_names = set(
        config.select_names(
            sorted({model["training_scenario"] for model in metadata["models"]}),
            run_config["training_scenarios"],
        )
    )
    selected_evaluation_set_names = [
        *selected_test_set_names,
        *selected_complete_set_names,
    ]
    evaluation_sets = {
        test_set_name: all_evaluation_sets[test_set_name]
        for test_set_name in selected_evaluation_set_names
    }
    metric_rows = []
    confusion_matrices = {}

    print("Cross-dataset experiment step 3: evaluate models")
    for model_payload in metadata["models"]:
        if model_payload["model"] not in selected_model_names:
            continue
        if model_payload["training_scenario"] not in selected_scenario_names:
            continue
        training_frame = training_scenarios[model_payload["training_scenario"]]
        for test_dataset_name, evaluation_set in evaluation_sets.items():
            row, matrix_payload = evaluate_one_model(
                model_payload,
                test_dataset_name,
                evaluation_set,
                training_frame,
            )
            metric_rows.append(row)
            key = (
                f"{model_payload['training_scenario']}::"
                f"{model_payload['model']}::{test_dataset_name}"
            )
            confusion_matrices[key] = matrix_payload
            print(
                f"{model_payload['training_scenario']} / "
                f"{model_payload['model']} / {test_dataset_name}: "
                f"scope={row['evaluation_scope']}, "
                f"accuracy={row['accuracy']:.4f}, "
                f"phishing_f1={row['phishing_f1']:.4f}"
            )

    write_reports(pd.DataFrame(metric_rows), confusion_matrices)
    print(f"Metrics: {(config.REPORTS_DIR / 'test_summary.csv').relative_to(config.PROJECT_ROOT)}")
    print(
        "Confusion matrices: "
        f"{(config.REPORTS_DIR / 'confusion_matrices.json').relative_to(config.PROJECT_ROOT)}"
    )
    print(
        "Summary JSON: "
        f"{(config.REPORTS_DIR / 'cross_dataset_summary.json').relative_to(config.PROJECT_ROOT)}"
    )
    print(f"Markdown report: {(config.REPORTS_DIR / 'report.md').relative_to(config.PROJECT_ROOT)}")
    print(f"HTML report: {(config.REPORTS_DIR / 'report.html').relative_to(config.PROJECT_ROOT)}")
    return metric_rows

def write_reports(metrics_frame, confusion_matrices):
    """Write CSV, JSON, Markdown, and HTML cross-dataset reports."""
    metrics_frame = metrics_frame.sort_values(
        by=["evaluation_scope", "training_scenario", "model", "test_dataset"],
        ascending=True,
    )
    summary_payload = build_summary_payload(metrics_frame)
    metrics_frame.to_csv(config.REPORTS_DIR / "test_summary.csv", index=False)
    (config.REPORTS_DIR / "confusion_matrices.json").write_text(
        json.dumps(confusion_matrices, indent=2),
        encoding="utf-8",
    )
    (config.REPORTS_DIR / "cross_dataset_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    (config.REPORTS_DIR / "report.md").write_text(
        build_markdown_report(metrics_frame, confusion_matrices),
        encoding="utf-8",
    )
    (config.REPORTS_DIR / "report.html").write_text(
        build_html_report(metrics_frame, confusion_matrices),
        encoding="utf-8",
    )


def best_rows(metrics_frame, group_columns):
    if metrics_frame.empty:
        return metrics_frame
    indexes = metrics_frame.groupby(group_columns)["phishing_f1"].idxmax()
    return metrics_frame.loc[indexes].sort_values(group_columns)


def metrics_for_scope(metrics_frame, scope):
    if "evaluation_scope" not in metrics_frame.columns:
        return metrics_frame.iloc[0:0]
    return metrics_frame[metrics_frame["evaluation_scope"] == scope].copy()


def scope_summary(metrics_frame):
    """Summarise how many clean and diagnostic evaluations were generated."""
    summary = {
        "clean_holdout": {"evaluations": 0, "prediction_rows": 0},
        "complete_dataset": {"evaluations": 0, "prediction_rows": 0},
    }
    if metrics_frame.empty or "evaluation_scope" not in metrics_frame.columns:
        return summary
    for scope, group in metrics_frame.groupby("evaluation_scope"):
        summary[str(scope)] = {
            "evaluations": int(len(group)),
            "prediction_rows": int(group["rows_tested"].sum()),
        }
    return summary


def build_summary_payload(metrics_frame):
    """Build machine-readable report summary data."""
    clean_metrics = metrics_for_scope(metrics_frame, "clean_holdout")
    complete_metrics = metrics_for_scope(metrics_frame, "complete_dataset")
    complete_best = best_rows(complete_metrics, ["test_dataset"])
    return {
        "scope_summary": scope_summary(metrics_frame),
        "decision_summary": build_decision_payload(metrics_frame),
        "best_clean_holdout_by_test_dataset": records_for_json(
            best_rows(clean_metrics, ["test_dataset"])
        ),
        "best_clean_holdout_by_training_scenario": records_for_json(
            best_rows(clean_metrics, ["training_scenario"])
        ),
        "combined_training_comparison": combined_training_comparison(clean_metrics),
        "best_complete_dataset_diagnostic": (
            record_for_json(complete_best.iloc[0]) if not complete_best.empty else None
        ),
    }


def build_decision_payload(metrics_frame):
    """Answer the dissertation/report decision questions from metric rows."""
    clean_metrics = metrics_for_scope(metrics_frame, "clean_holdout")
    combined_test = clean_metrics[clean_metrics["test_dataset"] == "combined_test"]
    single_source_combined_test = combined_test[
        combined_test["training_scenario"] != "combined_dataset"
    ]
    allrounders = allrounder_records(clean_metrics)
    training_datasets = training_dataset_records(clean_metrics)
    backend = allrounders[0].copy() if allrounders else None
    if backend:
        backend["reason"] = (
            f"{backend['training_scenario']} / {backend['model']} has the strongest "
            "clean-holdout all-round score using mean phishing F1, worst phishing F1, "
            "and phishing recall as tie-breakers. This is the most defensible backend "
            "choice because it is judged across multiple clean test datasets rather than "
            "one same-source split only."
        )
    return {
        "backend_recommendation": backend,
        "best_allrounder": allrounders[0] if allrounders else None,
        "best_training_dataset": training_datasets[0] if training_datasets else None,
        "best_overall_on_combined_test": best_metric_record(combined_test),
        "best_single_source_on_combined_test": best_metric_record(
            single_source_combined_test
        ),
        "allrounder_ranking": allrounders,
        "training_dataset_ranking": training_datasets,
    }


def best_metric_record(frame):
    """Return the strongest metric row as a JSON-safe dictionary."""
    if frame.empty:
        return None
    row = frame.sort_values(
        by=["phishing_f1", "phishing_recall", "accuracy"],
        ascending=False,
    ).iloc[0]
    return record_for_json(row)


def allrounder_records(metrics_frame):
    """Summarise each trained model artefact across clean holdout datasets."""
    if metrics_frame.empty:
        return []
    records = []
    for (training_scenario, model), group in metrics_frame.groupby(
        ["training_scenario", "model"],
        sort=True,
    ):
        best = best_metric_record(group)
        records.append(
            {
                "training_scenario": str(training_scenario),
                "model": str(model),
                "evaluation_count": int(len(group)),
                "mean_accuracy": float(group["accuracy"].mean()),
                "mean_phishing_recall": float(group["phishing_recall"].mean()),
                "worst_phishing_recall": float(group["phishing_recall"].min()),
                "mean_phishing_f1": float(group["phishing_f1"].mean()),
                "worst_phishing_f1": float(group["phishing_f1"].min()),
                "best_test_dataset": best["test_dataset"] if best else "",
                "best_phishing_f1": best["phishing_f1"] if best else 0.0,
            }
        )
    max_evaluation_count = max(row["evaluation_count"] for row in records)
    full_coverage_records = [
        row for row in records if row["evaluation_count"] == max_evaluation_count
    ]
    return sorted(
        full_coverage_records,
        key=lambda row: (
            row["mean_phishing_f1"],
            row["worst_phishing_f1"],
            row["mean_phishing_recall"],
            row["mean_accuracy"],
        ),
        reverse=True,
    )


def training_dataset_records(metrics_frame):
    """Rank training scenarios, independent of the specific model choice."""
    if metrics_frame.empty:
        return []
    records = []
    for training_scenario, group in metrics_frame.groupby("training_scenario", sort=True):
        best = best_metric_record(group)
        records.append(
            {
                "training_scenario": str(training_scenario),
                "evaluation_count": int(len(group)),
                "mean_accuracy": float(group["accuracy"].mean()),
                "mean_phishing_recall": float(group["phishing_recall"].mean()),
                "worst_phishing_recall": float(group["phishing_recall"].min()),
                "mean_phishing_f1": float(group["phishing_f1"].mean()),
                "worst_phishing_f1": float(group["phishing_f1"].min()),
                "best_model": best["model"] if best else "",
                "best_test_dataset": best["test_dataset"] if best else "",
                "best_phishing_f1": best["phishing_f1"] if best else 0.0,
            }
        )
    return sorted(
        records,
        key=lambda row: (
            row["mean_phishing_f1"],
            row["worst_phishing_f1"],
            row["mean_phishing_recall"],
            row["mean_accuracy"],
        ),
        reverse=True,
    )


def records_for_json(frame):
    return [record_for_json(row) for _, row in frame.iterrows()]


def record_for_json(row):
    return {
        "training_scenario": str(row["training_scenario"]),
        "model": str(row["model"]),
        "test_dataset": str(row["test_dataset"]),
        "evaluation_scope": str(row["evaluation_scope"]),
        "contains_training_rows": truthy(row["contains_training_rows"]),
        "training_url_overlap_removed": int(
            row.get("training_url_overlap_removed", 0)
        ),
        "row_source_note": str(row["row_source_note"]),
        "rows_tested": int(row["rows_tested"]),
        "accuracy": float(row["accuracy"]),
        "phishing_precision": float(row["phishing_precision"]),
        "phishing_recall": float(row["phishing_recall"]),
        "phishing_f1": float(row["phishing_f1"]),
        "model_file": str(row["model_file"]),
        "train_seconds": float(row["train_seconds"]),
        "predict_seconds": float(row["predict_seconds"]),
    }


def build_markdown_report(metrics_frame, confusion_matrices):
    """Build a concise Markdown report for dissertation evidence."""
    clean_metrics = metrics_for_scope(metrics_frame, "clean_holdout")
    complete_metrics = metrics_for_scope(metrics_frame, "complete_dataset")
    summary = scope_summary(metrics_frame)
    lines = [
        "# Experiment 2: Cross-Dataset Generalisation Report",
        "",
        "This report was generated by `machine_learning/scripts/2_3_evaluate_cross_dataset_models.py`.",
        "",
        "Experiment 2 trains every selected model on each dataset-specific training scenario: `phiusiil_main`, `legitphish`, `phishstorm`, and `combined_dataset`.",
        "",
        "Clean holdout tests use only untouched 20 percent test split rows. The `complete_combined_dataset` check is diagnostic and includes both training and test rows.",
        "",
        "## Scope Summary",
        "",
        "| Scope | Evaluations | Prediction Rows | Includes Training Rows? |",
        "|---|---:|---:|---|",
        f"| Clean holdout | {summary['clean_holdout']['evaluations']:,} | {summary['clean_holdout']['prediction_rows']:,} | No |",
        f"| Complete dataset diagnostic | {summary['complete_dataset']['evaluations']:,} | {summary['complete_dataset']['prediction_rows']:,} | Yes |",
        "",
        "## Best Result Per Clean Holdout Test Dataset",
        "",
        "| Test Dataset | Training Scenario | Model | Accuracy | Phishing Recall | Phishing F1 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in best_rows(clean_metrics, ["test_dataset"]).to_dict(orient="records"):
        lines.append(
            "| {test_dataset} | {training_scenario} | {model} | {accuracy:.4f} | "
            "{phishing_recall:.4f} | {phishing_f1:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Complete Combined Dataset Diagnostic",
            "",
            "This section tests against the full combined dataset, including rows that were used for training. Treat it as a diagnostic view, not clean generalisation evidence.",
            "",
            "| Training Scenario | Model | Rows | Accuracy | Phishing Recall | Phishing F1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in best_rows(complete_metrics, ["test_dataset"]).to_dict(orient="records"):
        lines.append(
            "| {training_scenario} | {model} | {rows_tested:,} | {accuracy:.4f} | "
            "{phishing_recall:.4f} | {phishing_f1:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Best Clean Holdout Result Per Training Scenario",
            "",
            "| Training Scenario | Test Dataset | Model | Accuracy | Phishing Recall | Phishing F1 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in best_rows(clean_metrics, ["training_scenario"]).to_dict(
        orient="records"
    ):
        lines.append(
            "| {training_scenario} | {test_dataset} | {model} | {accuracy:.4f} | "
            "{phishing_recall:.4f} | {phishing_f1:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Combined Training Comparison",
            "",
            "| Test Dataset | Combined Best Model | Combined F1 | Best Single-Source Scenario | Best Single-Source Model | Best Single-Source F1 | Difference |",
            "|---|---|---:|---|---|---:|---:|",
        ]
    )
    for row in combined_training_comparison(metrics_frame):
        lines.append(
            "| {test_dataset} | {combined_model} | {combined_f1:.4f} | "
            "{single_source_scenario} | {single_source_model} | "
            "{single_source_f1:.4f} | {difference:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Full Results",
            "",
            "| Scope | Training Scenario | Model | Test Dataset | Rows | URL Overlap Removed | Includes Training Rows? | Accuracy | Phishing Precision | Phishing Recall | Phishing F1 |",
            "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in metrics_frame.to_dict(orient="records"):
        row.setdefault("training_url_overlap_removed", 0)
        lines.append(
            "| {scope} | {training_scenario} | {model} | {test_dataset} | {rows_tested:,} | "
            "{training_url_overlap_removed:,} | {includes_training} | {accuracy:.4f} | {phishing_precision:.4f} | "
            "{phishing_recall:.4f} | {phishing_f1:.4f} |".format(
                scope=format_scope_label(row["evaluation_scope"]),
                includes_training="Yes"
                if truthy(row["contains_training_rows"])
                else "No",
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use this report to compare whether mixed-source training improves cross-dataset generalisation. Same-source results should be separated from cross-source results when discussing reliability.",
            "",
            "## Confusion Matrix Keys",
            "",
            "Detailed confusion matrices are saved in `confusion_matrices.json` using keys formatted as `training_scenario::model::test_dataset`.",
        ]
    )
    return "\n".join(lines)


def build_html_report(metrics_frame, confusion_matrices, generated_at=None):
    """Build an offline interactive cross-dataset report."""
    if generated_at is None:
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    evaluation_count = len(metrics_frame)
    prediction_rows = int(metrics_frame["rows_tested"].sum()) if evaluation_count else 0
    summary = scope_summary(metrics_frame)
    summary_payload = build_summary_payload(metrics_frame)
    decision_payload = summary_payload["decision_summary"]
    report_payload = {
        "generated_at": generated_at,
        "metrics": records_for_json(metrics_frame),
        "summary": summary_payload,
        "confusion_matrices": confusion_matrices,
        "metric_guide": METRIC_GUIDE,
        "model_guide": MODEL_GUIDE,
    }
    report_json = json.dumps(report_payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment 2: Cross-Dataset Generalisation Report</title>
{html_styles()}
</head>
<body>
<main>
<header>
<h1>Experiment 2: Cross-Dataset Generalisation Report</h1>
<p class="lede">Cross-Dataset Training Matrix for phishing URL model training and testing, generated {escape(generated_at)}.</p>
</header>

<section class="panel" id="decision-dashboard">
<div class="section-heading">
<div>
<h2>Decision Dashboard</h2>
<p class="section-note">Generated from the current <code>test_summary.csv</code>. These cards answer the questions needed for the dissertation and backend choice.</p>
</div>
<div class="view-summary">Generated {escape(generated_at)}</div>
</div>
{html_decision_dashboard(decision_payload)}
{html_decision_explanation(decision_payload)}
</section>

<section class="panel" id="filter-results">
<h2>Filter Results</h2>
<p class="section-note">Default view: clean holdout only, all training scenarios, all test datasets, all models, sorted by phishing F1.</p>
<div class="controls">
<label>Scope<select id="scopeFilter"></select></label>
<label>Training<select id="trainingFilter"></select></label>
<label>Tested On<select id="testedFilter"></select></label>
<label>Model<select id="modelFilter"></select></label>
<label>Sort By<select id="sortMetric"></select></label>
<button type="button" id="resetFilters">Reset</button>
</div>
<div class="preset-row">
<button type="button" data-preset="clean">Clean Holdout Overview</button>
<button type="button" data-preset="backend">Backend Decision</button>
<button type="button" data-preset="allrounder">All-Rounder</button>
<button type="button" data-preset="singleSourceCombined">Single-Source on Combined Test</button>
<button type="button" data-preset="combined">Combined Test</button>
<button type="button" data-preset="main">Main Test</button>
<button type="button" data-preset="external">External Tests</button>
<button type="button" data-preset="diagnostic">Complete Diagnostic</button>
</div>
</section>

<details class="report-section" id="how-to-read-section" open>
<summary>How to Read This Report</summary>
<div class="report-section-body">
{html_how_to_read_report()}
</div>
</details>

<details class="report-section" id="research-questions" open>
<summary>Research Questions Answered</summary>
<div class="report-section-body">
{html_research_answers(decision_payload)}
</div>
</details>

<details class="report-section" id="per-model-results-section" open>
<summary>Per Model Results</summary>
<div class="report-section-body">
<p class="section-note">Read this section model by model. Each matrix keeps the algorithm fixed, then changes what it was trained on and tested against.</p>
{html_per_model_results(metrics_frame)}
</div>
</details>

<details class="report-section" id="model-comparison-section" open>
<summary>Model Comparison</summary>
<div class="report-section-body">
{html_model_comparison(metrics_for_scope(metrics_frame, "clean_holdout"), decision_payload)}
</div>
</details>

<details class="report-section" id="model-summary-section" open>
<summary>Model Summary View</summary>
<div class="report-section-body">
<div class="section-heading">
<div>
<p class="section-note">Model-oriented summary across the currently selected scenarios and datasets. This is the main view for deciding which model is strongest overall.</p>
</div>
<div id="modelSummaryStats" class="view-summary"></div>
</div>
<div id="modelSummaryView"></div>
</div>
</details>

<details class="report-section" id="scenario-section">
<summary>Scenario View</summary>
<div class="report-section-body">
<div class="section-heading">
<div>
<p class="section-note">Scenario-oriented table showing which model wins inside each training/testing situation.</p>
</div>
<div id="scenarioSummary" class="view-summary"></div>
</div>
<div id="scenarioView"></div>
</div>
</details>

<details class="report-section" id="ranking-section">
<summary>Ranking View</summary>
<div class="report-section-body">
<div class="section-heading">
<div>
<p class="section-note">Sorted by the selected metric. The top row is the current winner under your filters.</p>
</div>
<div id="rankingSummary" class="view-summary"></div>
</div>
<div id="rankingView"></div>
</div>
</details>

<details class="report-section" id="matrix-section">
<summary>Heatmap Matrix View</summary>
<div class="report-section-body">
<div class="section-heading">
<div>
<p class="section-note">Training scenarios are rows, test datasets are columns. Each cell shows the best model for that pairing.</p>
</div>
<div id="matrixSummary" class="view-summary"></div>
</div>
<div id="matrixView"></div>
</div>
</details>

<details class="report-section" id="detail-section">
<summary>Per Model and Dataset Results View</summary>
<div class="report-section-body">
<div class="section-heading">
<div>
<p class="section-note">Use this when you need exact metrics for one model, several models, one dataset, or many datasets.</p>
</div>
<div id="detailSummary" class="view-summary"></div>
</div>
<div id="detailView"></div>
</div>
</details>

<details class="report-section" id="confusion-view-section">
<summary>Confusion Matrices</summary>
<div class="report-section-body">
<p class="section-note">Rows are actual labels and columns are predicted labels. Label 0 is phishing; label 1 is legitimate.</p>
<div id="confusionView"></div>
</div>
</details>

<details class="report-section" id="comparison-section">
<summary>Single-Source vs Combined Training (Combined Training Comparison)</summary>
<div class="report-section-body">
<p class="section-note">This compact table compares the best <code>combined_dataset</code> clean-holdout result against the best single-source result for each test dataset.</p>
{html_combined_comparison(metrics_for_scope(metrics_frame, "clean_holdout"))}
</div>
</details>

<details class="report-section" id="complete-diagnostic-section">
<summary>Complete Combined Dataset Diagnostic</summary>
<div class="report-section-body">
<p class="section-note">The Complete Combined Dataset includes training rows, so use it only as diagnostic evidence. The backend recommendation and dissertation headline conclusions should come from clean holdout rows.</p>
</div>
</details>

<details class="report-section" id="guide-section">
<summary>Metric and Model Explanations</summary>
<div class="report-section-body guide-grid">
{metric_guide_cards()}
{model_guide_cards()}
</div>
</details>

<details class="report-section" id="programmatic-summary-section">
<summary>Programmatic Summary</summary>
<div class="report-section-body">
<p>The same summary is written to <code>cross_dataset_summary.json</code>. Metrics are also written to <code>test_summary.csv</code>.</p>
{html_programmatic_summary(summary_payload)}
</div>
</details>

</main>
<script id="reportData" type="application/json">{report_json}</script>
{report_script()}
</body>
</html>"""


def html_headline_winners(summary_payload):
    clean_rows = summary_payload["best_clean_holdout_by_test_dataset"]
    best_clean = (
        max(
            clean_rows,
            key=lambda row: (
                row["phishing_f1"],
                row["phishing_recall"],
                row["accuracy"],
            ),
        )
        if clean_rows
        else None
    )
    complete_best = summary_payload["best_complete_dataset_diagnostic"]
    cards = []
    if best_clean:
        cards.append(
            headline_card(
                "Best clean holdout",
                best_clean["model"],
                f"{best_clean['training_scenario']} on {best_clean['test_dataset']}",
                best_clean["phishing_f1"],
                "Untouched 20 percent test rows",
            )
        )
    if complete_best:
        cards.append(
            headline_card(
                "Best complete diagnostic",
                complete_best["model"],
                f"{complete_best['training_scenario']} on {complete_best['test_dataset']}",
                complete_best["phishing_f1"],
                "Includes rows used during training",
            )
        )
    if not cards:
        return "<p>No winner rows are available yet.</p>"
    return '<div class="headline-grid">' + "\n".join(cards) + "</div>"


def html_decision_dashboard(decision_payload):
    """Render top-level decision cards from programmatic decision data."""
    backend = decision_payload.get("backend_recommendation")
    allrounder = decision_payload.get("best_allrounder")
    single_combined = decision_payload.get("best_single_source_on_combined_test")
    overall_combined = decision_payload.get("best_overall_on_combined_test")
    training_dataset = decision_payload.get("best_training_dataset")
    cards = [
        summary_decision_card(
            "Backend Recommendation",
            backend.get("model") if backend else "No result",
            f"trained on {backend.get('training_scenario')}" if backend else "",
            "Mean clean F1",
            backend.get("mean_phishing_f1") if backend else None,
            backend.get("reason") if backend else "No clean holdout rows were available.",
        ),
        summary_decision_card(
            "Best All-Rounder",
            allrounder.get("model") if allrounder else "No result",
            f"{allrounder.get('training_scenario')} across clean holdouts"
            if allrounder
            else "",
            "Worst clean F1",
            allrounder.get("worst_phishing_f1") if allrounder else None,
            "Ranked by mean phishing F1 first, then worst-case F1 and phishing recall.",
        ),
        summary_decision_card(
            "Best Single-Source on Combined Test",
            single_combined.get("model") if single_combined else "No result",
            f"trained on {single_combined.get('training_scenario')}"
            if single_combined
            else "",
            "Phishing F1",
            single_combined.get("phishing_f1") if single_combined else None,
            "This answers: if only one dataset is used for training, which model does best on the mixed combined test set?",
        ),
        summary_decision_card(
            "Best Overall on Combined Test",
            overall_combined.get("model") if overall_combined else "No result",
            f"trained on {overall_combined.get('training_scenario')}"
            if overall_combined
            else "",
            "Phishing F1",
            overall_combined.get("phishing_f1") if overall_combined else None,
            "This allows the single-source winner to be compared against combined training.",
        ),
        summary_decision_card(
            "Best Training Dataset",
            training_dataset.get("training_scenario") if training_dataset else "No result",
            f"best model: {training_dataset.get('best_model')}"
            if training_dataset
            else "",
            "Mean clean F1",
            training_dataset.get("mean_phishing_f1") if training_dataset else None,
            "This ranks the training scenario itself, independent of only one headline test result.",
        ),
    ]
    return '<div class="decision-grid">' + "\n".join(cards) + "</div>"


def html_decision_explanation(decision_payload):
    """Explain why the backend answer can differ from the single-source answer."""
    backend = decision_payload.get("backend_recommendation")
    single_combined = decision_payload.get("best_single_source_on_combined_test")
    overall_combined = decision_payload.get("best_overall_on_combined_test")
    if not backend or not single_combined:
        return ""

    backend_training = display_scenario_name(backend["training_scenario"])
    single_training = display_scenario_name(single_combined["training_scenario"])
    combined_test_delta = ""
    if overall_combined:
        difference = overall_combined["phishing_f1"] - single_combined["phishing_f1"]
        combined_test_delta = (
            f" On the Combined Test, the best combined-training result is "
            f"{escape(overall_combined['model'])} with Combined Test F1 "
            f"{overall_combined['phishing_f1']:.4f}, which is {difference:.4f} "
            "higher than the best single-source result."
        )

    return f"""
<div class="decision-explainer">
<h3>Why the backend recommendation is not the single-source winner</h3>
<p>
Linear SVM trained on {escape(single_training)} answers a narrower question:
which model performs best when training is limited to one dataset and testing is
done on the mixed combined test set. It scored phishing F1
{single_combined['phishing_f1']:.4f} for that question.
</p>
<p>
XGBoost trained on the {escape(backend_training)} is recommended for the backend
because it is judged across clean holdout tests, not just one single-source
comparison. The combined training scenario gives the model more diverse training rows
from PhiUSIIL, LegitPhish, and PhishStorm, so it learns URL patterns from multiple
sources instead of one dataset's habits. Its mean clean-holdout F1 is
{backend['mean_phishing_f1']:.4f} and its worst clean-holdout F1 is
{backend['worst_phishing_f1']:.4f}.{combined_test_delta}
</p>
</div>
"""


def summary_decision_card(title, primary, context, metric_label, metric_value, note):
    metric_text = "n/a" if metric_value is None else f"{float(metric_value):.4f}"
    return (
        '<article class="decision-card">'
        f"<span>{escape(title)}</span>"
        f"<strong>{escape(primary)}</strong>"
        f"<p>{escape(context)}</p>"
        f'<div class="score">{escape(metric_label)}: {metric_text}</div>'
        f"<small>{tooltip_markup('Why?', note)}</small>"
        "</article>"
    )


def html_research_answers(decision_payload):
    backend = decision_payload.get("backend_recommendation")
    single_combined = decision_payload.get("best_single_source_on_combined_test")
    overall_combined = decision_payload.get("best_overall_on_combined_test")
    training_dataset = decision_payload.get("best_training_dataset")
    allrounder = decision_payload.get("best_allrounder")
    return f"""
<section class="look-for">
<h3>What to Look For</h3>
<ul>
<li>Use the decision cards for the headline dissertation answers.</li>
<li>Use the filters when you want to compare one model, one training dataset, or one test dataset.</li>
<li>Use clean holdout rows for fair model selection. Treat complete-dataset rows as diagnostic only because they include training rows.</li>
</ul>
</section>
<div class="answer-grid">
<article>
<h3>Which model should go into the backend?</h3>
<p>{backend_sentence(backend)}</p>
</article>
<article>
<h3>Which single-dataset training result is best on the combined test set?</h3>
<p>{single_source_sentence(single_combined)}</p>
</article>
<article>
<h3>Does combined training help?</h3>
<p>{combined_training_sentence(overall_combined, single_combined)}</p>
</article>
<article>
<h3>Which training dataset is best overall?</h3>
<p>{training_dataset_sentence(training_dataset)}</p>
</article>
<article>
<h3>What matters most?</h3>
<p>{allrounder_sentence(allrounder)}</p>
</article>
</div>
"""


def backend_sentence(record):
    if not record:
        return "No clean holdout rows are available, so no backend recommendation can be made yet."
    return (
        f"Use {escape(record['model'])} trained on {escape(record['training_scenario'])}. "
        f"Its mean clean-holdout phishing F1 is {record['mean_phishing_f1']:.4f}, "
        f"with worst clean F1 {record['worst_phishing_f1']:.4f}. "
        f"{escape(record['reason'])}"
    )


def single_source_sentence(record):
    if not record:
        return "No single-source combined-test result is available."
    return (
        f"The best single-source result on the combined test set is {escape(record['model'])} "
        f"trained on {escape(record['training_scenario'])}, with phishing F1 "
        f"{record['phishing_f1']:.4f}."
    )


def combined_training_sentence(overall_record, single_record):
    if not overall_record or not single_record:
        return "The report needs both combined-training and single-source rows to answer this."
    difference = overall_record["phishing_f1"] - single_record["phishing_f1"]
    return (
        f"The best combined-test result is {escape(overall_record['model'])} trained on "
        f"{escape(overall_record['training_scenario'])}, with phishing F1 "
        f"{overall_record['phishing_f1']:.4f}. This is {difference:.4f} higher than the "
        "best single-source combined-test result."
    )


def training_dataset_sentence(record):
    if not record:
        return "No training-scenario ranking is available."
    return (
        f"The strongest training scenario is {escape(record['training_scenario'])}, "
        f"with mean clean-holdout phishing F1 {record['mean_phishing_f1']:.4f}. "
        f"Its best model in this scenario is {escape(record['best_model'])}."
    )


def allrounder_sentence(record):
    if not record:
        return "No all-rounder ranking is available."
    return (
        f"The leading all-round model artefact is {escape(record['model'])} trained on "
        f"{escape(record['training_scenario'])}. This supports a dataset-diversity argument: "
        "the backend choice should be justified using cross-dataset clean-holdout behaviour, "
        "not only the highest score on one same-source split."
    )


def html_how_to_read_report():
    return """
<div class="concept-grid">
<article class="guide-card">
<h3>Rows and Columns</h3>
<p>Rows in each model matrix are training scenarios: the dataset used to teach the model.</p>
<p>Columns are test scenarios: the dataset used to judge the trained model.</p>
</article>
<article class="guide-card">
<h3>Cell Values</h3>
<p>Each cell shows the phishing F1 score first because it balances catching phishing URLs with avoiding false alarms.</p>
<p><strong>Rows Tested</strong> is the number of URL records scored after any overlap removal.</p>
</article>
<article class="guide-card">
<h3>Training Scenarios</h3>
<p><strong>Single-source training</strong> means the model learned from one dataset only, such as PhiUSIIL or PhishStorm.</p>
<p><strong>Combined training</strong> means the model learned from the combined training split made from all usable datasets.</p>
</article>
<article class="guide-card">
<h3>Fairness Notes</h3>
<p><strong>URL Overlap Removed</strong> counts URLs removed from a test before scoring because the same normalized URL appeared in that model's training data.</p>
<p>The complete combined dataset is diagnostic because it includes training rows; use clean holdout results for model selection.</p>
</article>
</div>
<table class="plain-guide-table">
<thead><tr><th>Measurement</th><th>Meaning</th><th>Simple Example</th></tr></thead>
<tbody>
<tr><td>Accuracy</td><td>Overall correctness across phishing and legitimate URLs.</td><td>90 correct predictions out of 100 URLs gives accuracy 0.9000.</td></tr>
<tr><td>Phishing Precision</td><td>When the model predicts phishing, how often it is correct.</td><td>If 80 predicted-phishing URLs are truly phishing out of 100 predicted phishing, precision is 0.8000.</td></tr>
<tr><td>Phishing Recall</td><td>How many real phishing URLs the model catches.</td><td>If it catches 95 out of 100 real phishing URLs, recall is 0.9500.</td></tr>
<tr><td>Phishing F1</td><td>A balanced score combining phishing precision and phishing recall.</td><td>High F1 means the model catches phishing without creating too many false alarms.</td></tr>
</tbody>
</table>
"""


def html_per_model_results(metrics_frame):
    """Render one training-vs-testing matrix per model."""
    if metrics_frame.empty:
        return '<p class="empty">No model results are available yet.</p>'
    sections = []
    model_names = ordered_values(metrics_frame["model"].unique(), list(MODEL_GUIDE))
    for model_name in model_names:
        model_frame = metrics_frame[metrics_frame["model"] == model_name]
        sections.append(
            '<article class="model-block">'
            f"<h3>{escape(model_name)}: Training Scenario vs Test Scenario</h3>"
            f"<p>{escape(MODEL_GUIDE.get(model_name, 'Model explanation unavailable.'))}</p>"
            f"{html_single_model_matrix(model_frame)}"
            "</article>"
        )
    return "\n".join(sections)


def html_single_model_matrix(model_frame):
    training_names = ordered_values(
        model_frame["training_scenario"].unique(),
        TRAINING_SCENARIO_ORDER,
    )
    test_names = ordered_values(model_frame["test_dataset"].unique(), TEST_SCENARIO_ORDER)
    rows = [
        '<div class="matrix-scroll">',
        '<table class="model-matrix">',
        f"<thead><tr>{html_th('Trained On')}",
    ]
    for test_name in test_names:
        rows.append(
            "<th>"
            f"{tooltip_markup(display_scenario_name(test_name), COLUMN_GUIDE['Tested On'])}"
            f"{diagnostic_pill(test_name)}</th>"
        )
    rows.extend(["</tr></thead>", "<tbody>"])
    for training_name in training_names:
        rows.append(f"<tr><th>{escape(display_scenario_name(training_name))}</th>")
        for test_name in test_names:
            cell_frame = model_frame[
                (model_frame["training_scenario"] == training_name)
                & (model_frame["test_dataset"] == test_name)
            ]
            rows.append(html_model_matrix_cell(cell_frame))
        rows.append("</tr>")
    rows.extend(["</tbody>", "</table>", "</div>"])
    return "\n".join(rows)


def html_model_matrix_cell(cell_frame):
    if cell_frame.empty:
        return '<td class="muted-cell">Not tested</td>'
    row = cell_frame.sort_values(
        by=["evaluation_scope", "phishing_f1", "phishing_recall", "accuracy"],
        ascending=[True, False, False, False],
    ).iloc[0]
    overlap = int(row.get("training_url_overlap_removed", 0))
    includes_training = truthy(row["contains_training_rows"])
    notes = [
        f"Rows Tested: {int(row['rows_tested']):,}",
        f"URL Overlap Removed: {overlap:,}",
    ]
    if includes_training:
        notes.append("Diagnostic: includes training rows")
    return (
        '<td class="matrix-score-cell">'
        f"<strong>F1 {row['phishing_f1']:.4f}</strong>"
        f"<span>Recall {row['phishing_recall']:.4f}</span>"
        f"<span>Accuracy {row['accuracy']:.4f}</span>"
        f'<small>{" | ".join(escape(note) for note in notes)}</small>'
        "</td>"
    )


def html_model_comparison(metrics_frame, decision_payload):
    if metrics_frame.empty:
        return '<p class="empty">No clean holdout comparisons are available yet.</p>'
    return (
        '<div class="comparison-grid">'
        "<article>"
        "<h3>Best Model by Training Scenario on Combined Test</h3>"
        "<p>This answers: if the training source is fixed, which algorithm wins on the mixed combined test set?</p>"
        f"{html_combined_test_winners_by_training(metrics_frame)}"
        "</article>"
        "<article>"
        "<h3>Overall Model Ranking Across Clean Holdouts</h3>"
        "<p>This ranks complete trained model artefacts by mean phishing F1, then worst-case phishing F1.</p>"
        f"{html_allrounder_ranking(decision_payload)}"
        "</article>"
        "</div>"
        "<p class=\"section-note\">Use this section for model comparison. Use the per-model matrices above when you want to understand one algorithm in detail.</p>"
    )


def html_combined_test_winners_by_training(metrics_frame):
    combined_rows = metrics_frame[metrics_frame["test_dataset"] == "combined_test"]
    if combined_rows.empty:
        return "<p>No combined-test rows are available.</p>"
    best_rows_frame = best_rows(combined_rows, ["training_scenario"])
    rows = [
        "<table>",
        "<thead><tr>"
        f"{html_th('Trained On')}"
        f"{html_th('Winning Model')}"
        f"{html_th('Phishing F1', class_name='num')}"
        f"{html_th('Rows Tested', class_name='num')}"
        "</tr></thead>",
        "<tbody>",
    ]
    for row in best_rows_frame.to_dict(orient="records"):
        rows.append(
            "<tr>"
            f"<td>{escape(display_scenario_name(row['training_scenario']))}</td>"
            f"<td>{escape(row['model'])}</td>"
            f"<td class=\"num\">{row['phishing_f1']:.4f}</td>"
            f"<td class=\"num\">{int(row['rows_tested']):,}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def html_allrounder_ranking(decision_payload):
    ranking = decision_payload.get("allrounder_ranking", [])
    if not ranking:
        return "<p>No all-rounder ranking is available.</p>"
    rows = [
        "<table>",
        "<thead><tr>"
        f"{html_th('Rank')}"
        f"{html_th('Model')}"
        f"{html_th('Trained On')}"
        f"{html_th('Mean F1', 'Average phishing F1 across clean holdout datasets.', 'num')}"
        f"{html_th('Worst F1', 'Lowest phishing F1 across clean holdout datasets.', 'num')}"
        "</tr></thead>",
        "<tbody>",
    ]
    for index, row in enumerate(ranking, start=1):
        badge = ' <span class="winner">Backend candidate</span>' if index == 1 else ""
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{escape(row['model'])}{badge}</td>"
            f"<td>{escape(display_scenario_name(row['training_scenario']))}</td>"
            f"<td class=\"num\">{row['mean_phishing_f1']:.4f}</td>"
            f"<td class=\"num\">{row['worst_phishing_f1']:.4f}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def ordered_values(values, preferred_order):
    values = [str(value) for value in values]
    order = {name: index for index, name in enumerate(preferred_order)}
    return sorted(values, key=lambda value: (order.get(value, len(order)), value))


def display_scenario_name(name):
    return SCENARIO_LABELS.get(str(name), str(name).replace("_", " ").title())


def diagnostic_pill(test_name):
    if test_name != "complete_combined_dataset":
        return ""
    return ' <span class="diagnostic-pill">Diagnostic</span>'


def html_th(label, tip=None, class_name=""):
    """Build a table header with the same hover tooltip style as the JS tables."""
    class_attribute = f' class="{class_name}"' if class_name else ""
    tooltip = tip or METRIC_GUIDE.get(label) or COLUMN_GUIDE.get(label, "")
    if tooltip:
        content = tooltip_markup(label, tooltip)
    else:
        content = escape(label)
    return f"<th{class_attribute}>{content}</th>"


def tooltip_markup(label, tip):
    return (
        '<span class="tooltip" tabindex="0">'
        f"{escape(label)} <span class=\"hint\">?</span>"
        f'<span class="tooltip-text" role="tooltip">{escape(tip)}</span>'
        "</span>"
    )


def headline_card(label, model_name, context, phishing_f1, evidence):
    return (
        '<article class="headline-card">'
        f"<span>{escape(label)}</span>"
        f"<strong>{escape(model_name)}</strong>"
        f"<p>{escape(context)}</p>"
        f'<div class="score">Phishing F1: {phishing_f1:.4f}</div>'
        f"<small>{escape(evidence)}</small>"
        "</article>"
    )


def html_matrix(metrics_frame):
    summary = (
        metrics_frame.groupby(["training_scenario", "test_dataset"])["phishing_f1"]
        .max()
        .reset_index()
        .sort_values(["training_scenario", "test_dataset"])
    )
    rows = [
        "<table>",
        "<thead><tr>"
        f"{html_th('Trained On')}"
        f"{html_th('Tested On')}"
        f"{html_th('Phishing F1', 'Best phishing F1 for this training/testing pair.', 'num')}"
        f"{html_th('Chart')}"
        "</tr></thead>",
        "<tbody>",
    ]
    for row in summary.to_dict(orient="records"):
        width = max(row["phishing_f1"] * 100, 2)
        rows.append(
            "<tr>"
            f"<td>{escape(row['training_scenario'])}</td>"
            f"<td>{escape(row['test_dataset'])}</td>"
            f'<td class="num">{row["phishing_f1"]:.4f}</td>'
            f'<td><div class="track"><div class="bar" style="width:{width:.2f}%"></div></div></td>'
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def combined_training_comparison(metrics_frame):
    """Compare combined_dataset against the best single-dataset run per test set."""
    rows = []
    for test_dataset, group in metrics_frame.groupby("test_dataset", sort=True):
        combined_group = group[group["training_scenario"] == "combined_dataset"]
        single_group = group[group["training_scenario"] != "combined_dataset"]
        if combined_group.empty or single_group.empty:
            continue
        combined = combined_group.sort_values(
            by=["phishing_f1", "phishing_recall", "accuracy"],
            ascending=False,
        ).iloc[0]
        single = single_group.sort_values(
            by=["phishing_f1", "phishing_recall", "accuracy"],
            ascending=False,
        ).iloc[0]
        rows.append(
            {
                "test_dataset": str(test_dataset),
                "combined_model": str(combined["model"]),
                "combined_f1": float(combined["phishing_f1"]),
                "single_source_scenario": str(single["training_scenario"]),
                "single_source_model": str(single["model"]),
                "single_source_f1": float(single["phishing_f1"]),
                "difference": float(combined["phishing_f1"] - single["phishing_f1"]),
            }
        )
    return rows


def html_combined_comparison(metrics_frame):
    comparison_rows = combined_training_comparison(metrics_frame)
    if not comparison_rows:
        return (
            "<p>No single-source comparison rows are available for this selection.</p>"
        )
    rows = [
        "<table>",
        "<thead><tr>"
        f"{html_th('Tested On')}"
        f"{html_th('Combined Best Model', 'Best model trained on the combined training dataset.')}"
        f"{html_th('Phishing F1', 'Phishing F1 for the best combined-training result.', 'num')}"
        f"{html_th('Best Single-Source Scenario', 'Best one-dataset training source for the same test dataset.')}"
        f"{html_th('Best Single-Source Model', 'Best one-dataset trained model for the same test dataset.')}"
        f"{html_th('Single-Source F1', 'Phishing F1 for the best single-source result.', 'num')}"
        f"{html_th('Difference', class_name='num')}"
        "</tr></thead>",
        "<tbody>",
    ]
    for row in comparison_rows:
        badge = ' <span class="winner">Winner</span>' if row["difference"] >= 0 else ""
        rows.append(
            "<tr>"
            f"<td>{escape(row['test_dataset'])}</td>"
            f"<td>{escape(row['combined_model'])}{badge}</td>"
            f'<td class="num">{row["combined_f1"]:.4f}</td>'
            f"<td>{escape(row['single_source_scenario'])}</td>"
            f"<td>{escape(row['single_source_model'])}</td>"
            f'<td class="num">{row["single_source_f1"]:.4f}</td>'
            f'<td class="num">{row["difference"]:.4f}</td>'
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def html_table(metrics_frame, winner):
    best_index = (
        metrics_frame["phishing_f1"].idxmax() if winner and not metrics_frame.empty else None
    )
    rows = [
        "<table>",
        "<thead><tr>"
        f"{html_th('Scope')}"
        f"{html_th('Trained On')}"
        f"{html_th('Model')}"
        f"{html_th('Tested On')}"
        f"{html_th('Rows', class_name='num')}"
        f"{html_th('URL Overlap Removed', class_name='num')}"
        f"{html_th('Training Rows Included?')}"
        f"{html_th('Accuracy', class_name='num')}"
        f"{html_th('Phishing Precision', class_name='num')}"
        f"{html_th('Phishing Recall', class_name='num')}"
        f"{html_th('Phishing F1', class_name='num')}"
        "</tr></thead>",
        "<tbody>",
    ]
    for index, row in metrics_frame.iterrows():
        badge = ' <span class="winner">Winner</span>' if index == best_index else ""
        includes_training = "Yes" if truthy(row["contains_training_rows"]) else "No"
        rows.append(
            "<tr>"
            f"<td>{escape(format_scope_label(row['evaluation_scope']))}</td>"
            f"<td>{escape(row['training_scenario'])}{badge}</td>"
            f"<td>{escape(row['model'])}</td>"
            f"<td>{escape(row['test_dataset'])}</td>"
            f'<td class="num">{int(row["rows_tested"]):,}</td>'
            f'<td class="num">{int(row.get("training_url_overlap_removed", 0)):,}</td>'
            f"<td>{escape(includes_training)}</td>"
            f'<td class="num">{row["accuracy"]:.4f}</td>'
            f'<td class="num">{row["phishing_precision"]:.4f}</td>'
            f'<td class="num">{row["phishing_recall"]:.4f}</td>'
            f'<td class="num">{row["phishing_f1"]:.4f}</td>'
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def html_programmatic_summary(summary_payload):
    clean_rows = summary_payload["best_clean_holdout_by_test_dataset"]
    complete_best = summary_payload["best_complete_dataset_diagnostic"]
    rows = [
        "<table>",
        "<thead><tr>"
        f"{html_th('Summary Item', 'The dissertation question answered by this row.')}"
        f"{html_th('Tested On')}"
        f"{html_th('Trained On')}"
        f"{html_th('Model')}"
        f"{html_th('Phishing F1', class_name='num')}"
        f"{html_th('Evidence Type')}"
        "</tr></thead>",
        "<tbody>",
    ]
    for row in clean_rows:
        rows.append(
            "<tr>"
            "<td>Best clean holdout result</td>"
            f"<td>{escape(row['test_dataset'])}</td>"
            f"<td>{escape(row['training_scenario'])}</td>"
            f"<td>{escape(row['model'])}</td>"
            f"<td class=\"num\">{row['phishing_f1']:.4f}</td>"
            "<td>Untouched 20 percent test rows</td>"
            "</tr>"
        )
    if complete_best:
        rows.append(
            "<tr>"
            "<td>Best complete combined diagnostic</td>"
            f"<td>{escape(complete_best['test_dataset'])}</td>"
            f"<td>{escape(complete_best['training_scenario'])}</td>"
            f"<td>{escape(complete_best['model'])}</td>"
            f"<td class=\"num\">{complete_best['phishing_f1']:.4f}</td>"
            "<td>Includes training rows</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def html_confusion_matrices(confusion_matrices):
    if not confusion_matrices:
        return "<p>No confusion matrices are available.</p>"
    sections = []
    for key in sorted(confusion_matrices):
        payload = confusion_matrices[key]
        matrix = payload.get("matrix", [[0, 0], [0, 0]])
        title = escape(key.replace("::", " / "))
        sections.append(
            "<details>"
            f"<summary>{title}</summary>"
            "<table>"
            "<thead><tr><th>Actual / Predicted</th><th class=\"num\">0 phishing</th><th class=\"num\">1 legitimate</th></tr></thead>"
            "<tbody>"
            f"<tr><td>0 phishing</td><td class=\"num\">{int(matrix[0][0]):,}</td><td class=\"num\">{int(matrix[0][1]):,}</td></tr>"
            f"<tr><td>1 legitimate</td><td class=\"num\">{int(matrix[1][0]):,}</td><td class=\"num\">{int(matrix[1][1]):,}</td></tr>"
            "</tbody>"
            "</table>"
            "</details>"
        )
    return "\n".join(sections)


def metric_guide_cards():
    return "\n".join(
        f'<article class="guide-card" title="{escape(text)}"><h3>{escape(name)}</h3><p>{escape(text)}</p></article>'
        for name, text in METRIC_GUIDE.items()
    )


def model_guide_cards():
    return "\n".join(
        f'<article class="guide-card" title="{escape(text)}"><h3>{escape(name)}</h3><p>{escape(text)}</p></article>'
        for name, text in MODEL_GUIDE.items()
    )


def summary_card(label, value):
    return (
        '<div class="metric-card">'
        f"<span>{escape(label)}</span>"
        f"<strong>{escape(value)}</strong>"
        "</div>"
    )


def report_script():
    return """<script>
const report = JSON.parse(document.getElementById("reportData").textContent);
const rows = report.metrics || [];
const matrices = report.confusion_matrices || {};
const decision = report.summary?.decision_summary || {};
const metricLabels = {
  accuracy: "Accuracy",
  phishing_precision: "Phishing Precision",
  phishing_recall: "Phishing Recall",
  phishing_f1: "Phishing F1"
};
const scopeLabels = {
  clean_holdout: "Clean Holdout",
  complete_dataset: "Complete Dataset Diagnostic"
};
const columnTips = {
  rank: "Position after sorting by the selected metric. Rank 1 is the current winner.",
  training: "The dataset source used to train this model.",
  model: "The machine-learning algorithm that was trained and tested.",
  tested: "The dataset used to test predictions.",
  scope: "Clean holdout uses unseen 20 percent test rows; complete diagnostic includes training rows.",
  rows: "Number of URL rows scored in this result.",
  overlap: "Rows removed before scoring because their exact URL already appeared in this model's training scenario.",
  included: "Whether the test set includes rows that were also used during training.",
  meanF1: "Average phishing F1 across the selected rows for this model.",
  worstF1: "Lowest phishing F1 across the selected rows. Higher is more robust.",
  bestScenario: "The strongest training/testing pair for this model under the current filters."
};
let activePreset = "clean";

const controls = {
  scope: document.getElementById("scopeFilter"),
  training: document.getElementById("trainingFilter"),
  tested: document.getElementById("testedFilter"),
  model: document.getElementById("modelFilter"),
  metric: document.getElementById("sortMetric")
};

function unique(values) {
  return [...new Set(values.filter(Boolean))].sort();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function number(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function count(value) {
  return Number(value || 0).toLocaleString();
}

function metricInfo(metricKey) {
  return report.metric_guide?.[metricLabels[metricKey]] || "";
}

function modelInfo(modelName) {
  return report.model_guide?.[modelName] || "";
}

function fillSelect(select, values, allLabel, defaultValue = "all") {
  select.innerHTML = "";
  const all = document.createElement("option");
  all.value = "all";
  all.textContent = allLabel;
  select.appendChild(all);
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });
  select.value = defaultValue;
}

function initialiseControls() {
  fillSelect(
    controls.scope,
    unique(rows.map((row) => row.evaluation_scope)),
    "All scopes",
    "clean_holdout"
  );
  [...controls.scope.options].forEach((option) => {
    if (option.value === "clean_holdout") option.textContent = "Clean holdout only";
    if (option.value === "complete_dataset") option.textContent = "Complete diagnostic";
  });
  fillSelect(
    controls.training,
    unique(rows.map((row) => row.training_scenario)),
    "All training scenarios"
  );
  fillSelect(
    controls.tested,
    unique(rows.map((row) => row.test_dataset)),
    "All test datasets"
  );
  fillSelect(controls.model, unique(rows.map((row) => row.model)), "All models");
  Object.entries(metricLabels).forEach(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    option.title = metricInfo(value);
    controls.metric.appendChild(option);
  });
  controls.metric.value = "phishing_f1";
}

function selected(select) {
  return select.value === "all" ? null : select.value;
}

function applyPreset(name) {
  activePreset = name;
  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.classList.toggle("active", button.dataset.preset === name);
  });
  controls.scope.value = "clean_holdout";
  controls.training.value = "all";
  controls.tested.value = "all";
  controls.model.value = "all";
  controls.metric.value = "phishing_f1";
  if (name === "backend" && decision.backend_recommendation) {
    controls.training.value = decision.backend_recommendation.training_scenario || "all";
    controls.model.value = decision.backend_recommendation.model || "all";
  }
  if (name === "allrounder") controls.scope.value = "clean_holdout";
  if (name === "main") controls.tested.value = "phiusiil_main_test";
  if (name === "combined") controls.tested.value = "combined_test";
  if (name === "singleSourceCombined") controls.tested.value = "combined_test";
  if (name === "diagnostic") controls.scope.value = "complete_dataset";
  render();
}

function filteredRows() {
  const scope = selected(controls.scope);
  const training = selected(controls.training);
  const tested = selected(controls.tested);
  const model = selected(controls.model);
  return rows
    .filter((row) => !scope || row.evaluation_scope === scope)
    .filter((row) => !training || row.training_scenario === training)
    .filter((row) => !tested || row.test_dataset === tested)
    .filter((row) => !model || row.model === model)
    .filter((row) => {
      if (activePreset !== "external") return true;
      return (
        row.evaluation_scope === "clean_holdout" &&
        row.test_dataset !== "phiusiil_main_test" &&
        row.test_dataset !== "combined_test"
      );
    })
    .filter((row) => {
      if (activePreset !== "singleSourceCombined") return true;
      return row.training_scenario !== "combined_dataset";
    });
}

function sortRows(inputRows) {
  const metric = controls.metric.value;
  return [...inputRows].sort((a, b) => {
    const primary = Number(b[metric] || 0) - Number(a[metric] || 0);
    if (primary !== 0) return primary;
    return Number(b.phishing_recall || 0) - Number(a.phishing_recall || 0);
  });
}

function bestRow(inputRows) {
  return sortRows(inputRows)[0] || null;
}

function summaryText(inputRows, metric) {
  const best = bestRow(inputRows);
  if (!best) return "No rows";
  return `${count(inputRows.length)} rows. Winner: ${best.model} (${number(best[metric])})`;
}

function metricHeader(metricKey) {
  const tip = escapeHtml(metricInfo(metricKey));
  return `<span class="tooltip" data-tooltip="${tip}" tabindex="0">${escapeHtml(metricLabels[metricKey])} <span class="hint">?</span><span class="tooltip-text" role="tooltip">${tip}</span></span>`;
}

function columnHeader(label, tipKey) {
  const tip = escapeHtml(columnTips[tipKey] || "");
  return `<span class="tooltip" data-tooltip="${tip}" tabindex="0">${escapeHtml(label)} <span class="hint">?</span><span class="tooltip-text" role="tooltip">${tip}</span></span>`;
}

function modelCell(modelName) {
  const tip = escapeHtml(modelInfo(modelName));
  return `<span class="tooltip" data-tooltip="${tip}" tabindex="0">${escapeHtml(modelName)}<span class="tooltip-text" role="tooltip">${tip}</span></span>`;
}

function barCell(value) {
  const width = Math.max(2, Math.min(100, Number(value || 0) * 100));
  return `<div class="score-cell"><span>${number(value)}</span><div class="track"><div class="bar" style="width:${width}%"></div></div></div>`;
}

function groupedRows(inputRows, keyFn) {
  const groups = new Map();
  inputRows.forEach((row) => {
    const key = keyFn(row);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  });
  return groups;
}

function average(inputRows, field) {
  if (!inputRows.length) return 0;
  return inputRows.reduce((total, row) => total + Number(row[field] || 0), 0) / inputRows.length;
}

function minValue(inputRows, field) {
  if (!inputRows.length) return 0;
  return Math.min(...inputRows.map((row) => Number(row[field] || 0)));
}

function renderModelSummary(inputRows) {
  const groups = groupedRows(inputRows, (row) => row.model);
  const summaries = [...groups.entries()].map(([modelName, modelRows]) => {
    const best = bestRow(modelRows);
    return {
      model: modelName,
      rows: modelRows.length,
      meanF1: average(modelRows, "phishing_f1"),
      worstF1: minValue(modelRows, "phishing_f1"),
      meanPrecision: average(modelRows, "phishing_precision"),
      meanRecall: average(modelRows, "phishing_recall"),
      meanAccuracy: average(modelRows, "accuracy"),
      best
    };
  }).sort((a, b) => {
    const primary = b.meanF1 - a.meanF1;
    if (primary !== 0) return primary;
    return b.worstF1 - a.worstF1;
  });
  document.getElementById("modelSummaryStats").textContent = `${summaries.length} models compared`;
  if (!summaries.length) {
    document.getElementById("modelSummaryView").innerHTML = '<p class="empty">No matching model summary.</p>';
    return;
  }
  const html = [`<table><thead><tr>
    <th>${columnHeader("Model", "model")}</th>
    <th class="num">${columnHeader("Mean F1", "meanF1")}</th>
    <th class="num">${columnHeader("Worst F1", "worstF1")}</th>
    <th class="num">${metricHeader("phishing_precision")}</th>
    <th class="num">${metricHeader("phishing_recall")}</th>
    <th class="num">${metricHeader("accuracy")}</th>
    <th>${columnHeader("Best Scenario", "bestScenario")}</th>
  </tr></thead><tbody>`];
  summaries.forEach((summary, index) => {
    const winner = index === 0 ? ' <span class="winner">Best model</span>' : "";
    html.push(`<tr>
      <td>${modelCell(summary.model)}${winner}</td>
      <td class="num">${barCell(summary.meanF1)}</td>
      <td class="num">${number(summary.worstF1)}</td>
      <td class="num">${number(summary.meanPrecision)}</td>
      <td class="num">${number(summary.meanRecall)}</td>
      <td class="num">${number(summary.meanAccuracy)}</td>
      <td>${escapeHtml(summary.best.training_scenario)} / ${escapeHtml(summary.best.test_dataset)}</td>
    </tr>`);
  });
  html.push("</tbody></table>");
  document.getElementById("modelSummaryView").innerHTML = html.join("");
}

function renderScenarioView(inputRows) {
  const groups = groupedRows(
    inputRows,
    (row) => `${row.training_scenario}::${row.test_dataset}`
  );
  const summaries = [...groups.entries()].map(([key, scenarioRows]) => {
    const best = bestRow(scenarioRows);
    const [training, tested] = key.split("::");
    return { training, tested, rows: scenarioRows.length, best };
  }).sort((a, b) => a.training.localeCompare(b.training) || a.tested.localeCompare(b.tested));
  document.getElementById("scenarioSummary").textContent = `${summaries.length} scenarios under current filters`;
  if (!summaries.length) {
    document.getElementById("scenarioView").innerHTML = '<p class="empty">No matching scenarios.</p>';
    return;
  }
  const html = [`<table><thead><tr>
    <th>${columnHeader("Training", "training")}</th>
    <th>${columnHeader("Tested On", "tested")}</th>
    <th>${columnHeader("Winning Model", "model")}</th>
    <th class="num">${metricHeader("phishing_f1")}</th>
    <th class="num">${metricHeader("phishing_precision")}</th>
    <th class="num">${metricHeader("phishing_recall")}</th>
    <th class="num">${metricHeader("accuracy")}</th>
    <th class="num">${columnHeader("Rows", "rows")}</th>
  </tr></thead><tbody>`];
  summaries.forEach((summary) => {
    html.push(`<tr>
      <td>${escapeHtml(summary.training)}</td>
      <td>${escapeHtml(summary.tested)}</td>
      <td>${modelCell(summary.best.model)} <span class="winner">Winner</span></td>
      <td class="num">${barCell(summary.best.phishing_f1)}</td>
      <td class="num">${number(summary.best.phishing_precision)}</td>
      <td class="num">${number(summary.best.phishing_recall)}</td>
      <td class="num">${number(summary.best.accuracy)}</td>
      <td class="num">${count(summary.best.rows_tested)}</td>
    </tr>`);
  });
  html.push("</tbody></table>");
  document.getElementById("scenarioView").innerHTML = html.join("");
}

function renderRanking(inputRows) {
  const metric = controls.metric.value;
  const sorted = sortRows(inputRows);
  document.getElementById("rankingSummary").textContent = summaryText(sorted, metric);
  if (!sorted.length) {
    document.getElementById("rankingView").innerHTML = '<p class="empty">No matching rows.</p>';
    return;
  }
  const html = [`<table><thead><tr>
    <th>${columnHeader("Rank", "rank")}</th>
    <th>${columnHeader("Training", "training")}</th>
    <th>${columnHeader("Model", "model")}</th>
    <th>${columnHeader("Tested On", "tested")}</th>
    <th>${columnHeader("Scope", "scope")}</th>
    <th class="num">${metricHeader(metric)}</th>
    <th class="num">${metricHeader("phishing_precision")}</th>
    <th class="num">${metricHeader("phishing_recall")}</th>
    <th class="num">${metricHeader("accuracy")}</th>
  </tr></thead><tbody>`];
  sorted.forEach((row, index) => {
    const winner = index === 0 ? ' <span class="winner">Winner</span>' : "";
    html.push(`<tr>
      <td>${index + 1}</td>
      <td>${escapeHtml(row.training_scenario)}${winner}</td>
      <td>${modelCell(row.model)}</td>
      <td>${escapeHtml(row.test_dataset)}</td>
      <td>${escapeHtml(scopeLabels[row.evaluation_scope] || row.evaluation_scope)}</td>
      <td class="num">${barCell(row[metric])}</td>
      <td class="num">${number(row.phishing_precision)}</td>
      <td class="num">${number(row.phishing_recall)}</td>
      <td class="num">${number(row.accuracy)}</td>
    </tr>`);
  });
  html.push("</tbody></table>");
  document.getElementById("rankingView").innerHTML = html.join("");
}

function renderMatrix(inputRows) {
  const metric = controls.metric.value;
  const trainingNames = unique(inputRows.map((row) => row.training_scenario));
  const testNames = unique(inputRows.map((row) => row.test_dataset));
  document.getElementById("matrixSummary").textContent = `${trainingNames.length} training rows by ${testNames.length} test columns`;
  if (!trainingNames.length || !testNames.length) {
    document.getElementById("matrixView").innerHTML = '<p class="empty">No matching matrix cells.</p>';
    return;
  }
  const html = [`<table class="matrix-table"><thead><tr><th>${columnHeader("Training Scenario", "training")}</th>`];
  testNames.forEach((testName) => html.push(`<th>${escapeHtml(testName)}</th>`));
  html.push("</tr></thead><tbody>");
  trainingNames.forEach((trainingName) => {
    html.push(`<tr><th>${escapeHtml(trainingName)}</th>`);
    testNames.forEach((testName) => {
      const cellRows = inputRows.filter(
        (row) => row.training_scenario === trainingName && row.test_dataset === testName
      );
      const best = bestRow(cellRows);
      if (!best) {
        html.push('<td class="muted-cell">-</td>');
      } else {
        const intensity = Math.round(Math.max(8, Math.min(92, Number(best[metric] || 0) * 92)));
        html.push(`<td style="--heat:${intensity}%">
          <strong>${number(best[metric])}</strong>
          <span>${escapeHtml(best.model)}</span>
        </td>`);
      }
    });
    html.push("</tr>");
  });
  html.push("</tbody></table>");
  document.getElementById("matrixView").innerHTML = html.join("");
}

function renderDetails(inputRows) {
  const metric = controls.metric.value;
  const sorted = sortRows(inputRows);
  document.getElementById("detailSummary").textContent = `${count(sorted.length)} exact model/test rows`;
  if (!sorted.length) {
    document.getElementById("detailView").innerHTML = '<p class="empty">No matching details.</p>';
    return;
  }
  const html = [`<table><thead><tr>
    <th>${columnHeader("Training", "training")}</th>
    <th>${columnHeader("Model", "model")}</th>
    <th>${columnHeader("Tested On", "tested")}</th>
    <th>${columnHeader("Rows", "rows")}</th>
    <th>${columnHeader("URL Overlap Removed", "overlap")}</th>
    <th>${columnHeader("Training Rows Included?", "included")}</th>
    <th class="num">${metricHeader("accuracy")}</th>
    <th class="num">${metricHeader("phishing_precision")}</th>
    <th class="num">${metricHeader("phishing_recall")}</th>
    <th class="num">${metricHeader("phishing_f1")}</th>
  </tr></thead><tbody>`];
  sorted.forEach((row) => {
    html.push(`<tr>
      <td>${escapeHtml(row.training_scenario)}</td>
      <td>${modelCell(row.model)}</td>
      <td>${escapeHtml(row.test_dataset)}</td>
      <td>${count(row.rows_tested)}</td>
      <td>${count(row.training_url_overlap_removed || 0)}</td>
      <td>${row.contains_training_rows ? "Yes" : "No"}</td>
      <td class="num">${number(row.accuracy)}</td>
      <td class="num">${number(row.phishing_precision)}</td>
      <td class="num">${number(row.phishing_recall)}</td>
      <td class="num">${row[metric] === row.phishing_f1 ? barCell(row.phishing_f1) : number(row.phishing_f1)}</td>
    </tr>`);
  });
  html.push("</tbody></table>");
  document.getElementById("detailView").innerHTML = html.join("");
}

function renderConfusion(inputRows) {
  const sorted = sortRows(inputRows);
  if (!sorted.length) {
    document.getElementById("confusionView").innerHTML = '<p class="empty">No matching confusion matrices.</p>';
    return;
  }
  const html = [];
  sorted.forEach((row, index) => {
    const key = `${row.training_scenario}::${row.model}::${row.test_dataset}`;
    const payload = matrices[key];
    if (!payload) return;
    const matrix = payload.matrix || [[0, 0], [0, 0]];
    html.push(`<details ${index === 0 ? "open" : ""}>
      <summary>${escapeHtml(row.training_scenario)} / ${escapeHtml(row.model)} / ${escapeHtml(row.test_dataset)}</summary>
      <table>
        <thead><tr><th>Actual / Predicted</th><th class="num">0 phishing</th><th class="num">1 legitimate</th></tr></thead>
        <tbody>
          <tr><td>0 phishing</td><td class="num">${count(matrix[0][0])}</td><td class="num">${count(matrix[0][1])}</td></tr>
          <tr><td>1 legitimate</td><td class="num">${count(matrix[1][0])}</td><td class="num">${count(matrix[1][1])}</td></tr>
        </tbody>
      </table>
    </details>`);
  });
  document.getElementById("confusionView").innerHTML = html.join("") || '<p class="empty">No matching confusion matrices.</p>';
}

function render() {
  const matches = filteredRows();
  renderModelSummary(matches);
  renderScenarioView(matches);
  renderRanking(matches);
  renderMatrix(matches);
  renderDetails(matches);
  renderConfusion(matches);
}

initialiseControls();
Object.values(controls).forEach((control) => {
  control.addEventListener("change", () => {
    activePreset = "custom";
    document.querySelectorAll("[data-preset]").forEach((button) => button.classList.remove("active"));
    render();
  });
});
document.getElementById("resetFilters").addEventListener("click", () => applyPreset("clean"));
document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => applyPreset(button.dataset.preset));
});
applyPreset("clean");
</script>"""


def format_scope_label(scope):
    labels = {
        "clean_holdout": "Clean Holdout",
        "complete_dataset": "Complete Dataset Diagnostic",
    }
    return labels.get(str(scope), str(scope))


def truthy(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def html_styles():
    return """<style>
:root {
  --accent: #2563eb;
  --accent-soft: #dbeafe;
  --background: #f6f7f9;
  --line: #d6dbe3;
  --muted: #5f6b7a;
  --surface: #ffffff;
  --surface-soft: #f9fafb;
  --text: #17202a;
  --success: #047857;
  --warning: #a16207;
}
* {
  box-sizing: border-box;
}
body {
  background: var(--background);
  color: var(--text);
  font-family: Arial, Helvetica, sans-serif;
  margin: 0;
}
main {
  margin: 0 auto;
  max-width: 1240px;
  padding: 28px 22px 56px;
}
header {
  border-bottom: 1px solid var(--line);
  margin-bottom: 18px;
  padding: 10px 0 22px;
}
h1, h2, h3 {
  letter-spacing: 0;
  margin: 0;
}
h1 {
  font-size: clamp(30px, 5vw, 52px);
  line-height: 1.05;
  max-width: 860px;
}
.lede {
  color: var(--muted);
  font-size: 17px;
  line-height: 1.55;
  margin: 12px 0 0;
  max-width: 820px;
}
.panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin: 16px 0;
  padding: 18px;
}
.panel > h2,
.section-heading h2 {
  font-size: 21px;
}
.section-heading {
  align-items: start;
  display: flex;
  gap: 16px;
  justify-content: space-between;
}
.section-note,
.panel p {
  color: var(--muted);
  line-height: 1.5;
}
table {
  border-collapse: collapse;
  margin-top: 12px;
  width: 100%;
}
th, td {
  border-bottom: 1px solid var(--line);
  font-size: 14px;
  padding: 10px 8px;
  text-align: left;
  vertical-align: top;
}
th {
  color: #334155;
  font-size: 12px;
  text-transform: uppercase;
}
th.num, td.num {
  text-align: right;
}
code {
  background: var(--accent-soft);
  border-radius: 4px;
  padding: 1px 4px;
}
.metric-grid,
.guide-grid,
.headline-grid,
.decision-grid,
.answer-grid,
.concept-grid,
.comparison-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}
.metric-card,
.guide-card,
.headline-card,
.decision-card {
  background: var(--surface-soft);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.metric-card span {
  color: var(--muted);
  display: block;
  font-size: 13px;
}
.metric-card strong {
  display: block;
  font-size: 20px;
  margin-top: 4px;
}
.guide-card h3,
.headline-card h3 {
  font-size: 15px;
  margin: 0 0 6px;
}
.headline-card span,
.headline-card small {
  color: var(--muted);
  display: block;
  font-size: 12px;
}
.headline-card strong {
  display: block;
  font-size: 24px;
  margin-top: 6px;
}
.headline-card .score {
  color: var(--success);
  font-weight: 700;
  margin: 8px 0;
}
.decision-card > span,
.decision-card > small {
  color: var(--muted);
  display: block;
  font-size: 12px;
}
.decision-card strong {
  display: block;
  font-size: 21px;
  margin-top: 6px;
}
.decision-card .score {
  color: var(--success);
  font-weight: 700;
  margin: 8px 0;
}
.decision-explainer {
  background: var(--accent-soft);
  border: 1px solid #bfdbfe;
  border-radius: 8px;
  margin-top: 14px;
  padding: 14px;
}
.decision-explainer h3 {
  font-size: 16px;
  margin-bottom: 8px;
}
.decision-explainer p {
  color: #1e3a8a;
  margin: 8px 0 0;
}
.controls {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  margin-top: 14px;
}
.controls label {
  color: #334155;
  display: grid;
  font-size: 12px;
  font-weight: 700;
  gap: 6px;
  text-transform: uppercase;
}
select,
button {
  border: 1px solid var(--line);
  border-radius: 6px;
  color: var(--text);
  font: inherit;
  min-height: 38px;
}
select {
  background: #fff;
  padding: 0 10px;
}
button {
  background: #fff;
  cursor: pointer;
  font-weight: 700;
  padding: 0 12px;
}
button:hover,
button.active {
  border-color: var(--accent);
  color: var(--accent);
}
.preset-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}
.view-summary {
  background: var(--accent-soft);
  border-radius: 6px;
  color: #1e3a8a;
  font-size: 13px;
  font-weight: 700;
  padding: 8px 10px;
  text-align: right;
}
.winner {
  background: #fef3c7;
  border: 1px solid #f59e0b;
  border-radius: 999px;
  color: #78350f;
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  margin-left: 6px;
  padding: 1px 7px;
}
.hint {
  background: var(--accent-soft);
  border-radius: 999px;
  color: var(--accent);
  display: inline-grid;
  font-size: 10px;
  height: 16px;
  place-items: center;
  width: 16px;
}
.tooltip,
[data-tooltip] {
  cursor: help;
  display: inline-flex;
  gap: 4px;
  position: relative;
}
.tooltip-text {
  background: #111827;
  border-radius: 6px;
  bottom: calc(100% + 8px);
  box-shadow: 0 10px 20px rgba(15, 23, 42, 0.18);
  color: #fff;
  display: none;
  font-size: 12px;
  font-weight: 500;
  left: 0;
  line-height: 1.4;
  max-width: min(280px, 80vw);
  opacity: 0;
  padding: 8px 10px;
  pointer-events: none;
  position: absolute;
  text-transform: none;
  transform: translateY(4px);
  transition: opacity 120ms ease, transform 120ms ease;
  visibility: hidden;
  width: max-content;
  z-index: 20;
}
.tooltip:hover .tooltip-text,
.tooltip:focus .tooltip-text,
.tooltip:focus-within .tooltip-text {
  opacity: 1;
  display: block;
  transform: translateY(0);
  visibility: visible;
}
.score-cell {
  align-items: center;
  display: grid;
  gap: 8px;
  grid-template-columns: 58px minmax(90px, 1fr);
  min-width: 160px;
}
.track {
  background: #e5e7eb;
  border-radius: 999px;
  min-width: 90px;
}
.bar {
  background: var(--accent);
  border-radius: 999px;
  height: 12px;
}
.matrix-table td {
  background:
    linear-gradient(90deg, rgba(37, 99, 235, 0.16) var(--heat), transparent var(--heat));
  min-width: 140px;
}
.matrix-table td strong,
.matrix-table td span {
  display: block;
}
.matrix-table td span {
  color: var(--muted);
  font-size: 12px;
  margin-top: 3px;
}
.model-block {
  border-top: 1px solid var(--line);
  margin-top: 18px;
  padding-top: 18px;
}
.model-block:first-of-type {
  border-top: 0;
  margin-top: 0;
  padding-top: 0;
}
.model-block h3,
.comparison-grid h3 {
  font-size: 17px;
}
.matrix-scroll {
  overflow-x: auto;
}
.model-matrix {
  min-width: 760px;
}
.model-matrix th,
.model-matrix td {
  min-width: 150px;
}
.matrix-score-cell strong,
.matrix-score-cell span,
.matrix-score-cell small {
  display: block;
}
.matrix-score-cell strong {
  color: var(--success);
  font-size: 15px;
}
.matrix-score-cell span {
  color: var(--text);
  font-size: 12px;
  margin-top: 3px;
}
.matrix-score-cell small {
  color: var(--muted);
  font-size: 11px;
  line-height: 1.35;
  margin-top: 6px;
}
.diagnostic-pill {
  background: #fef3c7;
  border-radius: 999px;
  color: #78350f;
  display: inline-block;
  font-size: 10px;
  font-weight: 700;
  margin-left: 4px;
  padding: 1px 6px;
  text-transform: none;
}
.plain-guide-table td:nth-child(2),
.plain-guide-table td:nth-child(3) {
  color: var(--muted);
  line-height: 1.45;
}
#modelSummaryView,
#scenarioView,
#rankingView,
#matrixView,
#detailView {
  overflow-x: auto;
}
.muted-cell {
  color: var(--muted);
}
.empty {
  background: var(--surface-soft);
  border: 1px dashed var(--line);
  border-radius: 8px;
  padding: 14px;
}
details {
  border-bottom: 1px solid var(--line);
  padding: 8px 0;
}
summary {
  cursor: pointer;
  font-weight: 700;
}
.report-section {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin: 14px 0;
  padding: 0;
}
.report-section > summary {
  font-size: 18px;
  list-style-position: inside;
  padding: 16px 18px;
}
.report-section[open] > summary {
  border-bottom: 1px solid var(--line);
}
.report-section-body {
  padding: 16px 18px 18px;
}
.answer-grid article {
  background: var(--surface-soft);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.answer-grid h3 {
  font-size: 15px;
  margin-bottom: 6px;
}
.look-for {
  background: var(--accent-soft);
  border-radius: 8px;
  margin-bottom: 14px;
  padding: 12px 14px;
}
.look-for h3 {
  font-size: 15px;
  margin: 0 0 8px;
}
.look-for ul {
  color: #1e3a8a;
  margin: 0;
  padding-left: 18px;
}
.look-for li {
  line-height: 1.45;
  margin: 4px 0;
}
@media (max-width: 760px) {
  main {
    padding: 18px 12px 40px;
  }
  .section-heading {
    display: block;
  }
  .view-summary {
    margin-top: 10px;
    text-align: left;
  }
  table {
    display: block;
    overflow-x: auto;
  }
}
</style>"""


def escape(value):
    return html.escape(str(value))
