"""Build the single combined dissertation-facing experiment report.

This script reads the per-experiment outputs, derives the summary sections used
in the dissertation, and writes the top-level combined HTML and JSON reports.
"""

from datetime import datetime
import html
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.scripts.helpers.project_paths import (
    ALL_EXPERIMENTS_COMBINED_REPORT_HTML as OUTPUT_HTML,
    ALL_EXPERIMENTS_COMBINED_REPORT_JSON as OUTPUT_JSON,
    DOCS_ROOT,
    EXPERIMENT_1A_DIR,
    EXPERIMENT_1B_DIR,
    EXPERIMENT_1C_DIR,
    EXPERIMENT_2A_DIR,
    EXPERIMENT_2B_DIR,
    EXPERIMENT_3_DIR,
    EXPERIMENT_4_DIR,
)

MAIN_REPORTS = EXPERIMENT_1A_DIR
EXTERNAL_REPORTS = EXPERIMENT_1B_DIR
CROSS_REPORTS = EXPERIMENT_1C_DIR
URL_PROBE_REPORTS = EXPERIMENT_3_DIR
OUTPUT_MARKDOWN = DOCS_ROOT / "model_experiment_conclusions.md"

METRIC_COLUMNS = [
    "accuracy",
    "phishing_precision",
    "phishing_recall",
    "phishing_f1",
]

EXPERIMENT_DEFINITIONS = [
    {
        "id": "1a",
        "definition": "Trained on the main PhiUSIIL dataset and tested on the main PhiUSIIL held-out 20 percent test split.",
    },
    {
        "id": "1b",
        "definition": "Trained on the main PhiUSIIL dataset and tested against external datasets.",
    },
    {
        "id": "1c",
        "definition": "Trained on the remaining datasets and tested against other datasets, not the dataset used for training.",
    },
    {
        "id": "2a",
        "definition": "Trained on each single dataset and tested against the combined held-out test dataset.",
    },
    {
        "id": "2b",
        "definition": "Trained on the combined dataset and tested against the combined held-out test dataset.",
    },
    {
        "id": "3",
        "definition": "Saved models tested on 20 normal websites and 20 held-out phishing URLs to check practical false positives and missed phishing.",
    },
    {
        "id": "4",
        "definition": "Google trailing-slash brittleness check, comparing predictions for https://www.google.com and https://www.google.com/.",
    },
]

SINGLE_SOURCE_TRAINING = ["phiusiil_main", "legitphish", "phishstorm"]
EXTERNAL_TRAINING = ["legitphish", "phishstorm"]
SOURCE_TEST_SETS = {
    "phiusiil_main_test": "phiusiil_main",
    "legitphish_test": "legitphish",
    "phishstorm_test": "phishstorm",
}

DISPLAY_NAMES = {
    "phiusiil_main": "PhiUSIIL main",
    "phiusiil_main_test": "PhiUSIIL held-out test",
    "legitphish": "LegitPhish",
    "legitphish_test": "LegitPhish held-out test",
    "phishstorm": "PhishStorm",
    "phishstorm_test": "PhishStorm held-out test",
    "combined_dataset": "Combined dataset",
    "combined_test": "Combined held-out test",
}

METRIC_TOOLTIPS = {
    "Experiment": "The experiment section this row belongs to.",
    "Trained On": "The dataset used to teach the model before testing.",
    "Tested On": "The held-out dataset used to judge predictions.",
    "Model": "The machine-learning algorithm being compared.",
    "Rows": "Number of URL rows scored in this test.",
    "Accuracy": "Overall correctness across phishing and legitimate URLs.",
    "Precision": "When the model predicts phishing, how often that prediction is correct.",
    "Recall": "How many real phishing URLs the model catches.",
    "F1": "A balanced score combining phishing precision and phishing recall.",
    "Mean F1": "Average phishing F1 across the rows in this experiment section.",
    "Worst F1": "Lowest phishing F1 for this model in this experiment section.",
    "Best F1": "Best exact phishing F1 row for this model inside the experiment section.",
    "Evaluations": "Number of trained-model/test-dataset rows used in the average.",
    "Correct": "Number of URLs classified correctly in the sanity probe.",
    "False Positives": "Legitimate URLs incorrectly predicted as phishing.",
    "Missed Phishing": "Phishing URLs incorrectly predicted as legitimate.",
    "Google No Slash": "Prediction for https://www.google.com.",
    "Google Slash": "Prediction for https://www.google.com/.",
    "OpenAI Research": "Prediction for https://www.openai.com/research.",
    "Why": "Short explanation of why the model is treated as strongest in that section.",
}


def main():
    """Build one simple dissertation-facing HTML report."""
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    summary = build_summary_payload(generated_at)
    write_derived_experiment_outputs(summary)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    OUTPUT_HTML.write_text(build_html_report(summary), encoding="utf-8")
    OUTPUT_MARKDOWN.write_text(build_conclusions_markdown(summary), encoding="utf-8")
    print(f"Dissertation report: {OUTPUT_HTML.relative_to(PROJECT_ROOT)}")
    print(f"Summary JSON: {OUTPUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Conclusion doc: {OUTPUT_MARKDOWN.relative_to(PROJECT_ROOT)}")


def build_summary_payload(generated_at):
    main_metrics = read_csv(MAIN_REPORTS / "test_summary.csv")
    external_metrics = read_csv(EXTERNAL_REPORTS / "test_summary.csv")
    cross_metrics = clean_heldout_rows(
        read_csv(CROSS_REPORTS / "test_summary.csv")
    )
    sanity_summary = build_sanity_probe_summary(
        read_csv(URL_PROBE_REPORTS / "40_url_balanced_model_probe_summary.csv")
    )
    google_slash_summary = build_google_slash_summary(sanity_summary["rows"])

    experiments = [
        build_experiment(
            "1a",
            "Each model trained and tested on the main PhiUSIIL dataset",
            "80/20 split on PhiUSIIL only. This is the same-source baseline.",
            main_baseline_rows(main_metrics),
        ),
        build_experiment(
            "1b",
            "Main-trained models tested against external datasets",
            "Models are trained on PhiUSIIL and tested on external held-out datasets.",
            external_testing_rows(external_metrics),
        ),
        build_experiment(
            "1c",
            "External-trained models tested against other single-source datasets",
            "Models are trained on LegitPhish or PhishStorm, then tested on different held-out datasets only.",
            cross_source_rows(cross_metrics),
        ),
        build_experiment(
            "2a",
            "Single-source models tested against the combined held-out dataset",
            "Each model is trained on one source dataset and tested on the mixed combined test set.",
            single_source_combined_rows(cross_metrics),
        ),
        build_experiment(
            "2b",
            "Combined-trained models tested against the combined held-out dataset",
            "Each model is trained on the combined training dataset and tested on the combined held-out test set.",
            combined_training_rows(cross_metrics),
        ),
    ]

    return {
        "generated_at": generated_at,
        "heldout_policy": (
            "All sections use held-out test rows only. Diagnostic complete-dataset "
            "rows that include training data are excluded from this report."
        ),
        "experiment_definitions": EXPERIMENT_DEFINITIONS,
        "experiments": experiments,
        "decision_summary": decision_summary(experiments),
        "sanity_probe": sanity_summary,
        "google_slash": google_slash_summary,
        "source_files": source_files(),
    }


def clean_heldout_rows(frame):
    if frame.empty:
        return frame
    clean = frame[frame["evaluation_scope"] == "clean_holdout"].copy()
    if "contains_training_rows" in clean.columns:
        clean = clean[~clean["contains_training_rows"].map(truthy)].copy()
    return clean


def main_baseline_rows(metrics):
    if metrics.empty:
        return []
    matrices = read_json(MAIN_REPORTS / "confusion_matrices.json") or {}
    rows = []
    for row in metrics.to_dict(orient="records"):
        model = str(row["model"])
        rows.append(
            metric_row(
                experiment_id="1a",
                trained_on="phiusiil_main",
                tested_on="phiusiil_main_test",
                model=model,
                rows_tested=rows_tested_from_matrix(matrices.get(model, {})),
                row=row,
            )
        )
    return rows


def external_testing_rows(metrics):
    if metrics.empty:
        return []
    rows = []
    for row in metrics.to_dict(orient="records"):
        rows.append(
            metric_row(
                experiment_id="1b",
                trained_on=row.get("trained_on", "phiusiil_main"),
                tested_on=str(row["dataset"]),
                model=str(row["model"]),
                rows_tested=int(row.get("rows_tested", 0)),
                row=row,
            )
        )
    return rows


def cross_source_rows(clean_cross_metrics):
    if clean_cross_metrics.empty:
        return []
    frame = clean_cross_metrics[
        clean_cross_metrics["training_scenario"].isin(EXTERNAL_TRAINING)
        & clean_cross_metrics["test_dataset"].isin(SOURCE_TEST_SETS)
    ].copy()
    frame = frame[
        frame.apply(
            lambda row: SOURCE_TEST_SETS[row["test_dataset"]]
            != row["training_scenario"],
            axis=1,
        )
    ]
    return cross_metric_rows(frame, "1c")


def single_source_combined_rows(clean_cross_metrics):
    if clean_cross_metrics.empty:
        return []
    frame = clean_cross_metrics[
        clean_cross_metrics["training_scenario"].isin(SINGLE_SOURCE_TRAINING)
        & (clean_cross_metrics["test_dataset"] == "combined_test")
    ]
    return cross_metric_rows(frame, "2a")


def combined_training_rows(clean_cross_metrics):
    if clean_cross_metrics.empty:
        return []
    frame = clean_cross_metrics[
        (clean_cross_metrics["training_scenario"] == "combined_dataset")
        & (clean_cross_metrics["test_dataset"] == "combined_test")
    ]
    return cross_metric_rows(frame, "2b")


def cross_metric_rows(frame, experiment_id):
    rows = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            metric_row(
                experiment_id=experiment_id,
                trained_on=str(row["training_scenario"]),
                tested_on=str(row["test_dataset"]),
                model=str(row["model"]),
                rows_tested=int(row.get("rows_tested", 0)),
                row=row,
            )
        )
    return rows


def metric_row(experiment_id, trained_on, tested_on, model, rows_tested, row):
    return {
        "experiment_id": experiment_id,
        "trained_on": str(trained_on),
        "tested_on": str(tested_on),
        "trained_on_label": display_name(trained_on),
        "tested_on_label": display_name(tested_on),
        "model": str(model),
        "rows_tested": int(rows_tested),
        "accuracy": float(row["accuracy"]),
        "phishing_precision": float(row["phishing_precision"]),
        "phishing_recall": float(row["phishing_recall"]),
        "phishing_f1": float(row["phishing_f1"]),
        "model_file": str(row.get("model_file") or ""),
    }


def build_experiment(experiment_id, title, purpose, rows):
    rows = sorted(rows, key=score_key, reverse=True)
    ranking = model_ranking(rows)
    winner = rows[0] if rows else None
    experiment = {
        "id": experiment_id,
        "title": title,
        "purpose": purpose,
        "rows": rows,
        "model_ranking": ranking,
        "winner": winner,
        "row_count": len(rows),
        "model_count": len({row["model"] for row in rows}),
    }
    experiment["plain_english_summary"] = experiment_plain_english_summary(experiment)
    return experiment


def build_sanity_probe_summary(frame):
    """Build Experiment 3 rows from the 40-URL real-world sanity probe."""
    if frame.empty:
        return {
            "id": "3",
            "title": "40-URL real-world sanity probe",
            "purpose": (
                "Saved models are tested on 20 normal websites and 20 held-out "
                "phishing URLs to inspect false positives and missed phishing."
            ),
            "rows": [],
            "winner": None,
        }

    rows = []
    for row in frame.to_dict(orient="records"):
        phishing_correct = int(row.get("phishing_correct", 0))
        phishing_false_negatives = int(row.get("phishing_false_negatives", 0))
        legit_false_positives = int(row.get("legit_false_positives", 0))
        legit_correct = int(row.get("legit_correct", 0))
        precision = safe_divide(
            phishing_correct, phishing_correct + legit_false_positives
        )
        recall = safe_divide(
            phishing_correct, phishing_correct + phishing_false_negatives
        )
        f1 = safe_f1(precision, recall)
        rows.append(
            {
                "rank": int(row.get("rank", len(rows) + 1)),
                "training_scenario": str(row.get("training_scenario", "")),
                "training_scenario_label": display_name(
                    row.get("training_scenario", "")
                ),
                "model": str(row.get("model", "")),
                "model_file": str(row.get("model_file") or ""),
                "total_correct": int(row.get("total_correct", 0)),
                "accuracy": float(row.get("accuracy", 0)),
                "phishing_precision": precision,
                "phishing_recall": recall,
                "phishing_f1": f1,
                "legit_correct": legit_correct,
                "legit_false_positives": legit_false_positives,
                "phishing_correct": phishing_correct,
                "phishing_false_negatives": phishing_false_negatives,
                "google_no_slash_prediction": str(
                    row.get("google_no_slash_prediction", "")
                ),
                "google_slash_prediction": str(row.get("google_slash_prediction", "")),
                "openai_research_prediction": str(
                    row.get("openai_research_prediction", "")
                ),
                "false_positive_urls": clean_optional_text(
                    row.get("false_positive_urls")
                ),
                "false_negative_urls": clean_optional_text(
                    row.get("false_negative_urls")
                ),
            }
        )

    rows = sorted(
        rows,
        key=lambda item: (
            item["total_correct"],
            -item["legit_false_positives"],
            -item["phishing_false_negatives"],
            item["phishing_f1"],
        ),
        reverse=True,
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return {
        "id": "3",
        "title": "40-URL real-world sanity probe",
        "purpose": (
            "Saved models are tested on 20 normal websites and 20 held-out "
            "phishing URLs to inspect false positives and missed phishing."
        ),
        "rows": rows,
        "winner": rows[0] if rows else None,
    }


def build_google_slash_summary(sanity_rows):
    """Build Experiment 4 rows from the Google URL predictions."""
    rows = []
    for row in sanity_rows:
        no_slash = row.get("google_no_slash_prediction", "")
        slash = row.get("google_slash_prediction", "")
        openai = row.get("openai_research_prediction", "")
        google_false_positives = int(no_slash == "phishing") + int(slash == "phishing")
        rows.append(
            {
                "training_scenario": row["training_scenario"],
                "training_scenario_label": row["training_scenario_label"],
                "model": row["model"],
                "model_file": row.get("model_file", ""),
                "google_no_slash_prediction": no_slash,
                "google_slash_prediction": slash,
                "openai_research_prediction": openai,
                "prediction_changed": no_slash != slash,
                "google_false_positives": google_false_positives,
                "all_three_legitimate_correct": (
                    no_slash == "legitimate"
                    and slash == "legitimate"
                    and openai == "legitimate"
                ),
            }
        )

    rows = sorted(
        rows,
        key=lambda item: (
            item["all_three_legitimate_correct"],
            not item["prediction_changed"],
            -item["google_false_positives"],
        ),
        reverse=True,
    )
    highlighted = next(
        (
            row
            for row in rows
            if row["training_scenario"] == "combined_dataset"
            and row["model"] == "XGBoost"
        ),
        None,
    )
    return {
        "id": "4",
        "title": "Google trailing-slash brittleness check",
        "purpose": (
            "Checks whether a small harmless URL text change can flip the model's "
            "prediction for an obvious legitimate website."
        ),
        "rows": rows,
        "highlighted_row": highlighted,
        "feature_change": [
            {"feature": "url_length", "no_slash": 22, "with_slash": 23},
            {"feature": "path_length", "no_slash": 0, "with_slash": 1},
            {"feature": "special_char_count", "no_slash": 3, "with_slash": 4},
        ],
    }


def model_ranking(rows):
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    ranking = (
        frame.groupby("model")
        .agg(
            evaluations=("model", "count"),
            mean_accuracy=("accuracy", "mean"),
            mean_precision=("phishing_precision", "mean"),
            mean_recall=("phishing_recall", "mean"),
            mean_f1=("phishing_f1", "mean"),
            worst_f1=("phishing_f1", "min"),
        )
        .reset_index()
        .sort_values(
            ["mean_f1", "worst_f1", "mean_recall", "mean_precision", "mean_accuracy"],
            ascending=False,
        )
    )
    best_rows = {}
    for model, group in frame.groupby("model"):
        best_rows[model] = group.sort_values(
            ["phishing_f1", "phishing_recall", "phishing_precision", "accuracy"],
            ascending=False,
        ).iloc[0].to_dict()
    records = []
    for row in ranking.to_dict(orient="records"):
        record = row_record(row)
        record["best_row"] = row_record(best_rows[row["model"]])
        records.append(record)
    return records


def decision_summary(experiments):
    winners = []
    for experiment in experiments:
        winner = experiment.get("winner")
        if winner:
            ranking_winner = experiment["model_ranking"][0]
            ranking_best_row = ranking_winner["best_row"]
            winners.append(
                {
                    "experiment": experiment["id"],
                    "title": experiment["title"],
                    "winner_model": ranking_winner["model"],
                    "best_trained_on": ranking_best_row["trained_on"],
                    "best_tested_on": ranking_best_row["tested_on"],
                    "mean_f1": ranking_winner["mean_f1"],
                    "worst_f1": ranking_winner["worst_f1"],
                    "best_model_exact_f1": ranking_best_row["phishing_f1"],
                    "best_model_file": ranking_best_row.get("model_file", ""),
                    "exact_winner_model": winner["model"],
                    "exact_winner_trained_on": winner["trained_on"],
                    "exact_winner_tested_on": winner["tested_on"],
                    "exact_winner_f1": winner["phishing_f1"],
                    "exact_winner_model_file": winner.get("model_file", ""),
                    "why": conclusion_reason(experiment, ranking_winner, winner),
                }
            )

    backend = next(
        (
            item
            for item in winners
            if item["experiment"] == "2b"
        ),
        winners[-1] if winners else None,
    )
    return {
        "backend_recommendation": backend,
        "experiment_winners": winners,
        "overall_model_table": overall_model_table(experiments),
        "method": (
            "Models are ranked primarily by phishing F1. Recall, precision, and "
            "accuracy are used as supporting evidence. For experiments with more "
            "than one test row, the report uses mean F1 and also shows worst F1 so "
            "a model is not judged from one lucky result only."
        ),
    }


def conclusion_reason(experiment, ranking_winner, exact_winner):
    suffix = ""
    if ranking_winner["model"] != exact_winner["model"]:
        suffix = (
            f" Best single row: {exact_winner['model']} trained on "
            f"{display_name(exact_winner['trained_on'])}, tested on "
            f"{display_name(exact_winner['tested_on'])}."
        )
    if experiment["id"] == "2b":
        return (
            "This is the most backend-like scenario: combined training data and a "
            "separate combined held-out test set."
        )
    if experiment["id"] == "1a":
        return "Best same-source PhiUSIIL baseline on the 80/20 test split."
    if experiment["id"] == "1b":
        return (
            "Best average external-test performance for models trained only on PhiUSIIL."
            + suffix
        )
    if experiment["id"] == "1c":
        return (
            "Best average cross-source result when the test dataset is not the training dataset."
            + suffix
        )
    if experiment["id"] == "2a":
        return (
            "Best average single-source training result on the mixed combined held-out test set."
            + suffix
        )
    return (
        f"Highest mean F1 in this section ({ranking_winner['mean_f1']:.4f}); "
        f"best exact row reached {exact_winner['phishing_f1']:.4f}."
    )


def overall_model_table(experiments):
    rows = []
    for experiment in experiments:
        for row in experiment["model_ranking"]:
            rows.append(
                {
                    "experiment": experiment["id"],
                    "model": row["model"],
                    "mean_f1": row["mean_f1"],
                    "worst_f1": row["worst_f1"],
                    "evaluations": row["evaluations"],
                }
            )
    return sorted(rows, key=lambda row: (row["mean_f1"], row["worst_f1"]), reverse=True)


def experiment_plain_english_summary(experiment):
    if not experiment["model_ranking"]:
        return []
    ranking = experiment["model_ranking"]
    winner = ranking[0]
    winner_best_row = winner["best_row"]
    exact_winner = experiment["winner"]
    score_label = "mean F1" if winner["evaluations"] > 1 else "F1"
    summary = [
        (
            f"Winner: {winner['model']} is ranked first because it has the highest "
            f"{score_label} in this experiment: {fmt(winner['mean_f1'])}. "
            "F1 is the main score here because it balances catching phishing URLs "
            "with avoiding false phishing alarms."
        ),
        (
            f"For this winner, precision is {fmt(winner['mean_precision'])}, "
            f"recall is {fmt(winner['mean_recall'])}, and accuracy is "
            f"{fmt(winner['mean_accuracy'])}. Precision means how trustworthy its "
            "phishing warnings are; recall means how many real phishing URLs it catches."
        ),
    ]
    if len(ranking) > 1:
        runner_up = ranking[1]
        gap = winner["mean_f1"] - runner_up["mean_f1"]
        summary.append(
            f"Runner-up: {runner_up['model']} is second with {score_label} "
            f"{fmt(runner_up['mean_f1'])}. The gap from the winner is "
            f"{fmt(gap)}, so {gap_phrase(gap)}"
        )
    if winner["evaluations"] > 1:
        f1_range = winner["mean_f1"] - winner["worst_f1"]
        summary.append(
            f"Robustness check: the winner's worst F1 in this section is "
            f"{fmt(winner['worst_f1'])}. The drop from mean F1 is "
            f"{fmt(f1_range)}, which shows how much its performance changes across "
            "the tested datasets in this experiment."
        )
    if exact_winner and exact_winner["model"] != winner["model"]:
        summary.append(
            f"Best single held-out row: {exact_winner['model']} has the highest "
            f"individual F1 result, {fmt(exact_winner['phishing_f1'])}, when trained "
            f"on {display_name(exact_winner['trained_on'])} and tested on "
            f"{display_name(exact_winner['tested_on'])}. The section winner is still "
            f"{winner['model']} because the section ranking is based on overall "
            "model performance across the experiment rows."
        )
    else:
        summary.append(
            f"The best exact row for the winner is trained on "
            f"{display_name(winner_best_row['trained_on'])} and tested on "
            f"{display_name(winner_best_row['tested_on'])}, with F1 "
            f"{fmt(winner_best_row['phishing_f1'])}."
        )
    metric_warning = metric_balance_sentence(winner)
    if metric_warning:
        summary.append(metric_warning)
    summary.append(
        f"Conclusion: for Experiment {experiment['id']}, {winner['model']} gives "
        "the strongest current held-out evidence under this training/testing setup."
    )
    return summary


def gap_phrase(gap):
    if gap < 0.001:
        return "the result is almost tied and should be treated as very close."
    if gap < 0.01:
        return "the runner-up is close, so both models are competitive here."
    if gap < 0.05:
        return "the winner has a noticeable advantage in this section."
    return "the winner has a clear advantage in this section."


def metric_balance_sentence(row):
    precision = float(row["mean_precision"])
    recall = float(row["mean_recall"])
    if recall - precision >= 0.10:
        return (
            f"Measurement note: recall is higher than precision by "
            f"{fmt(recall - precision)}, so this model catches more phishing URLs "
            "but may create more false phishing alarms."
        )
    if precision - recall >= 0.10:
        return (
            f"Measurement note: precision is higher than recall by "
            f"{fmt(precision - recall)}, so its phishing warnings are very reliable "
            "but it may miss more phishing URLs than a higher-recall model."
        )
    return (
        "Measurement note: precision and recall are fairly balanced, so the F1 score "
        "is a reasonable single summary for this section."
    )


def build_html_report(summary):
    experiments = "\n".join(experiment_section(item) for item in summary["experiments"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phishing URL Detection: Simple Experiment Report</title>
{styles()}
</head>
<body>
<main>
<header>
<p class="eyebrow">Phishing URL Detection</p>
<h1>Simple Dissertation Experiment Report</h1>
<p>Generated {escape(summary["generated_at"])} from the current experiment outputs.</p>
<p class="policy">{escape(summary["heldout_policy"])}</p>
</header>
{experiment_definitions_section(summary["experiment_definitions"])}
{guide_section()}
{summary_section(summary)}
{experiments}
{sanity_probe_section(summary.get("sanity_probe", {}))}
{google_slash_section(summary.get("google_slash", {}))}
{source_section(summary["source_files"])}
</main>
</body>
</html>"""


def build_conclusions_markdown(summary):
    """Build a concise dissertation-facing conclusions note from current metrics."""
    decision = summary["decision_summary"]
    backend = decision.get("backend_recommendation")
    lines = [
        "# Model Experiment Conclusions",
        "",
        "This document is generated from the current experiment outputs. Rebuild it with:",
        "",
        "```bash",
        "python machine_learning/scripts/build_all_experiments_combined_report.py",
        "```",
        "",
        f"Generated from report data: {summary['generated_at']}",
        "",
        f"Held-out policy: {summary['heldout_policy']}",
        "",
        "## Short Answer",
        "",
    ]
    if backend:
        lines.extend(
            [
                (
                    f"The formal benchmark winner is **{backend['winner_model']}** "
                    f"trained on **{display_name(backend['best_trained_on'])}**."
                ),
                "",
                (
                    f"It is selected from Experiment {backend['experiment']} because "
                    f"that is the closest experiment to the final application setup: "
                    "the model is trained on the combined training dataset and tested "
                    "on a separate combined held-out test set."
                ),
                "",
                f"- Phishing F1: `{fmt(backend['best_model_exact_f1'])}`",
                f"- Mean F1: `{fmt(backend['mean_f1'])}`",
                f"- Worst F1: `{fmt(backend['worst_f1'])}`",
                f"- Model file: `{backend.get('best_model_file') or 'not recorded in current metrics'}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No backend candidate is available because no experiment winner rows were found.",
                "",
            ]
        )
    lines.extend(
        [
            "## Experiment Evidence Review",
            "",
            "| Experiment | Question Answered | Winner | Runner-up | F1 Evidence | Takeaway |",
            "|---|---|---|---|---|---|",
        ]
    )
    for experiment in summary["experiments"]:
        lines.append(markdown_experiment_row(experiment))
    sanity = summary.get("sanity_probe", {})
    google = summary.get("google_slash", {})
    if sanity.get("winner"):
        winner = sanity["winner"]
        lines.extend(
            [
                "",
                "## Experiment 3: 40-URL Real-World Sanity Probe",
                "",
                (
                    f"The practical sanity winner is **{winner['model']}** trained on "
                    f"**{display_name(winner['training_scenario'])}**."
                ),
                "",
                f"- Correct: `{winner['total_correct']}/40`",
                f"- Accuracy: `{fmt(winner['accuracy'])}`",
                f"- Phishing F1: `{fmt(winner['phishing_f1'])}`",
                f"- Legitimate false positives: `{winner['legit_false_positives']}`",
                f"- Missed phishing URLs: `{winner['phishing_false_negatives']}`",
                "",
                (
                    "This experiment is important because it checks whether the model "
                    "behaves sensibly on ordinary user-entered URLs, not only on cleaned "
                    "benchmark rows."
                ),
                "",
            ]
        )
    if google.get("highlighted_row"):
        highlighted = google["highlighted_row"]
        lines.extend(
            [
                "## Experiment 4: Google Slash Brittleness Check",
                "",
                (
                    f"The highlighted row is **{highlighted['model']}** trained on "
                    f"**{display_name(highlighted['training_scenario'])}**."
                ),
                "",
                (
                    f"- `https://www.google.com` prediction: "
                    f"`{highlighted['google_no_slash_prediction']}`"
                ),
                (
                    f"- `https://www.google.com/` prediction: "
                    f"`{highlighted['google_slash_prediction']}`"
                ),
                (
                    f"- `https://www.openai.com/research` prediction: "
                    f"`{highlighted['openai_research_prediction']}`"
                ),
                "",
                (
                    "This check shows whether a harmless lexical change, such as a "
                    "trailing slash, can flip a model prediction for an obvious "
                    "legitimate website."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Final ML Conclusion",
            "",
            final_ml_conclusion(summary),
            "",
            "## Dissertation Use",
            "",
            "- Use Experiment 1a as the same-dataset baseline, not as proof of real-world reliability.",
            "- Use Experiments 1b and 1c to discuss dataset dependency and cross-dataset generalisation.",
            "- Use Experiment 2a to show that a strong single-source result is still weaker than combined training for the mixed test set.",
            "- Use Experiment 2b as the main backend-selection evidence because it matches the intended deployment setting most closely.",
            "- Use Experiment 3 to discuss practical false positives and missed phishing on normal websites.",
            "- Use Experiment 4 to discuss lexical brittleness using the Google trailing-slash example.",
            "- Report F1 first, then use recall, precision, accuracy, and worst F1 to explain the result.",
            "",
            "## Source Evidence",
            "",
            "- `machine_learning/all_experiments_combined_report.html`",
            "- `machine_learning/all_experiments_combined_report.json`",
        ]
    )
    for path in summary.get("source_files", []):
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def markdown_experiment_row(experiment):
    ranking = experiment["model_ranking"]
    if not ranking:
        return (
            f"| {escape_markdown(experiment['id'])} | "
            f"{escape_markdown(experiment['title'])} | No rows | No rows | n/a | "
            "No evidence available. |"
        )
    winner = ranking[0]
    runner_up = ranking[1] if len(ranking) > 1 else None
    winner_score_name = "Mean F1" if winner["evaluations"] > 1 else "F1"
    runner_text = (
        f"{runner_up['model']} ({fmt(runner_up['mean_f1'])})"
        if runner_up
        else "n/a"
    )
    evidence = (
        f"{winner_score_name} {fmt(winner['mean_f1'])}; "
        f"worst F1 {fmt(winner['worst_f1'])}; "
        f"best exact F1 {fmt(winner['best_row']['phishing_f1'])}"
    )
    return (
        f"| {escape_markdown('Experiment ' + experiment['id'])} | "
        f"{escape_markdown(experiment['purpose'])} | "
        f"{escape_markdown(winner['model'] + ' (' + fmt(winner['mean_f1']) + ')')} | "
        f"{escape_markdown(runner_text)} | "
        f"{escape_markdown(evidence)} | "
        f"{escape_markdown(experiment_takeaway(experiment))} |"
    )


def experiment_takeaway(experiment):
    if experiment["id"] == "1a":
        return "Strong same-source baseline, but not enough by itself to prove real-world reliability."
    if experiment["id"] == "1b":
        return "External testing shows that main-dataset performance does not transfer equally to every dataset."
    if experiment["id"] == "1c":
        return "Cross-source transfer is difficult, so source distribution has a strong effect."
    if experiment["id"] == "2a":
        return "Single-source training is useful, but weaker than combined training for the mixed holdout."
    if experiment["id"] == "2b":
        return "Most backend-like evidence because combined training is tested on a separate mixed holdout."
    return "Use this section as supporting model-comparison evidence."


def final_ml_conclusion(summary):
    backend = summary["decision_summary"].get("backend_recommendation")
    sanity_winner = (summary.get("sanity_probe") or {}).get("winner")
    if not backend:
        return "The current experiment outputs do not contain enough rows to select a backend model."
    sanity_text = ""
    if sanity_winner:
        sanity_text = (
            f" However, Experiment 3 gives the practical sanity-check advantage to "
            f"{sanity_winner['model']} trained on "
            f"{display_name(sanity_winner['training_scenario'])}, with "
            f"{sanity_winner['total_correct']}/40 correct and "
            f"{sanity_winner['legit_false_positives']} false positives."
        )
    return (
        f"The formal benchmark evidence supports {backend['winner_model']} trained "
        f"on {display_name(backend['best_trained_on'])}, because it wins in the most "
        "deployment-like held-out dataset setting: combined training tested on a separate "
        f"combined test set.{sanity_text} The careful dissertation conclusion is that "
        "model choice matters, but evaluation scenario matters too: dataset-source tests "
        "answer formal generalisation questions, while sanity probes reveal practical "
        "false-positive and brittleness risks."
    )


def experiment_definitions_section(definitions):
    cards = "".join(
        '<article class="definition-card">'
        f"<strong>Experiment {escape(item['id'])}</strong>"
        f"<p>{escape(item['definition'])}</p>"
        "</article>"
        for item in definitions
    )
    return f"""
<section class="top-definitions" aria-label="Experiment definitions">
<h2>Experiment Definitions</h2>
<div class="definition-grid">{cards}</div>
</section>
"""


def guide_section():
    rows = [
        ("Accuracy", "Overall correctness."),
        ("Precision", "How trustworthy a phishing prediction is."),
        ("Recall", "How many real phishing URLs are caught."),
        ("F1", "Balanced phishing score used for the main ranking."),
        ("Mean F1", "Average F1 where a section has multiple test rows."),
        ("Worst F1", "Lowest F1, used to notice weak generalisation."),
    ]
    items = "".join(
        f"<tr><td>{escape(name)}</td><td>{escape(text)}</td></tr>"
        for name, text in rows
    )
    return f"""
<details class="panel" open>
<summary>How to Read This Report</summary>
<p>Every experiment section is collapsible. The charts rank models by phishing F1, because phishing detection cares about both catching phishing URLs and avoiding unnecessary false alarms.</p>
<table class="compact-table">
<thead><tr><th>Measurement</th><th>Meaning</th></tr></thead>
<tbody>{items}</tbody>
</table>
</details>
"""


def summary_section(summary):
    decision = summary["decision_summary"]
    backend = decision.get("backend_recommendation")
    sanity_winner = (summary.get("sanity_probe") or {}).get("winner")
    backend_card = (
        text_card(
            "Recommended backend model",
            backend["winner_model"],
            (
                f"Experiment {backend['experiment']}, trained on "
                f"{display_name(backend['best_trained_on'])}, tested on "
                f"{display_name(backend['best_tested_on'])}."
            ),
        )
        if backend
        else text_card("Recommended backend model", "Not available", "No experiment rows found.")
    )
    sanity_card = (
        text_card(
            "Practical sanity winner",
            sanity_winner["model"],
            (
                f"Experiment 3, trained on "
                f"{display_name(sanity_winner['training_scenario'])}: "
                f"{sanity_winner['total_correct']}/40 correct, "
                f"{sanity_winner['legit_false_positives']} false positives."
            ),
        )
        if sanity_winner
        else text_card("Practical sanity winner", "Not available", "No sanity probe rows found.")
    )
    return f"""
<details class="panel" open>
<summary>Final Summary and Recommendation</summary>
<div class="cards">
{backend_card}
{sanity_card}
{metric_card("Decision score", backend["best_model_exact_f1"] if backend else None, "F1")}
{text_card("Decision rule", "F1 first", "Recall, precision, and accuracy support the conclusion.")}
</div>
<p>{escape(decision["method"])}</p>
<h3>Which Model Is Better Where?</h3>
{winner_table(decision["experiment_winners"])}
<h3>Overall Model Ranking Rows</h3>
{overall_table(decision["overall_model_table"][:10])}
</details>
"""


def experiment_section(experiment):
    if not experiment["rows"]:
        body = "<p>No rows are available for this experiment.</p>"
    else:
        winner = experiment["winner"]
        ranking_winner = experiment["model_ranking"][0]
        body = f"""
<p>{escape(experiment["purpose"])}</p>
<div class="cards">
{text_card("Best model average", ranking_winner["model"], f"mean F1 {fmt(ranking_winner['mean_f1'])}; worst F1 {fmt(ranking_winner['worst_f1'])}")}
{text_card("Best exact result", winner["model"], f"trained on {display_name(winner['trained_on'])}; tested on {display_name(winner['tested_on'])}")}
{metric_card("Best exact F1", winner["phishing_f1"], "F1")}
{metric_card("Rows in section", experiment["row_count"], "rows", decimals=0)}
</div>
<h3>Model Ranking</h3>
{ranking_chart(experiment["model_ranking"])}
{plain_english_summary(experiment["plain_english_summary"])}
{model_ranking_table(experiment["model_ranking"])}
<h3>Exact Held-Out Results</h3>
{exact_results_table(experiment["rows"])}
"""
    return f"""
<details class="panel">
<summary>Experiment {escape(experiment["id"])}: {escape(experiment["title"])}</summary>
{body}
</details>
"""


def sanity_probe_section(payload):
    rows = payload.get("rows", []) if payload else []
    winner = payload.get("winner") if payload else None
    if not rows:
        body = "<p>No 40-URL sanity probe rows are available.</p>"
    else:
        body = f"""
<p>{escape(payload["purpose"])}</p>
<div class="cards">
{text_card("Best practical model", winner["model"], f"trained on {display_name(winner['training_scenario'])}")}
{metric_card("Correct answers", winner["total_correct"], "of 40", decimals=0)}
{metric_card("False positives", winner["legit_false_positives"], "legit URLs", decimals=0)}
{metric_card("Missed phishing", winner["phishing_false_negatives"], "URLs", decimals=0)}
</div>
<section class="plain-summary">
<h4>Plain-English Summary</h4>
<ul>
<li>Winner: {escape(winner["model"])} trained on {escape(display_name(winner["training_scenario"]))} is first because it scored {winner["total_correct"]}/40 with {winner["legit_false_positives"]} legitimate false positives and {winner["phishing_false_negatives"]} missed phishing URLs.</li>
<li>This experiment is practical rather than purely benchmark-style: false positives matter because the web app should not label normal websites as phishing.</li>
<li>This is the evidence that shifted the practical backend discussion toward Logistic Regression trained on LegitPhish, while keeping XGBoost as the formal benchmark winner.</li>
</ul>
</section>
{sanity_probe_table(rows)}
"""
    return f"""
<details class="panel">
<summary>Experiment 3: {escape(payload.get("title", "40-URL real-world sanity probe"))}</summary>
{body}
</details>
"""


def google_slash_section(payload):
    rows = payload.get("rows", []) if payload else []
    highlighted = payload.get("highlighted_row") if payload else None
    if not rows:
        body = "<p>No Google slash rows are available.</p>"
    else:
        highlighted_text = (
            f"{highlighted['model']} trained on {display_name(highlighted['training_scenario'])} "
            f"predicted no-slash as {highlighted['google_no_slash_prediction']} and slash as "
            f"{highlighted['google_slash_prediction']}."
            if highlighted
            else "The combined-dataset XGBoost row was not found in the current sanity probe."
        )
        body = f"""
<p>{escape(payload["purpose"])}</p>
<div class="cards">
{text_card("Highlighted finding", "Combined XGBoost", highlighted_text)}
{text_card("No slash URL", "https://www.google.com", "Expected class: legitimate.")}
{text_card("With slash URL", "https://www.google.com/", "Expected class: legitimate.")}
</div>
<h3>Only These URL Features Changed</h3>
{google_feature_change_table(payload.get("feature_change", []))}
<section class="plain-summary">
<h4>Plain-English Summary</h4>
<ul>
<li>A trailing slash should not normally make Google look like phishing.</li>
<li>The combined-dataset XGBoost model flipping this prediction is evidence of lexical brittleness.</li>
<li>This supports the dissertation limitation: URL-only lexical features can be useful, but they are not enough for production-grade reliability by themselves.</li>
</ul>
</section>
{google_slash_table(rows)}
"""
    return f"""
<details class="panel">
<summary>Experiment 4: {escape(payload.get("title", "Google trailing-slash brittleness check"))}</summary>
{body}
</details>
"""


def plain_english_summary(sentences):
    if not sentences:
        return ""
    items = "".join(f"<li>{escape(sentence)}</li>" for sentence in sentences)
    return f"""
<section class="plain-summary">
<h4>Plain-English Summary</h4>
<ul>{items}</ul>
</section>
"""


def sanity_probe_table(rows):
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td class=\"num\">{int(row['rank'])}</td>"
            f"<td>{escape(display_name(row['training_scenario']))}</td>"
            f"<td>{escape(row['model'])}</td>"
            f"<td class=\"num\">{int(row['total_correct'])}/40</td>"
            f"<td class=\"num\">{fmt(row['accuracy'])}</td>"
            f"<td class=\"num\">{fmt(row['phishing_f1'])}</td>"
            f"<td class=\"num\">{int(row['legit_false_positives'])}</td>"
            f"<td class=\"num\">{int(row['phishing_false_negatives'])}</td>"
            f"<td>{escape(row['google_slash_prediction'])}</td>"
            f"<td>{escape(row['openai_research_prediction'])}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        f"{th('Rank', 'num')}{th('Trained On')}{th('Model')}{th('Correct', 'num')}"
        f"{th('Accuracy', 'num')}{th('F1', 'num')}"
        f"{th('False Positives', 'num')}{th('Missed Phishing', 'num')}"
        f"{th('Google Slash')}{th('OpenAI Research')}"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def google_feature_change_table(rows):
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(row['feature'])}</td>"
            f"<td class=\"num\">{int(row['no_slash'])}</td>"
            f"<td class=\"num\">{int(row['with_slash'])}</td>"
            "</tr>"
        )
    return (
        "<table class=\"compact-table\">"
        "<thead><tr>"
        f"{th('Feature')}{th('No slash', 'num')}{th('With slash', 'num')}"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def google_slash_table(rows):
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(display_name(row['training_scenario']))}</td>"
            f"<td>{escape(row['model'])}</td>"
            f"<td>{escape(row['google_no_slash_prediction'])}</td>"
            f"<td>{escape(row['google_slash_prediction'])}</td>"
            f"<td>{escape(row['openai_research_prediction'])}</td>"
            f"<td>{'Yes' if row['prediction_changed'] else 'No'}</td>"
            f"<td class=\"num\">{int(row['google_false_positives'])}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        f"{th('Trained On')}{th('Model')}{th('Google No Slash')}"
        f"{th('Google Slash')}{th('OpenAI Research')}"
        f"{th('Prediction Changed')}{th('False Positives', 'num')}"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def winner_table(rows):
    if not rows:
        return "<p>No winner rows available.</p>"
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(row['experiment'])}</td>"
            f"<td>{escape(row['winner_model'])}</td>"
            f"<td>{escape(display_name(row['best_trained_on']))}</td>"
            f"<td>{escape(display_name(row['best_tested_on']))}</td>"
            f"<td class=\"num\">{fmt(row['mean_f1'])}</td>"
            f"<td class=\"num\">{fmt(row['worst_f1'])}</td>"
            f"<td class=\"num\">{fmt(row['best_model_exact_f1'])}</td>"
            f"<td>{escape(row['why'])}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        f"{th('Experiment')}{th('Model')}{th('Trained On')}{th('Tested On')}"
        f"{th('Mean F1', 'num')}{th('Worst F1', 'num')}{th('Best F1', 'num')}"
        f"{th('Why')}"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def overall_table(rows):
    if not rows:
        return "<p>No ranking rows available.</p>"
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(row['experiment'])}</td>"
            f"<td>{escape(row['model'])}</td>"
            f"<td class=\"num\">{fmt(row['mean_f1'])}</td>"
            f"<td class=\"num\">{fmt(row['worst_f1'])}</td>"
            f"<td class=\"num\">{int(row['evaluations'])}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        f"{th('Experiment')}{th('Model')}{th('Mean F1', 'num')}"
        f"{th('Worst F1', 'num')}{th('Evaluations', 'num')}"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def model_ranking_table(rows):
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(row['model'])}</td>"
            f"<td class=\"num\">{fmt(row['mean_f1'])}</td>"
            f"<td class=\"num\">{fmt(row['worst_f1'])}</td>"
            f"<td class=\"num\">{fmt(row['mean_recall'])}</td>"
            f"<td class=\"num\">{fmt(row['mean_precision'])}</td>"
            f"<td class=\"num\">{fmt(row['mean_accuracy'])}</td>"
            f"<td class=\"num\">{int(row['evaluations'])}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        f"{th('Model')}{th('Mean F1', 'num')}{th('Worst F1', 'num')}"
        f"{th('Recall', 'num')}{th('Precision', 'num')}{th('Accuracy', 'num')}"
        f"{th('Evaluations', 'num')}"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def exact_results_table(rows):
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(row['model'])}</td>"
            f"<td>{escape(row['trained_on_label'])}</td>"
            f"<td>{escape(row['tested_on_label'])}</td>"
            f"<td class=\"num\">{int(row['rows_tested']):,}</td>"
            f"<td class=\"num\">{fmt(row['accuracy'])}</td>"
            f"<td class=\"num\">{fmt(row['phishing_precision'])}</td>"
            f"<td class=\"num\">{fmt(row['phishing_recall'])}</td>"
            f"<td class=\"num\">{bar_value(row['phishing_f1'])}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        f"{th('Model')}{th('Trained On')}{th('Tested On')}{th('Rows', 'num')}"
        f"{th('Accuracy', 'num')}{th('Precision', 'num')}{th('Recall', 'num')}"
        f"{th('F1', 'num')}"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def ranking_chart(rows):
    if not rows:
        return ""
    bars = []
    for row in rows:
        bars.append(
            '<div class="bar-row">'
            f'<span class="bar-label">{escape(row["model"])}</span>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{percentage(row["mean_f1"])}%"></div>'
            "</div>"
            f'<span class="bar-score">{fmt(row["mean_f1"])}</span>'
            "</div>"
        )
    return f'<div class="chart">{"".join(bars)}</div>'


def source_section(paths):
    items = "".join(f"<li><code>{escape(path)}</code></li>" for path in paths)
    return f"""
<details class="panel">
<summary>Source Files Used</summary>
<ul>{items}</ul>
</details>
"""


def text_card(label, value, note):
    return (
        '<article class="card">'
        f"<span>{escape(label)}</span>"
        f"<strong>{escape(value)}</strong>"
        f"<p>{escape(note)}</p>"
        "</article>"
    )


def metric_card(label, value, suffix="", decimals=4):
    return (
        '<article class="card">'
        f"<span>{escape(label)}</span>"
        f"<strong>{fmt(value, decimals)} {escape(suffix)}</strong>"
        "</article>"
    )


def th(label, class_name=""):
    tooltip = METRIC_TOOLTIPS.get(label, label)
    class_attribute = f' class="{escape(class_name)}"' if class_name else ""
    return (
        f"<th{class_attribute}>{escape(label)} "
        f'<span class="tooltip" tabindex="0" data-tooltip="{escape(tooltip)}">?</span>'
        "</th>"
    )


def bar_value(value):
    return (
        '<div class="cell-bar">'
        f"<span>{fmt(value)}</span>"
        '<div class="mini-track">'
        f'<div class="mini-fill" style="width:{percentage(value)}%"></div>'
        "</div></div>"
    )


def percentage(value):
    return max(2, min(100, float(value or 0) * 100))


def rows_tested_from_matrix(payload):
    matrix = payload.get("matrix") if isinstance(payload, dict) else None
    if not matrix:
        return 0
    return int(sum(sum(row) for row in matrix))


def score_key(row):
    return (
        row["phishing_f1"],
        row["phishing_recall"],
        row["phishing_precision"],
        row["accuracy"],
    )


def truthy(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def write_derived_experiment_outputs(summary):
    experiment_lookup = {item["id"]: item for item in summary.get("experiments", [])}
    derived_experiments = [
        (EXPERIMENT_2A_DIR, experiment_lookup.get("2a")),
        (EXPERIMENT_2B_DIR, experiment_lookup.get("2b")),
    ]
    for directory, experiment in derived_experiments:
        if not experiment:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(experiment["rows"]).to_csv(directory / "test_summary.csv", index=False)
        (directory / "report.md").write_text(
            build_simple_experiment_markdown(experiment),
            encoding="utf-8",
        )

    sanity_probe = summary.get("sanity_probe") or {}
    if sanity_probe.get("rows"):
        EXPERIMENT_3_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sanity_probe["rows"]).to_csv(
            EXPERIMENT_3_DIR / "test_summary.csv",
            index=False,
        )
        (EXPERIMENT_3_DIR / "report.md").write_text(
            build_simple_experiment_markdown(sanity_probe),
            encoding="utf-8",
        )

    google_slash = summary.get("google_slash") or {}
    if google_slash.get("rows"):
        EXPERIMENT_4_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(google_slash["rows"]).to_csv(
            EXPERIMENT_4_DIR / "test_summary.csv",
            index=False,
        )
        (EXPERIMENT_4_DIR / "report.md").write_text(
            build_simple_experiment_markdown(google_slash),
            encoding="utf-8",
        )


def build_simple_experiment_markdown(experiment):
    rows = experiment.get("rows", [])
    lines = [
        f"# Experiment {experiment.get('id', '')}: {experiment.get('title', '')}",
        "",
        experiment.get("purpose", ""),
        "",
    ]
    winner = experiment.get("winner")
    if winner:
        lines.extend(
            [
                "## Winner",
                "",
                f"- Model: `{winner.get('model', '')}`",
                f"- Trained on: `{winner.get('training_scenario', winner.get('trained_on', ''))}`",
                f"- Phishing F1: `{winner.get('phishing_f1', 0):.4f}`"
                if "phishing_f1" in winner
                else f"- Correct: `{winner.get('total_correct', 0)}`",
                "",
            ]
        )
    if rows:
        frame = pd.DataFrame(rows)
        preferred_columns = [
            column
            for column in [
                "rank",
                "training_scenario",
                "tested_on",
                "test_dataset",
                "model",
                "rows_tested",
                "accuracy",
                "phishing_precision",
                "phishing_recall",
                "phishing_f1",
                "total_correct",
                "legit_false_positives",
                "phishing_false_negatives",
                "google_no_slash_prediction",
                "google_slash_prediction",
                "prediction_changed",
            ]
            if column in frame.columns
        ]
        lines.extend(
            [
                "## Summary Table",
                "",
                frame[preferred_columns].to_markdown(index=False),
                "",
            ]
        )
    return "\n".join(lines)


def source_files():
    paths = [
        MAIN_REPORTS / "test_summary.csv",
        MAIN_REPORTS / "confusion_matrices.json",
        EXTERNAL_REPORTS / "test_summary.csv",
        CROSS_REPORTS / "test_summary.csv",
        URL_PROBE_REPORTS / "40_url_balanced_model_probe_urls.csv",
        URL_PROBE_REPORTS / "40_url_balanced_model_probe_summary.csv",
        URL_PROBE_REPORTS / "40_url_balanced_model_probe_predictions.csv",
    ]
    return [str(path.relative_to(PROJECT_ROOT)) for path in paths if path.exists()]


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def row_record(row):
    output = {}
    for key, value in dict(row).items():
        if hasattr(value, "item"):
            value = value.item()
        output[key] = value
    return output


def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def safe_f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def clean_optional_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def display_name(value):
    value = str(value)
    return DISPLAY_NAMES.get(value, value.replace("_", " ").title())


def fmt(value, decimals=4):
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return escape(value)
    if decimals == 0:
        return f"{number:,.0f}"
    return f"{number:.{decimals}f}"


def escape(value):
    return html.escape(str(value))


def escape_markdown(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def styles():
    return """<style>
:root {
  color-scheme: light;
  --bg: #f5f7fb;
  --ink: #17202a;
  --muted: #5b6675;
  --line: #d8dee8;
  --panel: #ffffff;
  --accent: #126b5f;
  --accent-soft: #dff3ee;
  --gold: #a66a00;
}
* {
  box-sizing: border-box;
}
body {
  background: var(--bg);
  color: var(--ink);
  font-family: Arial, Helvetica, sans-serif;
  margin: 0;
}
main {
  margin: 0 auto;
  max-width: 1180px;
  padding: 28px 20px 56px;
}
header {
  border-bottom: 1px solid var(--line);
  margin-bottom: 18px;
  padding-bottom: 18px;
}
.eyebrow {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}
h1 {
  font-size: 32px;
  margin: 4px 0 8px;
}
h3 {
  font-size: 16px;
  margin: 18px 0 8px;
}
.policy {
  background: var(--accent-soft);
  border-left: 4px solid var(--accent);
  margin-top: 14px;
  padding: 10px 12px;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin: 14px 0;
}
summary {
  cursor: pointer;
  font-size: 17px;
  font-weight: 700;
  padding: 16px 18px;
}
.panel > *:not(summary) {
  margin-left: 18px;
  margin-right: 18px;
}
.panel > :last-child {
  margin-bottom: 18px;
}
.cards {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  margin: 12px 0;
}
.top-definitions {
  margin: 16px 0;
}
.top-definitions h2 {
  font-size: 18px;
  margin: 0 0 10px;
}
.definition-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
}
.definition-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
}
.definition-card strong {
  display: block;
  font-size: 14px;
  margin-bottom: 6px;
}
.definition-card p {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.4;
  margin: 0;
}
.card {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
}
.card span {
  color: var(--muted);
  display: block;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}
.card strong {
  display: block;
  font-size: 22px;
  margin-top: 6px;
}
.card p {
  color: var(--muted);
  margin: 8px 0 0;
}
.chart {
  border: 1px solid var(--line);
  border-radius: 8px;
  margin: 10px 0 14px;
  padding: 10px;
}
.plain-summary {
  background: #fbfcfe;
  border: 1px solid var(--line);
  border-radius: 8px;
  margin: 12px 0 16px;
  padding: 12px 14px;
}
.plain-summary h4 {
  font-size: 14px;
  margin: 0 0 8px;
}
.plain-summary ul {
  margin: 0;
  padding-left: 20px;
}
.plain-summary li {
  line-height: 1.45;
  margin: 6px 0;
}
.bar-row {
  align-items: center;
  display: grid;
  gap: 10px;
  grid-template-columns: minmax(140px, 220px) 1fr 62px;
  margin: 8px 0;
}
.bar-label {
  color: var(--ink);
  font-size: 13px;
  font-weight: 700;
}
.bar-track,
.mini-track {
  background: #e7ebf1;
  border-radius: 999px;
  overflow: hidden;
}
.bar-track {
  height: 14px;
}
.mini-track {
  height: 7px;
  margin-top: 4px;
}
.bar-fill,
.mini-fill {
  background: var(--accent);
  height: 100%;
}
.bar-score {
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  text-align: right;
}
table {
  border-collapse: collapse;
  margin: 10px 0 16px;
  width: 100%;
}
th, td {
  border-bottom: 1px solid var(--line);
  padding: 8px;
  text-align: left;
  vertical-align: top;
}
th {
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
}
.num {
  text-align: right;
}
.compact-table {
  max-width: 760px;
}
.cell-bar {
  min-width: 88px;
}
code {
  background: #eef2f7;
  border-radius: 4px;
  padding: 2px 4px;
}
.tooltip {
  background: #e9f2ff;
  border-radius: 999px;
  color: #145cc7;
  cursor: help;
  display: inline-grid;
  font-size: 10px;
  height: 16px;
  margin-left: 4px;
  place-items: center;
  position: relative;
  text-transform: none;
  width: 16px;
}
.tooltip::after {
  background: #17202a;
  border-radius: 6px;
  color: #fff;
  content: attr(data-tooltip);
  display: none;
  font-size: 12px;
  font-weight: 400;
  left: -8px;
  line-height: 1.35;
  padding: 8px 10px;
  position: absolute;
  text-align: left;
  text-transform: none;
  top: 22px;
  width: 240px;
  z-index: 10;
}
.tooltip:hover::after,
.tooltip:focus::after {
  display: block;
}
@media (max-width: 760px) {
  main {
    padding: 20px 12px 44px;
  }
  .bar-row {
    grid-template-columns: 1fr;
  }
  table {
    display: block;
    overflow-x: auto;
    white-space: nowrap;
  }
}
</style>"""


if __name__ == "__main__":
    main()
