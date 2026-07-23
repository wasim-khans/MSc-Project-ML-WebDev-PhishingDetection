import unittest

import pandas as pd

from machine_learning.scripts import build_all_experiments_combined_report as report


class DissertationReportTests(unittest.TestCase):
    def test_clean_heldout_rows_excludes_diagnostic_and_training_rows(self):
        frame = pd.DataFrame(
            [
                metric_frame_row("phiusiil_main", "combined_test"),
                metric_frame_row(
                    "combined_dataset",
                    "complete_combined_dataset",
                    scope="complete_dataset",
                    contains_training_rows=True,
                ),
                metric_frame_row(
                    "combined_dataset",
                    "combined_test",
                    contains_training_rows=True,
                ),
            ]
        )

        clean = report.clean_heldout_rows(frame)

        self.assertEqual(len(clean), 1)
        self.assertEqual(clean.iloc[0]["test_dataset"], "combined_test")
        self.assertFalse(bool(clean.iloc[0]["contains_training_rows"]))

    def test_report_experiment_filters_match_dissertation_plan(self):
        clean = pd.DataFrame(
            [
                metric_frame_row("legitphish", "legitphish_test"),
                metric_frame_row("legitphish", "phishstorm_test"),
                metric_frame_row("phishstorm", "phiusiil_main_test"),
                metric_frame_row("phiusiil_main", "combined_test"),
                metric_frame_row("legitphish", "combined_test"),
                metric_frame_row("phishstorm", "combined_test"),
                metric_frame_row("combined_dataset", "combined_test"),
            ]
        )

        one_c = report.cross_source_rows(clean)
        two_a = report.single_source_combined_rows(clean)
        two_b = report.combined_training_rows(clean)

        self.assertEqual(
            {(row["trained_on"], row["tested_on"]) for row in one_c},
            {
                ("legitphish", "phishstorm_test"),
                ("phishstorm", "phiusiil_main_test"),
            },
        )
        self.assertEqual(len(two_a), 3)
        self.assertTrue(
            all(row["trained_on"] != "combined_dataset" for row in two_a)
        )
        self.assertEqual(len(two_b), 1)
        self.assertEqual(two_b[0]["trained_on"], "combined_dataset")

    def test_html_report_has_collapsible_requested_sections_and_tooltips(self):
        experiments = [
            report.build_experiment(
                experiment_id,
                title,
                "Synthetic test section.",
                [
                    report.metric_row(
                        experiment_id,
                        "combined_dataset",
                        "combined_test",
                        "XGBoost",
                        100,
                        metric_payload(),
                    )
                ],
            )
            for experiment_id, title in [
                ("1a", "Each model trained and tested on the main PhiUSIIL dataset"),
                ("1b", "Main-trained models tested against external datasets"),
                ("1c", "External-trained models tested against other single-source datasets"),
                ("2a", "Single-source models tested against the combined held-out dataset"),
                ("2b", "Combined-trained models tested against the combined held-out dataset"),
            ]
        ]
        summary = {
            "generated_at": "2026-06-26 12:00",
            "heldout_policy": "All sections use held-out test rows only.",
            "experiment_definitions": report.EXPERIMENT_DEFINITIONS,
            "experiments": experiments,
            "decision_summary": report.decision_summary(experiments),
            "sanity_probe": {
                "id": "3",
                "title": "40-URL real-world sanity probe",
                "purpose": "Synthetic sanity probe.",
                "winner": sanity_row(),
                "rows": [sanity_row()],
            },
            "google_slash": {
                "id": "4",
                "title": "Google trailing-slash brittleness check",
                "purpose": "Synthetic slash check.",
                "highlighted_row": google_row(),
                "feature_change": [
                    {"feature": "path_length", "no_slash": 0, "with_slash": 1}
                ],
                "rows": [google_row()],
            },
            "source_files": [],
        }

        html = report.build_html_report(summary)

        self.assertIn("<details", html)
        self.assertIn("Experiment Definitions", html)
        self.assertIn("Trained on the main PhiUSIIL dataset", html)
        self.assertIn("Trained on the combined dataset", html)
        self.assertIn("Experiment 1a", html)
        self.assertIn("Experiment 1b", html)
        self.assertIn("Experiment 1c", html)
        self.assertIn("Experiment 2a", html)
        self.assertIn("Experiment 2b", html)
        self.assertIn("Experiment 3", html)
        self.assertIn("Experiment 4", html)
        self.assertNotIn("Experiment 5", html)
        self.assertIn("40-URL real-world sanity probe", html)
        self.assertIn("Google trailing-slash brittleness check", html)
        self.assertIn("Plain-English Summary", html)
        self.assertIn("Winner:", html)
        self.assertIn("data-tooltip", html)
        self.assertNotIn("complete_combined_dataset", html)

    def test_plain_english_summary_is_generated_from_current_metrics(self):
        experiment = report.build_experiment(
            "2b",
            "Synthetic experiment",
            "Synthetic purpose.",
            [
                report.metric_row(
                    "2b",
                    "combined_dataset",
                    "combined_test",
                    "Model A",
                    100,
                    metric_payload(f1=0.90, precision=0.91, recall=0.89),
                ),
                report.metric_row(
                    "2b",
                    "combined_dataset",
                    "combined_test",
                    "Model B",
                    100,
                    metric_payload(f1=0.80, precision=0.81, recall=0.79),
                ),
            ],
        )

        summary = " ".join(experiment["plain_english_summary"])

        self.assertIn("Winner: Model A", summary)
        self.assertIn("Runner-up: Model B", summary)
        self.assertIn("0.9000", summary)
        self.assertIn("0.1000", summary)

    def test_conclusions_markdown_is_generated_from_current_metrics(self):
        experiment = report.build_experiment(
            "2b",
            "Synthetic experiment",
            "Synthetic purpose.",
            [
                report.metric_row(
                    "2b",
                    "combined_dataset",
                    "combined_test",
                    "Model A",
                    100,
                    metric_payload(
                        f1=0.90,
                        precision=0.91,
                        recall=0.89,
                        model_file="models/model_a.joblib",
                    ),
                ),
                report.metric_row(
                    "2b",
                    "combined_dataset",
                    "combined_test",
                    "Model B",
                    100,
                    metric_payload(f1=0.80, precision=0.81, recall=0.79),
                ),
            ],
        )
        summary = {
            "generated_at": "2026-06-26 12:00",
            "heldout_policy": "All sections use held-out test rows only.",
            "experiment_definitions": report.EXPERIMENT_DEFINITIONS,
            "experiments": [experiment],
            "decision_summary": report.decision_summary([experiment]),
            "sanity_probe": {
                "winner": sanity_row(),
                "rows": [sanity_row()],
            },
            "google_slash": {
                "highlighted_row": google_row(),
                "rows": [google_row()],
            },
            "source_files": [],
        }

        markdown = report.build_conclusions_markdown(summary)

        self.assertIn("Model A", markdown)
        self.assertIn("Model B", markdown)
        self.assertIn("0.9000", markdown)
        self.assertIn("models/model_a.joblib", markdown)
        self.assertIn("Experiment 3", markdown)
        self.assertIn("Experiment 4", markdown)
        self.assertNotIn("Experiment 5", markdown)
        self.assertIn("Logistic Regression", markdown)


def metric_frame_row(
    training_scenario,
    test_dataset,
    scope="clean_holdout",
    contains_training_rows=False,
):
    row = metric_payload()
    row.update(
        {
            "training_scenario": training_scenario,
            "test_dataset": test_dataset,
            "evaluation_scope": scope,
            "contains_training_rows": contains_training_rows,
            "rows_tested": 100,
        }
    )
    return row


def metric_payload(f1=0.94, precision=0.92, recall=0.93, model_file=None):
    return {
        "model": "XGBoost",
        "accuracy": 0.91,
        "phishing_precision": precision,
        "phishing_recall": recall,
        "phishing_f1": f1,
        "model_file": model_file,
    }


def sanity_row():
    return {
        "rank": 1,
        "training_scenario": "legitphish",
        "training_scenario_label": "LegitPhish",
        "model": "Logistic Regression",
        "model_file": "models/logistic_regression.joblib",
        "total_correct": 40,
        "accuracy": 1.0,
        "phishing_precision": 1.0,
        "phishing_recall": 1.0,
        "phishing_f1": 1.0,
        "legit_correct": 20,
        "legit_false_positives": 0,
        "phishing_correct": 20,
        "phishing_false_negatives": 0,
        "google_no_slash_prediction": "legitimate",
        "google_slash_prediction": "legitimate",
        "openai_research_prediction": "legitimate",
    }


def google_row():
    return {
        "training_scenario": "combined_dataset",
        "training_scenario_label": "Combined dataset",
        "model": "XGBoost",
        "model_file": "models/xgboost.joblib",
        "google_no_slash_prediction": "legitimate",
        "google_slash_prediction": "phishing",
        "openai_research_prediction": "phishing",
        "prediction_changed": True,
        "google_false_positives": 1,
        "all_three_legitimate_correct": False,
    }


if __name__ == "__main__":
    unittest.main()
