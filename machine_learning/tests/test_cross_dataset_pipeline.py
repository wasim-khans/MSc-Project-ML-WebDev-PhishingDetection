import unittest

import pandas as pd

from machine_learning.scripts.helpers import cross_dataset_config as config
from machine_learning.scripts.helpers import cross_dataset_data_preparation as data_preparation
from machine_learning.scripts.helpers import cross_dataset_evaluation as evaluation
from machine_learning.scripts.helpers import cross_dataset_model_training as model_training


class Experiment2ConfigTests(unittest.TestCase):
    def test_config_defines_expected_datasets_and_scenarios(self):
        self.assertEqual(config.RANDOM_STATE, 42)
        self.assertEqual(config.TEST_SIZE, 0.2)
        self.assertEqual(config.LABEL_COLUMN, "label")
        self.assertEqual(config.LABEL_ORDER, [0, 1])
        self.assertEqual(
            list(config.DATASETS.keys()),
            ["main", "legitphish", "phishstorm"],
        )
        self.assertEqual(
            list(config.TRAINING_SCENARIOS.keys()),
            ["phiusiil_main", "legitphish", "phishstorm", "combined_dataset"],
        )
        self.assertEqual(
            list(config.TEST_SETS.keys()),
            [
                "phiusiil_main_test",
                "legitphish_test",
                "phishstorm_test",
                "combined_test",
            ],
        )
        self.assertEqual(
            list(config.COMPLETE_TEST_SETS.keys()),
            ["complete_combined_dataset"],
        )
        self.assertEqual(
            set(config.COMBINED_SPLIT_FILES),
            {
                "combined_dataset_train",
                "combined_test",
                "complete_combined_dataset",
            },
        )
        self.assertEqual(len(config.FEATURE_COLUMNS), 15)

    def test_cross_dataset_config_defaults_are_loaded_from_json(self):
        run_config = config.load_cross_dataset_config()

        self.assertEqual(run_config["test_size"], 0.2)
        self.assertEqual(run_config["models"], "all")
        self.assertEqual(
            run_config["training_scenarios"],
            ["phiusiil_main", "legitphish", "phishstorm", "combined_dataset"],
        )
        self.assertEqual(run_config["complete_test_sets"], ["complete_combined_dataset"])


class Experiment2DataPreparationTests(unittest.TestCase):
    def feature_row(self, source_row_id=1, label=0):
        row = {
            "source_dataset": "sample",
            "source_row_id": source_row_id,
            "label": label,
            "url_normalized": f"https://example.test/{source_row_id}",
        }
        for column in config.FEATURE_COLUMNS:
            row[column] = 0
        row["url_length"] = source_row_id + 10
        row["domain_length"] = 5
        row["tld_length"] = 3
        return row

    def test_external_features_and_labels_join_by_row_id(self):
        features = pd.DataFrame(
            [
                {"row_id": 2, "url": "https://b.example", **{column: 0 for column in config.FEATURE_COLUMNS}},
                {"row_id": 1, "url": "https://a.example", **{column: 1 for column in config.FEATURE_COLUMNS}},
            ]
        )
        labels = pd.DataFrame(
            [
                {"row_id": 1, "project_label": 0},
                {"row_id": 2, "project_label": 1},
            ]
        )

        merged = data_preparation.merge_external_features_and_labels(
            features,
            labels,
            dataset_name="sample",
        )

        self.assertEqual(list(merged["source_row_id"]), [2, 1])
        self.assertEqual(list(merged["label"]), [1, 0])
        self.assertNotIn("project_label", merged.columns)

    def test_validate_feature_columns_rejects_missing_feature(self):
        frame = pd.DataFrame({"url_length": [10], "label": [0]})

        with self.assertRaises(ValueError):
            data_preparation.validate_feature_columns(frame, dataset_name="sample")

    def test_split_dataset_keeps_source_rows_disjoint(self):
        frame = pd.DataFrame(
            [self.feature_row(source_row_id=index, label=index % 2) for index in range(20)]
        )

        train, test = data_preparation.split_dataset(frame, dataset_name="sample")

        train_keys = set(zip(train["source_dataset"], train["source_row_id"]))
        test_keys = set(zip(test["source_dataset"], test["source_row_id"]))
        self.assertEqual(len(train), 16)
        self.assertEqual(len(test), 4)
        self.assertTrue(train_keys.isdisjoint(test_keys))
        self.assertEqual(set(train["label"]), {0, 1})
        self.assertEqual(set(test["label"]), {0, 1})

    def test_split_dataset_keeps_duplicate_url_groups_disjoint(self):
        rows = []
        for index in range(30):
            row = self.feature_row(source_row_id=index, label=index % 2)
            row["url_normalized"] = f"https://example.test/group-{index // 2}"
            rows.append(row)
        frame = pd.DataFrame(rows)

        train, test = data_preparation.split_dataset(frame, dataset_name="sample")

        train_urls = set(train["url_normalized"])
        test_urls = set(test["url_normalized"])
        self.assertTrue(train_urls.isdisjoint(test_urls))

    def test_combined_split_frames_are_real_dataset_views(self):
        train_splits = {
            "main": pd.DataFrame([self.feature_row(1, 0)]),
            "legitphish": pd.DataFrame([self.feature_row(2, 1)]),
            "phishstorm": pd.DataFrame([self.feature_row(3, 0)]),
        }
        test_splits = {
            "main": pd.DataFrame([self.feature_row(4, 1)]),
            "legitphish": pd.DataFrame([self.feature_row(5, 0)]),
            "phishstorm": pd.DataFrame([self.feature_row(6, 1)]),
        }

        combined = data_preparation.build_combined_split_frames(
            train_splits, test_splits
        )

        self.assertEqual(len(combined["combined_dataset_train"]), 3)
        self.assertEqual(len(combined["combined_test"]), 3)
        self.assertEqual(len(combined["complete_combined_dataset"]), 6)
        self.assertEqual(
            set(combined["combined_dataset_train"]["source_dataset"]),
            {"sample"},
        )

    def test_combined_split_frames_remove_cross_source_duplicate_urls(self):
        main_train = self.feature_row(1, 0)
        main_train["source_dataset"] = "main"
        main_train["url_normalized"] = "https://shared.example"
        legit_train = self.feature_row(2, 0)
        legit_train["source_dataset"] = "legitphish"
        legit_train["url_normalized"] = "https://shared.example"
        phish_train = self.feature_row(3, 1)
        phish_train["source_dataset"] = "phishstorm"
        phish_train["url_normalized"] = "https://unique.example"

        combined = data_preparation.build_combined_split_frames(
            {
                "main": pd.DataFrame([main_train]),
                "legitphish": pd.DataFrame([legit_train]),
                "phishstorm": pd.DataFrame([phish_train]),
            },
            {
                "main": pd.DataFrame([self.feature_row(4, 1)]),
                "legitphish": pd.DataFrame([self.feature_row(5, 0)]),
                "phishstorm": pd.DataFrame([self.feature_row(6, 1)]),
            },
        )

        self.assertEqual(len(combined["combined_dataset_train"]), 2)
        self.assertEqual(
            set(combined["combined_dataset_train"]["source_dataset"]),
            {"main", "phishstorm"},
        )


class Experiment2TrainingTests(unittest.TestCase):
    def test_model_artifact_filename_includes_scenario_and_model(self):
        self.assertEqual(
            model_training.model_artifact_filename("combined_dataset", "Linear SVM"),
            "linear_svm_T_ON_combined_dataset_80_20.joblib",
        )
        self.assertEqual(
            model_training.model_artifact_filename("phiusiil_main", "XGBoost"),
            "xgboost_T_ON_phiusiil_main_80_20.joblib",
        )

    def test_build_training_scenarios_combines_only_train_frames(self):
        train_splits = {
            "main": pd.DataFrame(
                {"source_dataset": ["main"], "source_row_id": [1], "label": [0]}
            ),
            "legitphish": pd.DataFrame(
                {"source_dataset": ["legitphish"], "source_row_id": [2], "label": [1]}
            ),
            "phishstorm": pd.DataFrame(
                {"source_dataset": ["phishstorm"], "source_row_id": [3], "label": [0]}
            ),
        }

        scenarios = model_training.build_training_scenarios(train_splits)

        self.assertEqual(len(scenarios["phiusiil_main"]), 1)
        self.assertEqual(len(scenarios["combined_dataset"]), 3)
        self.assertEqual(
            set(scenarios["combined_dataset"]["source_dataset"]),
            {"main", "legitphish", "phishstorm"},
        )

    def test_combined_training_scenario_uses_saved_combined_file_when_available(self):
        train_splits = {
            "main": pd.DataFrame(
                {"source_dataset": ["main"], "source_row_id": [1], "label": [0]}
            ),
            "legitphish": pd.DataFrame(
                {"source_dataset": ["legitphish"], "source_row_id": [2], "label": [1]}
            ),
            "phishstorm": pd.DataFrame(
                {"source_dataset": ["phishstorm"], "source_row_id": [3], "label": [0]}
            ),
            "combined_dataset": pd.DataFrame(
                {
                    "source_dataset": ["combined_file", "combined_file"],
                    "source_row_id": [10, 11],
                    "label": [0, 1],
                }
            ),
        }

        scenarios = model_training.build_training_scenarios(train_splits)

        self.assertEqual(len(scenarios["combined_dataset"]), 2)
        self.assertEqual(
            set(scenarios["combined_dataset"]["source_dataset"]),
            {"combined_file"},
        )


class Experiment2EvaluationTests(unittest.TestCase):
    def test_metric_row_contains_training_scenario_model_and_test_dataset(self):
        row = evaluation.build_metric_row(
            training_scenario="combined_dataset",
            model_name="XGBoost",
            test_dataset="phiusiil_main_test",
            rows_tested=100,
            metrics={
                "accuracy": 0.9,
                "phishing_precision": 0.8,
                "phishing_recall": 0.7,
                "phishing_f1": 0.75,
            },
            model_file="machine_learning/trained_models/2_cross_dataset_generalisation/xgboost_T_ON_combined_dataset_80_20.joblib",
            evaluation_scope="clean_holdout",
            contains_training_rows=False,
            row_source_note="Untouched test split rows only.",
        )

        self.assertEqual(row["training_scenario"], "combined_dataset")
        self.assertEqual(row["model"], "XGBoost")
        self.assertEqual(row["test_dataset"], "phiusiil_main_test")
        self.assertEqual(row["rows_tested"], 100)
        self.assertEqual(row["phishing_f1"], 0.75)
        self.assertEqual(row["evaluation_scope"], "clean_holdout")
        self.assertFalse(row["contains_training_rows"])
        self.assertEqual(row["training_url_overlap_removed"], 0)

    def test_clean_evaluation_set_removes_urls_seen_in_training_scenario(self):
        training_frame = pd.DataFrame(
            [
                self.feature_row("main", 1, 0),
                self.feature_row("main", 2, 1),
            ]
        )
        test_frame = pd.DataFrame(
            [
                self.feature_row("legitphish", 3, 0),
                self.feature_row("legitphish", 4, 1),
            ]
        )
        test_frame.loc[0, "url_normalized"] = training_frame.loc[0, "url_normalized"]
        evaluation_set = {
            "frame": test_frame,
            "evaluation_scope": "clean_holdout",
            "contains_training_rows": False,
            "row_source_note": "Untouched test split rows only.",
        }

        filtered = evaluation.filter_evaluation_set_for_training_urls(
            evaluation_set,
            training_frame,
        )

        self.assertEqual(len(filtered["frame"]), 1)
        self.assertEqual(filtered["training_url_overlap_removed"], 1)
        self.assertIn("removed", filtered["row_source_note"])

    def test_evaluation_sets_include_clean_holdout_and_complete_combined_dataset(self):
        train_splits = {
            "main": pd.DataFrame([self.feature_row("main", 1, 0)]),
            "legitphish": pd.DataFrame([self.feature_row("legitphish", 2, 1)]),
            "phishstorm": pd.DataFrame([self.feature_row("phishstorm", 3, 0)]),
        }
        test_splits = {
            "main": pd.DataFrame([self.feature_row("main", 4, 1)]),
            "legitphish": pd.DataFrame([self.feature_row("legitphish", 5, 0)]),
            "phishstorm": pd.DataFrame([self.feature_row("phishstorm", 6, 1)]),
        }

        evaluation_sets = evaluation.build_evaluation_sets(train_splits, test_splits)

        self.assertEqual(evaluation_sets["combined_test"]["evaluation_scope"], "clean_holdout")
        self.assertFalse(evaluation_sets["combined_test"]["contains_training_rows"])
        self.assertEqual(len(evaluation_sets["combined_test"]["frame"]), 3)
        self.assertEqual(
            evaluation_sets["complete_combined_dataset"]["evaluation_scope"],
            "complete_dataset",
        )
        self.assertTrue(
            evaluation_sets["complete_combined_dataset"]["contains_training_rows"]
        )
        self.assertEqual(len(evaluation_sets["complete_combined_dataset"]["frame"]), 6)

    def test_evaluation_sets_prefer_saved_combined_files_when_available(self):
        train_splits = {
            "main": pd.DataFrame([self.feature_row("main", 1, 0)]),
            "legitphish": pd.DataFrame([self.feature_row("legitphish", 2, 1)]),
            "phishstorm": pd.DataFrame([self.feature_row("phishstorm", 3, 0)]),
        }
        test_splits = {
            "main": pd.DataFrame([self.feature_row("main", 4, 1)]),
            "legitphish": pd.DataFrame([self.feature_row("legitphish", 5, 0)]),
            "phishstorm": pd.DataFrame([self.feature_row("phishstorm", 6, 1)]),
            "combined_test": pd.DataFrame(
                [self.feature_row("combined_file", 7, 0)]
            ),
            "complete_combined_dataset": pd.DataFrame(
                [
                    self.feature_row("complete_file", 8, 0),
                    self.feature_row("complete_file", 9, 1),
                ]
            ),
        }

        evaluation_sets = evaluation.build_evaluation_sets(train_splits, test_splits)

        self.assertEqual(len(evaluation_sets["combined_test"]["frame"]), 1)
        self.assertEqual(
            set(evaluation_sets["combined_test"]["frame"]["source_dataset"]),
            {"combined_file"},
        )
        self.assertEqual(len(evaluation_sets["complete_combined_dataset"]["frame"]), 2)
        self.assertEqual(
            set(evaluation_sets["complete_combined_dataset"]["frame"]["source_dataset"]),
            {"complete_file"},
        )

    def test_scope_summary_counts_clean_and_complete_evaluations(self):
        metrics_frame = pd.DataFrame(
            [
                {"evaluation_scope": "clean_holdout", "rows_tested": 10},
                {"evaluation_scope": "clean_holdout", "rows_tested": 20},
                {"evaluation_scope": "complete_dataset", "rows_tested": 30},
            ]
        )

        summary = evaluation.scope_summary(metrics_frame)

        self.assertEqual(summary["clean_holdout"]["evaluations"], 2)
        self.assertEqual(summary["clean_holdout"]["prediction_rows"], 30)
        self.assertEqual(summary["complete_dataset"]["evaluations"], 1)
        self.assertEqual(summary["complete_dataset"]["prediction_rows"], 30)

    def test_summary_payload_keeps_clean_and_diagnostic_winners_separate(self):
        metrics_frame = pd.DataFrame(
            [
                {
                    "training_scenario": "phiusiil_main",
                    "model": "XGBoost",
                    "test_dataset": "combined_test",
                    "evaluation_scope": "clean_holdout",
                    "contains_training_rows": False,
                    "row_source_note": "Untouched test split rows only.",
                    "rows_tested": 10,
                    "accuracy": 0.8,
                    "phishing_precision": 0.8,
                    "phishing_recall": 0.8,
                    "phishing_f1": 0.8,
                    "model_file": "x.joblib",
                    "train_seconds": 1,
                    "predict_seconds": 1,
                },
                {
                    "training_scenario": "combined_dataset",
                    "model": "Random Forest",
                    "test_dataset": "combined_test",
                    "evaluation_scope": "clean_holdout",
                    "contains_training_rows": False,
                    "row_source_note": "Untouched test split rows only.",
                    "rows_tested": 10,
                    "accuracy": 0.9,
                    "phishing_precision": 0.9,
                    "phishing_recall": 0.9,
                    "phishing_f1": 0.9,
                    "model_file": "rf.joblib",
                    "train_seconds": 1,
                    "predict_seconds": 1,
                },
                {
                    "training_scenario": "combined_dataset",
                    "model": "Random Forest",
                    "test_dataset": "complete_combined_dataset",
                    "evaluation_scope": "complete_dataset",
                    "contains_training_rows": True,
                    "row_source_note": "Diagnostic full dataset check includes training rows.",
                    "rows_tested": 20,
                    "accuracy": 0.99,
                    "phishing_precision": 0.99,
                    "phishing_recall": 0.99,
                    "phishing_f1": 0.99,
                    "model_file": "rf.joblib",
                    "train_seconds": 1,
                    "predict_seconds": 1,
                },
            ]
        )

        summary = evaluation.build_summary_payload(metrics_frame)

        self.assertEqual(summary["scope_summary"]["clean_holdout"]["evaluations"], 2)
        self.assertEqual(
            summary["best_clean_holdout_by_test_dataset"][0]["model"],
            "Random Forest",
        )
        self.assertEqual(
            summary["best_complete_dataset_diagnostic"]["test_dataset"],
            "complete_combined_dataset",
        )

    def test_html_report_mentions_matrix_and_combined_training(self):
        metrics_frame = pd.DataFrame(
            [
                {
                    "training_scenario": "combined_dataset",
                    "model": "XGBoost",
                    "test_dataset": "phiusiil_main_test",
                    "rows_tested": 100,
                    "accuracy": 0.9,
                    "phishing_precision": 0.8,
                    "phishing_recall": 0.7,
                    "phishing_f1": 0.75,
                    "model_file": "machine_learning/trained_models/2_cross_dataset_generalisation/xgboost_T_ON_combined_dataset_80_20.joblib",
                    "train_seconds": 1.2,
                    "predict_seconds": 0.03,
                    "evaluation_scope": "clean_holdout",
                    "contains_training_rows": False,
                    "row_source_note": "Untouched test split rows only.",
                },
                {
                    "training_scenario": "combined_dataset",
                    "model": "XGBoost",
                    "test_dataset": "complete_combined_dataset",
                    "rows_tested": 300,
                    "accuracy": 0.95,
                    "phishing_precision": 0.91,
                    "phishing_recall": 0.93,
                    "phishing_f1": 0.92,
                    "model_file": "machine_learning/trained_models/2_cross_dataset_generalisation/xgboost_T_ON_combined_dataset_80_20.joblib",
                    "train_seconds": 1.2,
                    "predict_seconds": 0.05,
                    "evaluation_scope": "complete_dataset",
                    "contains_training_rows": True,
                    "row_source_note": "Diagnostic full dataset check includes training rows.",
                }
            ]
        )

        html = evaluation.build_html_report(
            metrics_frame=metrics_frame,
            confusion_matrices={
                "combined_dataset::XGBoost::phiusiil_main_test": {
                    "matrix": [[8, 2], [1, 9]],
                    "label_order": [0, 1],
                    "label_names": {0: "phishing", 1: "legitimate"},
                }
            },
            generated_at="2026-06-26 12:00",
        )

        self.assertIn("Experiment 2: Cross-Dataset Generalisation Report", html)
        self.assertIn("Cross-Dataset Training Matrix", html)
        self.assertIn("combined_dataset", html)
        self.assertIn("phiusiil_main_test", html)
        self.assertIn("Heatmap Matrix", html)
        self.assertIn("Combined Training Comparison", html)
        self.assertIn("Clean Holdout", html)
        self.assertIn("Complete Combined Dataset", html)
        self.assertIn("includes training rows", html)
        self.assertIn("Programmatic Summary", html)
        self.assertIn("What to Look For", html)
        self.assertIn("Filter Results", html)
        self.assertIn("Model Summary View", html)
        self.assertIn("Scenario View", html)
        self.assertIn("Ranking View", html)
        self.assertIn("Matrix View", html)
        self.assertIn("data-tooltip", html)
        self.assertIn("trainingFilter", html)
        self.assertIn("testedFilter", html)
        self.assertIn("modelFilter", html)
        self.assertIn("sortMetric", html)
        self.assertIn("Winner", html)
        self.assertIn("Accuracy", html)
        self.assertIn("Phishing F1", html)
        self.assertIn("Confusion Matrices", html)

    def test_decision_payload_answers_backend_and_single_source_questions(self):
        metrics_frame = pd.DataFrame(
            [
                self.metric_row("phiusiil_main", "Linear SVM", "combined_test", 0.91, 0.95, 0.90),
                self.metric_row("phiusiil_main", "Random Forest", "combined_test", 0.89, 0.99, 0.88),
                self.metric_row("legitphish", "Random Forest", "combined_test", 0.88, 0.97, 0.86),
                self.metric_row("combined_dataset", "Random Forest", "combined_test", 0.975, 0.973, 0.974),
                self.metric_row("combined_dataset", "XGBoost", "combined_test", 0.98, 0.971, 0.976),
                self.metric_row("combined_dataset", "XGBoost", "phiusiil_main_test", 0.99, 0.98, 0.99),
                self.metric_row("combined_dataset", "XGBoost", "legitphish_test", 0.99, 0.99, 0.99),
                self.metric_row("combined_dataset", "XGBoost", "phishstorm_test", 0.92, 0.89, 0.91),
                self.metric_row("phiusiil_main", "Linear SVM", "phishstorm_test", 0.67, 0.82, 0.59),
            ]
        )

        decision = evaluation.build_decision_payload(metrics_frame)

        self.assertEqual(
            decision["best_single_source_on_combined_test"]["training_scenario"],
            "phiusiil_main",
        )
        self.assertEqual(
            decision["best_single_source_on_combined_test"]["model"],
            "Linear SVM",
        )
        self.assertEqual(
            decision["best_overall_on_combined_test"]["training_scenario"],
            "combined_dataset",
        )
        self.assertEqual(decision["best_overall_on_combined_test"]["model"], "XGBoost")
        self.assertEqual(decision["backend_recommendation"]["model"], "XGBoost")
        self.assertEqual(
            decision["backend_recommendation"]["training_scenario"],
            "combined_dataset",
        )
        self.assertIn("combined_dataset", decision["backend_recommendation"]["reason"])
        self.assertEqual(
            decision["best_training_dataset"]["training_scenario"],
            "combined_dataset",
        )

    def test_html_report_is_decision_first_collapsible_and_uses_visible_tooltips(self):
        metrics_frame = pd.DataFrame(
            [
                self.metric_row("phiusiil_main", "Linear SVM", "combined_test", 0.91, 0.95, 0.90),
                self.metric_row("combined_dataset", "XGBoost", "combined_test", 0.98, 0.971, 0.976),
                self.metric_row("combined_dataset", "XGBoost", "phiusiil_main_test", 0.99, 0.98, 0.99),
            ]
        )

        html = evaluation.build_html_report(
            metrics_frame=metrics_frame,
            confusion_matrices={},
            generated_at="2099-01-01 09:30 Europe/London",
        )

        self.assertIn('id="decision-dashboard"', html)
        self.assertIn("Backend Recommendation", html)
        self.assertIn("Best Single-Source on Combined Test", html)
        self.assertIn("Best Training Dataset", html)
        self.assertLess(html.index('id="filter-results"'), html.index('id="research-questions"'))
        self.assertIn('<details class="report-section"', html)
        self.assertIn('<summary>Research Questions Answered</summary>', html)
        self.assertIn('class="tooltip-text"', html)
        self.assertIn("2099-01-01 09:30 Europe/London", html)
        self.assertNotIn("2026-06-26 06:16", html)

    def test_html_report_is_model_first_and_explains_rows_columns_and_metrics(self):
        metrics_frame = pd.DataFrame(
            [
                self.metric_row("phiusiil_main", "XGBoost", "phiusiil_main_test", 0.99, 0.98, 0.99),
                self.metric_row("phiusiil_main", "XGBoost", "combined_test", 0.89, 0.92, 0.88),
                self.metric_row("combined_dataset", "XGBoost", "phiusiil_main_test", 0.98, 0.97, 0.98),
                self.metric_row("combined_dataset", "XGBoost", "combined_test", 0.97, 0.96, 0.97),
                self.metric_row("phiusiil_main", "Linear SVM", "combined_test", 0.91, 0.95, 0.90),
                self.metric_row("combined_dataset", "Linear SVM", "combined_test", 0.92, 0.93, 0.91),
            ]
        )

        html = evaluation.build_html_report(
            metrics_frame=metrics_frame,
            confusion_matrices={},
            generated_at="2099-01-01 09:30 Europe/London",
        )

        self.assertIn('id="how-to-read-section"', html)
        self.assertIn("How to Read This Report", html)
        self.assertIn("Rows in each model matrix are training scenarios", html)
        self.assertIn("Columns are test scenarios", html)
        self.assertIn("Rows Tested", html)
        self.assertIn("URL Overlap Removed", html)
        self.assertIn("Single-source training", html)
        self.assertIn("Combined training", html)
        self.assertIn('id="per-model-results-section"', html)
        self.assertIn("Per Model Results", html)
        self.assertIn("XGBoost: Training Scenario vs Test Scenario", html)
        self.assertIn("Linear SVM: Training Scenario vs Test Scenario", html)
        self.assertIn("Trained On", html)
        self.assertIn("Combined Test", html)
        self.assertIn("F1 0.9700", html)
        self.assertIn('id="model-comparison-section"', html)
        self.assertIn("Model Comparison", html)
        self.assertIn("Best Model by Training Scenario on Combined Test", html)
        self.assertLess(
            html.index('id="per-model-results-section"'),
            html.index('id="model-summary-section"'),
        )

    def test_html_report_explains_backend_recommendation_vs_single_source_winner(self):
        metrics_frame = pd.DataFrame(
            [
                self.metric_row("phiusiil_main", "Linear SVM", "combined_test", 0.9070, 0.9524, 0.8953),
                self.metric_row("phiusiil_main", "Linear SVM", "phiusiil_main_test", 0.93, 0.94, 0.92),
                self.metric_row("combined_dataset", "XGBoost", "combined_test", 0.9780, 0.9711, 0.9748),
                self.metric_row("combined_dataset", "XGBoost", "phiusiil_main_test", 0.9947, 0.9899, 0.9948),
                self.metric_row("combined_dataset", "XGBoost", "legitphish_test", 0.9978, 0.9993, 0.9960),
                self.metric_row("combined_dataset", "XGBoost", "phishstorm_test", 0.9144, 0.8929, 0.9175),
            ]
        )

        html = evaluation.build_html_report(
            metrics_frame=metrics_frame,
            confusion_matrices={},
            generated_at="2099-01-01 09:30 Europe/London",
        )

        self.assertIn("Why the backend recommendation is not the single-source winner", html)
        self.assertIn("Linear SVM trained on PhiUSIIL answers a narrower question", html)
        self.assertIn("XGBoost trained on the Combined Dataset is recommended", html)
        self.assertIn("more diverse training rows", html)
        self.assertIn("mean clean-holdout F1", html)
        self.assertIn("worst clean-holdout F1", html)
        self.assertIn("Combined Test F1", html)

    def test_html_report_defaults_to_clean_holdout_all_models_and_phishing_f1(self):
        metrics_frame = pd.DataFrame(
            [
                self.metric_row("phiusiil_main", "Linear SVM", "combined_test", 0.9070, 0.9524, 0.8953),
                self.metric_row("combined_dataset", "XGBoost", "combined_test", 0.9780, 0.9711, 0.9748),
            ]
        )

        html = evaluation.build_html_report(
            metrics_frame=metrics_frame,
            confusion_matrices={},
            generated_at="2099-01-01 09:30 Europe/London",
        )

        self.assertIn('"Clean holdout only"', html)
        self.assertIn('controls.scope.value = "clean_holdout";', html)
        self.assertIn('controls.training.value = "all";', html)
        self.assertIn('controls.tested.value = "all";', html)
        self.assertIn('controls.model.value = "all";', html)
        self.assertIn('controls.metric.value = "phishing_f1";', html)
        self.assertIn('applyPreset("clean");', html)

    def feature_row(self, source_dataset, source_row_id, label):
        row = {
            "source_dataset": source_dataset,
            "source_row_id": source_row_id,
            "label": label,
            "url_normalized": f"https://{source_dataset}.example/{source_row_id}",
        }
        for column in config.FEATURE_COLUMNS:
            row[column] = 0
        row["url_length"] = source_row_id + 10
        row["domain_length"] = 5
        row["tld_length"] = 3
        return row

    def metric_row(
        self,
        training_scenario,
        model,
        test_dataset,
        phishing_f1,
        phishing_recall,
        accuracy,
        evaluation_scope="clean_holdout",
    ):
        return {
            "training_scenario": training_scenario,
            "model": model,
            "test_dataset": test_dataset,
            "rows_tested": 100,
            "accuracy": accuracy,
            "phishing_precision": phishing_f1,
            "phishing_recall": phishing_recall,
            "phishing_f1": phishing_f1,
            "model_file": f"{model}.joblib",
            "train_seconds": 1.0,
            "predict_seconds": 0.01,
            "evaluation_scope": evaluation_scope,
            "contains_training_rows": evaluation_scope == "complete_dataset",
            "training_url_overlap_removed": 0,
            "row_source_note": "Synthetic test row.",
        }


if __name__ == "__main__":
    unittest.main()
