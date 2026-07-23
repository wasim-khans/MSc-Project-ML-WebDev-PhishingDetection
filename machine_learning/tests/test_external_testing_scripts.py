import importlib.util
import unittest
from pathlib import Path

import pandas as pd


def load_script(script_name):
    machine_learning_root = Path(__file__).resolve().parents[1]
    script_path = machine_learning_root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExternalTestingPreparationTests(unittest.TestCase):
    def test_phishstorm_labels_are_mapped_to_project_label_meaning(self):
        module = load_script("1b_1_prepare_external_testing_datasets.py")

        self.assertEqual(module.map_phishstorm_label_to_project_label(1), 0)
        self.assertEqual(module.map_phishstorm_label_to_project_label(0), 1)

    def test_processed_external_features_keep_labels_in_a_separate_table(self):
        module = load_script("1b_1_prepare_external_testing_datasets.py")
        records = [
            {"domain": "https://secure.example.com/login", "label": 1},
            {"domain": "https://roehampton.ac.uk", "label": 0},
        ]

        features, labels, cleaning = module.build_phishstorm_processed_tables(records)

        self.assertIn("url_length", features.columns)
        self.assertNotIn("project_label", features.columns)
        self.assertEqual(list(labels["project_label"]), [0, 1])
        self.assertEqual(list(features["row_id"]), list(labels["row_id"]))
        self.assertEqual(cleaning["duplicate_rows_removed"], 0)

    def test_processed_external_features_deduplicate_exact_urls(self):
        module = load_script("1b_1_prepare_external_testing_datasets.py")
        records = [
            {"domain": "https://secure.example.com/login", "label": 1},
            {"domain": " https://secure.example.com/login ", "label": 1},
            {"domain": "https://roehampton.ac.uk", "label": 0},
        ]

        features, labels, cleaning = module.build_phishstorm_processed_tables(records)

        self.assertEqual(len(features), 2)
        self.assertEqual(len(labels), 2)
        self.assertEqual(cleaning["duplicate_rows_removed"], 1)
        self.assertIn("url_normalized", labels.columns)

    def test_active_dataset_choices_exclude_feature_only_iscx_dataset(self):
        module = load_script("1b_1_prepare_external_testing_datasets.py")

        self.assertEqual(module.DATASET_CHOICES, ["phishstorm", "legitphish"])
        self.assertNotIn("iscx_url_2016", module.normalise_dataset_selection("all"))


class ExternalModelEvaluationTests(unittest.TestCase):
    def test_model_selection_accepts_all_or_specific_model_names(self):
        module = load_script("1b_2_test_main_models_on_external_datasets.py")
        available = {
            "Random Forest": Path("random_forest.joblib"),
            "XGBoost": Path("xgboost.joblib"),
        }

        self.assertEqual(
            module.normalise_model_selection("all", available),
            ["Random Forest", "XGBoost"],
        )
        self.assertEqual(
            module.normalise_model_selection("xgboost", available),
            ["XGBoost"],
        )

    def test_model_selection_rejects_unknown_model_name(self):
        module = load_script("1b_2_test_main_models_on_external_datasets.py")

        with self.assertRaises(ValueError):
            module.normalise_model_selection(
                "unknown",
                {"Random Forest": Path("random_forest.joblib")},
            )

    def test_external_testing_config_defaults_are_loaded_from_json(self):
        module = load_script("1b_2_test_main_models_on_external_datasets.py")

        config = module.load_external_testing_config()

        self.assertEqual(config["models"], "all")
        self.assertEqual(config["datasets"], "all")

    def test_display_path_shows_project_local_config_paths_neatly(self):
        module = load_script("1b_2_test_main_models_on_external_datasets.py")

        self.assertEqual(
            module.display_path(
                module.PROJECT_ROOT
                / "machine_learning/configs/external_testing_config.json"
            ),
            Path("machine_learning/configs/external_testing_config.json"),
        )

    def test_display_name_from_context_model_slug_uses_model_part_only(self):
        module = load_script("1b_2_test_main_models_on_external_datasets.py")

        self.assertEqual(
            module.display_name_from_model_slug(
                "random_forest_T_ON_phiusiil_main_80_20"
            ),
            "Random Forest",
        )

    def test_model_summary_includes_average_and_robustness_metrics(self):
        module = load_script("1b_2_test_main_models_on_external_datasets.py")
        metrics_frame = pd.DataFrame(
            [
                {
                    "dataset": "legitphish",
                    "model": "Stable Model",
                    "rows_tested": 100,
                    "accuracy": 0.90,
                    "phishing_precision": 0.91,
                    "phishing_recall": 0.92,
                    "phishing_f1": 0.90,
                    "model_file": "stable.joblib",
                },
                {
                    "dataset": "phishstorm",
                    "model": "Stable Model",
                    "rows_tested": 100,
                    "accuracy": 0.88,
                    "phishing_precision": 0.89,
                    "phishing_recall": 0.90,
                    "phishing_f1": 0.88,
                    "model_file": "stable.joblib",
                },
                {
                    "dataset": "legitphish",
                    "model": "Swingy Model",
                    "rows_tested": 100,
                    "accuracy": 0.99,
                    "phishing_precision": 0.99,
                    "phishing_recall": 0.99,
                    "phishing_f1": 0.99,
                    "model_file": "swingy.joblib",
                },
                {
                    "dataset": "phishstorm",
                    "model": "Swingy Model",
                    "rows_tested": 100,
                    "accuracy": 0.50,
                    "phishing_precision": 0.50,
                    "phishing_recall": 1.00,
                    "phishing_f1": 0.66,
                    "model_file": "swingy.joblib",
                },
            ]
        )

        summary = module.build_model_summary(metrics_frame)

        self.assertEqual(set(summary["model"]), {"Stable Model", "Swingy Model"})
        stable_score = summary.loc[
            summary["model"] == "Stable Model", "robustness_score"
        ].iloc[0]
        swingy_score = summary.loc[
            summary["model"] == "Swingy Model", "robustness_score"
        ].iloc[0]
        self.assertGreater(stable_score, swingy_score)

    def test_html_report_shows_run_summary_and_per_model_sections(self):
        module = load_script("1b_2_test_main_models_on_external_datasets.py")
        metrics_frame = pd.DataFrame(
            [
                {
                    "dataset": "legitphish",
                    "model": "XGBoost",
                    "rows_tested": 101218,
                    "accuracy": 0.9938,
                    "phishing_precision": 0.9902,
                    "phishing_recall": 1.0,
                    "phishing_f1": 0.9951,
                    "model_file": (
                        "machine_learning/trained_models/1a_train_on_main_test_on_main/"
                        "xgboost.joblib"
                    ),
                },
                {
                    "dataset": "phishstorm",
                    "model": "XGBoost",
                    "rows_tested": 95912,
                    "accuracy": 0.4994,
                    "phishing_precision": 0.4994,
                    "phishing_recall": 1.0,
                    "phishing_f1": 0.6662,
                    "model_file": (
                        "machine_learning/trained_models/1a_train_on_main_test_on_main/"
                        "xgboost.joblib"
                    ),
                },
            ]
        )
        confusion_matrices = {
            "legitphish::XGBoost": {"matrix": [[10, 0], [1, 9]]},
            "phishstorm::XGBoost": {"matrix": [[8, 2], [7, 3]]},
        }
        main_metrics_frame = pd.DataFrame(
            [
                {
                    "model": "XGBoost",
                    "accuracy": 0.9952,
                    "phishing_precision": 0.9986,
                    "phishing_recall": 0.9902,
                    "phishing_f1": 0.9944,
                    "train_seconds": 0.5,
                    "predict_seconds": 0.01,
                },
            ]
        )
        main_confusion_matrices = {
            "XGBoost": {"matrix": [[19992, 197], [28, 26942]]},
        }

        html = module.build_external_html_report(
            metrics_frame=metrics_frame,
            confusion_matrices=confusion_matrices,
            main_metrics_frame=main_metrics_frame,
            main_confusion_matrices=main_confusion_matrices,
            generated_at="2026-06-25 12:00",
        )

        self.assertIn("External Testing Report", html)
        self.assertIn("1 model tested", html)
        self.assertIn("2 external datasets tested", html)
        self.assertIn("3 model-dataset evaluations", html)
        self.assertIn("Overall Model Summary", html)
        self.assertIn("Per Dataset Leaderboard", html)
        self.assertIn("Per Model Detail", html)
        self.assertIn("Confusion Matrices", html)
        self.assertIn("Robustness Score", html)
        self.assertIn("Choose Models", html)
        self.assertIn("Choose Datasets", html)
        self.assertIn("Summary View", html)
        self.assertIn("Detailed Explorer", html)
        self.assertIn("Dissertation Summary", html)
        self.assertIn("How to Read the Metrics", html)
        self.assertIn("Models Used", html)
        self.assertIn("simple baseline", html)
        self.assertIn("data-tooltip=\"Accuracy:", html)
        self.assertIn("data-tooltip=\"Recall:", html)
        self.assertIn("data-tooltip=\"Confusion matrix:", html)
        self.assertIn("chart-panel", html)
        self.assertIn("winner-badge", html)
        self.assertIn("renderSummaryCharts", html)
        self.assertIn("renderExplorerChart", html)
        self.assertIn("Generalisation Drop Chart", html)
        self.assertIn("PhiUSIIL held-out test split", html)
        self.assertIn('id="summaryExternalDatasetFilters"', html)
        self.assertIn('id="explorerModelFilters"', html)
        self.assertIn('id="explorerDatasetFilters"', html)
        self.assertIn("renderSummaryView", html)
        self.assertIn("renderExplorerView", html)
        self.assertIn("XGBoost", html)
        self.assertIn("phishstorm", html)

    def test_html_report_data_separates_main_and_external_datasets(self):
        module = load_script("1b_2_test_main_models_on_external_datasets.py")
        external_metrics = pd.DataFrame(
            [
                {
                    "dataset": "legitphish",
                    "model": "XGBoost",
                    "rows_tested": 101218,
                    "accuracy": 0.9938,
                    "phishing_precision": 0.9902,
                    "phishing_recall": 1.0,
                    "phishing_f1": 0.9951,
                    "model_file": (
                        "machine_learning/trained_models/1a_train_on_main_test_on_main/"
                        "xgboost.joblib"
                    ),
                },
            ]
        )
        main_metrics = pd.DataFrame(
            [
                {
                    "model": "XGBoost",
                    "accuracy": 0.9952,
                    "phishing_precision": 0.9986,
                    "phishing_recall": 0.9902,
                    "phishing_f1": 0.9944,
                    "train_seconds": 0.5,
                    "predict_seconds": 0.01,
                },
            ]
        )

        report_data = module.build_html_report_data(
            metrics_frame=external_metrics,
            confusion_matrices={},
            generated_at="2026-06-25 12:00",
            main_metrics_frame=main_metrics,
            main_confusion_matrices={},
        )

        dataset_types = {row["dataset"]: row["dataset_type"] for row in report_data["metrics"]}

        self.assertEqual(dataset_types["PhiUSIIL held-out test split"], "main")
        self.assertEqual(dataset_types["legitphish"], "external")
        self.assertEqual(report_data["main_dataset"], "PhiUSIIL held-out test split")
        self.assertEqual(report_data["external_datasets"], ["legitphish"])


if __name__ == "__main__":
    unittest.main()
