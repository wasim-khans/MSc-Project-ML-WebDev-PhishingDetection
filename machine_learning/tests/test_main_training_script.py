import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


def load_train_script_module():
    machine_learning_root = Path(__file__).resolve().parents[1]
    script_path = machine_learning_root / "scripts" / "1a_4_train_main_models.py"
    spec = importlib.util.spec_from_file_location("train_models_script", script_path)
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


class TrainScriptTests(unittest.TestCase):
    def test_train_script_imports_project_modules_when_loaded_by_path(self):
        machine_learning_root = Path(__file__).resolve().parents[1]
        project_root = Path(__file__).resolve().parents[2]
        script_path = machine_learning_root / "scripts" / "1a_4_train_main_models.py"
        original_path = sys.path[:]

        try:
            sys.path = [
                str(machine_learning_root / "scripts"),
                *[
                    entry
                    for entry in sys.path
                    if entry and Path(entry).resolve() != project_root
                ],
            ]
            spec = importlib.util.spec_from_file_location("train_models_script", script_path)
            module = importlib.util.module_from_spec(spec)

            spec.loader.exec_module(module)

            self.assertTrue(hasattr(module, "main"))
        finally:
            sys.path = original_path

    def test_model_artifact_filename_is_stable_and_readable(self):
        module = load_train_script_module()

        self.assertEqual(
            module.model_artifact_filename("Logistic Regression"),
            "logistic_regression_T_ON_phiusiil_main_80_20.joblib",
        )
        self.assertEqual(
            module.model_artifact_filename("Linear SVM"),
            "linear_svm_T_ON_phiusiil_main_80_20.joblib",
        )
        self.assertEqual(
            module.model_artifact_filename("XGBoost"),
            "xgboost_T_ON_phiusiil_main_80_20.joblib",
        )
        self.assertEqual(
            module.best_model_artifact_filename("XGBoost"),
            "best_model_xgboost_T_ON_phiusiil_main_80_20.joblib",
        )

    def test_training_config_defaults_are_loaded_from_json(self):
        module = load_train_script_module()

        config = module.load_training_config()

        self.assertEqual(config["dataset_name"], "phiusiil_main")
        self.assertEqual(config["test_size"], 0.2)
        self.assertEqual(config["models"], "all")

    def test_markdown_report_includes_timing_and_all_saved_models(self):
        module = load_train_script_module()
        metrics_frame = pd.DataFrame(
            [
                {
                    "model": "Random Forest",
                    "accuracy": 0.99,
                    "phishing_precision": 0.98,
                    "phishing_recall": 0.97,
                    "phishing_f1": 0.975,
                    "train_seconds": 0.62,
                    "predict_seconds": 0.03,
                }
            ]
        )
        metadata = {
            "best_model": "Random Forest",
            "selection_rule": "Highest phishing_f1, then phishing_recall, then accuracy.",
            "dataset": "machine_learning/datasets/main/processed/phishing_url_features.csv",
            "test_size": 0.2,
            "random_state": 42,
            "best_model_file": (
                "machine_learning/trained_models/1a_train_on_main_test_on_main/"
                "best_model_xgboost.joblib"
            ),
            "saved_model_files": {
                "Random Forest": (
                    "machine_learning/trained_models/1a_train_on_main_test_on_main/"
                    "random_forest.joblib"
                ),
            },
        }
        confusion_matrices = {
            "Random Forest": {
                "matrix": [[10, 1], [2, 20]],
            }
        }

        report = module.build_markdown_report(
            metrics_frame=metrics_frame,
            confusion_matrices=confusion_matrices,
            metadata=metadata,
        )

        self.assertIn("Training Time", report)
        self.assertIn("Testing Phase", report)
        self.assertIn("0.6200s", report)
        self.assertIn(
            "`machine_learning/trained_models/1a_train_on_main_test_on_main/random_forest.joblib`",
            report,
        )
        self.assertIn(
            "`machine_learning/trained_models/1a_train_on_main_test_on_main/best_model_xgboost.joblib`",
            report,
        )
        self.assertNotIn("best_model.joblib", report)


if __name__ == "__main__":
    unittest.main()
