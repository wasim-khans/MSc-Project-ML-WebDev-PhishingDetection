import tempfile
import unittest
from pathlib import Path

from machine_learning.scripts.helpers.interactive_workflow import (
    default_output_name,
    evaluate_model_file,
    load_dataset,
    normalise_joblib_filename,
    train_and_save_model,
)


class InteractiveWorkflowTests(unittest.TestCase):
    def test_normalise_joblib_filename_appends_extension(self):
        self.assertEqual(normalise_joblib_filename("my_model"), "my_model.joblib")
        self.assertEqual(
            normalise_joblib_filename("my_model.joblib"),
            "my_model.joblib",
        )

    def test_default_output_name_uses_model_and_dataset(self):
        self.assertEqual(
            default_output_name("Random Forest", "custom_dataset"),
            "random_forest_T_ON_custom_dataset_interactive.joblib",
        )

    def test_load_dataset_extracts_features_from_url_and_label_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "urls.csv"
            dataset_path.write_text(
                "url,label\n"
                "https://www.google.com,1\n"
                "http://192.168.1.10/bank-update,0\n",
                encoding="utf-8",
            )

            features, labels = load_dataset(dataset_path)

            self.assertEqual(len(features), 2)
            self.assertIn("url_length", features.columns)
            self.assertEqual(labels.tolist(), [1, 0])

    def test_train_and_test_decision_tree_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "features.csv"
            dataset_path.write_text(
                "url_length,domain_length,path_length,dot_count,hyphen_count,"
                "digit_count,special_char_count,has_https,has_ip_address,"
                "has_at_symbol,subdomain_count,query_param_count,"
                "suspicious_word_count,tld_length,has_url_shortener,label\n"
                "20,10,0,1,0,0,2,1,0,0,0,0,0,3,0,1\n"
                "21,10,0,1,0,0,2,1,0,0,0,0,0,3,0,1\n"
                "60,30,15,4,2,5,8,0,1,1,2,3,2,3,0,0\n"
                "62,31,15,4,2,5,8,0,1,1,2,3,2,3,0,0\n",
                encoding="utf-8",
            )

            result = train_and_save_model(
                model_name="Decision Tree",
                dataset_path=dataset_path,
                output_name="unit_test_model.joblib",
                dataset_name="unit_test_dataset",
                output_dir=Path(temp_dir) / "models",
            )
            evaluation = evaluate_model_file(
                model_path=result["model_path"],
                dataset_path=dataset_path,
            )

            self.assertTrue(result["model_path"].exists())
            self.assertEqual(result["rows_trained"], 4)
            self.assertEqual(evaluation["rows_tested"], 4)
            self.assertGreaterEqual(evaluation["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
