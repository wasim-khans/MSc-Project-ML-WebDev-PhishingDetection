import unittest

from machine_learning.scripts.core.model_artifacts import (
    best_model_artifact_filename,
    model_artifact_filename,
    split_suffix,
    slugify,
)


class ModelArtifactNamingTests(unittest.TestCase):
    def test_slugify_makes_names_stable_for_files(self):
        self.assertEqual(slugify("Linear SVM"), "linear_svm")
        self.assertEqual(slugify("PhiUSIIL main"), "phiusiil_main")

    def test_split_suffix_is_derived_from_test_size(self):
        self.assertEqual(split_suffix(0.2), "80_20")
        self.assertEqual(split_suffix(0.4), "60_40")
        self.assertEqual(split_suffix(0.25), "75_25")

    def test_model_artifact_filename_includes_model_dataset_and_split(self):
        self.assertEqual(
            model_artifact_filename("XGBoost", "PhiUSIIL main", 0.2),
            "xgboost_T_ON_phiusiil_main_80_20.joblib",
        )
        self.assertEqual(
            model_artifact_filename("Linear SVM", "combined_all", 0.4),
            "linear_svm_T_ON_combined_all_60_40.joblib",
        )

    def test_best_model_artifact_filename_keeps_same_training_context(self):
        self.assertEqual(
            best_model_artifact_filename("Random Forest", "PhiUSIIL main", 0.2),
            "best_model_random_forest_T_ON_phiusiil_main_80_20.joblib",
        )


if __name__ == "__main__":
    unittest.main()
