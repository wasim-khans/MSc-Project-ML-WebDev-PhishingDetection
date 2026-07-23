import json
import tempfile
import unittest
from pathlib import Path

import joblib

from machine_learning.core.url_feature_extractor import FEATURE_NAMES

from backend.app.repositories.model_repository import ModelRepository
from backend.app.services.prediction_service import PredictionService


class DummyClassifier:
    def predict(self, frame):
        return [0 if int(frame.iloc[0]["has_at_symbol"]) == 1 else 1]

    def predict_proba(self, frame):
        if int(frame.iloc[0]["has_at_symbol"]) == 1:
            return [[0.93, 0.07]]
        return [[0.12, 0.88]]


class PredictionServiceTests(unittest.TestCase):
    def test_prediction_service_reads_backend_candidate_from_generated_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            summary_path, metadata_path = build_backend_artifacts(project_root)

            repository = ModelRepository(
                project_root=project_root,
                summary_path=summary_path,
                model_metadata_path=metadata_path,
            )
            service = PredictionService(repository)

            model_info = service.get_model_info()

            self.assertEqual(model_info["model_name"], "XGBoost")
            self.assertEqual(model_info["training_scenario"], "combined_dataset")
            self.assertEqual(model_info["label_mapping"][0], "phishing")
            self.assertTrue(model_info["model_file"].endswith(".joblib"))

    def test_prediction_service_returns_prediction_features_and_confidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            summary_path, metadata_path = build_backend_artifacts(project_root)

            repository = ModelRepository(
                project_root=project_root,
                summary_path=summary_path,
                model_metadata_path=metadata_path,
            )
            service = PredictionService(repository)

            result = service.predict_url(
                "https://user@example.test/verify/account?next=home"
            )

            self.assertEqual(result["predicted_label"], 0)
            self.assertEqual(result["predicted_class"], "phishing")
            self.assertAlmostEqual(result["confidence"], 0.93)
            self.assertEqual(result["feature_values"]["has_at_symbol"], 1)
            self.assertEqual(result["model_name"], "XGBoost")


def build_backend_artifacts(project_root: Path):
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "dummy_model.joblib"
    joblib.dump(DummyClassifier(), model_path)

    summary_path = project_root / "summary.json"
    summary_payload = {
        "decision_summary": {
            "backend_recommendation": {
                "winner_model": "XGBoost",
                "best_trained_on": "combined_dataset",
                "best_model_file": str(model_path.relative_to(project_root)),
            }
        }
    }
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

    metadata_path = project_root / "metadata.json"
    metadata_payload = {
        "feature_columns": FEATURE_NAMES,
        "label_mapping": {"0": "phishing", "1": "legitimate"},
    }
    metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")
    return summary_path, metadata_path


if __name__ == "__main__":
    unittest.main()
