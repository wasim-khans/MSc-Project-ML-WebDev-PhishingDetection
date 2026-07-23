import unittest
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Using `httpx` with `starlette\.testclient` is deprecated; install `httpx2` instead\.",
    category=Warning,
)

from fastapi.testclient import TestClient

from backend.app.core.dependencies import get_prediction_service
from backend.app.main import create_app


class FakePredictionService:
    def get_model_info(self, model_id=None):
        return {
            "model_id": model_id or "combined_dataset__xgboost",
            "model_name": "XGBoost",
            "training_scenario": "combined_dataset",
            "model_file": "machine_learning/trained_models/2_cross_dataset_generalisation/xgboost_T_ON_combined_dataset_80_20.joblib",
            "feature_columns": ["url_length", "has_https"],
            "label_mapping": {0: "phishing", 1: "legitimate"},
        }

    def list_model_options(self):
        return {
            "default_model_id": "combined_dataset__xgboost",
            "models": [
                {
                    "model_id": "combined_dataset__xgboost",
                    "model_name": "XGBoost",
                    "training_scenario": "combined_dataset",
                    "model_file": "machine_learning/trained_models/2_cross_dataset_generalisation/xgboost_T_ON_combined_dataset_80_20.joblib",
                    "train_rows": 100,
                    "source_datasets": ["main", "legitphish", "phishstorm"],
                    "is_recommended": True,
                },
                {
                    "model_id": "combined_dataset__logistic_regression",
                    "model_name": "Logistic Regression",
                    "training_scenario": "combined_dataset",
                    "model_file": "machine_learning/trained_models/2_cross_dataset_generalisation/logistic_regression_T_ON_combined_dataset_80_20.joblib",
                    "train_rows": 100,
                    "source_datasets": ["main", "legitphish", "phishstorm"],
                    "is_recommended": False,
                },
            ],
        }

    def predict_url(self, url, model_id=None):
        return {
            "url": url,
            "predicted_label": 1,
            "predicted_class": "legitimate",
            "confidence": 0.88,
            "model_id": model_id or "combined_dataset__xgboost",
            "model_name": "XGBoost",
            "training_scenario": "combined_dataset",
            "feature_values": {"url_length": len(url), "has_https": 1},
        }


class PredictionApiTests(unittest.TestCase):
    def create_client(self):
        app = create_app()
        app.dependency_overrides[get_prediction_service] = lambda: FakePredictionService()
        return TestClient(app)

    def test_predict_endpoint_returns_prediction_payload(self):
        client = self.create_client()

        response = client.post("/api/v1/predict", json={"url": "https://example.test"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["predicted_class"], "legitimate")
        self.assertEqual(payload["model_id"], "combined_dataset__xgboost")
        self.assertEqual(payload["model_name"], "XGBoost")
        self.assertIn("feature_values", payload)

    def test_predict_endpoint_accepts_selected_model_id(self):
        client = self.create_client()

        response = client.post(
            "/api/v1/predict",
            json={
                "url": "https://example.test",
                "model_id": "combined_dataset__logistic_regression",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["model_id"],
            "combined_dataset__logistic_regression",
        )

    def test_model_info_endpoint_returns_selected_model(self):
        client = self.create_client()

        response = client.get("/api/v1/model-info")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["training_scenario"], "combined_dataset")
        self.assertEqual(payload["model_id"], "combined_dataset__xgboost")
        self.assertEqual(payload["label_mapping"]["0"], "phishing")

    def test_models_endpoint_returns_available_models(self):
        client = self.create_client()

        response = client.get("/api/v1/models")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["default_model_id"], "combined_dataset__xgboost")
        self.assertEqual(len(payload["models"]), 2)
        self.assertTrue(payload["models"][0]["is_recommended"])

    def test_predict_endpoint_rejects_blank_url(self):
        client = self.create_client()

        response = client.post("/api/v1/predict", json={"url": ""})

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
