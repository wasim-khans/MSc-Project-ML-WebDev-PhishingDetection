import math

import pandas as pd

from machine_learning.scripts.core.url_feature_extractor import extract_features

from backend.app.repositories.model_repository import ModelRepository


class PredictionService:
    """Coordinate feature extraction and model inference for one URL."""

    def __init__(self, repository: ModelRepository):
        self.repository = repository

    def get_model_info(self, model_id: str | None = None) -> dict:
        assets = self.repository.load_prediction_assets(model_id=model_id)
        return {
            "model_id": assets.model_id,
            "model_name": assets.model_name,
            "training_scenario": assets.training_scenario,
            "model_file": assets.model_file,
            "feature_columns": assets.feature_columns,
            "label_mapping": assets.label_mapping,
        }

    def list_model_options(self) -> dict:
        return self.repository.list_model_options()

    def predict_url(self, url: str, model_id: str | None = None) -> dict:
        cleaned_url = str(url).strip()
        if not cleaned_url:
            raise ValueError("URL must not be blank.")

        assets = self.repository.load_prediction_assets(model_id=model_id)
        feature_values = extract_features(cleaned_url)
        feature_frame = pd.DataFrame(
            [[feature_values[column] for column in assets.feature_columns]],
            columns=assets.feature_columns,
        )
        predicted_label = int(assets.model.predict(feature_frame)[0])
        confidence = self._prediction_confidence(
            model=assets.model,
            feature_frame=feature_frame,
            predicted_label=predicted_label,
        )
        return {
            "url": cleaned_url,
            "predicted_label": predicted_label,
            "predicted_class": assets.label_mapping[predicted_label],
            "confidence": confidence,
            "model_id": assets.model_id,
            "model_name": assets.model_name,
            "training_scenario": assets.training_scenario,
            "feature_values": feature_values,
        }

    def _prediction_confidence(self, model, feature_frame, predicted_label: int):
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(feature_frame)[0]
            classes = list(getattr(model, "classes_", []))
            if predicted_label in classes:
                return float(probabilities[classes.index(predicted_label)])
            return float(max(probabilities))
        if hasattr(model, "decision_function"):
            decision = model.decision_function(feature_frame)
            if hasattr(decision, "__len__"):
                decision_value = decision[0]
                if hasattr(decision_value, "__len__"):
                    decision_value = max(abs(value) for value in decision_value)
            else:
                decision_value = decision
            return float(1.0 / (1.0 + math.exp(-abs(float(decision_value)))))
        return None
