import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib


@dataclass(frozen=True)
class PredictionAssets:
    model_id: str
    model_name: str
    training_scenario: str
    model_file: str
    feature_columns: list[str]
    label_mapping: dict[int, str]
    model: object


class ModelRepository:
    """Load the backend-selected trained model and its supporting metadata."""

    def __init__(self, project_root: Path, summary_path: Path, model_metadata_path: Path):
        self.project_root = Path(project_root)
        self.summary_path = Path(summary_path)
        self.model_metadata_path = Path(model_metadata_path)
        self._cached_assets = {}

    def list_model_options(self) -> dict:
        metadata = self._read_json(self.model_metadata_path)
        backend = self._read_backend_recommendation()
        default_model_id = self._model_id(
            training_scenario=str(backend["best_trained_on"]),
            model_name=str(backend["winner_model"]),
        )

        models = []
        for row in metadata.get("models", []):
            model_name = str(row.get("model"))
            training_scenario = str(row.get("training_scenario"))
            model_id = self._model_id(
                training_scenario=training_scenario,
                model_name=model_name,
            )
            models.append(
                {
                    "model_id": model_id,
                    "model_name": model_name,
                    "training_scenario": training_scenario,
                    "model_file": str(row.get("model_file")),
                    "train_rows": row.get("train_rows"),
                    "source_datasets": list(row.get("source_datasets", [])),
                    "is_recommended": model_id == default_model_id,
                }
            )

        models.sort(
            key=lambda model: (
                not model["is_recommended"],
                str(model["training_scenario"]),
                str(model["model_name"]),
            )
        )

        return {
            "default_model_id": default_model_id,
            "models": models,
        }

    def load_prediction_assets(self, model_id: str | None = None) -> PredictionAssets:
        metadata = self._read_json(self.model_metadata_path)
        selected_model = self._select_model(metadata=metadata, model_id=model_id)
        selected_model_id = str(selected_model["model_id"])

        if selected_model_id in self._cached_assets:
            return self._cached_assets[selected_model_id]

        model_file = str(selected_model["model_file"])
        model_path = self._resolve_project_path(model_file)
        if not model_path.exists():
            raise FileNotFoundError(f"Missing trained model file: {model_path}")

        feature_columns = list(metadata["feature_columns"])
        label_mapping = {
            int(label): name for label, name in metadata["label_mapping"].items()
        }
        assets = PredictionAssets(
            model_id=selected_model_id,
            model_name=str(selected_model["model_name"]),
            training_scenario=str(selected_model["training_scenario"]),
            model_file=str(model_path.relative_to(self.project_root)),
            feature_columns=feature_columns,
            label_mapping=label_mapping,
            model=joblib.load(model_path),
        )
        self._cached_assets[selected_model_id] = assets
        return assets

    def _select_model(self, metadata: dict, model_id: str | None) -> dict:
        if model_id:
            for row in metadata.get("models", []):
                row_model_id = self._model_id(
                    training_scenario=str(row.get("training_scenario")),
                    model_name=str(row.get("model")),
                )
                if row_model_id == model_id:
                    return {
                        "model_id": row_model_id,
                        "model_name": str(row.get("model")),
                        "training_scenario": str(row.get("training_scenario")),
                        "model_file": str(row.get("model_file")),
                    }
            raise ValueError(f"Unknown model_id: {model_id}")

        backend = self._read_backend_recommendation()
        model_name = str(backend["winner_model"])
        training_scenario = str(backend["best_trained_on"])
        model_file = str(
            backend.get("best_model_file")
            or self._lookup_model_file(
                metadata=metadata,
                training_scenario=training_scenario,
                model_name=model_name,
            )
        )
        return {
            "model_id": self._model_id(
                training_scenario=training_scenario,
                model_name=model_name,
            ),
            "model_name": model_name,
            "training_scenario": training_scenario,
            "model_file": model_file,
        }

    def _read_backend_recommendation(self) -> dict:
        summary = self._read_json(self.summary_path)
        backend = summary.get("decision_summary", {}).get("backend_recommendation")
        if not backend:
            raise RuntimeError(
                "No backend recommendation found in dissertation_experiments_summary.json."
            )
        return backend

    def _model_id(self, training_scenario: str, model_name: str) -> str:
        return f"{self._slug(training_scenario)}__{self._slug(model_name)}"

    def _slug(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    def _lookup_model_file(self, metadata: dict, training_scenario: str, model_name: str) -> str:
        for row in metadata.get("models", []):
            if (
                str(row.get("training_scenario")) == training_scenario
                and str(row.get("model")) == model_name
            ):
                return str(row["model_file"])
        raise RuntimeError(
            "Could not find the backend-selected model file in cross_dataset_model_metadata.json."
        )

    def _read_json(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Missing JSON file: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _resolve_project_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path
