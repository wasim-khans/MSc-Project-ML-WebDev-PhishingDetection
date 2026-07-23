from functools import lru_cache

from backend.app.core.settings import get_settings
from backend.app.repositories.model_repository import ModelRepository
from backend.app.services.prediction_service import PredictionService


@lru_cache
def get_model_repository() -> ModelRepository:
    settings = get_settings()
    return ModelRepository(
        project_root=settings.project_root,
        summary_path=settings.summary_path,
        model_metadata_path=settings.model_metadata_path,
    )


@lru_cache
def get_prediction_service() -> PredictionService:
    return PredictionService(get_model_repository())
