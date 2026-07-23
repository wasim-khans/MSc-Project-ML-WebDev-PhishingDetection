from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.dependencies import get_prediction_service
from backend.app.schemas.prediction import (
    HealthResponse,
    ModelInfoResponse,
    ModelOptionsResponse,
    PredictionRequest,
    PredictionResponse,
)
from backend.app.services.prediction_service import PredictionService


router = APIRouter(tags=["prediction"])


@router.get("/health", response_model=HealthResponse)
def health_check(
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    model_info = prediction_service.get_model_info()
    return {
        "status": "ok",
        "model_id": model_info["model_id"],
        "model_name": model_info["model_name"],
        "training_scenario": model_info["training_scenario"],
    }


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(
    model_id: str | None = None,
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    try:
        return prediction_service.get_model_info(model_id=model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/models", response_model=ModelOptionsResponse)
def model_options(
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    return prediction_service.list_model_options()


@router.post("/predict", response_model=PredictionResponse)
def predict_url(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    try:
        return prediction_service.predict_url(request.url, model_id=request.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
