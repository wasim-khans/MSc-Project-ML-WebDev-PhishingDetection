from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    url: str = Field(..., min_length=1, description="URL string to score.")
    model_id: str | None = Field(
        default=None,
        description="Optional backend model id selected from /api/v1/models.",
    )


class PredictionResponse(BaseModel):
    url: str
    predicted_label: int
    predicted_class: str
    confidence: float | None = None
    model_id: str
    model_name: str
    training_scenario: str
    feature_values: dict[str, int]


class ModelInfoResponse(BaseModel):
    model_id: str
    model_name: str
    training_scenario: str
    model_file: str
    feature_columns: list[str]
    label_mapping: dict[int, str]


class ModelOption(BaseModel):
    model_id: str
    model_name: str
    training_scenario: str
    model_file: str
    train_rows: int | None = None
    source_datasets: list[str] = Field(default_factory=list)
    is_recommended: bool = False


class ModelOptionsResponse(BaseModel):
    default_model_id: str
    models: list[ModelOption]


class HealthResponse(BaseModel):
    status: str
    model_id: str
    model_name: str
    training_scenario: str
