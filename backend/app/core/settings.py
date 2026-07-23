import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_title: str
    api_version: str
    api_prefix: str
    project_root: Path
    summary_path: Path
    model_metadata_path: Path


@lru_cache
def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[3]
    summary_path = _resolve_project_path(
        project_root,
        os.getenv(
            "BACKEND_SUMMARY_PATH",
            "machine_learning/all_experiments_combined_report.json",
        ),
    )
    model_metadata_path = _resolve_project_path(
        project_root,
        os.getenv(
            "BACKEND_MODEL_METADATA_PATH",
            "machine_learning/trained_models/2_cross_dataset_generalisation/cross_dataset_model_metadata.json",
        ),
    )
    return Settings(
        api_title="Phishing URL Detection API",
        api_version="1.0.0",
        api_prefix="/api/v1",
        project_root=project_root,
        summary_path=summary_path,
        model_metadata_path=model_metadata_path,
    )


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path
