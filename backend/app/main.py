from fastapi import FastAPI

from backend.app.controllers.prediction_controller import router as prediction_router
from backend.app.core.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="Backend API for the phishing URL detection dissertation prototype.",
    )
    app.include_router(prediction_router, prefix=settings.api_prefix)
    return app


app = create_app()
