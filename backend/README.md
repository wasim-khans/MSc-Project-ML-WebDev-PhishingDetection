# Backend API

FastAPI backend for the phishing URL detection prototype.

## Structure

- `backend/app/main.py`: FastAPI application factory.
- `backend/app/controllers/`: HTTP endpoints for health, model info, and prediction.
- `backend/app/services/`: prediction logic that turns one URL into one model result.
- `backend/app/repositories/`: loads the selected model, feature columns, and label mapping from project artefacts.
- `backend/app/schemas/`: request and response models.
- `backend/app/core/`: settings and dependency wiring.
- `backend/tests/`: API and service tests.

## Model Selection

The backend does not hardcode the chosen model file. It reads the current backend recommendation from:

- `machine_learning/all_experiments_combined_report.json`

It reads feature columns and label mapping from:

- `machine_learning/trained_models/2_cross_dataset_generalisation/cross_dataset_model_metadata.json`

This keeps the backend aligned with the dissertation evidence and the current best model selection.

## Endpoints

- `GET /api/v1/health`
- `GET /api/v1/model-info`
- `GET /api/v1/models`
- `POST /api/v1/predict`

Example request body:

```json
{
  "url": "https://example.com/login",
  "model_id": "combined_dataset__xgboost"
}
```

`model_id` is optional. If it is omitted, the backend uses the current model recommended by the dissertation experiment summary. Use `GET /api/v1/models` to list valid ids for switching models.

## Run

First-time setup from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cd frontend
npm install
cd ..
```

Run backend only:

```bash
source .venv/bin/activate
BACKEND_PORT=7330 python3 backend/run.py
```

Run the full app:

```bash
./run.sh
```
