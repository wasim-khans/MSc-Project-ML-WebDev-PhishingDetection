# Frontend

React + Vite + Tailwind frontend for the phishing URL detection dissertation prototype.

## Structure

- `src/pages/`: top-level page views.
- `src/components/`: reusable UI sections for the analyzer screen.
- `src/services/`: API calls to the FastAPI backend.
- `src/hooks/`: stateful frontend logic for model loading and prediction requests.
- `src/router/`: route wiring.
- `src/lib/`: formatting helpers, feature metadata, and sample data.

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

Run backend and frontend together:

```bash
./run.sh
```

Run frontend only, assuming the backend is already running on `7330`:

```bash
cd frontend
VITE_PROXY_TARGET=http://127.0.0.1:7330 npm run dev -- --host 127.0.0.1 --port 7331
```

By default, the launcher uses:

- backend: `http://127.0.0.1:7330`
- frontend: `http://127.0.0.1:7331`

If those ports are occupied, it tries `9519` and `9520`. The Vite dev server proxies `/api/*` requests to the backend URL selected by the launcher, so the backend must be running for live predictions.

## Model Switching

The analyzer screen has an `Active Model` dropdown. It loads available models from:

- `GET /api/v1/models`

When you submit a URL, the selected `model_id` is sent to:

- `POST /api/v1/predict`

This is useful for comparing cases where the default recommended model produces a false positive.

The launcher starts:
- backend on `7330` and frontend on `7331` when free
- otherwise backend on `9519` and frontend on `9520`
- otherwise it frees `7330/7331` and starts there
