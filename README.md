# Phishing URL Detection

MSc Web Development dissertation project by Muhammad Wasim Khan.

This project compares machine-learning models for phishing URL detection using
URL-only lexical features. It also includes a FastAPI backend and React frontend
where a user can enter a URL and receive a phishing/legitimate prediction.

Models compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Linear SVM
- XGBoost

## Main Folders

- `backend/` - FastAPI API used by the frontend.
- `frontend/` - React, Vite, and Tailwind URL-input interface.
- `machine_learning/` - feature extraction, datasets structure, training,
  testing, experiment reports, and model metadata.
- `machine_learning/scripts/` - numbered scripts for reproducing the ML work.
- `machine_learning/experiments/` - experiment result summaries.
- `machine_learning/trained_models/` - model metadata and the small default
  model used by the backend.

## 1. Download and Place the Datasets

Raw datasets are not committed to Git. Download the three CSV files and place or
rename them exactly as shown.

| Dataset | Download source | Put the raw CSV here |
|---|---|---|
| PhiUSIIL | [UCI PhiUSIIL dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) | `machine_learning/datasets/main/raw/PhiUSIIL_Phishing_URL_Dataset.csv` |
| LegitPhish | [Mendeley LegitPhish dataset](https://data.mendeley.com/datasets/hx4m73v2sf/1) | `machine_learning/datasets/external_testing/legitphish/raw/LegitPhish_dataset_url_features_extracted1.csv` |
| PhishStorm | [Aalto PhishStorm dataset](https://research.aalto.fi/en/datasets/phishstorm-phishing-legitimate-url-dataset/) | `machine_learning/datasets/external_testing/phishstorm/raw/PhishStorm_urlset.csv` |

Project label rule:

```text
0 = phishing
1 = legitimate
```

Important note: PhishStorm uses the opposite label meaning in its original file.
The preparation script remaps it automatically.

## 2. Install Dependencies

Run this once from the project root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

cd frontend
npm install
cd ..
```

## 3. Build Datasets, Run Experiments, and Open the Report

Run these scripts from the project root, in this order.

1. Optional: `python3 machine_learning/scripts/1a_1_inspect_main_dataset.py` -
   inspects the main raw PhiUSIIL dataset for human checking.
2. Required: `python3 machine_learning/scripts/1a_2_build_main_feature_dataset.py` -
   builds the main processed URL-feature dataset.
3. Optional: `python3 machine_learning/scripts/1a_3_validate_main_feature_dataset.py` -
   validates extracted features against the source URLs.
4. Required: `python3 machine_learning/scripts/1b_1_prepare_external_testing_datasets.py` -
   prepares LegitPhish and PhishStorm in the project label format.
5. Required: `python3 machine_learning/scripts/2_1_build_cross_dataset_splits.py` -
   creates clean 80/20 train/test splits for all datasets.
6. Required: `python3 machine_learning/scripts/1a_4_train_main_models.py` -
   trains and tests models on the main dataset split.
7. Required: `python3 machine_learning/scripts/1b_2_test_main_models_on_external_datasets.py` -
   tests main-trained models on external datasets.
8. Required: `python3 machine_learning/scripts/2_2_train_cross_dataset_models.py` -
   trains models on each dataset and on the combined dataset.
9. Required: `python3 machine_learning/scripts/2_3_evaluate_cross_dataset_models.py` -
   evaluates trained models across held-out test datasets.
10. Optional reporting step: `python3 machine_learning/scripts/build_all_experiments_combined_report.py` -
    rebuilds the final combined HTML/JSON experiment report from saved experiment outputs.

Then open the generated report in a browser:

```text
machine_learning/all_experiments_combined_report.html
```

The report summarises model comparisons, cross-dataset tests, practical sanity
checks, and final model interpretation.

More workflow detail is available in:

```text
machine_learning/WORKFLOW.md
```

## 4. Run the Web App

Quick run:

```bash
./run.sh
```

The script starts both apps:

- Backend: `http://127.0.0.1:7330`
- Frontend: `http://127.0.0.1:7331`

If those ports are busy, it tries `9519` and `9520`. On first run, it may ask
permission to create `.venv`, install Python dependencies, and install frontend
packages.

Manual backend:

```bash
source .venv/bin/activate
BACKEND_PORT=7330 python3 backend/run.py
```

Manual frontend:

```bash
cd frontend
VITE_PROXY_TARGET=http://127.0.0.1:7330 npm run dev -- --host 127.0.0.1 --port 7331
```

Important note: the web app can run with the committed default model.
Reproducing all ML experiments requires the dataset setup above.

## Important Git Notes

Large local/generated files are intentionally not committed to Git:

- raw datasets
- processed datasets
- train/test split CSV files
- most trained `.joblib` model files
- dissertation drafts in `docs/`
- `.venv`, `node_modules`, cache files, and local generated clutter

The repository keeps the code, configuration, experiment reports, metadata, and
one small default trained model so the web app can run after setup.
