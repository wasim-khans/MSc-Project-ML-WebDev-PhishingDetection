"""Train all configured models for the cross-dataset experiment matrix.

This script loads the saved train splits, trains one fresh model per training
scenario, and saves the resulting joblib files and metadata.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.scripts.helpers.cross_dataset_model_training import (
    train_cross_dataset_models,
)


if __name__ == "__main__":
    train_cross_dataset_models()
