"""Evaluate all saved cross-dataset models against the configured test sets.

This script scores every trained model on the held-out and diagnostic datasets
and produces the experiment-matrix CSV plus detailed experiment reports.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.scripts.helpers.cross_dataset_evaluation import (
    evaluate_cross_dataset_models,
)


if __name__ == "__main__":
    evaluate_cross_dataset_models()
