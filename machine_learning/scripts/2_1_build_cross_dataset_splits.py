"""Build the saved train/test split files for all active datasets.

This script creates the 80/20 splits used by the cross-dataset experiment and
saves dataset-specific plus combined split CSV files in one central location.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.scripts.helpers.cross_dataset_data_preparation import (
    build_cross_dataset_splits,
)


if __name__ == "__main__":
    build_cross_dataset_splits()
