"""Interactively train one chosen model on one chosen dataset."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.scripts.helpers.interactive_workflow import train_interactive_main


if __name__ == "__main__":
    try:
        train_interactive_main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
