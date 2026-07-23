"""Inspect the raw PhiUSIIL dataset and write a concise dataset report.

This script confirms the URL/label columns, label distribution, and duplicate
URL situation before any feature extraction or model training happens.
"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.scripts.helpers.project_paths import (
    EXPERIMENT_1A_DIR,
    MAIN_RAW_DATASET_PATH,
)

DATASET_PATH = MAIN_RAW_DATASET_PATH
REPORT_PATH = EXPERIMENT_1A_DIR / "dataset_inspection.md"


def main() -> None:
    print("Step 1: Inspect main PhiUSIIL dataset")
    print(f"Source: {DATASET_PATH.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(DATASET_PATH)

    required_columns = ["URL", "label"]
    missing_required = [column for column in required_columns if column not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    label_counts = df["label"].value_counts(dropna=False).sort_index()
    url_label_missing = df[required_columns].isna().sum()
    url_normalized = df["URL"].dropna().astype(str).str.strip()
    duplicate_url_rows = int(len(url_normalized) - url_normalized.nunique())
    duplicate_urls = int(url_normalized.value_counts().gt(1).sum())
    conflicting_duplicate_urls = int(
        df.dropna(subset=required_columns)
        .assign(url_normalized=lambda frame: frame["URL"].astype(str).str.strip())
        .groupby("url_normalized")["label"]
        .nunique()
        .gt(1)
        .sum()
    )

    report = [
        "# Dataset Inspection Report",
        "",
        "## Source File",
        "",
        f"`{DATASET_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## Shape",
        "",
        f"- Rows: {df.shape[0]:,}",
        f"- Columns: {df.shape[1]:,}",
        "",
        "## Confirmed Project Columns",
        "",
        "- URL column: `URL`",
        "- Label column: `label`",
        "- Label meaning: `1` = legitimate, `0` = phishing",
        "",
        "## Label Distribution",
        "",
        "| Label | Meaning | Rows |",
        "|---:|---|---:|",
        f"| 0 | phishing | {int(label_counts.get(0, 0)):,} |",
        f"| 1 | legitimate | {int(label_counts.get(1, 0)):,} |",
        "",
        "## Missing Values in Project Columns",
        "",
        "| Column | Missing values |",
        "|---|---:|",
        f"| URL | {int(url_label_missing['URL']):,} |",
        f"| label | {int(url_label_missing['label']):,} |",
        "",
        "## Exact URL Duplicate Check",
        "",
        "| Item | Count |",
        "|---|---:|",
        f"| Duplicate URL rows after whitespace trimming | {duplicate_url_rows:,} |",
        f"| URLs that appear more than once | {duplicate_urls:,} |",
        f"| Duplicate URLs with conflicting labels | {conflicting_duplicate_urls:,} |",
        "",
        "The generated processed dataset removes exact duplicate URLs only when their labels agree. The full cleaning rule is documented in `docs/dataset_cleaning.md`.",
        "",
        "## All Columns",
        "",
        "```text",
        "\n".join(df.columns),
        "```",
        "",
        "## First Five URL/Label Rows",
        "",
        "```text",
        df[required_columns].head(5).to_string(index=False),
        "```",
        "",
        "## Project Decision",
        "",
        "Use only `URL` and `label` from the raw dataset, then generate fresh URL-only lexical features in project code.",
        "",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")
    print(f"Label counts: {label_counts.to_dict()} (0=phishing, 1=legitimate)")
    print(f"Missing URL values: {int(url_label_missing['URL']):,}")
    print(f"Missing label values: {int(url_label_missing['label']):,}")
    print(f"Duplicate URL rows after trimming: {duplicate_url_rows:,}")
    print(f"Conflicting duplicate URLs: {conflicting_duplicate_urls:,}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
