from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def normalise_url_for_deduplication(url) -> str:
    """Normalise only enough to catch exact URL duplicates safely."""
    if pd.isna(url):
        return ""
    return str(url).strip()


def deduplicate_url_rows(
    frame: pd.DataFrame,
    url_column: str,
    label_column: str,
    dataset_name: str,
) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicate URLs when their labels agree."""
    required_columns = {url_column, label_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required cleaning columns: "
            f"{sorted(missing_columns)}"
        )

    working = frame.copy()
    input_rows = int(len(working))
    working["url_normalized"] = working[url_column].map(normalise_url_for_deduplication)
    working[label_column] = pd.to_numeric(working[label_column], errors="coerce")
    working = working.dropna(subset=["url_normalized", label_column]).copy()
    working = working[working["url_normalized"] != ""].copy()
    rows_after_missing = int(len(working))

    label_counts_per_url = working.groupby("url_normalized")[label_column].nunique()
    conflicting_urls = label_counts_per_url[label_counts_per_url > 1]
    if not conflicting_urls.empty:
        sample = ", ".join(list(conflicting_urls.index[:5]))
        raise ValueError(
            f"{dataset_name} has {len(conflicting_urls):,} duplicate URL(s) with "
            f"conflicting labels. Sample: {sample}"
        )

    duplicate_counts = working.groupby("url_normalized").size().rename("duplicate_count")
    cleaned = (
        working.sort_index()
        .drop_duplicates(subset=["url_normalized"], keep="first")
        .merge(duplicate_counts, on="url_normalized", how="left")
        .reset_index(drop=True)
    )
    cleaned[label_column] = cleaned[label_column].astype(int)
    report = {
        "dataset": dataset_name,
        "input_rows": input_rows,
        "rows_after_missing_url_or_label_drop": rows_after_missing,
        "missing_or_empty_rows_removed": input_rows - rows_after_missing,
        "output_rows": int(len(cleaned)),
        "unique_urls": int(cleaned["url_normalized"].nunique()),
        "duplicate_rows_removed": int(rows_after_missing - len(cleaned)),
        "duplicate_urls": int((duplicate_counts > 1).sum()),
        "conflicting_duplicate_urls": int(len(conflicting_urls)),
    }
    return cleaned, report


def remove_cross_source_duplicate_urls(
    frame: pd.DataFrame,
    source_priority: Iterable[str],
    url_column: str = "url_normalized",
    source_column: str = "source_dataset",
    label_column: str = "label",
) -> tuple[pd.DataFrame, dict]:
    """Keep one row per URL across sources using a deterministic source priority."""
    required_columns = {url_column, source_column, label_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "Cannot remove cross-source duplicates because columns are missing: "
            f"{sorted(missing_columns)}"
        )

    priority = {source_name: index for index, source_name in enumerate(source_priority)}
    working = frame.copy()
    working["_source_priority"] = working[source_column].map(
        lambda name: priority.get(name, len(priority))
    )

    label_counts_per_url = working.groupby(url_column)[label_column].nunique()
    conflicting_urls = label_counts_per_url[label_counts_per_url > 1]
    if not conflicting_urls.empty:
        sample = ", ".join(list(conflicting_urls.index[:5]))
        raise ValueError(
            "Cross-source duplicate URLs have conflicting labels. "
            f"Count: {len(conflicting_urls):,}. Sample: {sample}"
        )

    duplicate_counts = working.groupby(url_column).size()
    cleaned = (
        working.sort_values(["_source_priority", source_column])
        .drop_duplicates(subset=[url_column], keep="first")
        .drop(columns=["_source_priority"])
        .sort_index()
        .reset_index(drop=True)
    )
    report = {
        "input_rows": int(len(frame)),
        "output_rows": int(len(cleaned)),
        "unique_urls": int(cleaned[url_column].nunique()),
        "cross_source_duplicate_urls": int((duplicate_counts > 1).sum()),
        "cross_source_duplicate_rows_removed": int(len(frame) - len(cleaned)),
        "conflicting_duplicate_urls": int(len(conflicting_urls)),
    }
    return cleaned, report
