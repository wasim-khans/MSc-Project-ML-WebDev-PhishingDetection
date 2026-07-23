import unittest

import pandas as pd

from machine_learning.scripts.core import dataset_cleaning


class DatasetCleaningTests(unittest.TestCase):
    def test_deduplicate_url_rows_keeps_one_copy_and_reports_removed_rows(self):
        frame = pd.DataFrame(
            [
                {"url": " https://example.com/login ", "label": 0},
                {"url": "https://example.com/login", "label": 0},
                {"url": "https://roehampton.ac.uk", "label": 1},
            ]
        )

        cleaned, report = dataset_cleaning.deduplicate_url_rows(
            frame,
            url_column="url",
            label_column="label",
            dataset_name="sample",
        )

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(report["duplicate_rows_removed"], 1)
        self.assertEqual(report["conflicting_duplicate_urls"], 0)
        self.assertIn("url_normalized", cleaned.columns)
        self.assertEqual(
            cleaned.loc[0, "url_normalized"],
            "https://example.com/login",
        )

    def test_deduplicate_url_rows_rejects_conflicting_labels(self):
        frame = pd.DataFrame(
            [
                {"url": "https://example.com/login", "label": 0},
                {"url": "https://example.com/login", "label": 1},
            ]
        )

        with self.assertRaises(ValueError):
            dataset_cleaning.deduplicate_url_rows(
                frame,
                url_column="url",
                label_column="label",
                dataset_name="sample",
            )

    def test_deduplicate_url_rows_drops_missing_or_empty_url_rows(self):
        frame = pd.DataFrame(
            [
                {"url": None, "label": 0},
                {"url": "   ", "label": 1},
                {"url": "https://example.com/login", "label": 0},
            ]
        )

        cleaned, report = dataset_cleaning.deduplicate_url_rows(
            frame,
            url_column="url",
            label_column="label",
            dataset_name="sample",
        )

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report["missing_or_empty_rows_removed"], 2)
        self.assertEqual(cleaned.loc[0, "url_normalized"], "https://example.com/login")

    def test_remove_cross_source_duplicate_urls_keeps_first_source_priority(self):
        frame = pd.DataFrame(
            [
                {
                    "source_dataset": "main",
                    "url_normalized": "https://shared.example",
                    "label": 0,
                },
                {
                    "source_dataset": "legitphish",
                    "url_normalized": "https://shared.example",
                    "label": 0,
                },
                {
                    "source_dataset": "phishstorm",
                    "url_normalized": "https://unique.example",
                    "label": 1,
                },
            ]
        )

        cleaned, report = dataset_cleaning.remove_cross_source_duplicate_urls(
            frame,
            source_priority=["main", "legitphish", "phishstorm"],
        )

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(report["cross_source_duplicate_rows_removed"], 1)
        self.assertEqual(
            set(cleaned["source_dataset"]),
            {"main", "phishstorm"},
        )


if __name__ == "__main__":
    unittest.main()
