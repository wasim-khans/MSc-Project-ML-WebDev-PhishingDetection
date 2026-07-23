import unittest

from machine_learning.scripts.core.model_evaluation import (
    evaluate_predictions,
    select_best_model,
)


class ModelTrainingTests(unittest.TestCase):
    def test_evaluate_predictions_reports_phishing_class_metrics(self):
        y_true = [0, 0, 0, 1, 1]
        y_pred = [0, 0, 1, 0, 1]

        metrics = evaluate_predictions(y_true, y_pred)

        self.assertEqual(metrics["accuracy"], 0.6)
        self.assertAlmostEqual(metrics["phishing_precision"], 2 / 3)
        self.assertAlmostEqual(metrics["phishing_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["phishing_f1"], 2 / 3)

    def test_select_best_model_prioritises_phishing_f1_then_recall(self):
        metrics = [
            {
                "model": "High Accuracy",
                "accuracy": 0.99,
                "phishing_precision": 0.80,
                "phishing_recall": 0.65,
                "phishing_f1": 0.70,
            },
            {
                "model": "Better Phishing Balance",
                "accuracy": 0.95,
                "phishing_precision": 0.82,
                "phishing_recall": 0.78,
                "phishing_f1": 0.80,
            },
        ]

        best = select_best_model(metrics)

        self.assertEqual(best["model"], "Better Phishing Balance")


if __name__ == "__main__":
    unittest.main()
