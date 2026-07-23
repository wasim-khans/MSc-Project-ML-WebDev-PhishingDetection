from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# In this dataset, 0 means phishing and 1 means legitimate.
PHISHING_LABEL = 0


def evaluate_predictions(y_true, y_pred):
    """Calculate the evaluation metrics used in the model comparison table."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        # Treat phishing as the positive class because catching phishing is the priority.
        "phishing_precision": precision_score(
            y_true, y_pred, pos_label=PHISHING_LABEL, zero_division=0
        ),
        "phishing_recall": recall_score(
            y_true, y_pred, pos_label=PHISHING_LABEL, zero_division=0
        ),
        "phishing_f1": f1_score(
            y_true, y_pred, pos_label=PHISHING_LABEL, zero_division=0
        ),
    }


def select_best_model(metrics):
    """Select the winning model using the same rule documented in the report."""
    if not metrics:
        raise ValueError("metrics must contain at least one model result")

    # F1 comes first, then recall, then accuracy as a final tie-breaker.
    return max(
        metrics,
        key=lambda row: (
            row["phishing_f1"],
            row["phishing_recall"],
            row["accuracy"],
        ),
    )
