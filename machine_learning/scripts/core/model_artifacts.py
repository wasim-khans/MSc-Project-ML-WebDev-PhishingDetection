import re


def slugify(value):
    """Convert display names into stable lowercase filename slugs."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def split_suffix(test_size):
    """Return train/test split text such as 80_20 from a test-size fraction."""
    test_percent = round(float(test_size) * 100)
    train_percent = 100 - test_percent
    return f"{train_percent}_{test_percent}"


def model_artifact_filename(model_name, dataset_name, test_size):
    """Create the standard trained-model filename."""
    return (
        f"{slugify(model_name)}_T_ON_{slugify(dataset_name)}_"
        f"{split_suffix(test_size)}.joblib"
    )


def best_model_artifact_filename(model_name, dataset_name, test_size):
    """Create the standard best-model alias filename."""
    return (
        f"best_model_{slugify(model_name)}_T_ON_{slugify(dataset_name)}_"
        f"{split_suffix(test_size)}.joblib"
    )
