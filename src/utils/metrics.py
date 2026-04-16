"""Metrics calculation utilities."""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """Calculate core evaluation metrics with sensible fallbacks."""
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        # Happens when there is a single class in y_true or probabilities are degenerate
        roc_auc = 0.5

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }

