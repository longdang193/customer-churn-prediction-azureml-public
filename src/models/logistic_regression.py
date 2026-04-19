"""Logistic Regression model configuration.

@meta
name: logistic_regression
type: module
domain: models
responsibility:
  - Provide models behavior for `src/models/logistic_regression.py`.
inputs: []
outputs: []
tags:
  - models
lifecycle:
  status: active
"""

from sklearn.linear_model import LogisticRegression


def get_logistic_regression(class_weight='balanced', random_state=42):
    """
    Get Logistic Regression model with default hyperparameters.
    
    Args:
        class_weight: Strategy for handling class imbalance
        random_state: Random seed for reproducibility
        
    Returns:
        Configured LogisticRegression model
    """
    return LogisticRegression(
        class_weight=class_weight,
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs',
        C=1.0,
    )
