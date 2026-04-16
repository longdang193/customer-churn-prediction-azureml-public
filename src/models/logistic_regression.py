"""Logistic Regression model configuration."""

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
