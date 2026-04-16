"""Random Forest model configuration."""

from sklearn.ensemble import RandomForestClassifier


def get_random_forest(class_weight='balanced', random_state=42):
    """
    Get Random Forest model with default hyperparameters.
    
    Args:
        class_weight: Strategy for handling class imbalance
        random_state: Random seed for reproducibility
        
    Returns:
        Configured RandomForestClassifier model
    """
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )