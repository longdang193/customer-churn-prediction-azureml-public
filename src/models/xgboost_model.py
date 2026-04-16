"""XGBoost model configuration."""

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_xgboost(class_weight=None, random_state=42):
    """
    Get XGBoost model with default hyperparameters.
    
    Args:
        class_weight: Not used for XGBoost (uses scale_pos_weight)
        random_state: Random seed for reproducibility
        
    Returns:
        Configured XGBClassifier model
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    return XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss',
    )
