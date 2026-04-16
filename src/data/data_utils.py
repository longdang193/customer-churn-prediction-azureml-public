"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import Tuple

import pandas as pd

REQUIRED_FILES = ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv")


def load_prepared_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load preprocessed training and test data.
    
    Args:
        data_dir: Directory containing preprocessed CSV files
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    data_path = Path(data_dir)
    missing = [fname for fname in REQUIRED_FILES if not (data_path / fname).exists()]
    if missing:
        raise FileNotFoundError(
            f"Preprocessed data directory {data_path} is missing files: {missing}"
        )

    return (
        pd.read_csv(data_path / "X_train.csv"),
        pd.read_csv(data_path / "X_test.csv"),
        pd.read_csv(data_path / "y_train.csv").squeeze("columns"),
        pd.read_csv(data_path / "y_test.csv").squeeze("columns"),
    )


def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to balance training labels.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_balanced, y_balanced)
        
    Raises:
        ImportError: If imbalanced-learn is not installed
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "SMOTE is not available. Install with `pip install imbalanced-learn`."
        ) from exc
    
    smote = SMOTE(random_state=random_state)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    y_series = pd.Series(y_bal, name=y_train.name)
    return pd.DataFrame(X_bal, columns=X_train.columns), y_series

