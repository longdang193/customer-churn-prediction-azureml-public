"""Core preprocessing transformation utilities."""

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def remove_columns(df: pd.DataFrame, columns: Iterable[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Drop any columns present in the frame and return the removed columns.
    
    Args:
        df: Input DataFrame
        columns: Iterable of column names to remove
        
    Returns:
        Tuple of (DataFrame with columns removed, list of removed column names)
    """
    to_drop = [col for col in columns if col in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
    return df, to_drop


def encode_categoricals(
    df: pd.DataFrame,
    *,
    categorical_cols: Iterable[str],
    encoders: Optional[Dict[str, LabelEncoder]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode categorical columns, reusing encoders when provided.
    
    Args:
        df: Input DataFrame
        categorical_cols: Iterable of categorical column names to encode
        encoders: Optional dictionary of existing encoders to reuse
        
    Returns:
        Tuple of (encoded DataFrame, dictionary of encoders)
    """
    encoders = encoders or {}
    df_encoded = df.copy()
    for col in categorical_cols:
        if col not in df.columns:
            continue
        if col not in encoders:
            encoders[col] = LabelEncoder().fit(df[col])
        df_encoded[col] = encoders[col].transform(df[col])
    return df_encoded, encoders


def scale_features(
    df: pd.DataFrame,
    *,
    scaler: Optional[StandardScaler] = None,
    columns: Optional[Iterable[str]] = None,
    exclude_cols: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    """Standard-score the numeric columns.
    
    Args:
        df: Input DataFrame
        scaler: Optional pre-fitted StandardScaler to reuse
        columns: Optional specific columns to scale
        exclude_cols: Optional columns to exclude from scaling
        
    Returns:
        Tuple of (scaled DataFrame, scaler, list of scaled column names)
    """
    df_scaled = df.copy()
    if columns is not None:
        numeric_cols = [col for col in columns if col in df.columns]
    else:
        numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        if exclude_cols:
            exclude_set = {col for col in exclude_cols if col in df.columns}
            numeric_cols = [col for col in numeric_cols if col not in exclude_set]

    if not numeric_cols:
        scaler = scaler or StandardScaler()
        return df_scaled, scaler, []

    if scaler is None:
        scaler = StandardScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])

    return df_scaled, scaler, list(numeric_cols)

