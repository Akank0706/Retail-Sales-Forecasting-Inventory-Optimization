"""Utility helpers and metrics for retail forecasting."""
from __future__ import annotations
import numpy as np
import pandas as pd
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""; return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""; return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def mase(y_true: np.ndarray, y_pred: np.ndarray, train_series: np.ndarray) -> float:
    """Compute mean absolute scaled error."""
    if len(train_series) < 2: return float("nan")
    denom = np.mean(np.abs(np.diff(train_series)))
    if denom == 0: return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / denom)
def ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Ensure a DataFrame date column is parsed as datetime."""; out = df.copy(); out[col] = pd.to_datetime(out[col], errors="coerce"); return out
