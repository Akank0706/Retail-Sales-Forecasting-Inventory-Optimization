"""Preprocessing pipeline for retail sales data."""
from __future__ import annotations
import pandas as pd
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by parsing dates, handling missing values, and filtering invalid rows."""
    out = df.copy(); out["date"] = pd.to_datetime(out["date"], errors="coerce"); out = out.dropna(subset=["date"])
    out = out[out["stockout_flag"] == 0]
    numeric_cols = ["qty_sold","price","discount_pct","stock_on_hand","supplier_lead_time_days","unit_cost","holding_cost_rate","ordering_cost"]
    for col in numeric_cols: out[col] = pd.to_numeric(out[col], errors="coerce")
    out[numeric_cols] = out[numeric_cols].fillna(out[numeric_cols].median(numeric_only=True))
    for col in ["qty_sold","price","stock_on_hand","unit_cost","ordering_cost"]: out = out[out[col] >= 0]
    return out.sort_values(["store_id","item_id","date"]).reset_index(drop=True)
