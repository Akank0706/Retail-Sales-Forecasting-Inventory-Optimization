"""Load and validate retail sales data."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
REQUIRED_COLUMNS = ["date","store_id","item_id","category","qty_sold","price","discount_pct","on_promo","stock_on_hand","stockout_flag","supplier_lead_time_days","unit_cost","holding_cost_rate","ordering_cost","holiday_flag"]
def load_retail_data(path: str | Path) -> pd.DataFrame:
    """Load retail data from CSV and validate required schema."""
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any(): raise ValueError("Found invalid date rows in input data")
    return df
