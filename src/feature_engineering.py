"""Feature engineering for retail forecasting models."""
from __future__ import annotations
import pandas as pd
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling, and calendar features for each store-item group."""
    out = df.copy().sort_values(["store_id","item_id","date"]).reset_index(drop=True)
    grp = out.groupby(["store_id","item_id"], group_keys=False)
    for lag in [1,7,14]: out[f"lag_{lag}"] = grp["qty_sold"].shift(lag)
    for window in [7,14,28]:
        out[f"rolling_mean_{window}"] = grp["qty_sold"].transform(lambda s: s.rolling(window).mean())
        out[f"rolling_std_{window}"] = grp["qty_sold"].transform(lambda s: s.rolling(window).std())
    out["day_of_week"] = out["date"].dt.dayofweek
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    out["month"] = out["date"].dt.month
    out["is_weekend"] = out["day_of_week"].isin([5,6]).astype(int)
    return out.dropna().reset_index(drop=True)
