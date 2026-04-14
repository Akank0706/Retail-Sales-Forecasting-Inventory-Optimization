"""Inventory optimization logic for retail SKUs."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm


def optimize_inventory(pred_df: pd.DataFrame, base_df: pd.DataFrame, service_level: float = 0.95) -> pd.DataFrame:
    """Compute SS, ROP, EOQ, and order recommendations for each store-item."""
    z = float(norm.ppf(service_level))
    latest = base_df.sort_values("date").groupby(["store_id", "item_id", "category"], as_index=False).tail(1).reset_index(drop=True)
    demand_stats = pred_df.groupby(["store_id", "item_id"], as_index=False).agg(avg_daily_forecast=("predicted", "mean"), resid_std=("residual", "std"))
    annual = base_df.groupby(["store_id", "item_id"], as_index=False)["qty_sold"].sum().rename(columns={"qty_sold": "annual_demand"})
    merged = latest.merge(demand_stats, on=["store_id", "item_id"], how="left").merge(annual, on=["store_id", "item_id"], how="left")
    merged["resid_std"] = merged["resid_std"].fillna(0.0)
    merged["avg_daily_forecast"] = merged["avg_daily_forecast"].fillna(0.0)
    merged["lead_time"] = merged["supplier_lead_time_days"].clip(lower=1)
    merged["mu_L"] = merged["avg_daily_forecast"] * merged["lead_time"]
    merged["sigma_L"] = merged["resid_std"] * np.sqrt(merged["lead_time"])
    merged["Safety_Stock"] = z * merged["sigma_L"]
    merged["ROP"] = merged["mu_L"] + merged["Safety_Stock"]
    holding_cost = (merged["unit_cost"] * merged["holding_cost_rate"]).replace(0, np.nan)
    merged["EOQ"] = np.sqrt((2 * merged["annual_demand"] * merged["ordering_cost"]) / holding_cost).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    merged["Order_Qty"] = np.maximum(0.0, np.maximum(merged["EOQ"], merged["ROP"] - merged["stock_on_hand"]))
    merged["reorder_flag"] = np.where(merged["stock_on_hand"] <= merged["ROP"], "REORDER NOW", "OK")
    merged["status"] = np.where(merged["stock_on_hand"] < merged["Safety_Stock"], "REORDER NOW", np.where(merged["stock_on_hand"] <= merged["ROP"], "LOW", "OK"))
    cols = ["store_id", "item_id", "category", "stock_on_hand", "Safety_Stock", "ROP", "EOQ", "Order_Qty", "lead_time", "annual_demand", "reorder_flag", "status"]
    return merged[cols].round(2)
