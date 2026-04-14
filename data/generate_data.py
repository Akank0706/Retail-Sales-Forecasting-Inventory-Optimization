"""Synthetic retail time-series data generator."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
    """Generate two years of synthetic retail sales data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=730, freq="D")
    stores = [f"S{i}" for i in range(1, 6)]
    category_map = {
        "Grocery": [f"SKU{i:03d}" for i in range(1, 6)],
        "Electronics": [f"SKU{i:03d}" for i in range(6, 11)],
        "Apparel": [f"SKU{i:03d}" for i in range(11, 16)],
        "Pharma": [f"SKU{i:03d}" for i in range(16, 21)],
    }
    cat_params = {
        "Grocery": {"base": 32, "price": 8.0, "cost": 4.5, "lead": 2},
        "Electronics": {"base": 8, "price": 120.0, "cost": 80.0, "lead": 10},
        "Apparel": {"base": 14, "price": 35.0, "cost": 18.0, "lead": 7},
        "Pharma": {"base": 22, "price": 18.0, "cost": 9.0, "lead": 4},
    }
    holidays = set(pd.date_range(dates.min(), dates.max(), freq="MS").date)
    records = []
    for category, skus in category_map.items():
        params = cat_params[category]
        for item_id in skus:
            sku_factor = rng.uniform(0.8, 1.2)
            for store_id in stores:
                store_factor = rng.uniform(0.85, 1.15)
                stock = int(rng.integers(80, 220))
                for dt in dates:
                    dow = dt.dayofweek
                    trend = 1 + 0.0008 * (dt - dates.min()).days
                    weekly = 1 + 0.18 * np.sin(2 * np.pi * dow / 7)
                    yearly = 1 + 0.12 * np.sin(2 * np.pi * dt.dayofyear / 365)
                    holiday_flag = int(dt.date() in holidays)
                    on_promo = int(rng.random() < (0.08 + 0.05 * (dow in [4, 5])))
                    discount_pct = np.round(rng.uniform(0.05, 0.3), 3) if on_promo else 0.0
                    promo_lift = 1 + (0.25 + discount_pct) * on_promo
                    expected = params["base"] * sku_factor * store_factor * trend * weekly * yearly * promo_lift
                    qty = max(0, int(rng.normal(expected, max(1.5, expected * 0.18))))
                    lead_time = max(1, int(rng.normal(params["lead"], 1.5)))
                    ordering_cost = float(np.round(rng.uniform(25, 90), 2))
                    unit_cost = float(np.round(params["cost"] * rng.uniform(0.9, 1.1), 2))
                    holding_cost_rate = float(np.round(rng.uniform(0.12, 0.28), 3))
                    price = float(np.round(params["price"] * (1 - discount_pct) * rng.uniform(0.96, 1.04), 2))
                    stockout_flag = int(stock <= 0 or (rng.random() < 0.02 and qty > 0))
                    if stockout_flag:
                        qty = int(max(0, qty * rng.uniform(0.2, 0.6)))
                    stock = max(0, stock - qty)
                    if rng.random() < 0.12:
                        stock += int(rng.integers(40, 140))
                    records.append({
                        "date": dt, "store_id": store_id, "item_id": item_id, "category": category,
                        "qty_sold": qty, "price": price, "discount_pct": discount_pct, "on_promo": on_promo,
                        "stock_on_hand": stock, "stockout_flag": stockout_flag, "supplier_lead_time_days": lead_time,
                        "unit_cost": unit_cost, "holding_cost_rate": holding_cost_rate, "ordering_cost": ordering_cost,
                        "holiday_flag": holiday_flag
                    })
    df = pd.DataFrame(records)
    for col in ["price", "unit_cost", "ordering_cost", "holding_cost_rate"]:
        miss_idx = df.sample(frac=0.003, random_state=seed).index
        df.loc[miss_idx, col] = np.nan
    return df

def main() -> None:
    """Generate and save synthetic retail time-series data."""
    out_path = Path(__file__).resolve().parent / "retail_timeseries.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv(out_path, index=False)
    print(f"Saved synthetic data to {out_path} with shape {df.shape}")

if __name__ == "__main__":
    main()
