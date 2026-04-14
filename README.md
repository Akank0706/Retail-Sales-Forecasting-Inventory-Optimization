# Retail Sales Forecasting & Inventory Optimization System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![XGBoost](https://img.shields.io/badge/XGBoost-Forecasting-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Demo GIF
![Demo](docs/demo.gif)

## Architecture
```text
                +-----------------------+
                | data/generate_data.py |
                +-----------+-----------+
                            |
                            v
+-------------+    +--------------------+    +--------------------------+
| data_loader | -> | preprocessor.py    | -> | feature_engineering.py   |
+-------------+    +--------------------+    +-------------+------------+
                                                          |
                                                          v
                                          +-------------------------------+
                                          | forecasting.py (RF + XGBoost) |
                                          +---------------+---------------+
                                                          |
                                   +----------------------+---------------------+
                                   v                                            v
                     outputs/forecasts.csv                       models/best_model.pkl
                                   |
                                   v
                         +--------------------+
                         | inventory.py        |
                         +---------+----------+
                                   |
                                   v
                  outputs/reorder_recommendations.csv
                                   |
                                   v
                        app/dashboard.py (Streamlit)
```

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How To Run
```bash
python data/generate_data.py
python main.py
streamlit run app/dashboard.py
```

## Screenshots
- `docs/overview.png`
- `docs/forecasting.png`
- `docs/inventory.png`
- `docs/store_analytics.png`
- `docs/eda.png`

## Tech Stack
| Layer | Tools |
|---|---|
| Data | Pandas, NumPy |
| ML | Scikit-learn, XGBoost |
| Stats | SciPy, Statsmodels |
| Dashboard | Streamlit, Plotly |
| Serialization | Joblib |

## Results (Sample)
- MAE: ~4.5 to 6.5
- RMSE: ~6.0 to 9.0
- MASE: < 1.0 for strong SKUs
- Reorder recommendations for all store-SKU combinations
