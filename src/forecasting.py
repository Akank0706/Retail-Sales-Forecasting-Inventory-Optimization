"""Model training and prediction for retail demand forecasting."""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from src.utils import mae, rmse, mase


def _build_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    """Build column preprocessor."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )


def train_and_forecast(df_feat: pd.DataFrame, model_dir: str | Path = "models") -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    """Train RF and XGB models, evaluate, save best model, and return predictions."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    target = "qty_sold"
    feature_cols = [c for c in df_feat.columns if c != target]
    cat_cols = ["store_id", "item_id", "category"]
    num_cols = [c for c in feature_cols if c not in cat_cols + ["date"]]
    max_date = df_feat["date"].max()
    split_date = max_date - pd.Timedelta(weeks=8)
    train_df = df_feat[df_feat["date"] <= split_date].copy()
    test_df = df_feat[df_feat["date"] > split_date].copy()
    X_train, y_train = train_df[feature_cols], train_df[target].values
    X_test, y_test = test_df[feature_cols], test_df[target].values
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=400, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, objective="reg:squarederror", random_state=42, n_jobs=-1
        ),
    }
    rows = []
    best_name, best_pipe, best_pred, best_mae = None, None, None, float("inf")
    for name, model in models.items():
        pipe = Pipeline([("prep", _build_preprocessor(cat_cols, num_cols)), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        row = {"model": name, "mae": mae(y_test, pred), "rmse": rmse(y_test, pred), "mase": mase(y_test, pred, y_train), "r2": float(r2_score(y_test, pred))}
        rows.append(row)
        if row["mae"] < best_mae:
            best_name, best_pipe, best_pred, best_mae = name, pipe, pred, row["mae"]
    metrics_df = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    joblib.dump(best_pipe, model_dir / "best_model.pkl")
    pred_df = test_df[["date", "store_id", "item_id", target]].rename(columns={target: "actual"}).copy()
    pred_df["predicted"] = np.round(best_pred, 2)
    pred_df["residual"] = pred_df["actual"] - pred_df["predicted"]
    pred_df["best_model"] = best_name
    feat_names = best_pipe.named_steps["prep"].get_feature_names_out()
    importances = getattr(best_pipe.named_steps["model"], "feature_importances_", np.zeros(len(feat_names)))
    pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).to_csv(model_dir / "feature_importance.csv", index=False)
    return pred_df, metrics_df, best_pipe
