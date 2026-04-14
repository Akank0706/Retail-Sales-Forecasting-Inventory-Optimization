"""CLI entrypoint for retail forecasting and inventory optimization pipeline."""
from __future__ import annotations
from pathlib import Path
from src.data_loader import load_retail_data
from src.preprocessor import preprocess_data
from src.feature_engineering import create_features
from src.forecasting import train_and_forecast
from src.inventory import optimize_inventory


def run_pipeline() -> None:
    """Run full data, model, and inventory pipeline and save outputs."""
    root = Path(__file__).resolve().parent
    data_path = root / "data" / "retail_timeseries.csv"
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        from data.generate_data import main as gen_main
        gen_main()
    df = load_retail_data(data_path)
    df_clean = preprocess_data(df)
    df_feat = create_features(df_clean)
    pred_df, metrics_df, _ = train_and_forecast(df_feat, root / "models")
    inv_df = optimize_inventory(pred_df, df_clean)
    pred_out = outputs / "forecasts.csv"
    inv_out = outputs / "reorder_recommendations.csv"
    pred_df.to_csv(pred_out, index=False)
    inv_df.to_csv(inv_out, index=False)
    metrics_df.to_csv(outputs / "model_metrics.csv", index=False)
    print("Pipeline run completed")
    print(f"Input rows: {len(df):,}")
    print(f"Clean rows: {len(df_clean):,}")
    print(f"Feature rows: {len(df_feat):,}")
    print(f"Forecast rows: {len(pred_df):,}")
    print(f"Reorder rows: {len(inv_df):,}")
    print("\nModel metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved: {pred_out}")
    print(f"Saved: {inv_out}")


if __name__ == "__main__":
    run_pipeline()
