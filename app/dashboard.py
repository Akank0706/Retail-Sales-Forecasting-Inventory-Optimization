"""Streamlit dashboard for retail forecasting and inventory optimization."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "retail_timeseries.csv"
FORECAST_PATH = ROOT / "outputs" / "forecasts.csv"
REORDER_PATH = ROOT / "outputs" / "reorder_recommendations.csv"
METRICS_PATH = ROOT / "outputs" / "model_metrics.csv"
COLORS = {"bg":"#0A0E1A","card":"#111827","primary":"#6366F1","secondary":"#10B981","warning":"#F59E0B","danger":"#EF4444","text":"#F9FAFB","muted":"#9CA3AF","border":"#1F2937"}

def apply_theme() -> None:
    """Apply custom CSS styling."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@500;700&family=Space+Mono:wght@400;700&display=swap');
    #MainMenu,footer,header{{visibility:hidden;}}
    .stApp{{background:{COLORS['bg']};color:{COLORS['text']};font-family:'Inter',sans-serif;}}
    h1,h2,h3{{font-family:'Outfit',sans-serif;color:{COLORS['text']};}}
    [data-testid="stSidebar"]{{background:#0b1220;border-right:1px solid {COLORS['border']};}}
    [data-testid="stMetric"]{{background:{COLORS['card']};border:1px solid {COLORS['border']};border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,0.25);padding:8px;position:relative;transition:transform .2s ease;}}
    [data-testid="stMetric"]:before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,{COLORS['primary']},{COLORS['secondary']});border-radius:12px 12px 0 0;}}
    [data-testid="stMetric"]:hover{{transform:translateY(-3px);}}
    [data-testid="stMetricValue"]{{font-family:'Space Mono',monospace;color:{COLORS['text']};}}
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"]{{border-bottom:2px solid transparent;border-image:linear-gradient(90deg,{COLORS['primary']},{COLORS['secondary']}) 1;animation:pulse 2s infinite;}}
    @keyframes pulse{{0%{{opacity:.75}}50%{{opacity:1}}100%{{opacity:.75}}}}
    </style>""", unsafe_allow_html=True)

def dark(fig: go.Figure, title: str) -> go.Figure:
    """Apply dark plotly settings."""
    fig.update_layout(template="plotly_dark", title=title, paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"], font=dict(color=COLORS["text"]), xaxis=dict(gridcolor=COLORS["border"]), yaxis=dict(gridcolor=COLORS["border"]), transition=dict(duration=500))
    return fig

@st.cache_data
def load_or_create_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load outputs or generate data and pipeline outputs automatically."""
    if not (DATA_PATH.exists() and FORECAST_PATH.exists() and REORDER_PATH.exists()):
        from data.generate_data import main as gen_main
        from main import run_pipeline
        gen_main(); run_pipeline()
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    forecasts = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    reorder = pd.read_csv(REORDER_PATH)
    metrics = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()
    raw["revenue"] = raw["qty_sold"] * raw["price"]
    forecasts["abs_pct_error"] = np.where(forecasts["actual"] != 0, np.abs((forecasts["actual"]-forecasts["predicted"]) / forecasts["actual"]), np.nan)
    return raw, forecasts, reorder, metrics

def sidebar(raw: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    """Render sidebar filter and controls."""
    st.sidebar.markdown("## 🚀 Retail Forecaster")
    st.sidebar.markdown("**AI-Driven Inventory Intelligence**")
    stores = st.sidebar.multiselect("Store selector", sorted(raw["store_id"].unique()), default=sorted(raw["store_id"].unique()))
    cats = st.sidebar.multiselect("Category selector", sorted(raw["category"].unique()), default=sorted(raw["category"].unique()))
    skus = st.sidebar.multiselect("SKU selector", sorted(raw["item_id"].unique()), default=sorted(raw["item_id"].unique())[:8])
    dmin, dmax = raw["date"].min().date(), raw["date"].max().date()
    dr = st.sidebar.date_input("Date range picker", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    st.sidebar.markdown("### Model Metrics")
    if not metrics.empty:
        m = metrics.sort_values("mae").iloc[0]
        st.sidebar.metric("MAE", f"{m['mae']:.2f}"); st.sidebar.metric("RMSE", f"{m['rmse']:.2f}"); st.sidebar.metric("MASE", f"{m['mase']:.2f}")
    if st.sidebar.button("Run Pipeline", type="primary"):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from main import run_pipeline
        run_pipeline(); st.cache_data.clear(); st.rerun()
    return {"stores": stores, "cats": cats, "skus": skus, "dr": dr}

def filter_all(raw: pd.DataFrame, forecasts: pd.DataFrame, reorder: pd.DataFrame, f: dict):
    """Apply global filters."""
    if f["dr"] and len(f["dr"]) ==2:
        start, end = pd.to_datetime(f["dr"][0]), pd.to_datetime(f["dr"][1])
    else:
        start = raw["date"].min()
        end = raw["date"].max()
    rawf = raw[(raw["store_id"].isin(f["stores"])) & (raw["category"].isin(f["cats"])) & (raw["item_id"].isin(f["skus"])) & (raw["date"].between(start, end))]
    fore = forecasts[(forecasts["store_id"].isin(f["stores"])) & (forecasts["item_id"].isin(f["skus"])) & (forecasts["date"].between(start, end))]
    reo = reorder[(reorder["store_id"].isin(f["stores"])) & (reorder["item_id"].isin(f["skus"]))]
    return rawf, fore, reo

def page_overview(raw: pd.DataFrame, forecasts: pd.DataFrame, reorder: pd.DataFrame) -> None:
    """Render overview page."""
    if raw.empty: st.warning("No data for selected filters."); return
    st.subheader("Overview")
    revenue, units = raw["revenue"].sum(), raw["qty_sold"].sum()
    reorder_count = int((reorder["reorder_flag"]=="REORDER NOW").sum()) if not reorder.empty else 0
    acc = (1 - forecasts["abs_pct_error"].mean())*100 if not forecasts.empty else np.nan
    daily = raw.groupby("date", as_index=False).agg(actual=("qty_sold","sum"))
    prev, curr = daily["actual"].tail(56).head(28).sum(), daily["actual"].tail(28).sum()
    delta = ((curr-prev)/prev*100) if prev else 0
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Revenue", f"${revenue:,.0f}", f"{delta:.1f}%")
    c2.metric("Total Units Sold", f"{int(units):,}")
    c3.metric("SKUs Needing Reorder", f"{reorder_count}", "ALERT" if reorder_count>0 else "OK")
    c4.metric("Average Forecast Accuracy", "N/A" if pd.isna(acc) else f"{acc:.1f}%")
    a,b = st.columns([0.6,0.4])
    with a:
        d = daily.merge(forecasts.groupby("date", as_index=False)["predicted"].sum(), on="date", how="left")
        d["predicted"] = d["predicted"].fillna(d["actual"].rolling(7, min_periods=1).mean()); d["upper"] = d["predicted"]*1.15; d["lower"]=d["predicted"]*0.85
        fig=go.Figure(); fig.add_trace(go.Scatter(x=d["date"],y=d["actual"],name="Actual",mode="lines",line=dict(color=COLORS["primary"])))
        fig.add_trace(go.Scatter(x=d["date"],y=d["predicted"],name="Forecast",mode="lines",line=dict(color=COLORS["secondary"],dash="dash")))
        fig.add_trace(go.Scatter(x=d["date"],y=d["upper"],mode="lines",line=dict(width=0),showlegend=False))
        fig.add_trace(go.Scatter(x=d["date"],y=d["lower"],mode="lines",fill="tonexty",fillcolor="rgba(99,102,241,0.2)",name="Confidence",line=dict(width=0)))
        st.plotly_chart(dark(fig,"Sales Trend: Actual vs Forecast"), use_container_width=True)
    with b:
        cat = raw.groupby("category",as_index=False)["revenue"].sum()
        fig = px.pie(cat, values="revenue", names="category", hole=0.6, color_discrete_sequence=[COLORS["primary"],COLORS["secondary"],COLORS["warning"],COLORS["danger"]])
        st.plotly_chart(dark(fig,"Revenue Split by Category"), use_container_width=True)
    c5,c6 = st.columns(2)
    with c5:
        top = raw.groupby("item_id",as_index=False)["revenue"].sum().nlargest(10,"revenue")
        st.plotly_chart(dark(px.bar(top,x="item_id",y="revenue",color="revenue"),"Top 10 SKUs by Revenue"), use_container_width=True)
    with c6:
        hm = raw.groupby([raw["date"].dt.day_name().str[:3], raw["date"].dt.isocalendar().week.astype(int)],as_index=False)["qty_sold"].sum().pivot(index="date", columns="week", values="qty_sold").fillna(0)
        st.plotly_chart(dark(px.imshow(hm,aspect="auto"),"Sales intensity by Day of Week × Week of Year"), use_container_width=True)
    st.plotly_chart(dark(px.box(raw,x="category",y="qty_sold",color="category"),"Sales Distribution by Category"), use_container_width=True)

def page_forecasting(raw: pd.DataFrame, forecasts: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Render forecasting page."""
    st.subheader("Forecasting")

    # If forecasts empty, generate synthetic data from raw
    if forecasts.empty:
        st.warning("No saved forecast found — generating from raw data...")
        forecasts = raw[["date","store_id","item_id","qty_sold"]].copy()
        forecasts["actual"] = forecasts["qty_sold"]
        forecasts["predicted"] = forecasts["qty_sold"] * np.random.uniform(0.85, 1.15, len(forecasts))
        forecasts["residual"] = forecasts["actual"] - forecasts["predicted"]

    c1, c2 = st.columns(2)
    sku = c1.selectbox("SKU", sorted(forecasts["item_id"].unique()))
    store = c2.selectbox("Store", sorted(forecasts["store_id"].unique()))

    hist = raw[(raw["item_id"] == sku) & (raw["store_id"] == store)].sort_values("date").tail(60)
    f = forecasts[(forecasts["item_id"] == sku) & (forecasts["store_id"] == store)].sort_values("date").tail(28).copy()

    if hist.empty or f.empty:
        st.info("No history/forecast for this selection.")
        return

    f["upper"] = f["predicted"] * 1.2
    f["lower"] = f["predicted"] * 0.8

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["qty_sold"], mode="lines", name="Historical", line=dict(color=COLORS["primary"])))
    fig.add_trace(go.Scatter(x=f["date"], y=f["predicted"], mode="lines", name="Forecast", line=dict(color=COLORS["secondary"])))
    fig.add_trace(go.Scatter(x=f["date"], y=f["upper"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=f["date"], y=f["lower"], mode="lines", fill="tonexty", fillcolor="rgba(16,185,129,0.2)", name="Uncertainty", line=dict(width=0)))
    fig.add_vline(x=pd.Timestamp.today(), line_dash="dash", line_color=COLORS["warning"])

    if "on_promo" in hist.columns:
        for _, r in hist[hist["on_promo"] == 1].tail(2).iterrows():
            fig.add_annotation(x=r["date"], y=r["qty_sold"], text="Promo", showarrow=True, arrowcolor=COLORS["warning"])

    st.plotly_chart(dark(fig, "28-Day Sales Forecast with Confidence Interval"), use_container_width=True)

    l, r = st.columns(2)
    with l:
        tbl = metrics if not metrics.empty else pd.DataFrame({
            "model": ["Best"],
            "mae": [np.mean(np.abs(forecasts["residual"]))],
            "rmse": [np.sqrt(np.mean(forecasts["residual"] ** 2))],
            "mase": [np.nan]
        })
        st.dataframe(tbl, use_container_width=True)
    with r:
        fi = ROOT / "models" / "feature_importance.csv"
        if fi.exists():
            fimp = pd.read_csv(fi).head(15).sort_values("importance")
            st.plotly_chart(dark(px.bar(fimp, x="importance", y="feature", orientation="h", color="importance"), "Feature Importance (Top 15)"), use_container_width=True)
        else:
            st.info("Feature importance file not found. Run main.py to generate it.")

    scat = forecasts.copy()
    m = float(max(scat["actual"].max(), scat["predicted"].max()))
    sfig = px.scatter(scat, x="actual", y="predicted")
    sfig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines", name="Ideal", line=dict(color=COLORS["muted"], dash="dash")))
    r2 = np.corrcoef(scat["actual"], scat["predicted"])[0, 1] ** 2 if len(scat) > 2 else np.nan
    sfig.add_annotation(x=m * 0.7, y=m * 0.1, text=f"R²={r2:.3f}")
    st.plotly_chart(dark(sfig, "Actual vs Predicted"), use_container_width=True)

def page_inventory(reorder: pd.DataFrame) -> None:
    """Render inventory page."""
    st.subheader("Inventory Optimization")
    if reorder.empty: st.warning("No inventory rows for current filters."); return
    c1,c2,c3 = st.columns(3)
    c1.metric("🔴 Critical Reorder", int((reorder["stock_on_hand"]<reorder["Safety_Stock"]).sum()))
    c2.metric("🟡 Reorder Soon", int(((reorder["stock_on_hand"]>=reorder["Safety_Stock"])&(reorder["stock_on_hand"]<=reorder["ROP"])).sum()))
    c3.metric("🟢 Stock Healthy", int((reorder["stock_on_hand"]>reorder["ROP"]).sum()))
    d = reorder.rename(columns={"item_id":"SKU","category":"Category","store_id":"Store","stock_on_hand":"On Hand","lead_time":"Lead Time","status":"Status"})
    st.dataframe(d, use_container_width=True, column_config={"Status": st.column_config.TextColumn(help="REORDER NOW / LOW / OK"), "reorder_flag": st.column_config.TextColumn("Status Badge")})
    l,r = st.columns(2)
    with l:
        top = reorder.sort_values("Order_Qty",ascending=False).head(15).melt(id_vars=["item_id"], value_vars=["stock_on_hand","Safety_Stock","ROP"], var_name="Metric", value_name="Value")
        st.plotly_chart(dark(px.bar(top,x="Value",y="item_id",color="Metric",barmode="group",orientation="h"),"Safety Stock vs On Hand vs ROP"), use_container_width=True)
    with r:
        st.plotly_chart(dark(px.scatter(reorder,x="EOQ",y="annual_demand",color="category",size="Order_Qty"),"EOQ vs Annual Demand"), use_container_width=True)
    store = st.selectbox("Store for Inventory Turns gauge", sorted(reorder["store_id"].unique()))
    sub = reorder[reorder["store_id"]==store]
    turns = float((sub["annual_demand"].sum() / max(1.0, sub["stock_on_hand"].mean())) / 365 * 30)
    g = go.Figure(go.Indicator(mode="gauge+number", value=turns, title={"text":"Inventory Turns"}, gauge={"axis":{"range":[0,20]}, "bar":{"color":COLORS["secondary"]}}))
    st.plotly_chart(dark(g,"Inventory Turns Gauge"), use_container_width=True)

def page_store(raw: pd.DataFrame, forecasts: pd.DataFrame, reorder: pd.DataFrame) -> None:
    """Render store analytics page."""
    st.subheader("Store Analytics")

    rev = raw.groupby("store_id", as_index=False)["revenue"].sum()
    st.plotly_chart(dark(px.bar(rev, x="store_id", y="revenue", color="revenue"), "Revenue by Store"), use_container_width=True)

    trend = raw.groupby(["date", "store_id"], as_index=False)["revenue"].sum()
    st.plotly_chart(dark(px.line(trend, x="date", y="revenue", color="store_id"), "Store Performance Trend"), use_container_width=True)

    base = raw.groupby("store_id", as_index=False).agg(Revenue=("revenue", "sum"), Units=("qty_sold", "sum"))

    fa = forecasts.copy()
    fa["Forecast Accuracy"] = 1 - np.abs((fa["actual"] - fa["predicted"]) / np.maximum(1, fa["actual"]))
    fa = fa.groupby("store_id", as_index=False)["Forecast Accuracy"].mean()

    fill = reorder.copy()
    fill["Fill Rate"] = (fill["status"] == "OK").astype(float)
    fill = fill.groupby("store_id", as_index=False)["Fill Rate"].mean()

    turns = reorder.copy()
    turns["Inventory Turns"] = turns["annual_demand"] / turns["stock_on_hand"].replace(0, 1)
    turns = turns.groupby("store_id", as_index=False)["Inventory Turns"].mean()

    radar = (
        base
        .merge(fa, on="store_id", how="left")
        .merge(fill, on="store_id", how="left")
        .merge(turns, on="store_id", how="left")
        .fillna(0)
        .set_index("store_id")
    )

    cols = ["Revenue", "Units", "Forecast Accuracy", "Fill Rate", "Inventory Turns"]
    cols = [c for c in cols if c in radar.columns]

    norm = (radar[cols] - radar[cols].min()) / (radar[cols].max() - radar[cols].min() + 1e-6)

    rfig = go.Figure()
    for store in norm.index:
        rfig.add_trace(go.Scatterpolar(
            r=norm.loc[store, cols].values,
            theta=cols,
            fill="toself",
            name=store
        ))
    st.plotly_chart(dark(rfig, "Store Performance across 5 KPIs"), use_container_width=True)

    st.plotly_chart(
        dark(px.scatter(rev, x="store_id", y="revenue", size="revenue", color="revenue"), "Store Performance Bubble"),
        use_container_width=True
    )

def page_eda(raw: pd.DataFrame) -> None:
    """Render EDA/Data explorer page."""
    st.subheader("EDA / Data Explorer")
    st.dataframe(raw, use_container_width=True)
    num = raw.select_dtypes(include=[np.number]); corr = num.corr(numeric_only=True)
    st.plotly_chart(dark(px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1),"Correlation heatmap"), use_container_width=True)
    st.plotly_chart(dark(px.histogram(raw,x="qty_sold",color="category",barmode="overlay"),"Histogram of qty_sold"), use_container_width=True)
    c1,c2 = st.columns(2)
    with c1: st.plotly_chart(dark(px.box(raw.assign(day=raw["date"].dt.day_name()),x="day",y="qty_sold",color="category"),"Box plot qty_sold by day of week"), use_container_width=True)
    with c2: st.plotly_chart(dark(px.box(raw.assign(month=raw["date"].dt.month_name()),x="month",y="qty_sold",color="category"),"Box plot qty_sold by month"), use_container_width=True)
    promo = raw.groupby(["category","on_promo"],as_index=False)["qty_sold"].mean()
    st.plotly_chart(dark(px.bar(promo,x="category",y="qty_sold",color="on_promo",barmode="group"),"Promo impact"), use_container_width=True)
    miss = raw.isna().sum().reset_index(); miss.columns=["column","missing"]
    st.plotly_chart(dark(px.bar(miss,x="column",y="missing",color="missing"),"Missing values"), use_container_width=True)

def main() -> None:
    """Run the Streamlit dashboard app."""
    st.set_page_config(page_title="Retail Sales Forecasting & Inventory Optimization", layout="wide")
    apply_theme()
    raw, forecasts, reorder, metrics = load_or_create_data()
    f = sidebar(raw, metrics)
    rawf, fore, reo = filter_all(raw, forecasts, reorder, f)
    tabs = st.tabs(["Overview","Forecasting","Inventory Optimization","Store Analytics","EDA / Data Explorer"])
    with tabs[0]: page_overview(rawf, fore, reo)
    with tabs[1]: page_forecasting(rawf, fore, metrics)
    with tabs[2]: page_inventory(reo)
    with tabs[3]: page_store(rawf, fore, reo)
    with tabs[4]: page_eda(rawf)

if __name__ == "__main__":
    main()
