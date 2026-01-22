import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import storage
import os
from pathlib import Path

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# Custom CSS
# ============================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Config
# ============================
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
GCS_BUCKET = os.getenv("GCS_BUCKET", "housing-price-ml-e2e-xcluo")

storage_client = storage.Client()

def load_from_gcs(blob_name, local_path):
    """Download from GCS if not already cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(str(local_path))
    return str(local_path)

def batch_predict(payload, batch_size=100):
    """Send predictions in batches to avoid 413 error"""
    all_preds = []
    total_batches = (len(payload) + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, start in enumerate(range(0, len(payload), batch_size)):
        batch = payload[start:start + batch_size]
        resp = requests.post(API_URL, json=batch, timeout=60)
        resp.raise_for_status()
        preds = resp.json().get("predictions", [])
        all_preds.extend(preds)
        
        # Update progress
        progress = (i + 1) / total_batches
        progress_bar.progress(progress)
        status_text.text(f"Processing batch {i+1}/{total_batches}")
    
    progress_bar.empty()
    status_text.empty()
    return all_preds

# Paths
HOLDOUT_ENGINEERED_PATH = load_from_gcs(
    "processed/feature_engineered_holdout.csv",
    "data/processed/feature_engineered_holdout.csv"
)
HOLDOUT_META_PATH = load_from_gcs(
    "processed/cleaning_holdout.csv",
    "data/processed/cleaning_holdout.csv"
)

# ============================
# Data loading
# ============================
@st.cache_data
def load_data():
    fe = pd.read_csv(HOLDOUT_ENGINEERED_PATH)
    meta = pd.read_csv(HOLDOUT_META_PATH, parse_dates=["date"])[["date", "city_full"]]

    if len(fe) != len(meta):
        min_len = min(len(fe), len(meta))
        fe = fe.iloc[:min_len].copy()
        meta = meta.iloc[:min_len].copy()

    disp = pd.DataFrame(index=fe.index)
    disp["date"] = meta["date"]
    disp["region"] = meta["city_full"]
    disp["year"] = disp["date"].dt.year
    disp["month"] = disp["date"].dt.month
    disp["actual_price"] = fe["price"]

    return fe, disp

fe_df, disp_df = load_data()

# ============================
# Header
# ============================
st.markdown('<h1 class="main-header">🏠 Housing Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Explore ML-powered price predictions across regions and time")

# ============================
# Sidebar Filters
# ============================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/home.png", width=80)
    st.title("🎯 Filters")
    
    years = sorted(disp_df["year"].unique())
    months = list(range(1, 13))
    regions = ["All Regions"] + sorted(disp_df["region"].dropna().unique())
    
    year = st.selectbox("📅 Year", years, index=len(years)-1)
    month = st.selectbox("📆 Month", months, index=0, format_func=lambda x: f"{x:02d}")
    region = st.selectbox("🌍 Region", regions, index=0)
    
    st.divider()
    
    run_prediction = st.button("🚀 Run Prediction", use_container_width=True)
    
    st.divider()
    
    # Stats
    st.markdown("### 📊 Dataset Info")
    st.metric("Total Records", f"{len(disp_df):,}")
    st.metric("Regions", f"{disp_df['region'].nunique()}")
    st.metric("Date Range", f"{disp_df['year'].min()}-{disp_df['year'].max()}")

# ============================
# Main Content
# ============================
if run_prediction:
    mask = (disp_df["year"] == year) & (disp_df["month"] == month)
    if region != "All Regions":
        mask &= (disp_df["region"] == region)

    idx = disp_df.index[mask]

    if len(idx) == 0:
        st.warning("⚠️ No data found for these filters.")
    else:
        # Info banner
        st.info(f"📍 Analyzing **{len(idx):,} properties** | {year}-{month:02d} | {region}")
        
        payload = fe_df.loc[idx].to_dict(orient="records")

        try:
            with st.spinner("🔮 Making predictions..."):
                resp = requests.post(API_URL, json=payload, timeout=60)
                resp.raise_for_status()
                out = resp.json()
                preds = out.get("predictions", [])
                actuals = out.get("actuals", None)

                view = disp_df.loc[idx, ["date", "region", "actual_price"]].copy()
                view = view.sort_values("date")
                view["prediction"] = pd.Series(preds, index=view.index).astype(float)

                if actuals is not None and len(actuals) == len(view):
                    view["actual_price"] = pd.Series(actuals, index=view.index).astype(float)

            # ============================
            # Metrics Cards
            # ============================
            mae = (view["prediction"] - view["actual_price"]).abs().mean()
            rmse = ((view["prediction"] - view["actual_price"]) ** 2).mean() ** 0.5
            avg_pct_error = ((view["prediction"] - view["actual_price"]).abs() / view["actual_price"]).mean() * 100

            st.markdown("### 🎯 Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Absolute Error", f"${mae:,.0f}", delta=None)
            with col2:
                st.metric("RMSE", f"${rmse:,.0f}", delta=None)
            with col3:
                st.metric("Avg % Error", f"{avg_pct_error:.2f}%", delta=None)
            with col4:
                accuracy = 100 - avg_pct_error
                st.metric("Accuracy", f"{accuracy:.2f}%", delta=None)

            st.divider()

            # ============================
            # Predictions Table
            # ============================
            st.markdown("### 📋 Predictions vs Actuals")
            
            display_view = view[["date", "region", "actual_price", "prediction"]].copy()
            display_view["error"] = display_view["prediction"] - display_view["actual_price"]
            display_view["error_pct"] = (display_view["error"] / display_view["actual_price"] * 100).round(2)
            
            st.dataframe(
                display_view.reset_index(drop=True).style.format({
                    "actual_price": "${:,.0f}",
                    "prediction": "${:,.0f}",
                    "error": "${:,.0f}",
                    "error_pct": "{:.2f}%"
                }).background_gradient(subset=["error_pct"], cmap="RdYlGn_r"),
                use_container_width=True,
                height=400
            )

            # ============================
            # Scatter Plot
            # ============================
            st.markdown("### 📈 Prediction Accuracy")
            
            fig_scatter = px.scatter(
                view,
                x="actual_price",
                y="prediction",
                color=(view["prediction"] - view["actual_price"]).abs(),
                color_continuous_scale="Viridis",
                labels={"actual_price": "Actual Price ($)", "prediction": "Predicted Price ($)"},
                title="Predicted vs Actual Prices"
            )
            
            # Add perfect prediction line
            max_val = max(view["actual_price"].max(), view["prediction"].max())
            min_val = min(view["actual_price"].min(), view["prediction"].min())
            fig_scatter.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Perfect Prediction",
                    line=dict(color="red", dash="dash", width=2)
                )
            )
            
            fig_scatter.update_layout(height=500, template="plotly_white")
            st.plotly_chart(fig_scatter, use_container_width=True)

            # ============================
            # Yearly Trend Chart
            # ============================
            st.markdown("### 📊 Yearly Trend Analysis")
            
            with st.spinner("📈 Generating yearly trends..."):
                if region == "All Regions":
                    yearly_data = disp_df[disp_df["year"] == year].copy()
                    idx_all = yearly_data.index
                    payload_all = fe_df.loc[idx_all].to_dict(orient="records")
                    preds_all = batch_predict(payload_all, batch_size=100)
                    yearly_data["prediction"] = pd.Series(preds_all, index=yearly_data.index).astype(float)
                else:
                    yearly_data = disp_df[(disp_df["year"] == year) & (disp_df["region"] == region)].copy()
                    idx_region = yearly_data.index
                    payload_region = fe_df.loc[idx_region].to_dict(orient="records")
                    preds_region = batch_predict(payload_region, batch_size=100)
                    yearly_data["prediction"] = pd.Series(preds_region, index=yearly_data.index).astype(float)

            monthly_avg = yearly_data.groupby("month")[["actual_price", "prediction"]].mean().reset_index()

            fig_trend = go.Figure()
            
            fig_trend.add_trace(go.Scatter(
                x=monthly_avg["month"],
                y=monthly_avg["actual_price"],
                mode="lines+markers",
                name="Actual Price",
                line=dict(color="#667eea", width=3),
                marker=dict(size=10)
            ))
            
            fig_trend.add_trace(go.Scatter(
                x=monthly_avg["month"],
                y=monthly_avg["prediction"],
                mode="lines+markers",
                name="Predicted Price",
                line=dict(color="#764ba2", width=3, dash="dash"),
                marker=dict(size=10)
            ))
            
            # Highlight selected month
            fig_trend.add_vrect(
                x0=month - 0.5,
                x1=month + 0.5,
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
                annotation_text="Selected",
                annotation_position="top left"
            )
            
            fig_trend.update_layout(
                title=f"Monthly Price Trends — {year}" + (f" — {region}" if region != "All Regions" else ""),
                xaxis_title="Month",
                yaxis_title="Price ($)",
                height=500,
                template="plotly_white",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)

        except Exception as e:
            st.error(f"❌ API call failed: {e}")
            with st.expander("📋 Error Details"):
                st.exception(e)

else:
    # ============================
    # Welcome Screen
    # ============================
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>👋 Welcome!</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Select your filters in the sidebar and click 
                <strong>Run Prediction</strong> to explore housing price predictions.
            </p>
            <br>
            <p style="color: #999;">
                Powered by ML | Built with ❤️
            </p>
        </div>
        """, unsafe_allow_html=True)