import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib

# Set matplotlib style for clean light aesthetic with neutral slate/grey tones
plt.style.use('default')
matplotlib.rcParams['font.sans-serif'] = 'Inter, DejaVu Sans, Arial, Helvetica'
matplotlib.rcParams['axes.edgecolor'] = '#cbd5e1'
matplotlib.rcParams['axes.linewidth'] = 0.8

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features import (engineer_features, NUMERIC_FEATURES,
                           CATEGORICAL_FEATURES)

MODEL_PATH = ROOT / "models" / "best_model.joblib"
DATASET_PATH = ROOT / "data" / "processed_dataset.csv"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sydney Housing Price Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Basic Design & Grey Sidebar ────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }

    /* Basic Grey Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #e9ecef !important;
        border-right: 1px solid #ced4da;
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #1e293b !important;
        font-weight: 600;
        font-size: 0.88rem;
    }

    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }

    /* Basic Main Container */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Header styling */
    .basic-header {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    .basic-header h1 {
        color: #0f172a;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }

    .basic-header p {
        color: #475569;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Price card styling */
    .basic-price-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #1e293b;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    .basic-price-card .label {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .basic-price-card .amount {
        color: #0f172a;
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0.3rem 0;
    }

    .basic-price-card .range {
        color: #475569;
        font-size: 0.88rem;
    }

    /* Metric cards */
    .basic-metric {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.85rem;
        text-align: center;
    }

    .basic-metric .metric-label {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .basic-metric .metric-value {
        color: #0f172a;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    /* Disclaimer box */
    .basic-disclaimer {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #64748b;
        border-radius: 6px;
        padding: 0.9rem 1.2rem;
        color: #64748b;
        font-size: 0.82rem;
        margin-top: 2rem;
        line-height: 1.5;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: background-color 0.15s ease-in-out;
    }

    .stButton > button:hover {
        background-color: #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model & dataset ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    if not DATASET_PATH.exists():
        return None
    return pd.read_csv(DATASET_PATH)

model = load_model()
df_data = load_dataset()

# ── Main Page Header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="basic-header">
    <h1>Sydney Housing Price Predictor</h1>
    <p>Decision Support Tool & Market Analytics · Surry Hills · Parramatta · Penrith</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Property Features")
    st.markdown("---")

    suburb = st.selectbox(
        "Suburb",
        options=["Surry Hills", "Parramatta", "Penrith"],
        index=0,
        help="Select target Sydney suburb.",
    )

    property_type = st.selectbox(
        "Property Type",
        options=["Unit", "Townhouse", "House"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Dwelling Details**")

    bedrooms = st.slider("Bedrooms", min_value=1, max_value=6, value=3)
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=4, value=2)
    car_spaces = st.slider("Car Spaces", min_value=0, max_value=3, value=1)

    st.markdown("---")
    st.markdown("**Dimensions & Age**")

    land_size = st.number_input(
        "Land Size (m²) — (0 for units)",
        min_value=0, max_value=2000, value=0, step=10,
    )
    floor_area = st.number_input(
        "Floor Area (m²)",
        min_value=30, max_value=500, value=120, step=5,
    )
    year_built = st.number_input(
        "Year Built",
        min_value=1900, max_value=2024, value=2000, step=1,
    )
    distance_to_cbd = st.number_input(
        "Distance to CBD (km)",
        min_value=1.0, max_value=80.0,
        value={"Surry Hills": 3.5, "Parramatta": 25.0, "Penrith": 54.0}.get(suburb, 25.0),
        step=0.5,
    )

    st.markdown("---")
    predict_btn = st.button("Predict Price", type="primary")

# ── Main Panel Top: Price Prediction ──────────────────────────────────────────
input_data = pd.DataFrame([{
    "suburb": suburb,
    "postcode": {"Surry Hills": 2010, "Parramatta": 2150, "Penrith": 2750}[suburb],
    "property_type": property_type,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "car_spaces": car_spaces,
    "land_size_m2": float(land_size) if land_size > 0 else np.nan,
    "floor_area_m2": float(floor_area),
    "year_built": int(year_built),
    "sale_date": pd.Timestamp("2024-06-01"),
    "distance_to_cbd_km": float(distance_to_cbd),
    "agent_description": "",
}])
input_data = engineer_features(input_data, reference_year=2024)

features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
X_pred = input_data[features]

if model is not None:
    log_pred = model.predict(X_pred)[0]
    pred_price = np.exp(log_pred)
    low = pred_price * 0.85
    high = pred_price * 1.15
    price_per_m2 = pred_price / floor_area if floor_area > 0 else 0
else:
    pred_price = None

col_pred1, col_pred2 = st.columns([3, 2])

with col_pred1:
    if pred_price is not None:
        st.markdown(f"""
        <div class="basic-price-card">
            <div class="label">Estimated Sale Price</div>
            <div class="amount">${pred_price:,.0f}</div>
            <div class="range">Indicative Valuation Range: ${low:,.0f} – ${high:,.0f} (±15%)</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Model file not found. Please train model using script or notebook.")

with col_pred2:
    if pred_price is not None:
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="basic-metric">
                <div class="metric-label">Location & Type</div>
                <div class="metric-value">{suburb}<br><span style="font-size:0.85rem; font-weight:500; color:#64748b;">{property_type}</span></div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="basic-metric">
                <div class="metric-label">Unit Rate</div>
                <div class="metric-value">${price_per_m2:,.0f}<br><span style="font-size:0.85rem; font-weight:500; color:#64748b;">per m²</span></div>
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Page Graphs Section ──────────────────────────────────────────────────
st.markdown("### Main Page Market Graphs & Insights")
st.markdown("Explore feature drivers, suburb price distributions, and market trends below.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Feature Importance", 
    "Suburb Price Distribution", 
    "Price vs. Floor Area", 
    "Property Type Comparison"
])

# ── Tab 1: Feature Importance ─────────────────────────────────────────────────
with tab1:
    col_g1, col_t1 = st.columns([3, 2])
    with col_g1:
        if model is not None:
            try:
                preprocessor = model.named_steps["preprocessor"]
                cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
                cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
                all_names = NUMERIC_FEATURES + cat_names
                importances = model.named_steps["model"].feature_importances_

                imp_df = pd.DataFrame({"Feature": all_names, "Importance": importances})
                imp_df = imp_df.sort_values("Importance", ascending=False).head(10)

                fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#ffffff')

                y_pos = np.arange(len(imp_df))
                bars = ax.barh(y_pos, imp_df["Importance"][::-1], color="#334155", alpha=0.9, height=0.6)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(imp_df["Feature"][::-1], fontsize=9, color="#1e293b")
                ax.set_xlabel("Relative Importance", fontsize=9, color="#475569")
                ax.set_title("Gradient Boosting Model — Top Feature Importances", fontsize=11, fontweight="bold", color="#0f172a", pad=12)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#cbd5e1')
                ax.spines['bottom'].set_color('#cbd5e1')
                ax.grid(axis="x", color="#f1f5f9", linestyle="--")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error plotting feature importances: {e}")
        else:
            st.info("Train model to display feature importance graph.")

    with col_t1:
        st.markdown("#### Key Drivers Summary")
        st.markdown("""
        - **Distance to CBD**: Primary driver of baseline property valuation in Sydney.
        - **Floor Area ($m^2$)**: Strong positive correlation with property price.
        - **Property Type**: Units exhibit lower median prices compared to freestanding houses.
        - **Bathrooms & Bedrooms**: Additional bathrooms contribute significantly to premium valuation.
        """)

# ── Tab 2: Suburb Price Distribution ──────────────────────────────────────────
with tab2:
    if df_data is not None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')

        suburbs = ["Surry Hills", "Parramatta", "Penrith"]
        data_to_plot = [df_data[df_data["suburb"] == s]["sale_price_aud"] / 1e6 for s in suburbs]

        bp = ax.boxplot(data_to_plot, patch_artist=True, labels=suburbs, widths=0.4)
        
        colors = ["#1e293b", "#64748b", "#cbd5e1"]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_edgecolor("#0f172a")
        
        for median in bp['medians']:
            median.set(color='#0f172a', linewidth=2)

        ax.set_ylabel("Sale Price ($ Millions AUD)", fontsize=9, color="#475569")
        ax.set_title("Sale Price Distribution across Suburbs", fontsize=11, fontweight="bold", color="#0f172a", pad=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.grid(axis="y", color="#f1f5f9", linestyle="--")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Dataset not loaded.")

# ── Tab 3: Price vs Floor Area ────────────────────────────────────────────────
with tab3:
    if df_data is not None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')

        suburb_colors = {"Surry Hills": "#1e293b", "Parramatta": "#64748b", "Penrith": "#cbd5e1"}
        for s, color in suburb_colors.items():
            sub_df = df_data[df_data["suburb"] == s]
            ax.scatter(sub_df["floor_area_m2"], sub_df["sale_price_aud"] / 1e6, 
                       label=s, color=color, alpha=0.85, edgecolors='#0f172a', linewidth=0.5, s=45)

        ax.set_xlabel("Floor Area ($m^2$)", fontsize=9, color="#475569")
        ax.set_ylabel("Sale Price ($ Millions AUD)", fontsize=9, color="#475569")
        ax.set_title("Property Sale Price vs. Floor Area", fontsize=11, fontweight="bold", color="#0f172a", pad=12)
        ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#e2e8f0", fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.grid(color="#f1f5f9", linestyle="--")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Dataset not loaded.")

# ── Tab 4: Property Type Comparison ───────────────────────────────────────────
with tab4:
    if df_data is not None:
        grouped = df_data.groupby(["suburb", "property_type"])["sale_price_aud"].median().unstack() / 1e6
        
        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')

        grey_colors = ["#1e293b", "#64748b", "#cbd5e1"]
        grouped.plot(kind="bar", ax=ax, color=grey_colors, width=0.7, edgecolor="#0f172a", linewidth=0.5, alpha=0.85)

        ax.set_xlabel("Suburb", fontsize=9, color="#475569")
        ax.set_ylabel("Median Price ($ Millions AUD)", fontsize=9, color="#475569")
        ax.set_title("Median Sale Price by Suburb & Property Type", fontsize=11, fontweight="bold", color="#0f172a", pad=12)
        ax.tick_params(axis='x', rotation=0)
        ax.legend(title="Property Type", frameon=True, facecolor="#f8fafc", edgecolor="#e2e8f0", fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.grid(axis="y", color="#f1f5f9", linestyle="--")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Dataset not loaded.")

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="basic-disclaimer">
    <strong>Educational Use Only.</strong>
    This decision support tool is trained on ~120 property listings across Surry Hills, Parramatta, and Penrith.
    Estimates are indicative and provided for academic evaluation (Deakin University Task 8.1). 
    Do not rely on estimates for actual commercial or financial transactions.
</div>
""", unsafe_allow_html=True)
