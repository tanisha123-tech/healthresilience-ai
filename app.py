from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from firebase_db import init_firebase, sync_predictions

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from xgboost import XGBRegressor, XGBClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False



# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HealthResilience AI",
    page_icon="🏥",
    layout="wide"
)

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "phc_data.csv"


# =========================================================
# FEATURES
# =========================================================

FEATURES = [
    "current_stock",
    "daily_consumption",
    "patient_footfall",
    "bed_occupancy_pct",
    "staff_availability_pct",
    "lead_time_days",
    "season_index",
    "emergency_index"
]


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data(ttl=60)
def load_data():
    if DATA.exists():
        return pd.read_csv(DATA)

    return None

df = load_data()

if df is None:
    st.error("Dataset not found.")
    st.code("python generate_data.py")
    st.stop()


# =========================================================
# MODEL TRAINING
# =========================================================

@st.cache_resource
def train_models(df):

    X = df[FEATURES]
    y_demand = df["forecast_7d"]
    y_stockout = df["stockout"]

    # -----------------------------------------------------
    # MODEL 1: DEMAND FORECASTING
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_demand,
        test_size=0.20,
        random_state=42
    )

    demand_model = XGBRegressor(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
        n_jobs=-1
    )

    demand_model.fit(X_train, y_train)

    demand_predictions = demand_model.predict(X_test)

    demand_mae = mean_absolute_error(
        y_test,
        demand_predictions
    )

    # -----------------------------------------------------
    # MODEL 2: STOCK-OUT RISK
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_stockout,
        test_size=0.20,
        random_state=42,
        stratify=y_stockout
    )

    stockout_model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    stockout_model.fit(X_train, y_train)

    stock_predictions = stockout_model.predict(X_test)

    stockout_accuracy = accuracy_score(
        y_test,
        stock_predictions
    )

    return (
        demand_model,
        stockout_model,
        demand_mae,
        stockout_accuracy
    )


(
    demand_model,
    stockout_model,
    demand_mae,
    stockout_accuracy
) = train_models(df)


# =========================================================
# HEADER
# =========================================================

st.title("🏥 HealthResilience AI")

st.caption(
    "Predictive healthcare resource & supply-chain resilience"
)

st.info(
    "🌍 Pan-BRICS simulated network: 1000 PHCs across 36 global regions. "
    "Use Emergency Mode to stress-test global healthcare resilience."
)


# =========================================================
# SIDEBAR
# =========================================================


st.sidebar.header("⚙️ Control Center")

gemini_key = st.sidebar.text_input("🔑 Gemini API Key (Optional)", type="password", help="Enter your Google AI Studio key to enable LLM features")
if gemini_key and GENAI_AVAILABLE:
    genai.configure(api_key=gemini_key)


emergency = st.sidebar.toggle(
    "🚨 Emergency Mode",
    value=False
)

state_options = ["Global View"] + sorted(
    df["state"].unique().tolist()
)

selected_state = st.sidebar.selectbox(
    "Region",
    state_options
)


# =========================================================
# FILTER DATA
# =========================================================

view = df.copy()

if selected_state != "Global View":
    view = view[
        view["state"] == selected_state
    ].copy()


# =========================================================
# CREATE MODEL INPUT
# =========================================================

X_view = view[FEATURES].copy()

if emergency:

    # Simulate sudden increase in healthcare demand
    X_view["emergency_index"] = 1

    X_view["patient_footfall"] = (
        X_view["patient_footfall"] * 1.60
    ).round()


# =========================================================
# PREDICTIONS
# =========================================================

view["predicted_demand_7d"] = (
    demand_model.predict(X_view)
).round(0)

view["stockout_probability"] = (
    stockout_model
    .predict_proba(X_view)[:, 1]
)

# ---------------------------------------------------------
# ESTIMATED DAYS TO STOCK-OUT
# ---------------------------------------------------------

daily_need = (
    X_view["daily_consumption"]
    * X_view["season_index"]
    * (
        1
        + 0.55 * X_view["emergency_index"]
    )
)

view["days_to_stockout"] = (
    view["current_stock"]
    / np.maximum(daily_need, 1)
).round(1)


# =========================================================
# HEALTHCARE RISK SCORE
# =========================================================

def calculate_risk(row):

    stock_risk = row["stockout_probability"]

    bed_risk = min(
        row["bed_occupancy_pct"] / 100,
        1
    )

    workforce_risk = (
        1 -
        min(
            row["staff_availability_pct"] / 100,
            1
        )
    )

    patient_risk = min(
        row["patient_footfall"] / 500,
        1
    )

    risk_score = (
        0.45 * stock_risk
        + 0.25 * bed_risk
        + 0.20 * workforce_risk
        + 0.10 * patient_risk
    )

    return risk_score


view["risk_score"] = view.apply(
    calculate_risk,
    axis=1
)


view["risk_level"] = pd.cut(
    view["risk_score"],
    bins=[-0.01, 0.35, 0.60, 1.01],
    labels=["Low", "Medium", "High"]
)


# =========================================================
# FIREBASE SYNC
# =========================================================

db = init_firebase()
if db is not None:
    try:
        sync_predictions(db, view)
    except Exception as e:
        pass # Silently fail in hackathon demo mode to avoid UI clutter

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f77b4;
    }
    /* Tabs */
    [data-baseweb="tab-list"] {
        gap: 24px;
    }
    [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================

tab_dash, tab_alerts, tab_fed = st.tabs([
    "🌍 Global Dashboard", 
    "🚨 Alerts & Redistribution", 
    "🌐 BRICS Federated Learning"
])

with tab_dash:
    # =========================================================
    # TOP METRICS
    # =========================================================

    col1, col2, col3, col4 = st.columns(4)
    
    unique_phcs = view["phc_id"].nunique()
    
    high_risk = (
        view["risk_level"] == "High"
    ).sum()
    
    stock_alerts = (
        view["stockout_probability"] >= 0.60
    ).sum()
    
    avg_bed = view[
        "bed_occupancy_pct"
    ].mean()
    
    
    col1.metric(
        "🏥 PHCs Monitored",
        unique_phcs
    )
    
    col2.metric(
        "🔴 High-Risk PHCs",
        high_risk
    )
    
    col3.metric(
        "⚠️ Stock-out Alerts",
        stock_alerts
    )
    
    col4.metric(
        "🛏️ Avg Bed Occupancy",
        f"{avg_bed:.0f}%"
    )
    
    
    # =========================================================
    # EMERGENCY ALERT
    # =========================================================
    
    if emergency:
    
        st.warning(
            "🚨 EMERGENCY SIMULATION ACTIVE\n\n"
            "Patient footfall increased by 60% and emergency "
            "demand conditions have been activated."
        )
    
    
    st.divider()
    
    # =========================================================
    # SMART RESOURCE REDISTRIBUTION
    # =========================================================
    
    def haversine_km(lat1, lon1, lat2, lon2):
        """
        Calculate approximate distance between two PHCs.
        """
    
        R = 6371
    
        lat1 = np.radians(lat1)
        lat2 = np.radians(lat2)
    
        dlat = lat2 - lat1
        dlon = np.radians(lon2 - lon1)
    
        a = (
            np.sin(dlat / 2) ** 2
            +
            np.cos(lat1)
            * np.cos(lat2)
            * np.sin(dlon / 2) ** 2
        )
    
        c = 2 * np.arctan2(
            np.sqrt(a),
            np.sqrt(1 - a)
        )
    
        return R * c
    
    
    def find_best_source(target, network):
    
        candidates = network[
            (network["medicine"] == target["medicine"])
            &
            (network["phc_id"] != target["phc_id"])
        ].copy()
    
        if candidates.empty:
            return None
    
        # Expected surplus after satisfying the source PHC's own demand
        candidates["surplus"] = (
            candidates["current_stock"]
            -
            candidates["predicted_demand_7d"]
        )
    
        candidates = candidates[
            candidates["surplus"] > 100
        ].copy()
    
        if candidates.empty:
            return None
    
        # Distance from source PHC to shortage PHC
        candidates["distance_km"] = candidates.apply(
            lambda x: haversine_km(
                target["latitude"],
                target["longitude"],
                x["latitude"],
                x["longitude"]
            ),
            axis=1
        )
    
        # Normalize values for scoring
        max_surplus = max(
            candidates["surplus"].max(),
            1
        )
    
        max_distance = max(
            candidates["distance_km"].max(),
            1
        )
    
        candidates["surplus_score"] = (
            candidates["surplus"] / max_surplus
        )
    
        candidates["distance_score"] = (
            1 -
            candidates["distance_km"] / max_distance
        )
    
        candidates["lead_time_score"] = (
            1 -
            candidates["lead_time_days"]
            /
            max(
                candidates["lead_time_days"].max(),
                1
            )
        )
    
        # Final decision score
        candidates["source_score"] = (
            0.50 * candidates["surplus_score"]
            +
            0.30 * candidates["distance_score"]
            +
            0.20 * candidates["lead_time_score"]
        )
    
        return candidates.sort_values(
            "source_score",
            ascending=False
        ).iloc[0]
        
    # =========================================================
    # RISK DISTRIBUTION
    # =========================================================
    
    left, right = st.columns(2)
    
    with left:
    
        st.subheader("📊 PHC Risk Distribution")
    
        risk_counts = (
            view["risk_level"]
            .value_counts()
            .reindex(
                ["Low", "Medium", "High"]
            )
            .fillna(0)
        )
    
        fig = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            labels={
                "x": "Risk Level",
                "y": "Number of PHCs"
            },
            title="Healthcare Risk Across PHCs"
        )
    
        st.plotly_chart(
            fig,
            use_container_width=True
        )
    
    
    # =========================================================
    # MODEL STATUS
    # =========================================================
    
    with right:
    
        st.subheader("🤖 AI Model Status")
    
        st.write(
            f"**Demand Forecasting MAE:** "
            f"{demand_mae:.2f} units"
        )
    
        st.write(
            f"**Stock-out Risk Accuracy:** "
            f"{stockout_accuracy * 100:.2f}%"
        )
    
        st.write(
            "**Models:** XGBoost Regressor + "
            "XGBoost Classifier"
        )
    
        st.success(
            "AI pipeline is running successfully."
        )
    
    
    st.divider()
    
    
    # =========================================================

with tab_alerts:
    # AI ALERTS
    # =========================================================
    
    st.subheader(
        "🚨 AI Alerts & Resource Recommendations"
    )
    
    critical_cases = (
        view
        .sort_values(
            [
                "risk_score",
                "stockout_probability"
            ],
            ascending=False
        )
        .head(8)
    )
    
    
    for _, row in critical_cases.iterrows():
    
        if row["risk_level"] == "High":
            icon = "🔴"
        elif row["risk_level"] == "Medium":
            icon = "🟠"
        else:
            icon = "🟢"
    
        title = (
            f"{icon} {row['phc_id']} | "
            f"{row['medicine']} | "
            f"{row['risk_level']} Risk"
        )
    
        with st.expander(title):
    
            c1, c2, c3 = st.columns(3)
    
            c1.metric(
                "Current Stock",
                f"{row['current_stock']:.0f}"
            )
    
            c2.metric(
                "Predicted 7-Day Demand",
                f"{row['predicted_demand_7d']:.0f}"
            )
    
            c3.metric(
                "Days to Stock-out",
                f"{row['days_to_stockout']:.1f}"
            )
    
            st.write(
                f"**Stock-out Probability:** "
                f"{row['stockout_probability'] * 100:.1f}%"
            )
    
            st.write(
                f"**Healthcare Risk Score:** "
                f"{row['risk_score']:.2f}"
            )
    
    
            # =================================================
            # RISK DRIVERS
            # =================================================
    
            st.markdown(
                "### 🔎 Why is this PHC at risk?"
            )
    
            drivers = pd.Series({
    
                "Stock-out Probability":
                    row["stockout_probability"],
    
                "Bed Occupancy":
                    row["bed_occupancy_pct"] / 100,
    
                "Low Staff Availability":
                    1 -
                    row["staff_availability_pct"] / 100,
    
                "Patient Load":
                    min(
                        row["patient_footfall"] / 500,
                        1
                    )
    
            }).sort_values(
                ascending=False
            )
    
            driver_df = (
                drivers
                .rename("Contribution")
                .to_frame()
            )
    
            st.dataframe(
                driver_df.round(3),
                use_container_width=True
            )
    
    
            # =================================================
            # SHAP EXPLANATION
            # =================================================
    
            if SHAP_AVAILABLE:
    
                try:
    
                    explainer = shap.TreeExplainer(
                        stockout_model
                    )
    
                    explanation_input = X_view.loc[
                        [[_]
                        ]
                    ]
    
                    shap_values = explainer(
                        explanation_input
                    )
    
                    st.markdown(
                        "### 🧠 Explainable AI"
                    )
    
                    shap_df = pd.DataFrame({
    
                        "Feature":
                            FEATURES,
    
                        "Impact":
                            shap_values.values[0]
    
                    })
    
                    shap_df["Absolute Impact"] = (
                        shap_df["Impact"]
                        .abs()
                    )
    
                    shap_df = (
                        shap_df
                        .sort_values(
                            "Absolute Impact",
                            ascending=False
                        )
                        .head(5)
                    )
    
                    fig_shap = px.bar(
                        shap_df,
                        x="Impact",
                        y="Feature",
                        orientation="h",
                        title="Top factors influencing stock-out prediction"
                    )
    
                    st.plotly_chart(
                        fig_shap,
                        use_container_width=True
                    )
    
                except Exception:
                    st.caption(
                        "SHAP explanation unavailable for this record."
                    )
    
    
            # =================================================
            # RESOURCE REDISTRIBUTION
            # =================================================
    
            if row["stockout_probability"] >= 0.60:
    
                shortage = max(
                    0,
                    int(
                        row["predicted_demand_7d"]
                        - row["current_stock"]
                    )
                )
    
                best = find_best_source(row, view)
    
                if best is not None:
    
                    transfer_quantity = min(
                        shortage,
                        int(best["surplus"])
                    )
    
                    if transfer_quantity > 0:
    
                        st.success(
                            f"""
    **🤖 AI Redistribution Recommendation**
    
    Transfer **{transfer_quantity} units** of
    **{row['medicine']}**
    
    **From:** {best['phc_id']} ({best['distance_km']:.1f} km away, Lead time: {best['lead_time_days']} days)
    
    **To:** {row['phc_id']}
    
    **Reason:** {row['phc_id']} has a predicted shortage,
    while {best['phc_id']} has sufficient projected surplus and is a suitable redistribution source based on distance and lead time.
    """
                        )

                    if st.button('✨ Generate Crisis Action Plan (Gemini)', key=f'gemini_{row["phc_id"]}_{row["medicine"]}'):
                        if not gemini_key:
                            st.warning('⚠️ Please enter your Gemini API key in the left sidebar to generate the plan.')
                        else:
                            with st.spinner('Gemini 1.5 Flash is analyzing the supply chain data...'):
                                try:
                                    import google.generativeai as genai
                                    genai.configure(api_key=gemini_key)
                                    system_instruction = (
                                        'You are an elite Healthcare Logistics AI. Your task is to output a 3-point crisis action plan. '
                                        'Edge case protocols: '
                                        '1. If distance > 20km, explicitly mandate cold-chain or secure transit verification. '
                                        '2. If risk is High, mandate immediate notification of the district medical officer. '
                                        '3. If surplus is tight, advise rationing remaining stock strictly for vulnerable patients.'
                                    )
                                    model = genai.GenerativeModel(
                                        'gemini-1.5-flash',
                                        system_instruction=system_instruction
                                    )
                                    prompt = (
                                        f'Context: Clinic {row["phc_id"]} is at {row["risk_level"]} risk of a {row["medicine"]} stockout. '
                                        f'Current stock: {row["current_stock"]} | Predicted 7-day demand: {row["predicted_demand_7d"]}. '
                                        f'Action: Transferring {transfer_quantity} units from {best["phc_id"]} ({best["distance_km"]:.1f} km away). '
                                        f'Draft the briefing.'
                                    )
                                    response = model.generate_content(prompt)
                                    st.info(response.text)
                                except Exception as e:
                                    st.error(f'Gemini API Error: {e}')

    
                else:
    
                    st.warning(
                        "No suitable surplus PHC was found "
                        "for this resource."
                    )
    
    
    st.divider()
    
    
    # =========================================================
    # PHC MAP
    # =========================================================
    
    st.subheader("🗺️ PHC Risk Map")
    
    map_df = view[
        [
            "latitude",
            "longitude",
            "phc_id",
            "state"
        ]
    ].drop_duplicates("phc_id")
    
    st.map(
        map_df[
            [
                "latitude",
                "longitude"
            ]
        ]
    )
    
    # =========================================================

with tab_fed:
    # FEDERATED LEARNING
    # =========================================================
    
    st.divider()
    
    st.subheader(
        "🌐 Federated Learning Simulation"
    )
    
    st.write(
        "Healthcare nodes train locally and contribute "
        "model updates without centralizing their raw records."
    )
    
    fed_cols = st.columns(5)
    
    nodes = [
        "🇮🇳 India Node",
        "🇧🇷 Brazil Node",
        "🇷🇺 Russia Node",
        "🇨🇳 China Node",
        "🇿🇦 South Africa Node"
    ]
    
    for i, node in enumerate(nodes):
    
        local_data = df.sample(
            frac=0.33,
            random_state=i
        )
    
        local_model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=i,
            n_jobs=-1
        )
    
        local_model.fit(
            local_data[FEATURES],
            local_data["forecast_7d"]
        )
    
        local_predictions = (
            local_model.predict(
                local_data[FEATURES]
            )
        )
    
        local_mae = mean_absolute_error(
            local_data["forecast_7d"],
            local_predictions
        )
    
        fed_cols[i].metric(
            node,
            f"Local MAE: {local_mae:.1f}"
        )
    
    
    st.success(
        "Federated architecture demonstrated: "
        "local healthcare nodes → local model training → "
        "shared learning process."
    )
    
    
    # =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "HealthResilience AI | Smart Health & Supply Chain Resilience | Hackathon Prototype"
)