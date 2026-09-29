"""
streamlit_app.py
-----------------
Road Accident Risk Prediction System — Streamlit Web Application

This app has 3 core tabs:
1. 🗺️ City Risk Map — Visualizes road risk levels (Red = High, Orange = Medium, Green = Low)
2. 🔮 Predict Risk — Enter road and environmental conditions to predict accident danger
3. 📊 Model Performance — Compares models and displays evaluation metrics
"""

import json
from pathlib import Path
import folium
from folium.plugins import HeatMap
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Page Configuration
st.set_page_config(
    page_title="Road Accident Risk Prediction System",
    page_icon="🚧",
    layout="wide"
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "processed" / "grid_risk.csv"


@st.cache_resource
def load_ml_models():
    """Loads pre-trained model and preprocessing objects."""
    model = joblib.load(MODELS_DIR / "risk_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_columns.pkl")

    # Load summary metrics
    results = {}
    if (MODELS_DIR / "results_summary.json").exists():
        with open(MODELS_DIR / "results_summary.json") as f:
            results = json.load(f)

    return model, scaler, label_encoder, feature_cols, results


@st.cache_data
def load_road_data():
    """Loads processed road grid risk table."""
    return pd.read_csv(DATA_FILE)


def get_color(risk_level: str) -> str:
    """Returns color code for risk levels."""
    if risk_level == "High":
        return "#dc3545"  # Red
    elif risk_level == "Medium":
        return "#fd7e14"  # Orange
    else:
        return "#28a745"  # Green


def main():
    st.title("🚧 Road Accident Risk Prediction System")
    st.caption("A machine learning system to identify and predict high-risk accident locations.")

    # Check if files exist
    if not (MODELS_DIR / "risk_model.pkl").exists() or not DATA_FILE.exists():
        st.error("Model artifacts or processed data not found. Please run preprocessing and training first.")
        return

    model, scaler, label_encoder, feature_cols, results = load_ml_models()
    df = load_road_data()

    # Top KPI Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    total_roads = len(df)
    high_risk_roads = int((df["Risk_Level"] == "High").sum())
    med_risk_roads = int((df["Risk_Level"] == "Medium").sum())
    low_risk_roads = int((df["Risk_Level"] == "Low").sum())

    col1.metric("Total Monitored Segments", f"{total_roads:,}")
    col2.metric("🔴 High Risk (Red Alert)", f"{high_risk_roads:,}", f"{(high_risk_roads/total_roads)*100:.1f}%")
    col3.metric("🟠 Medium Risk", f"{med_risk_roads:,}")
    col4.metric("🟢 Low Risk", f"{low_risk_roads:,}")

    # 3 Simple Tabs
    tab1, tab2, tab3 = st.tabs(["🗺️ City Risk Map", "🔮 Predict Road Risk", "📊 Model Performance & Metrics"])

    # =========================================================================
    # TAB 1: CITY RISK MAP
    # =========================================================================
    with tab1:
        st.subheader("Interactive Accident Risk Map")
        st.write("Explore road accident hotspots. Red circles mark dangerous high-risk locations.")

        filter_col1, filter_col2 = st.columns([1, 1])
        cities = ["All Cities"] + sorted(df["City"].unique().tolist())

        with filter_col1:
            selected_city = st.selectbox("Select City", cities)
        with filter_col2:
            show_only_danger = st.checkbox("Show Only High Risk (Red Alert) Hotspots", value=False)

        # Filter data
        city_df = df.copy()
        if selected_city != "All Cities":
            city_df = city_df[city_df["City"] == selected_city]

        if show_only_danger:
            city_df = city_df[city_df["Risk_Level"] == "High"]

        # Build Map
        center_lat = city_df["GridLat"].mean()
        center_lng = city_df["GridLng"].mean()
        zoom_level = 12 if selected_city != "All Cities" else 5

        risk_map = folium.Map(location=[center_lat, center_lng], zoom_start=zoom_level, tiles="OpenStreetMap")

        # Plot markers
        for _, row in city_df.iterrows():
            marker_color = get_color(row["Risk_Level"])
            radius = 6 if row["Risk_Level"] == "High" else 4

            popup_text = f"""
            <b>Corridor:</b> {row['PrimaryRoad']}<br>
            <b>City:</b> {row['City']}, {row['State']}<br>
            <b>Risk Level:</b> {row['Risk_Level']}<br>
            <b>Historical Crashes:</b> {int(row['AccidentCount'])}<br>
            <b>Average Severity:</b> {row['AvgSeverity']:.2f} / 4.0
            """

            folium.CircleMarker(
                location=[row["GridLat"], row["GridLng"]],
                radius=radius,
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.8,
                popup=folium.Popup(popup_text, max_width=250),
                tooltip=f"{row['PrimaryRoad']} ({row['Risk_Level']} Risk)"
            ).add_to(risk_map)

        # Render Map with explicit dimensions (just like Claude's code)
        st_folium(risk_map, width=1100, height=550, returned_objects=[])

        # Top Dangerous Roads in this City
        st.markdown(f"#### 🚨 Top Dangerous Road Corridors ({selected_city})")
        danger_roads = city_df.sort_values(by=["RiskScore", "AccidentCount"], ascending=False).head(5)

        summary_table = pd.DataFrame({
            "Road / Intersection": danger_roads["PrimaryRoad"],
            "City": danger_roads["City"] + ", " + danger_roads["State"],
            "Past Crashes": danger_roads["AccidentCount"].astype(int),
            "Risk Category": danger_roads["Risk_Level"].apply(
                lambda x: f"🔴 {x}" if x == "High" else (f"🟠 {x}" if x == "Medium" else f"🟢 {x}")
            ),
            "Major Hazard": danger_roads["PrimaryDangerFactor"],
        })
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: PREDICT ROAD RISK
    # =========================================================================
    with tab2:
        st.subheader("Predict Accident Risk for Any Road / Conditions")
        st.write("Enter road infrastructure and environmental conditions to predict whether the road is Low, Medium, or High risk.")

        p_col1, p_col2 = st.columns(2)

        with p_col1:
            st.markdown("##### 🚦 Road Features")
            is_junction = st.checkbox("Highway Junction / Ramp present?", value=True)
            is_signal = st.checkbox("Traffic Signal present?", value=False)
            is_crossing = st.checkbox("Pedestrian Crossing present?", value=False)

            st.markdown("##### ⏰ Time of Day")
            is_rush_hour = st.checkbox("Rush Hour Traffic (7-9 AM or 4-7 PM)", value=True)
            is_weekend = st.checkbox("Weekend driving", value=False)

        with p_col2:
            st.markdown("##### 🌧️ Weather Conditions")
            weather = st.selectbox(
                "Weather Condition",
                ["Clear / Sunny", "Rain / Wet Road", "Heavy Rain / Storm", "Fog / Low Visibility", "Snow / Ice"]
            )

            # Map weather dropdown to model features
            if "Heavy Rain" in weather:
                vis = 3.0
                precip = 0.60
                bad_weather = 1.0
            elif "Rain" in weather:
                vis = 6.0
                precip = 0.20
                bad_weather = 1.0
            elif "Fog" in weather:
                vis = 1.5
                precip = 0.05
                bad_weather = 1.0
            elif "Snow" in weather:
                vis = 2.0
                precip = 0.35
                bad_weather = 1.0
            else:
                vis = 10.0
                precip = 0.0
                bad_weather = 0.0

            wind = st.slider("Wind Speed (mph)", min_value=0.0, max_value=35.0, value=8.0)

        st.markdown("---")
        if st.button("🔮 Predict Risk Level", type="primary", use_container_width=True):
            input_data = {
                "PctJunction": 1.0 if is_junction else 0.0,
                "PctCrossing": 1.0 if is_crossing else 0.0,
                "PctTrafficSignal": 1.0 if is_signal else 0.0,
                "PctRushHour": 1.0 if is_rush_hour else 0.0,
                "PctWeekend": 1.0 if is_weekend else 0.0,
                "AvgVisibility": vis,
                "AvgWindSpeed": wind,
                "AvgPrecipitation": precip,
                "PctBadWeather": bad_weather,
            }

            input_df = pd.DataFrame([input_data])[feature_cols]
            input_scaled = scaler.transform(input_df)

            # Model prediction
            prediction_encoded = model.predict(input_scaled)[0]
            prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

            st.markdown("### Prediction Result")
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if prediction_label == "High":
                    st.error("### 🔴 HIGH RISK (RED ALERT)")
                    st.write("**High probability of vehicle collision.** Drivers should reduce speed and traffic authorities should consider targeted patrols or warning signage.")
                elif prediction_label == "Medium":
                    st.warning("### 🟠 MEDIUM RISK (WARNING)")
                    st.write("**Moderate crash probability.** Caution advised during peak traffic hours or wet weather.")
                else:
                    st.success("### 🟢 LOW RISK (SAFE)")
                    st.write("**Standard safety profile.** Normal driving conditions.")

            with res_col2:
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_scaled)[0]
                    prob_df = pd.DataFrame({
                        "Risk Level": label_encoder.classes_,
                        "Probability": probabilities
                    }).sort_values("Probability", ascending=False)
                    st.write("**Model Confidence:**")
                    st.bar_chart(prob_df.set_index("Risk Level"))

    # =========================================================================
    # TAB 3: MODEL PERFORMANCE & METRICS
    # =========================================================================
    with tab3:
        st.subheader("Model Performance & Evaluation Metrics")

        st.info("""
        💡 **Research & Methodology Highlight: Preventing Target Leakage**:
        - In baseline models, using accident counts as inputs resulted in an artificial **0.99 Macro F1** (Target Leakage).
        - We eliminated proxy features to retain only **independent physical road features (Junctions, Signals), Time, and Atmospheric Conditions**.
        - The resulting **~0.49 Macro F1** (with XGBoost) demonstrates robust real-world generalization without data contamination.
        """)

        eval_col1, eval_col2 = st.columns(2)

        with eval_col1:
            st.markdown("#### 1. Model Comparison (Macro F1 Score)")
            if results:
                comparison_df = pd.DataFrame([
                    {"Model": name, "Macro F1 Score": metrics["macro_f1"]}
                    for name, metrics in results.items()
                ]).sort_values("Macro F1 Score", ascending=False)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                st.bar_chart(comparison_df.set_index("Model"))

        with eval_col2:
            st.markdown("#### 2. System Architecture & Methodology")
            st.markdown("""
            * **Problem Formulation**: Multi-class Spatial Classification (`Low`, `Medium`, `High` Risk).
            * **Class Balancing**: Applied **SMOTE** (Synthetic Minority Over-sampling Technique) on training data to balance minority high-risk segments.
            * **Evaluation Metric**: **Macro F1-Score** (Accounts for class imbalance better than standard accuracy).
            * **Top Performing Algorithm**: **XGBoost Classifier** (Captures complex non-linear interactions across features).
            * **Spatial Aggregation**: **0.05° Grid Cells (~5 km)** for continuous corridor risk mapping.
            """)


if __name__ == "__main__":
    main()
