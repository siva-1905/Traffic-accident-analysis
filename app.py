import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import folium

from folium.plugins import HeatMap
from folium import FeatureGroup, LayerControl
from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Traffic Accident Analysis & Prediction",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("traffic_accidents.csv")


@st.cache_resource
def load_models():

    models = {
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "XGBoost": joblib.load("models/xgboost.pkl"),
        "Best Model": joblib.load("models/best_model.pkl")
    }

    target_mapping = joblib.load(
        "models/target_mapping.pkl"
    )

    return models, target_mapping


@st.cache_data
def load_model_results():

    comparison = pd.read_csv(
        "models/model_comparison.csv"
    )

    rf_importance = pd.read_csv(
        "models/random_forest_feature_importance.csv"
    )

    xgb_importance = pd.read_csv(
        "models/xgboost_feature_importance.csv"
    )

    hotspots = pd.read_csv(
        "models/hotspot_summary.csv"
    )

    return comparison, rf_importance, xgb_importance, hotspots


df = load_data()

models, target_mapping = load_models()

# st.cache_resource keeps the loaded model objects in memory.
# Streamlit can rerun this script after widget interaction without
# calling joblib.load() again unless the cache is cleared or the app restarts.

comparison, rf_importance, xgb_importance, hotspots = (
    load_model_results()
)


# Reverse target mapping
reverse_mapping = {
    value: key
    for key, value in target_mapping.items()
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚦 Traffic Accident System")

st.sidebar.markdown(
    """
    ### Navigation

    Use the sections below to explore:

    - Accident overview
    - Geographic hotspots
    - Accident severity prediction
    - ML model performance
    - Feature importance
    """
)

selected_model = st.sidebar.selectbox(
    "Prediction Model",
    [
        "Best Model",
        "Random Forest",
        "XGBoost"
    ]
)

model = models[selected_model]


# ============================================================
# HEADER
# ============================================================

st.title("🚦 Traffic Accident Analysis & Prediction")

st.markdown(
    """
    **Machine Learning based traffic accident severity analysis,
    geographic hotspot detection and prediction system.**
    """
)

st.info(
    "Dataset note: This project uses a synthetic dataset generated "
    "for demonstration and educational purposes."
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Dashboard Filters")

cities = sorted(df["City"].unique())

selected_cities = st.sidebar.multiselect(
    "City",
    cities,
    default=cities
)

severity_options = [
    "Low",
    "Medium",
    "High"
]

selected_severity = st.sidebar.multiselect(
    "Accident Severity",
    severity_options,
    default=severity_options
)

weather_options = sorted(
    df["Weather"].unique()
)

selected_weather = st.sidebar.multiselect(
    "Weather",
    weather_options,
    default=weather_options
)


# Apply filters
filtered_df = df[
    df["City"].isin(selected_cities)
    & df["Accident_Severity"].isin(selected_severity)
    & df["Weather"].isin(selected_weather)
]


# ============================================================
# KPI SECTION
# ============================================================

total_accidents = len(filtered_df)

high_accidents = (
    filtered_df["Accident_Severity"] == "High"
).sum()

medium_accidents = (
    filtered_df["Accident_Severity"] == "Medium"
).sum()

low_accidents = (
    filtered_df["Accident_Severity"] == "Low"
).sum()


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Accidents",
    f"{total_accidents:,}"
)

col2.metric(
    "🔴 High Severity",
    f"{high_accidents:,}"
)

col3.metric(
    "🟡 Medium Severity",
    f"{medium_accidents:,}"
)

col4.metric(
    "🟢 Low Severity",
    f"{low_accidents:,}"
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Overview",
        "🗺️ Accident Hotspots",
        "🤖 Prediction",
        "📈 Model Performance"
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.header("Accident Overview")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # Severity distribution
    # --------------------------------------------------------

    with col1:

        severity_counts = (
            filtered_df["Accident_Severity"]
            .value_counts()
            .reindex(
                ["Low", "Medium", "High"],
                fill_value=0
            )
            .reset_index()
        )

        severity_counts.columns = [
            "Severity",
            "Count"
        ]

        fig = px.bar(
            severity_counts,
            x="Severity",
            y="Count",
            title="Accident Severity Distribution",
            text="Count"
        )

        fig.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Weather distribution
    # --------------------------------------------------------

    with col2:

        weather_counts = (
            filtered_df["Weather"]
            .value_counts()
            .reset_index()
        )

        weather_counts.columns = [
            "Weather",
            "Count"
        ]

        fig = px.pie(
            weather_counts,
            names="Weather",
            values="Count",
            title="Accidents by Weather"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Accidents by city
    # --------------------------------------------------------

    city_counts = (
        filtered_df["City"]
        .value_counts()
        .reset_index()
    )

    city_counts.columns = [
        "City",
        "Accidents"
    ]

    fig = px.bar(
        city_counts,
        x="City",
        y="Accidents",
        title="Accidents by City",
        text="Accidents"
    )

    fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Hourly accident pattern
    # --------------------------------------------------------

    hourly = (
        filtered_df
        .groupby("Hour")
        .size()
        .reset_index(name="Accidents")
    )

    fig = px.line(
        hourly,
        x="Hour",
        y="Accidents",
        markers=True,
        title="Accidents by Hour of Day"
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
            dtick=1
        ),
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Traffic density vs severity
    # --------------------------------------------------------

    density_severity = pd.crosstab(
        filtered_df["Traffic_Density"],
        filtered_df["Accident_Severity"]
    )

    density_severity = density_severity.reindex(
        columns=["Low", "Medium", "High"],
        fill_value=0
    )

    fig = px.bar(
        density_severity,
        barmode="group",
        title="Traffic Density vs Accident Severity"
    )

    fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TAB 2 — ACCIDENT HOTSPOTS
# ============================================================

with tab2:

    st.header("🗺️ Geographic Accident Hotspots")

    st.markdown(
        """
        Explore accident locations, geographic concentration,
        and DBSCAN-detected accident hotspots using the interactive map.
        The map and hotspot statistics respond to the sidebar filters.
        """
    )

    # ========================================================
    # 1. HOTSPOT DATA FOR CURRENT FILTERS
    # ========================================================

    if "Hotspot_Cluster" in filtered_df.columns:

        cluster_df = filtered_df[
            filtered_df["Hotspot_Cluster"] != -1
        ].copy()

    else:

        cluster_df = pd.DataFrame()

    # --------------------------------------------------------
    # Calculate dynamic hotspot summary
    # --------------------------------------------------------

    if not cluster_df.empty:

        filtered_hotspots = (
            cluster_df
            .groupby("Hotspot_Cluster")
            .agg(
                Latitude=("Latitude", "mean"),
                Longitude=("Longitude", "mean"),
                Total_Accidents=("Accident_ID", "count"),

                High_Severity=(
                    "Accident_Severity",
                    lambda x: (x == "High").sum()
                ),

                Medium_Severity=(
                    "Accident_Severity",
                    lambda x: (x == "Medium").sum()
                ),

                Low_Severity=(
                    "Accident_Severity",
                    lambda x: (x == "Low").sum()
                )
            )
            .reset_index()
        )

        # Project-defined hotspot risk score
        filtered_hotspots["Risk_Score"] = (
            filtered_hotspots["Total_Accidents"] * 0.50
            + filtered_hotspots["High_Severity"] * 0.35
            + filtered_hotspots["Medium_Severity"] * 0.15
        )

        filtered_hotspots = (
            filtered_hotspots
            .sort_values(
                "Risk_Score",
                ascending=False
            )
            .reset_index(drop=True)
        )

        filtered_hotspots["Rank"] = (
            filtered_hotspots.index + 1
        )

    else:

        filtered_hotspots = pd.DataFrame(
            columns=[
                "Hotspot_Cluster",
                "Latitude",
                "Longitude",
                "Total_Accidents",
                "High_Severity",
                "Medium_Severity",
                "Low_Severity",
                "Risk_Score",
                "Rank"
            ]
        )

    # ========================================================
    # 2. HOTSPOT KPI CARDS
    # ========================================================

    hotspot_count = len(filtered_hotspots)

    if hotspot_count > 0:
        top_hotspot = filtered_hotspots.iloc[0]
    else:
        top_hotspot = None

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.metric(
            "Detected Hotspots",
            hotspot_count
        )

    with kpi2:

        if top_hotspot is not None:
            st.metric(
                "Top Hotspot Accidents",
                int(top_hotspot["Total_Accidents"])
            )
        else:
            st.metric(
                "Top Hotspot Accidents",
                0
            )

    with kpi3:

        if top_hotspot is not None:
            st.metric(
                "Top Hotspot Risk Score",
                f"{top_hotspot['Risk_Score']:.1f}"
            )
        else:
            st.metric(
                "Top Hotspot Risk Score",
                "0.0"
            )

    # ========================================================
    # 3. INTERACTIVE FOLIUM MAP
    # ========================================================

    st.subheader("Interactive Accident & Hotspot Map")

    if filtered_df.empty:

        st.warning(
            "No accident records match the selected filters."
        )

    else:

        # ----------------------------------------------------
        # Determine map center
        # ----------------------------------------------------

        map_lat = filtered_df["Latitude"].mean()
        map_lon = filtered_df["Longitude"].mean()

        # ----------------------------------------------------
        # CREATE BASE MAP FIRST
        # ----------------------------------------------------

        accident_map = folium.Map(
            location=[map_lat, map_lon],
            zoom_start=7,
            tiles=None,
            control_scale=True
        )

        # ====================================================
        # 4. OPENSTREETMAP BASE MAP
        # ====================================================

        folium.TileLayer(
            tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr="© OpenStreetMap contributors",
            name="OpenStreetMap",
            overlay=False,
            control=True,
            show=True
        ).add_to(accident_map)

        # ====================================================
        # 5. ACCIDENT LOCATION LAYER
        # ====================================================

        accident_layer = folium.FeatureGroup(
            name="Accident Locations",
            show=True
        )

        # Limit markers so the browser does not become slow
        # with thousands of points.
        marker_df = filtered_df.sample(
            n=min(500, len(filtered_df)),
            random_state=42
        )

        for _, row in marker_df.iterrows():

            severity = row["Accident_Severity"]

            # Marker colour according to severity
            if severity == "High":
                marker_color = "red"

            elif severity == "Medium":
                marker_color = "orange"

            else:
                marker_color = "green"

            popup_html = f"""
            <div style="font-family:Arial; font-size:13px;">

                <h4 style="margin-bottom:8px;">
                    🚗 Accident Details
                </h4>

                <b>Accident ID:</b>
                {row["Accident_ID"]}<br>

                <b>City:</b>
                {row["City"]}<br>

                <b>Date:</b>
                {row["Date"]}<br>

                <b>Time:</b>
                {row["Time"]}<br>

                <b>Severity:</b>
                {severity}<br>

                <b>Weather:</b>
                {row["Weather"]}<br>

                <b>Road Condition:</b>
                {row["Road_Condition"]}<br>

                <b>Light Condition:</b>
                {row["Light_Condition"]}<br>

                <b>Traffic Density:</b>
                {row["Traffic_Density"]}<br>

                <b>Speed Limit:</b>
                {row["Speed_Limit"]} km/h<br>

                <b>Vehicle Count:</b>
                {row["Vehicle_Count"]}

            </div>
            """

            folium.CircleMarker(
                location=[
                    row["Latitude"],
                    row["Longitude"]
                ],

                radius=4,

                color=marker_color,

                fill=True,

                fill_color=marker_color,

                fill_opacity=0.75,

                weight=1,

                tooltip=(
                    f"{row['City']} | "
                    f"{severity} Severity"
                ),

                popup=folium.Popup(
                    popup_html,
                    max_width=320
                )
            ).add_to(accident_layer)

        accident_layer.add_to(accident_map)

        # ====================================================
        # 6. ACCIDENT HEATMAP
        # ====================================================

        heatmap_layer = folium.FeatureGroup(
            name="Accident Heatmap",
            show=True
        )

        heat_data = (
            filtered_df[
                ["Latitude", "Longitude"]
            ]
            .dropna()
            .values
            .tolist()
        )

        if len(heat_data) > 0:

            HeatMap(
                heat_data,

                radius=20,

                blur=18,

                min_opacity=0.35,

                max_zoom=12
            ).add_to(heatmap_layer)

        heatmap_layer.add_to(accident_map)

        # ====================================================
        # 7. DBSCAN HOTSPOT LAYER
        # ====================================================

        hotspot_layer = folium.FeatureGroup(
            name="DBSCAN Hotspots",
            show=True
        )

        for _, row in filtered_hotspots.head(20).iterrows():

            risk_score = float(
                row["Risk_Score"]
            )

            # Risk colour
            if risk_score >= 70:

                hotspot_color = "red"

            elif risk_score >= 40:

                hotspot_color = "orange"

            else:

                hotspot_color = "blue"

            popup_html = f"""
            <div style="font-family:Arial; font-size:13px;">

                <h4 style="margin-bottom:8px;">
                    🔥 Accident Hotspot
                </h4>

                <b>Rank:</b>
                #{int(row["Rank"])}<br>

                <b>Cluster ID:</b>
                {int(row["Hotspot_Cluster"])}<br>

                <b>Total Accidents:</b>
                {int(row["Total_Accidents"])}<br>

                <b>High Severity:</b>
                {int(row["High_Severity"])}<br>

                <b>Medium Severity:</b>
                {int(row["Medium_Severity"])}<br>

                <b>Low Severity:</b>
                {int(row["Low_Severity"])}<br>

                <b>Risk Score:</b>
                {risk_score:.1f}

            </div>
            """

            folium.Circle(
                location=[
                    row["Latitude"],
                    row["Longitude"]
                ],

                radius=1000,

                color=hotspot_color,

                fill=True,

                fill_color=hotspot_color,

                fill_opacity=0.15,

                weight=3,

                tooltip=(
                    f"Hotspot #{int(row['Rank'])} | "
                    f"Risk Score: {risk_score:.1f}"
                ),

                popup=folium.Popup(
                    popup_html,
                    max_width=300
                )
            ).add_to(hotspot_layer)

        hotspot_layer.add_to(accident_map)

        # ====================================================
        # 8. MAP LEGEND
        # ====================================================

        legend_html = """
        <div style="
            position: fixed;
            bottom: 25px;
            left: 25px;
            z-index: 9999;
            background-color: white;
            padding: 12px 15px;
            border: 2px solid #777;
            border-radius: 6px;
            font-family: Arial;
            font-size: 13px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        ">

            <b>Accident Severity</b><br>

            <span style="color:red; font-size:18px;">
                ●
            </span>
            High<br>

            <span style="color:orange; font-size:18px;">
                ●
            </span>
            Medium<br>

            <span style="color:green; font-size:18px;">
                ●
            </span>
            Low<br>

            <br>

            <b>Hotspot Risk</b><br>

            <span style="color:red;">
                ■
            </span>
            High<br>

            <span style="color:orange;">
                ■
            </span>
            Medium<br>

            <span style="color:blue;">
                ■
            </span>
            Lower

        </div>
        """

        accident_map.get_root().html.add_child(
            folium.Element(legend_html)
        )

        # ====================================================
        # 9. LAYER CONTROL
        # ====================================================

        folium.LayerControl(
            collapsed=False
        ).add_to(accident_map)

        # ====================================================
        # 10. RENDER THE INTERACTIVE MAP
        # ====================================================

        map_result = st_folium(
            accident_map,
            width=1200,
            height=650,
            returned_objects=["last_clicked"],
            key="accident_hotspot_map"
        )

        # ----------------------------------------------------
        # Show clicked coordinates
        # ----------------------------------------------------

        if (
            map_result
            and map_result.get("last_clicked")
        ):

            clicked = map_result["last_clicked"]

            st.caption(
                f"📍 Selected location: "
                f"{clicked['lat']:.6f}, "
                f"{clicked['lng']:.6f}"
            )

    # ========================================================
    # 11. HOTSPOT TABLE
    # ========================================================

    st.subheader("🔥 Top Accident Hotspots")

    if not filtered_hotspots.empty:

        display_hotspots = (
            filtered_hotspots
            .head(15)
            .copy()
        )

        display_hotspots = display_hotspots[
            [
                "Rank",
                "Hotspot_Cluster",
                "Latitude",
                "Longitude",
                "Total_Accidents",
                "High_Severity",
                "Medium_Severity",
                "Low_Severity",
                "Risk_Score"
            ]
        ]

        # Round coordinates and risk score
        display_hotspots["Latitude"] = (
            display_hotspots["Latitude"].round(5)
        )

        display_hotspots["Longitude"] = (
            display_hotspots["Longitude"].round(5)
        )

        display_hotspots["Risk_Score"] = (
            display_hotspots["Risk_Score"].round(2)
        )

        st.dataframe(
            display_hotspots,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No DBSCAN hotspots are available "
            "for the current filter selection."
        )


# ============================================================
# TAB 3 — PREDICTION
# ============================================================

with tab3:

    st.header("🤖 Accident Severity Prediction")

    st.markdown(
        """
        Enter the environmental and traffic conditions.
        The selected machine learning model will estimate
        the accident severity.
        """
    )

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    with col1:

        city = st.selectbox(
            "City",
            sorted(df["City"].unique())
        )

        weather = st.selectbox(
            "Weather",
            sorted(df["Weather"].unique())
        )

        road_condition = st.selectbox(
            "Road Condition",
            sorted(df["Road_Condition"].unique())
        )


    with col2:

        light_condition = st.selectbox(
            "Light Condition",
            sorted(df["Light_Condition"].unique())
        )

        traffic_density = st.selectbox(
            "Traffic Density",
            sorted(df["Traffic_Density"].unique())
        )

        speed_limit = st.selectbox(
            "Speed Limit",
            sorted(df["Speed_Limit"].unique())
        )


    with col3:

        vehicle_count = st.slider(
            "Vehicle Count",
            int(df["Vehicle_Count"].min()),
            int(df["Vehicle_Count"].max()),
            int(df["Vehicle_Count"].median())
        )

        hour = st.slider(
            "Hour of Day",
            0,
            23,
            12
        )

        day_of_week = st.selectbox(
            "Day of Week",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Accident Severity",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "City": [city],
            "Weather": [weather],
            "Road_Condition": [road_condition],
            "Light_Condition": [light_condition],
            "Speed_Limit": [speed_limit],
            "Traffic_Density": [traffic_density],
            "Vehicle_Count": [vehicle_count],
            "Hour": [hour],
            "Day_of_Week": [day_of_week]
        })


        prediction = model.predict(input_data)[0]

        predicted_severity = reverse_mapping[
            int(prediction)
        ]


        # ========================================================
        # PROJECT-DEFINED ACCIDENT RISK SCORE
        # ========================================================

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            low_probability = probabilities[0]
            medium_probability = probabilities[1]
            high_probability = probabilities[2]

        else:

            low_probability = 0
            medium_probability = 0
            high_probability = 0


        # Environmental risk contribution
        environment_risk = 0

        if weather == "Rain":
            environment_risk += 10
        elif weather == "Fog":
            environment_risk += 15
        elif weather == "Cloudy":
            environment_risk += 5

        if road_condition == "Wet":
            environment_risk += 8
        elif road_condition == "Slippery":
            environment_risk += 15

        # Lighting risk contribution
        if light_condition == "Night":
            environment_risk += 12
        elif light_condition == "Dusk":
            environment_risk += 6

        # Traffic risk contribution
        if traffic_density == "High":
            environment_risk += 15
        elif traffic_density == "Medium":
            environment_risk += 8

        # Speed risk contribution
        if speed_limit >= 80:
            environment_risk += 12
        elif speed_limit >= 60:
            environment_risk += 7

        # Vehicle-count contribution
        vehicle_risk = min(
            (vehicle_count / df["Vehicle_Count"].max()) * 10,
            10
        )

        # Rush-hour contribution
        time_risk = 0

        if 7 <= hour <= 10:
            time_risk += 8
        elif 17 <= hour <= 20:
            time_risk += 10

        # ML probability contribution
        ml_risk = (
            low_probability * 10
            + medium_probability * 50
            + high_probability * 100
        )

        # Final project-defined 0–100 score
        risk_score = (
            ml_risk * 0.60
            + environment_risk * 0.20
            + vehicle_risk
            + time_risk
        )

        risk_score = min(
            max(risk_score, 0),
            100
        )


        # Probability
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            probability_df = pd.DataFrame({
                "Severity": [
                    reverse_mapping[i]
                    for i in range(len(probabilities))
                ],
                "Probability": probabilities
            })

            probability_df["Probability"] *= 100

        else:

            probability_df = pd.DataFrame()


        st.markdown("---")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if predicted_severity == "High":

                st.error(
                    f"🔴 Predicted Severity: **{predicted_severity}**"
                )

            elif predicted_severity == "Medium":

                st.warning(
                    f"🟡 Predicted Severity: **{predicted_severity}**"
                )

            else:

                st.success(
                    f"🟢 Predicted Severity: **{predicted_severity}**"
                )

        with result_col2:

            if risk_score >= 70:

                st.error(
                    f"🚨 Risk Score: **{risk_score:.1f}/100**"
                )

            elif risk_score >= 40:

                st.warning(
                    f"⚠️ Risk Score: **{risk_score:.1f}/100**"
                )

            else:

                st.success(
                    f"✅ Risk Score: **{risk_score:.1f}/100**"
                )


        # Risk gauge
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={
                    "text": "Accident Risk Score"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "threshold": {
                        "line": {
                            "width": 4
                        },
                        "value": risk_score
                    }
                }
            )
        )

        gauge.update_layout(
            height=350
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )


        if not probability_df.empty:

            fig = px.bar(
                probability_df,
                x="Severity",
                y="Probability",
                text=probability_df["Probability"].round(2),
                title="Prediction Probability"
            )

            fig.update_layout(
                yaxis_title="Probability (%)",
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# TAB 4 — MODEL PERFORMANCE
# ============================================================

with tab4:

    st.header("📈 Machine Learning Model Performance")

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    st.subheader("Model Comparison")

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Performance chart
    # --------------------------------------------------------

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score"
    ]

    available_metrics = [
        col
        for col in metric_columns
        if col in comparison.columns
    ]

    if available_metrics:

        melted = comparison.melt(
            id_vars=["Model"],
            value_vars=available_metrics,
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            melted,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Machine Learning Model Performance"
        )

        fig.update_layout(
            yaxis_range=[0, 1],
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    st.subheader("Feature Importance")

    importance_model = st.selectbox(
        "Select Model",
        [
            "Random Forest",
            "XGBoost"
        ]
    )

    if importance_model == "Random Forest":

        importance_df = rf_importance.head(15)

    else:

        importance_df = xgb_importance.head(15)


    fig = px.bar(
        importance_df.sort_values(
            "Importance",
            ascending=True
        ),
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"{importance_model} Feature Importance"
    )

    fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Traffic Accident Analysis & Prediction | "
    "Random Forest + XGBoost + DBSCAN + Folium"
)