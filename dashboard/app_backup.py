import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Waste Generation Analytics",
    page_icon="♻️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5faf7;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* HERO */
    .hero {
        background: linear-gradient(135deg, #075e38, #198754);
        padding: 35px 40px;
        border-radius: 22px;
        margin-bottom: 30px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.10);
    }

    .hero-title {
        color: white;
        font-size: 38px;
        font-weight: 800;
        margin: 0;
    }

    .hero-subtitle {
        color: #e9f7ef;
        font-size: 17px;
        margin-top: 10px;
        margin-bottom: 0;
    }

    /* SECTION TITLE */
    .section-title {
        color: #075e38;
        font-size: 25px;
        font-weight: 750;
        margin-top: 20px;
        margin-bottom: 18px;
    }

    /* KPI CARDS */
    .kpi-card {
        background-color: white;
        padding: 20px;
        border-radius: 17px;
        border: 1px solid #d8e8de;
        min-height: 125px;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.06);
    }

    .kpi-icon {
        font-size: 25px;
    }

    .kpi-label {
        color: #6b7280;
        font-size: 13px;
        margin-top: 7px;
        font-weight: 600;
    }

    .kpi-value {
        color: #123b29;
        font-size: 24px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* PREDICTION RESULT */
    .prediction-result {
        background: linear-gradient(135deg, #e8f7ee, #d7f1e1);
        border: 1px solid #b7dfc5;
        border-radius: 17px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-label {
        color: #397052;
        font-size: 14px;
        font-weight: 600;
    }

    .prediction-value {
        color: #075e38;
        font-size: 36px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* INSIGHT CARDS */
    .insight-card {
        background-color: white;
        border-left: 5px solid #198754;
        padding: 17px 20px;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.05);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("dataset/waste_data.csv")

df["Date"] = pd.to_datetime(df["Date"])

# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    """
<div class="hero">
<div class="hero-title">♻️ Smart Waste Generation Analytics</div>
<div class="hero-subtitle">Turning waste data into smarter and more sustainable waste-management decisions.</div>
</div>
""",
    unsafe_allow_html=True
)
# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.title("🔍 Dashboard Filters")

st.sidebar.markdown(
    "Use the filters below to explore the waste data."
)

area_options = ["All Areas"] + sorted(
    df["Area"].unique().tolist()
)

selected_area = st.sidebar.selectbox(
    "Select Area",
    area_options
)


waste_options = ["All Waste Types"] + sorted(
    df["Waste_Type"].unique().tolist()
)

selected_waste = st.sidebar.selectbox(
    "Select Waste Type",
    waste_options
)


# =========================================================
# APPLY FILTERS
# =========================================================

filtered_df = df.copy()


if selected_area != "All Areas":

    filtered_df = filtered_df[
        filtered_df["Area"] == selected_area
    ]


if selected_waste != "All Waste Types":

    filtered_df = filtered_df[
        filtered_df["Waste_Type"] == selected_waste
    ]


# =========================================================
# BASIC STATISTICS
# =========================================================

total_waste = filtered_df[
    "Waste_Collected_kg"
].sum()


average_daily_waste = (
    filtered_df
    .groupby("Date")["Waste_Collected_kg"]
    .sum()
    .mean()
)


if len(filtered_df) > 0:

    highest_area = (
        filtered_df
        .groupby("Area")["Waste_Collected_kg"]
        .sum()
        .idxmax()
    )

    highest_waste_type = (
        filtered_df
        .groupby("Waste_Type")["Waste_Collected_kg"]
        .sum()
        .idxmax()
    )

else:

    highest_area = "N/A"
    highest_waste_type = "N/A"


# =========================================================
# KEY STATISTICS
# =========================================================

st.markdown(
    '<div class="section-title">📊 Key Statistics</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">♻️</div>
            <div class="kpi-label">TOTAL WASTE</div>
            <div class="kpi-value">
                {total_waste:,.2f} kg
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">📅</div>
            <div class="kpi-label">AVERAGE DAILY WASTE</div>
            <div class="kpi-value">
                {average_daily_waste:,.2f} kg
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">📍</div>
            <div class="kpi-label">HIGHEST WASTE AREA</div>
            <div class="kpi-value">
                {highest_area}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">🗑️</div>
            <div class="kpi-label">TOP WASTE TYPE</div>
            <div class="kpi-value">
                {highest_waste_type}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# =========================================================
# WASTE INTELLIGENCE
# =========================================================

st.markdown(
    '<div class="section-title">🧠 Waste Intelligence</div>',
    unsafe_allow_html=True
)


chart_col1, chart_col2 = st.columns(2)


# =========================================================
# AREA ANALYSIS
# =========================================================

with chart_col1:

    st.subheader("📍 Waste Generation by Area")

    area_waste = (
        filtered_df
        .groupby("Area")["Waste_Collected_kg"]
        .sum()
        .sort_values(ascending=False)
    )

    fig1, ax1 = plt.subplots(figsize=(8, 5))

    area_waste.plot(
        kind="bar",
        ax=ax1
    )

    ax1.set_title(
        "Total Waste Generation by Area"
    )

    ax1.set_xlabel("Area")

    ax1.set_ylabel(
        "Waste Collected (kg)"
    )

    plt.xticks(rotation=35)

    plt.tight_layout()

    st.pyplot(
        fig1,
        width="stretch"
    )


# =========================================================
# WASTE TYPE ANALYSIS
# =========================================================

with chart_col2:

    st.subheader("♻️ Waste Composition")

    # Composition uses the selected AREA,
    # but ignores the selected WASTE TYPE.
    composition_df = df.copy()

    if selected_area != "All Areas":
        composition_df = composition_df[
            composition_df["Area"] == selected_area
        ]

    waste_type_total = (
        composition_df
        .groupby("Waste_Type")["Waste_Collected_kg"]
        .sum()
        .sort_values(ascending=False)
    )
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    wedges, texts, autotexts = ax2.pie(
        waste_type_total,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.72
    )

    ax2.set_ylabel("")

    ax2.set_title(
        "Waste Distribution by Type"
    )

    ax2.legend(
        wedges,
        waste_type_total.index,
        title="Waste Type",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=9
    )
    plt.tight_layout()

    st.pyplot(
        fig2,
        width="stretch"
    )
# =========================================================
# HADOOP MAPREDUCE ANALYSIS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🐘 Hadoop MapReduce Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "This section displays the area-wise waste analysis "
    "generated using Hadoop MapReduce from the dataset "
    "stored in HDFS."
)

try:

    mapreduce_output = pd.read_csv(
        "results/mapreduce_area_output.txt",
        sep="\t",
        header=None,
        names=["Area", "Total_Waste_kg"]
    )

    # Convert waste values to numeric
    mapreduce_output["Total_Waste_kg"] = pd.to_numeric(
        mapreduce_output["Total_Waste_kg"],
        errors="coerce"
    )

    # Remove invalid rows if any
    mapreduce_output = mapreduce_output.dropna()

    # Sort by total waste
    mapreduce_output = mapreduce_output.sort_values(
        "Total_Waste_kg",
        ascending=False
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # MAPREDUCE KPIs
    # -----------------------------------------------------

    mr_col1, mr_col2, mr_col3 = st.columns(3)

    highest_mr_area = mapreduce_output.iloc[0]["Area"]
    highest_mr_waste = mapreduce_output.iloc[0]["Total_Waste_kg"]

    lowest_mr_area = mapreduce_output.iloc[-1]["Area"]
    lowest_mr_waste = mapreduce_output.iloc[-1]["Total_Waste_kg"]

    with mr_col1:

        st.metric(
            "📍 Areas Processed",
            len(mapreduce_output)
        )

    with mr_col2:

        st.metric(
            "🔥 Highest Waste Area",
            highest_mr_area
        )

    with mr_col3:

        st.metric(
            "♻️ Highest Area Waste",
            f"{highest_mr_waste:,.2f} kg"
        )

    # -----------------------------------------------------
    # MAPREDUCE BAR CHART
    # -----------------------------------------------------

    st.subheader("📊 Total Waste by Area — Hadoop MapReduce")

    fig_mr, ax_mr = plt.subplots(
        figsize=(12, 6)
    )

    mapreduce_output.sort_values(
        "Total_Waste_kg",
        ascending=True
    ).plot(
        kind="barh",
        x="Area",
        y="Total_Waste_kg",
        ax=ax_mr,
        legend=False
    )

    ax_mr.set_xlabel(
        "Total Waste Collected (kg)"
    )

    ax_mr.set_ylabel(
        "Area"
    )

    ax_mr.set_title(
        "Area-wise Waste Generation using Hadoop MapReduce"
    )

    plt.tight_layout()

    st.pyplot(
        fig_mr,
        width="stretch"
    )

    # -----------------------------------------------------
    # MAPREDUCE RESULT TABLE
    # -----------------------------------------------------

    st.subheader("📋 MapReduce Results")

    display_mr = mapreduce_output.copy()

    display_mr["Total_Waste_kg"] = (
        display_mr["Total_Waste_kg"]
        .round(2)
    )

    st.dataframe(
        display_mr,
        width="stretch",
        hide_index=True
    )

    # -----------------------------------------------------
    # MAPREDUCE SUMMARY
    # -----------------------------------------------------

    st.markdown(
        f"""
        <div class="insight-card">
            🐘 Hadoop MapReduce processed the waste dataset
            and generated results for <b>{len(mapreduce_output)}</b>
            areas.
        </div>

        <div class="insight-card">
            🔥 The highest total waste was generated in
            <b>{highest_mr_area}</b> with
            <b>{highest_mr_waste:,.2f} kg</b>.
        </div>

        <div class="insight-card">
            📉 The lowest total waste was generated in
            <b>{lowest_mr_area}</b> with
            <b>{lowest_mr_waste:,.2f} kg</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

except FileNotFoundError:

    st.warning(
        "Hadoop MapReduce output file was not found. "
        "Run the MapReduce program first."
    )

except Exception as e:

    st.error(
        "Unable to load Hadoop MapReduce results."
    )

    st.code(str(e))

# =========================================================
# MONTHLY TREND
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">📈 Waste Generation Trend</div>',
    unsafe_allow_html=True
)


filtered_df["Month"] = (
    filtered_df["Date"]
    .dt.to_period("M")
    .astype(str)
)


monthly_waste = (
    filtered_df
    .groupby("Month")["Waste_Collected_kg"]
    .sum()
)


fig3, ax3 = plt.subplots(
    figsize=(12, 5)
)


monthly_waste.plot(
    kind="line",
    marker="o",
    ax=ax3
)


ax3.set_title(
    "Monthly Waste Generation"
)

ax3.set_xlabel("Month")

ax3.set_ylabel(
    "Waste Collected (kg)"
)

plt.xticks(rotation=45)

plt.tight_layout()


st.pyplot(
    fig3,
    width="stretch"
)


# =========================================================
# WASTE HOTSPOTS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🔥 Waste Generation Hotspots</div>',
    unsafe_allow_html=True
)


hotspot_data = (
    filtered_df
    .groupby("Area")["Waste_Collected_kg"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)


hotspot_data.columns = [
    "Area",
    "Total_Waste_kg"
]


if len(hotspot_data) > 0:

    maximum_waste = hotspot_data[
        "Total_Waste_kg"
    ].max()

    def get_status(value):

        if value >= maximum_waste * 0.75:
            return "High"

        elif value >= maximum_waste * 0.45:
            return "Medium"

        else:
            return "Low"


    hotspot_data["Status"] = (
        hotspot_data["Total_Waste_kg"]
        .apply(get_status)
    )


    hotspot_data[
        "Total_Waste_kg"
    ] = hotspot_data[
        "Total_Waste_kg"
    ].round(2)


    st.dataframe(
        hotspot_data,
        width="stretch",
        hide_index=True
    )


# =========================================================
# WEATHER ANALYSIS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🌦️ Weather and Waste Analysis</div>',
    unsafe_allow_html=True
)


weather_corr = filtered_df[
    [
        "Waste_Collected_kg",
        "Temperature_C",
        "Rainfall_mm"
    ]
].corr()


st.dataframe(
    weather_corr.round(2),
    width="stretch"
)


st.caption(
    "Correlation shows the strength of a linear relationship. "
    "It does not prove that weather causes waste generation."
)


# =========================================================
# SMART INSIGHTS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">💡 Smart Insights</div>',
    unsafe_allow_html=True
)


if len(filtered_df) > 0:

    top_area_waste = (
        filtered_df
        .groupby("Area")["Waste_Collected_kg"]
        .sum()
        .max()
    )

    top_type_waste = (
        filtered_df
        .groupby("Waste_Type")["Waste_Collected_kg"]
        .sum()
        .max()
    )

    total_for_percentage = (
        filtered_df["Waste_Collected_kg"].sum()
    )

    if total_for_percentage > 0:

        top_type_percentage = (
            top_type_waste /
            total_for_percentage
        ) * 100

    else:

        top_type_percentage = 0


    st.markdown(
        f"""
        <div class="insight-card">
            📍 <b>{highest_area}</b> has the highest
            total waste generation among the selected data.
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="insight-card">
            ♻️ <b>{highest_waste_type}</b> is the
            highest-contributing waste category.
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="insight-card">
            📊 The highest waste category contributes
            approximately <b>{top_type_percentage:.1f}%</b>
            of the selected waste.
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="insight-card">
            🚛 Waste-generation patterns can help
            planners identify areas that may require
            closer monitoring and collection planning.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# AI WASTE PREDICTION
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🤖 AI Waste Predictor</div>',
    unsafe_allow_html=True
)


st.write(
    "Enter population, collection and environmental "
    "conditions to estimate waste generation."
)


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = joblib.load(
        "src/waste_prediction_model.pkl"
    )

except Exception as e:

    st.error(
        "Unable to load the prediction model."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# PREDICTION INPUTS
# =========================================================

pred_col1, pred_col2 = st.columns(2)


with pred_col1:

    prediction_area = st.selectbox(
        "Select Area for Prediction",
        sorted(
            df["Area"].unique()
        ),
        key="prediction_area"
    )


    prediction_waste_type = st.selectbox(
        "Select Waste Type",
        sorted(
            df["Waste_Type"].unique()
        ),
        key="prediction_waste_type"
    )


    prediction_population = st.number_input(
        "Population",
        min_value=1000,
        value=25000,
        step=1000
    )


with pred_col2:

    prediction_vehicles = st.number_input(
        "Collection Vehicles",
        min_value=1,
        value=1,
        step=1
    )


    prediction_temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=28.0,
        step=0.5
    )


    prediction_rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=10.0,
        step=1.0
    )


# =========================================================
# PREPARE PREDICTION DATA
# =========================================================

prediction_input = pd.DataFrame(
    {
        "Population": [
            prediction_population
        ],

        "Collection_Vehicles": [
            prediction_vehicles
        ],

        "Temperature_C": [
            prediction_temperature
        ],

        "Rainfall_mm": [
            prediction_rainfall
        ]
    }
)


# =========================================================
# AREA ENCODING
# =========================================================

for area in df["Area"].unique():

    if area != "Vidyanagar":

        prediction_input[
            f"Area_{area}"
        ] = [
            1
            if prediction_area == area
            else 0
        ]


# =========================================================
# WASTE TYPE ENCODING
# =========================================================

for waste_type in df["Waste_Type"].unique():

    if waste_type != "Glass":

        prediction_input[
            f"Waste_Type_{waste_type}"
        ] = [
            1
            if prediction_waste_type == waste_type
            else 0
        ]


# =========================================================
# MATCH MODEL FEATURES
# =========================================================

try:

    prediction_input = prediction_input.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

except Exception:

    st.error(
        "The prediction model does not contain "
        "the expected feature information."
    )

    st.stop()

# =========================================================
# PREDICT
# =========================================================

if st.button(
    "🔮 Predict Waste Generation",
    type="primary"
):

    try:

        prediction = model.predict(
            prediction_input
        )[0]

        st.success(
            f"Estimated Waste Generation: {prediction:.2f} kg"
        )

    except Exception as e:

        st.error(
            "Prediction could not be generated."
        )

        st.code(str(e))

# =========================================================
# DOWNLOAD DATA
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">📥 Download Data</div>',
    unsafe_allow_html=True
)


csv_data = filtered_df.to_csv(
    index=False
)


st.download_button(
    label="Download Filtered Dataset",
    data=csv_data,
    file_name="filtered_waste_data.csv",
    mime="text/csv",
    width="stretch"
)


# =========================================================
# DATASET
# =========================================================

st.markdown(
    '<div class="section-title">📋 Waste Dataset</div>',
    unsafe_allow_html=True
)


st.dataframe(
    filtered_df.drop(
        columns=["Month"],
        errors="ignore"
    ),
    width="stretch",
    height=400
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Smart Waste Generation Analytics | "
    "Python • Pandas • Scikit-learn • Streamlit"
)