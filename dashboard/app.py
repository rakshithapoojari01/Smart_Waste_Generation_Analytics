
import os
import base64
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import base64

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Waste Generation Analytics",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "waste_data.csv"
MODEL_PATH = BASE_DIR / "src" / "waste_prediction_model.pkl"
HEADER_PATH = BASE_DIR / "assets" / "waste_header.png"
MAPREDUCE_PATH = BASE_DIR / "results" / "mapreduce_area_output.txt"


# =========================================================
# GLOBAL CSS
# =========================================================

st.markdown(
    """
<style>

    /* ---------- APP ---------- */

    .stApp {
        background: #061c1a;
        color: #f4f8f7;
    }

    [data-testid="stHeader"] {
        background: #061c1a;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1500px;
        padding-top: 18px;
        padding-left: 24px;
        padding-right: 24px;
    }


    /* ---------- SIDEBAR ---------- */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #031614 0%,
                #062522 55%,
                #041b19 100%
            );
        border-right: 1px solid #164b44;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 22px;
    }

    .sidebar-brand {
        text-align: center;
        padding: 12px 8px 20px 8px;
        border-bottom: 1px solid #16453f;
        margin-bottom: 18px;
    }

    .sidebar-logo {
        font-size: 42px;
        line-height: 1;
        margin-bottom: 8px;
    }

    .sidebar-title {
        color: white;
        font-size: 19px;
        font-weight: 800;
        line-height: 1.12;
    }

    .sidebar-subtitle {
        color: #72d8bf;
        font-size: 11px;
        margin-top: 8px;
    }

    .sidebar-tagline {
        color: #8cebd5;
        font-size: 11px;
        line-height: 1.6;
        margin-top: 14px;
    }

    .sidebar-heading {
        color: white;
        font-size: 16px;
        font-weight: 800;
        margin: 22px 0 10px 2px;
    }

    .sidebar-info {
        background: #082d29;
        border: 1px solid #15574e;
        border-radius: 12px;
        padding: 10px 12px;
        margin-top: 8px;
    }

    .sidebar-info-value {
        color: white;
        font-size: 16px;
        font-weight: 800;
    }

    .sidebar-info-label {
        color: #77cdbb;
        font-size: 10px;
        margin-top: 2px;
    }

/* ---------- HEADER IMAGE ---------- */


    /* ---------- GENERAL ---------- */

    .section-heading {
        color: #ffffff;
        font-size: 19px;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 2px;
    }

    .section-subheading {
        color: #8ab3aa;
        font-size: 11px;
        margin-bottom: 12px;
    }

    .date-badge {
        display: inline-block;
        float: right;
        background: #092d29;
        border: 1px solid #245e55;
        color: #d7efea;
        border-radius: 10px;
        padding: 8px 12px;
        font-size: 11px;
        margin-top: -32px;
    }


    /* ---------- KPI CARDS ---------- */

    .kpi-card {
        min-height: 125px;
        border-radius: 11px;
        padding: 15px 16px;
        border: 1px solid #1b665b;
        background:
            linear-gradient(
                145deg,
                #0a3b35 0%,
                #082923 100%
            );
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.18);
    }

    .kpi-icon {
        width: 46px;
        height: 46px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        font-size: 25px;
        margin-bottom: 10px;
        background: #0c624f;
    }

    .kpi-label {
        color: #d1e7e2;
        font-size: 11px;
        font-weight: 600;
    }

    .kpi-value {
        color: #ffffff;
        font-size: 22px;
        font-weight: 850;
        margin-top: 4px;
        line-height: 1.15;
    }

    .kpi-foot {
        color: #62d6b9;
        font-size: 10px;
        margin-top: 7px;
    }


    /* ---------- CONTENT CARDS ---------- */

    .chart-card {
        background: #072a27;
        border: 1px solid #18534b;
        border-radius: 11px;
        padding: 10px 12px 4px 12px;
        min-height: 100%;
    }

    .card-title {
        color: white;
        font-size: 15px;
        font-weight: 800;
    }

    .card-subtitle {
        color: #8fb6ae;
        font-size: 10px;
        margin-top: 2px;
        margin-bottom: 8px;
    }


    /* ---------- HADOOP ---------- */

    .hadoop-card {
        background: linear-gradient(
            135deg,
            #102d4b 0%,
            #24194a 100%
        );
        border: 1px solid #6259a5;
        border-radius: 12px;
        padding: 12px 15px;
        margin-bottom: 10px;
    }

    .hadoop-title {
        color: white;
        font-size: 17px;
        font-weight: 850;
    }

    .hadoop-subtitle {
        color: #b8c4ed;
        font-size: 10px;
        margin-top: 3px;
    }

    .hadoop-stat {
        min-height: 82px;
        background: #f4f0ff;
        border-radius: 8px;
        padding: 11px 14px;
        margin-bottom: 10px;
    }

    .hadoop-stat-value {
        color: #30239a;
        font-size: 19px;
        font-weight: 850;
    }

    .hadoop-stat-label {
        color: #5c52a8;
        font-size: 10px;
        margin-top: 2px;
    }


    /* ---------- AI ---------- */

    .ai-card {
        background: linear-gradient(
            145deg,
            #21184a 0%,
            #092b2b 100%
        );
        border: 1px solid #6845a6;
        border-radius: 12px;
        padding: 13px 15px;
        margin-bottom: 10px;
    }

    .ai-title {
        color: white;
        font-size: 17px;
        font-weight: 850;
    }

    .ai-subtitle {
        color: #c1b5df;
        font-size: 10px;
        margin-top: 3px;
    }

    .prediction-result {
    background: #083d31;
    border: 1px solid #18a66f;
    border-radius: 9px;
    padding: 13px 15px;
    margin-top: 12px;
    margin-bottom: 15px;
}

    .prediction-label {
        color: #8ce7c8;
        font-size: 10px;
        font-weight: 700;
    }

    .prediction-value {
        color: white;
        font-size: 25px;
        font-weight: 850;
        margin-top: 3px;
    }


    /* ---------- LOWER CARDS ---------- */

    .small-card {
        background: #072a27;
        border: 1px solid #18534b;
        border-radius: 11px;
        padding: 12px;
    }

    .small-title {
        color: white;
        font-size: 14px;
        font-weight: 800;
    }

    .small-subtitle {
        color: #8fb6ae;
        font-size: 9px;
    }


    /* ---------- STREAMLIT WIDGETS ---------- */

    .stSelectbox label,
    .stNumberInput label {
        color: #d9ece8 !important;
        font-size: 11px !important;
    }

    .stSelectbox > div > div,
    .stNumberInput input {
        background: #082d2a !important;
        border-color: #28675e !important;
        color: white !important;
    }

    .stButton > button {
        background: #08a866;
        color: white;
        border: none;
        border-radius: 7px;
        font-weight: 750;
        min-height: 38px;
    }

    .stButton > button:hover {
        background: #0bc47a;
        color: white;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #1c514a;
        border-radius: 9px;
    }

    hr {
        border-color: #16453f;
    }

    .footer {
        text-align: center;
        color: #789e97;
        font-size: 10px;
        padding: 18px 0 8px 0;
    }

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# LOAD DATA
# =========================================================

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error("Dataset could not be loaded.")
    st.code(str(e))
    st.stop()


df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Waste_Collected_kg"] = pd.to_numeric(
    df["Waste_Collected_kg"], errors="coerce"
)
df["Population"] = pd.to_numeric(df["Population"], errors="coerce")
df["Collection_Vehicles"] = pd.to_numeric(
    df["Collection_Vehicles"], errors="coerce"
)
df["Temperature_C"] = pd.to_numeric(
    df["Temperature_C"], errors="coerce"
)
df["Rainfall_mm"] = pd.to_numeric(
    df["Rainfall_mm"], errors="coerce"
)

df = df.dropna(
    subset=[
        "Date",
        "Area",
        "Waste_Type",
        "Waste_Collected_kg",
    ]
).copy()


# =========================================================
# LOAD MODEL
# =========================================================

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">♻️</div>
            <div class="sidebar-title">
                Smart Waste<br>
                Generation Analytics
            </div>
            <div class="sidebar-subtitle">
                Hubli-Dharwad Smart City
            </div>
            <div class="sidebar-tagline">
                🌱 Data Today<br>
                Cleaner Tomorrow
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-heading">🧭 Navigation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="color:#d7ece8; font-size:12px; line-height:2.25;">
            🏠 Dashboard<br>
            ▦ Dataset Overview<br>
            📊 Data Analysis<br>
            🐘 Big Data (Hadoop)<br>
            🧠 AI Prediction<br>
            ⬇️ Download Data<br>
            ⓘ About Project
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-heading">🔽 Filters</div>',
        unsafe_allow_html=True,
    )

    area_options = ["All Areas"] + sorted(df["Area"].unique().tolist())
    waste_options = ["All Waste Types"] + sorted(
        df["Waste_Type"].unique().tolist()
    )

    selected_area = st.selectbox(
        "Select Area",
        area_options,
        key="sidebar_area",
    )

    selected_waste = st.selectbox(
        "Select Waste Type",
        waste_options,
        key="sidebar_waste",
    )

    if st.button("⟳ Reset Filters", use_container_width=True):
        st.session_state["sidebar_area"] = "All Areas"
        st.session_state["sidebar_waste"] = "All Waste Types"
        st.rerun()

    st.markdown(
        '<div class="sidebar-heading">📊 Dataset Information</div>',
        unsafe_allow_html=True,
    )

    info_items = [
        (f"{len(df):,}", "Total Records"),
        (f"{df['Area'].nunique()}", "Areas"),
        (f"{df['Waste_Type'].nunique()}", "Waste Types"),
        (f"{df['Date'].nunique()}", f"Days ({df['Date'].dt.year.min()})"),
    ]

    for value, label in info_items:
        st.markdown(
            f"""
            <div class="sidebar-info">
                <div class="sidebar-info-value">{value}</div>
                <div class="sidebar-info-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
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


if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()
# =========================================================
# HEADER IMAGE
# =========================================================

if HEADER_PATH.exists():

    with open(HEADER_PATH, "rb") as image_file:
        image_base64 = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    st.markdown(
        f"""
        <div style="
            width: 100%;
            margin: 0 0 22px 0;
            overflow: hidden;
            border: 1px solid #18534b;
            border-radius: 10px;
            line-height: 0;
        ">
            <img
                src="data:image/png;base64,{image_base64}"
                style="
                    width: 100%;
                    height: auto;
                    display: block;
                "
            >
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.error(
        "Header image not found. Expected: assets/waste_header.png"
    )
# =========================================================
# DATE BADGE
# =========================================================

date_start = filtered_df["Date"].min().strftime("%Y-%m-%d")
date_end = filtered_df["Date"].max().strftime("%Y-%m-%d")

st.markdown(
    f"""
    <div class="date-badge">
        📅 {date_start} → {date_end}
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# KEY STATISTICS
# =========================================================

total_waste = filtered_df["Waste_Collected_kg"].sum()

number_of_days = filtered_df["Date"].nunique()

average_daily_waste = (
    total_waste / number_of_days
    if number_of_days
    else 0
)

area_totals = (
    filtered_df.groupby("Area")["Waste_Collected_kg"]
    .sum()
    .sort_values(ascending=False)
)

waste_type_totals = (
    filtered_df.groupby("Waste_Type")["Waste_Collected_kg"]
    .sum()
    .sort_values(ascending=False)
)

highest_area = (
    area_totals.index[0]
    if not area_totals.empty
    else "N/A"
)

highest_area_value = (
    area_totals.iloc[0]
    if not area_totals.empty
    else 0
)

top_waste_type = (
    waste_type_totals.index[0]
    if not waste_type_totals.empty
    else "N/A"
)

top_waste_type_value = (
    waste_type_totals.iloc[0]
    if not waste_type_totals.empty
    else 0
)

top_waste_percentage = (
    top_waste_type_value / total_waste * 100
    if total_waste
    else 0
)


st.markdown(
    """
    <div class="section-heading">📊 Key Statistics</div>
    <div class="section-subheading">
        Overview of waste generation based on selected filters
    </div>
    """,
    unsafe_allow_html=True,
)


k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">♻️</div>
            <div class="kpi-label">Total Waste Collected</div>
            <div class="kpi-value">{total_waste:,.2f} kg</div>
            <div class="kpi-foot">↗ Current filtered period</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">🗓️</div>
            <div class="kpi-label">Average Daily Waste</div>
            <div class="kpi-value">{average_daily_waste:,.2f} kg</div>
            <div class="kpi-foot">↗ Based on daily records</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">📍</div>
            <div class="kpi-label">Highest Waste Area</div>
            <div class="kpi-value">{highest_area}</div>
            <div class="kpi-foot">{highest_area_value:,.2f} kg</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">🗑️</div>
            <div class="kpi-label">Top Waste Type</div>
            <div class="kpi-value">{top_waste_type}</div>
            <div class="kpi-foot">↗ {top_waste_percentage:.1f}% of total waste</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# WASTE GENERATION ANALYSIS
# =========================================================

st.markdown(
    """
    <div class="section-heading">📊 Waste Generation Analysis</div>
    <div class="section-subheading">
        Area-wise generation and waste-type composition
    </div>
    """,
    unsafe_allow_html=True,
)

chart_left, chart_right = st.columns([2.05, 1])


# ---------- AREA CHART ----------

with chart_left:

    st.markdown(
        """
        <div class="chart-card">
            <div class="card-title">📊 Waste Generation by Area</div>
            <div class="card-subtitle">
                Total waste collected in each area based on selected filters
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    area_plot = area_totals.sort_values(ascending=True)

    fig_area, ax_area = plt.subplots(
        figsize=(8, 5)
   )

    fig_area.patch.set_facecolor("#072a27")
    ax_area.set_facecolor("#072a27")

    area_plot.plot(
        kind="barh",
        ax=ax_area,
        color="#32d583",
        edgecolor="#73f2bd",
        linewidth=0.4,
    )

    ax_area.tick_params(
        colors="#d2e8e3",
        labelsize=8,
    )

    ax_area.set_xlabel(
        "Waste Collected (kg)",
        color="#a9c8c1",
        fontsize=9,
    )

    ax_area.set_ylabel(
        "Area",
        color="#a9c8c1",
        fontsize=9,
    )

    ax_area.grid(
        axis="x",
        color="#4b8278",
        alpha=0.22,
        linewidth=0.7,
    )

    for spine in ax_area.spines.values():
        spine.set_color("#19574e")

    fig_area.tight_layout()

    st.pyplot(
        fig_area,
        width="content",
    )

    plt.close(fig_area)


# ---------- WASTE COMPOSITION ----------

with chart_right:

    st.markdown(
        """
        <div class="chart-card">
            <div class="card-title">♻️ Waste Composition</div>
            <div class="card-subtitle">
                Distribution of waste types
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_pie, ax_pie = plt.subplots(
        figsize=(5, 4.5)
    )
    fig_pie.patch.set_facecolor("#072a27")
    ax_pie.set_facecolor("#072a27")

    wedges, _ = ax_pie.pie(
        waste_type_totals,
        startangle=90,
        wedgeprops={
            "width": 0.40,
            "edgecolor": "#072a27",
        },
    )

    legend_labels = [
        f"{name}  {value / total_waste * 100:.1f}%"
        for name, value in waste_type_totals.items()
    ]

    legend = ax_pie.legend(
        wedges,
        legend_labels,
        title="Waste Type",
        loc="center left",
        bbox_to_anchor=(0.98, 0.5),
        fontsize=7.5,
        title_fontsize=8,
        frameon=False,
        labelcolor="#d8ece7",
    )

    legend.get_title().set_color("#ffffff")

    ax_pie.text(
        0,
        0.05,
        f"{total_waste / 1000:,.0f}",
        ha="center",
        va="center",
        color="white",
        fontsize=17,
        fontweight="bold",
    )

    ax_pie.text(
        0,
        -0.10,
        "kg total",
        ha="center",
        va="center",
        color="#8eb4ac",
        fontsize=8,
    )

    fig_pie.tight_layout()

    st.pyplot(
        fig_pie,
        width="content",
    )

    plt.close(fig_pie)


# =========================================================
# HADOOP + AI
# =========================================================

st.markdown(
    """
    <div class="section-heading">🐘 Hadoop MapReduce + 🧠 AI Prediction</div>
    <div class="section-subheading">
        Big Data processing from HDFS and machine-learning based waste prediction
    </div>
    """,
    unsafe_allow_html=True,
)

hadoop_col, ai_col = st.columns([2.05, 1])


# =========================================================
# HADOOP MAPREDUCE
# =========================================================

with hadoop_col:

    try:

        mr = pd.read_csv(
            MAPREDUCE_PATH,
            sep="\t",
            header=None,
            names=["Area", "Total_Waste_kg"],
        )

        mr["Total_Waste_kg"] = pd.to_numeric(
            mr["Total_Waste_kg"],
            errors="coerce",
        )

        mr = mr.dropna()

        mr = mr.sort_values(
            "Total_Waste_kg",
            ascending=False,
        ).reset_index(drop=True)

        mr_top = mr.head(10)

        st.markdown(
            """
            <div class="hadoop-card">
                <div class="hadoop-title">
                    🐘 Hadoop MapReduce Analysis
                </div>
                <div class="hadoop-subtitle">
                    HDFS → MapReduce → Area-wise Waste Analytics
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        h1, h2, h3 = st.columns(3)

        with h1:
            st.markdown(
                f"""
                <div class="hadoop-stat">
                    <div class="hadoop-stat-value">{len(mr)}</div>
                    <div class="hadoop-stat-label">AREAS PROCESSED</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with h2:
            st.markdown(
                f"""
                <div class="hadoop-stat">
                    <div class="hadoop-stat-value">{mr.iloc[0]["Area"]}</div>
                    <div class="hadoop-stat-label">HIGHEST WASTE AREA</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with h3:
            st.markdown(
                f"""
                <div class="hadoop-stat">
                    <div class="hadoop-stat-value">
                        {mr.iloc[0]["Total_Waste_kg"]:,.2f} kg
                    </div>
                    <div class="hadoop-stat-label">HIGHEST AREA WASTE</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        h_chart, h_table = st.columns([1.35, 1])

        with h_chart:

            fig_mr, ax_mr = plt.subplots(
                figsize=(7, 4)
            )

            fig_mr.patch.set_facecolor("#072a27")
            ax_mr.set_facecolor("#072a27")

            mr_top.sort_values(
                "Total_Waste_kg"
            ).plot(
                kind="barh",
                x="Area",
                y="Total_Waste_kg",
                ax=ax_mr,
                legend=False,
                color="#8c6cff",
                edgecolor="#b8aaff",
            )

            ax_mr.tick_params(
                colors="#d3e5e1",
                labelsize=7,
            )

            ax_mr.set_xlabel(
                "Total Waste Collected (kg)",
                color="#a9c8c1",
                fontsize=8,
            )

            ax_mr.set_ylabel(
                "Area",
                color="#a9c8c1",
                fontsize=8,
            )

            ax_mr.set_title(
                "Top 10 Areas by Total Waste (Hadoop MapReduce)",
                color="white",
                fontsize=9,
                fontweight="bold",
            )

            ax_mr.grid(
                axis="x",
                color="#695da1",
                alpha=0.18,
            )

            for spine in ax_mr.spines.values():
                spine.set_color("#3b3970")

            fig_mr.tight_layout()

            st.pyplot(
                fig_mr,
                width="content",
            )

            plt.close(fig_mr)

        with h_table:

            st.markdown(
                """
                <div class="small-title">🐘 MapReduce Results</div>
                <div class="small-subtitle">
                    Output generated by the Hadoop MapReduce job
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.dataframe(
                mr.head(10).round(2),
                width="stretch",
                hide_index=True,
            )

    except Exception as e:

        st.error(
            "Hadoop MapReduce output could not be loaded."
        )
        st.code(str(e))

# =========================================================
# AI WASTE PREDICTOR
# =========================================================

with ai_col:

    st.markdown(
        """
        <div class="ai-card">
            <div class="ai-title">🧠 AI Waste Predictor</div>
            <div class="ai-subtitle">
                Predict waste generation using trained ML model
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prediction_area = st.selectbox(
        "Area",
        sorted(df["Area"].unique()),
        key="prediction_area",
    )

    prediction_type = st.selectbox(
        "Waste Type",
        sorted(df["Waste_Type"].unique()),
        key="prediction_type",
    )

    p1, p2 = st.columns(2)

    with p1:

        population = st.number_input(
            "Population",
            min_value=1000,
            value=25000,
            step=1000,
            key="prediction_population",
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=28.0,
            step=0.5,
            key="prediction_temperature",
        )

    with p2:

        vehicles = st.number_input(
            "Collection Vehicles",
            min_value=1,
            value=4,
            step=1,
            key="prediction_vehicles",
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=10.0,
            step=1.0,
            key="prediction_rainfall",
        )

    # Keep button and result together
    predict_clicked = st.button(
        "🔮 Predict Waste Generation",
        use_container_width=True,
        key="predict_waste_button",
    )

    if predict_clicked:

        if model is None:

            st.error(
                "ML model could not be loaded."
            )

        else:

            input_data = pd.DataFrame(
                {
                    "Population": [population],
                    "Collection_Vehicles": [vehicles],
                    "Temperature_C": [temperature],
                    "Rainfall_mm": [rainfall],
                    "Area": [prediction_area],
                    "Waste_Type": [prediction_type],
                }
            )

            prediction = model.predict(
                input_data
            )[0]

            prediction = max(
                0,
                float(prediction)
            )

            # Prediction result stays directly
            # below the predictor button
            st.html(
    f"""
    <div class="prediction-result">
        <div class="prediction-label">
            ESTIMATED WASTE GENERATION
        </div>

        <div class="prediction-value">
            {prediction:,.2f} kg
        </div>
    </div>
    """
)
# =========================================================
# LOWER ANALYTICS
# =========================================================

st.markdown(
    """
    <div class="section-heading">📈 Additional Analytics</div>
    <div class="section-subheading">
        Trends, hotspots and weather relationships in the selected data
    </div>
    """,
    unsafe_allow_html=True,
)

trend_col, hotspot_col, weather_col = st.columns(
    [1.5, 1, 1]
)


# ---------- MONTHLY TREND ----------

with trend_col:

    monthly = (
        filtered_df
        .assign(Month=filtered_df["Date"].dt.to_period("M").astype(str))
        .groupby("Month")["Waste_Collected_kg"]
        .sum()
        .reset_index()
    )

    fig_trend, ax_trend = plt.subplots(
        figsize=(7, 3.2)
    )

    fig_trend.patch.set_facecolor("#072a27")
    ax_trend.set_facecolor("#072a27")

    ax_trend.plot(
        monthly["Month"],
        monthly["Waste_Collected_kg"],
        marker="o",
        linewidth=2,
        markersize=4,
        color="#36d995",
    )

    ax_trend.fill_between(
        range(len(monthly)),
        monthly["Waste_Collected_kg"].values,
        alpha=0.12,
        color="#36d995",
    )

    ax_trend.tick_params(
        colors="#cfe5df",
        labelsize=7,
    )

    ax_trend.set_title(
        "Monthly Waste Generation Trend",
        color="white",
        fontsize=10,
        fontweight="bold",
    )

    ax_trend.set_xlabel(
        "Month",
        color="#9dbbb4",
        fontsize=8,
    )

    ax_trend.set_ylabel(
        "Waste (kg)",
        color="#9dbbb4",
        fontsize=8,
    )

    ax_trend.grid(
        alpha=0.16,
        color="#5b9489",
    )

    for spine in ax_trend.spines.values():
        spine.set_color("#19574e")

    plt.xticks(rotation=45)

    fig_trend.tight_layout()

    st.pyplot(
        fig_trend,
        width="content",
    )

    plt.close(fig_trend)


# ---------- HOTSPOTS ----------

with hotspot_col:

    hotspot = (
        df.groupby("Area")["Waste_Collected_kg"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )

    hotspot["Status"] = "High"

    st.markdown(
        """
        <div class="small-card">
            <div class="small-title">🔥 Waste Generation Hotspots</div>
            <div class="small-subtitle">
                Areas with highest total waste generation
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        hotspot.round(2),
        width="stretch",
        hide_index=True,
    )


# ---------- WEATHER ----------

with weather_col:

    weather_cols = [
        "Waste_Collected_kg",
        "Temperature_C",
        "Rainfall_mm",
    ]

    weather_corr = filtered_df[
        weather_cols
    ].corr().round(2)

    st.markdown(
        """
        <div class="small-card">
            <div class="small-title">🌦️ Weather and Waste Analysis</div>
            <div class="small-subtitle">
                Correlation between waste, temperature and rainfall
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        weather_corr,
        width="stretch",
    )

    st.caption(
        "Correlation shows the strength of a linear relationship; "
        "it does not prove that weather causes waste generation."
    )


# =========================================================
# DOWNLOAD DATA
# =========================================================

st.markdown(
    """
    <div class="section-heading">⬇️ Download Data</div>
    """,
    unsafe_allow_html=True,
)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download Filtered Dataset",
    data=csv_data,
    file_name="filtered_waste_data.csv",
    mime="text/csv",
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        ♻️ Smart Waste Generation Analytics
        &nbsp; | &nbsp;
        Big Data Analytics • Machine Learning • Hadoop • HDFS • MapReduce • Streamlit
        <br>
        Built for academic project demonstration using simulated waste-generation data.
    </div>
    """,
    unsafe_allow_html=True,
)
