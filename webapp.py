import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)

# Read CSV from a GitHub repo
dataset_url = "https://raw.githubusercontent.com/Lexie88rus/bank-marketing-analysis/master/bank.csv"

# Memoize data loading for efficiency
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_url)

# Load data
df = get_data()

# Dashboard title
st.title("Real-Time / Live Data Science Dashboard")

# Top-level filter (job selection)
job_filter = st.selectbox("Select the Job", pd.unique(df["job"]))

# Filter DataFrame based on the selected job
df_filtered = df[df["job"] == job_filter]

# Create a container for dynamic content updates
placeholder = st.empty()

# Near real-time / live feed simulation
for seconds in range(200):
    # Random adjustments for simulation
    df_filtered["age_new"] = df_filtered["age"] * np.random.choice(range(1, 5), size=len(df_filtered))
    df_filtered["balance_new"] = df_filtered["balance"] * np.random.choice(range(1, 5), size=len(df_filtered))

    # KPI calculations
    avg_age = np.mean(df_filtered["age_new"])
    count_married = int(df_filtered[df_filtered["marital"] == "married"].shape[0] + np.random.choice(range(1, 30)))
    balance = np.mean(df_filtered["balance_new"])

    with placeholder.container():
        # Create columns for KPIs
        kpi1, kpi2, kpi3 = st.columns(3)

        # Display KPIs
        kpi1.metric(
            label="Age ⏳",
            value=round(avg_age),
            delta=round(avg_age) - 10,
        )

        kpi2.metric(
            label="Married Count 💍",
            value=int(count_married),
            delta=-10 + count_married,
        )

        kpi3.metric(
            label="A/C Balance ＄",
            value=f"$ {round(balance, 2)}",
            delta=-round(balance / (count_married or 1)) * 100,  # Prevent divide by zero
        )

        # Create two columns for charts
        fig_col1, fig_col2 = st.columns(2)
        
        # First chart: Heatmap
        with fig_col1:
            st.markdown("### First Chart")
            fig = px.density_heatmap(
                data_frame=df_filtered, y="age_new", x="marital"
            )
            st.write(fig)

        # Second chart: Histogram
        with fig_col2:
            st.markdown("### Second Chart")
            fig2 = px.histogram(data_frame=df_filtered, x="age_new")
            st.write(fig2)

        # Detailed Data View
        st.markdown("### Detailed Data View")
        st.dataframe(df_filtered)

    # Sleep for simulation (real-time effect)
    time.sleep(1)
