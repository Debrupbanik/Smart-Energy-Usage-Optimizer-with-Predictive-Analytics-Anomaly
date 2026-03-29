import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
from src.preprocessing import create_sequences


def load_models():
    """Load trained models from disk."""
    rf_model = joblib.load("models/rf_model.pkl")
    iso_forest = joblib.load("models/iso_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return rf_model, iso_forest, scaler


@st.cache_data
def load_data():
    """Load energy data from CSV."""
    return pd.read_csv(
        "energy_data.csv", parse_dates=["timestamp"], index_col="timestamp"
    )


@st.cache_data
def load_or_generate_data():
    """Load existing data or generate new data if not available."""
    if os.path.exists("energy_data.csv"):
        return load_data()
    return None


st.set_page_config(page_title="Smart Energy Optimizer", page_icon="⚡", layout="wide")

st.title("⚡ Smart Energy Usage Optimizer")
st.markdown("### Predictive Analytics & Anomaly Detection")

# Sidebar
st.sidebar.header("Settings")
days_to_show = st.sidebar.slider("Days to visualize", 1, 30, 7)
forecast_hours = st.sidebar.slider("Hours to forecast", 6, 72, 24)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard uses Random Forest for energy consumption forecasting "
    "and Isolation Forest for anomaly detection."
)

# Check if models and data exist
models_exist = (
    os.path.exists("models/rf_model.pkl")
    and os.path.exists("models/iso_forest.pkl")
    and os.path.exists("models/scaler.pkl")
)
data_exists = os.path.exists("energy_data.csv")

if not models_exist or not data_exists:
    st.warning("⚠️ Models or data not found. Please train the models first.")
    st.code("python src/train.py", language="bash")
    st.markdown("---")

    # Offer to generate sample data
    if st.button("Generate Sample Data & Train"):
        with st.spinner("Training models..."):
            import subprocess

            result = subprocess.run(
                ["python", "src/train.py"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )
            if result.returncode == 0:
                st.success("Models trained successfully!")
                st.rerun()
            else:
                st.error(f"Training failed: {result.stderr}")
        st.stop()

try:
    rf_model, iso_forest, scaler = load_models()
    df = load_data()

    # Anomaly Detection
    df["anomaly"] = iso_forest.predict(df[["consumption"]])
    df["anomaly_score"] = iso_forest.score_samples(df[["consumption"]])
    anomalies = df[df["anomaly"] == -1]

    # Forecasting
    last_sequence = df["consumption"].values[-24:].reshape(-1, 1)
    last_sequence_scaled = scaler.transform(last_sequence)

    # Generate forecast
    forecast = []
    current_seq = last_sequence_scaled.reshape(1, 24)

    for _ in range(forecast_hours):
        pred = rf_model.predict(current_seq)
        forecast.append(pred[0])
        current_seq = np.append(current_seq[:, 1:], [[pred[0]]], axis=1)

    forecast_actual = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Avg Consumption", f"{df['consumption'].mean():.1f} kWh")
    with col3:
        st.metric("Anomalies Detected", f"{len(anomalies)}", delta_color="inverse")
    with col4:
        st.metric("Forecast Period", f"{forecast_hours}h")

    st.markdown("---")

    # Visualization - Historical Data
    st.subheader("📊 Energy Consumption & Anomalies")

    fig = go.Figure()

    # Historical Data
    subset = df.tail(days_to_show * 24)
    fig.add_trace(
        go.Scatter(
            x=subset.index,
            y=subset["consumption"],
            mode="lines",
            name="Consumption",
            line=dict(color="#3498db", width=2),
        )
    )

    # Anomalies
    anomalies_subset = subset[subset["anomaly"] == -1]
    if len(anomalies_subset) > 0:
        fig.add_trace(
            go.Scatter(
                x=anomalies_subset.index,
                y=anomalies_subset["consumption"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#e74c3c", size=10, symbol="x"),
            )
        )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Energy Consumption (kWh)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Forecast
    st.markdown("---")
    st.subheader("🔮 24-Hour Forecast")

    forecast_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(hours=1), periods=forecast_hours, freq="h"
    )
    forecast_df = pd.DataFrame(
        {"timestamp": forecast_dates, "forecast": forecast_actual.flatten()}
    )

    fig_forecast = go.Figure()

    # Forecast line
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["forecast"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#27ae60", width=3),
            marker=dict(size=6),
        )
    )

    # Confidence interval (simple estimation)
    std_dev = df["consumption"].std()
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["forecast"] + std_dev,
            mode="lines",
            name="Upper Bound",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["forecast"] - std_dev,
            mode="lines",
            name="Confidence Interval",
            fill="tonexty",
            fillcolor="rgba(39, 174, 96, 0.2)",
            line=dict(width=0),
        )
    )

    fig_forecast.update_layout(
        xaxis_title="Time",
        yaxis_title="Forecasted Consumption (kWh)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        if len(anomalies_subset) > 0:
            st.warning(
                f"⚠️ Detected **{len(anomalies_subset)} anomalies** in the last "
                f"{days_to_show} days. Check equipment for faults or unusual consumption patterns."
            )
        else:
            st.success("✅ No recent anomalies detected. System operating normally.")

    with col2:
        peak_hour = forecast_df.loc[forecast_df["forecast"].idxmax()]
        st.info(
            f"📈 Peak usage expected at **{peak_hour['timestamp'].strftime('%H:%M')}** "
            f"with **{peak_hour['forecast']:.1f} kWh**. "
            "Consider load shifting to off-peak hours."
        )

    # Download forecast
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast Data",
        data=csv,
        file_name="energy_forecast.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.markdown("Please run the training script first:")
    st.code("python src/train.py", language="bash")
