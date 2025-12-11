import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from src.preprocessing import create_sequences

# Load models and data
@st.cache_resource
def load_models():
    rf_model = joblib.load('models/rf_model.pkl')
    iso_forest = joblib.load('models/iso_forest.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return rf_model, iso_forest, scaler

@st.cache_data
def load_data():
    return pd.read_csv('energy_data.csv', parse_dates=['timestamp'], index_col='timestamp')

st.set_page_config(page_title="Smart Energy Optimizer", layout="wide")

st.title("⚡ Smart Energy Usage Optimizer")
st.markdown("### Predictive Analytics & Anomaly Detection")

try:
    rf_model, iso_forest, scaler = load_models()
    df = load_data()
    
    # Sidebar
    st.sidebar.header("Settings")
    days_to_show = st.sidebar.slider("Days to visualize", 1, 30, 7)
    
    # Anomaly Detection
    df['anomaly'] = iso_forest.predict(df[['consumption']])
    anomalies = df[df['anomaly'] == -1]
    
    # Forecasting
    last_sequence = df['consumption'].values[-24:].reshape(-1, 1)
    last_sequence_scaled = scaler.transform(last_sequence)
    
    # Generate next 24 hours forecast
    forecast = []
    current_seq = last_sequence_scaled.reshape(1, 24) # (1, 24) for RF
    
    for _ in range(24):
        pred = rf_model.predict(current_seq)
        forecast.append(pred[0])
        # Update sequence: remove first, add new pred
        current_seq = np.append(current_seq[:, 1:], [[pred[0]]], axis=1)
        
    forecast_actual = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    
    # Visualization
    st.subheader("Energy Consumption & Anomalies")
    
    fig = go.Figure()
    
    # Historical Data
    subset = df.tail(days_to_show * 24)
    fig.add_trace(go.Scatter(x=subset.index, y=subset['consumption'], mode='lines', name='Consumption'))
    
    # Anomalies
    anomalies_subset = subset[subset['anomaly'] == -1]
    fig.add_trace(go.Scatter(x=anomalies_subset.index, y=anomalies_subset['consumption'], mode='markers', name='Anomaly', marker=dict(color='red', size=8)))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast
    st.subheader("24-Hour Forecast")
    
    forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1), periods=24, freq='H')
    forecast_df = pd.DataFrame({'timestamp': forecast_dates, 'forecast': forecast_actual.flatten()})
    
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=forecast_df['timestamp'], y=forecast_df['forecast'], mode='lines+markers', name='Forecast', line=dict(color='green')))
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Recommendations
    st.subheader("💡 Recommendations")
    if len(anomalies_subset) > 0:
        st.warning(f"Detected {len(anomalies_subset)} anomalies in the last {days_to_show} days. Check equipment for faults.")
    else:
        st.success("No recent anomalies detected.")
        
    peak_hour = forecast_df.loc[forecast_df['forecast'].idxmax()]
    st.info(f"Peak usage expected at {peak_hour['timestamp'].strftime('%H:%M')} with {peak_hour['forecast']:.2f} kWh. Consider load shifting.")

except Exception as e:
    st.error(f"Error loading models or data. Please run training script first. Error: {e}")
    st.code("python src/train.py")
