# Smart Energy Usage Optimizer

Predictive analytics + anomaly detection for household/industrial energy consumption.

## Features

- **Machine Learning Forecasting** - Random Forest-based prediction for 24-hour energy forecasts
- **Anomaly Detection** - Isolation Forest for detecting unusual consumption patterns
- **Interactive Dashboard** - Streamlit UI with real-time visualization
- **Synthetic Data Generation** - Generate realistic energy consumption data with patterns

## Installation

```bash
# Clone the repository
git clone https://github.com/Debrupbanik/Smart-Energy-Usage-Optimizer-with-Predictive-Analytics-Anomaly.git
cd Smart-Energy-Usage-Optimizer-with-Predictive-Analytics-Anomaly

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train Models

```bash
python src/train.py
```

This will:
- Generate synthetic energy data
- Train Random Forest forecasting model
- Train Isolation Forest anomaly detector
- Save models to `models/` directory

### 2. Run Dashboard

```bash
streamlit run app.py
```

Visit `http://localhost:8501` to see the dashboard.

## Project Structure

```
├── app.py                    # Streamlit dashboard
├── energy_data.csv          # Sample energy data
├── models/                   # Trained models
│   ├── rf_model.pkl
│   ├── iso_forest.pkl
│   └── scaler.pkl
├── src/
│   ├── data_generator.py    # Synthetic data generation
│   ├── preprocessing.py     # Data preprocessing
│   ├── models.py            # ML model definitions
│   └── train.py            # Training script
└── tests/                   # Test suite
```

## Usage

### Generate Custom Data

```python
from src.data_generator import generate_synthetic_data

df = generate_synthetic_data(days=30)
df.to_csv('energy_data.csv')
```

### Use Models Programmatically

```python
import joblib
import pandas as pd
import numpy as np

# Load models
rf_model = joblib.load('models/rf_model.pkl')
iso_forest = joblib.load('models/iso_forest.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load data
df = pd.read_csv('energy_data.csv', parse_dates=['timestamp'], index_col='timestamp')

# Detect anomalies
df['anomaly'] = iso_forest.predict(df[['consumption']])
anomalies = df[df['anomaly'] == -1]

# Make predictions
last_seq = df['consumption'].values[-24:].reshape(-1, 1)
last_seq_scaled = scaler.transform(last_seq)

forecast = []
current_seq = last_seq_scaled.reshape(1, 24)
for _ in range(24):
    pred = rf_model.predict(current_seq)
    forecast.append(pred[0])
    current_seq = np.append(current_seq[:, 1:], [[pred[0]]], axis=1)
```

## Testing

```bash
pytest tests/ -v
```

## Requirements

- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- streamlit>=1.28.0
- matplotlib>=3.7.0
- plotly>=5.15.0
- joblib>=1.3.0

## Screenshots

### Energy Consumption & Anomalies
![Dashboard](https://github.com/user-attachments/assets/33054ab7-741e-44f8-a3e2-635e933ea638)

### 24-Hour Forecast
![Forecast](https://github.com/user-attachments/assets/f36c15f4-c8b4-4718-886f-e68dbd476c9d)

## License

MIT License
