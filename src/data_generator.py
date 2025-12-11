import numpy as np
import pandas as pd

def generate_synthetic_data(days=30):
    """
    Generates synthetic hourly energy consumption data.
    Includes:
    - Daily seasonality (peak in evening)
    - Weekly seasonality (lower on weekends)
    - Yearly trend (sinusoidal for temperature effect)
    - Random noise
    - Anomalies
    """
    hours = days * 24
    dates = pd.date_range(start='2024-01-01', periods=hours, freq='H')
    
    # Base consumption
    base = 50
    
    # Daily pattern (24 hours) - Peak around 18:00 - 20:00
    daily = 10 * np.sin(2 * np.pi * dates.hour / 24 - np.pi/2)
    
    # Weekly pattern - Weekends lower
    weekly = np.where(dates.dayofweek >= 5, -5, 0)
    
    # Yearly pattern - Seasonal (higher in winter/summer)
    yearly = 15 * np.sin(2 * np.pi * dates.dayofyear / 365)
    
    # Noise
    noise = np.random.normal(0, 2, hours)
    
    consumption = base + daily + weekly + yearly + noise
    consumption = np.array(consumption)
    
    # Inject anomalies
    anomalies_indices = np.random.choice(hours, size=int(hours * 0.01), replace=False)
    consumption[anomalies_indices] += np.random.choice([-20, 20], size=len(anomalies_indices))
    
    df = pd.DataFrame({'timestamp': dates, 'consumption': consumption})
    df.set_index('timestamp', inplace=True)
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    print(df.head())
    df.to_csv('energy_data.csv')
