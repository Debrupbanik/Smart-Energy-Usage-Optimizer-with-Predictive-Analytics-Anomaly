import os
import joblib
import numpy as np
import pandas as pd
from data_generator import generate_synthetic_data
from preprocessing import preprocess_data
from models import build_rf_model, train_anomaly_detector

def train():
    print("Generating data...")
    df = generate_synthetic_data(days=30)
    df.to_csv('energy_data.csv')
    
    print("Preprocessing data...")
    X, y, scaler = preprocess_data(df)
    
    # Flatten X for Random Forest (samples, features)
    # X shape is (samples, seq_len, 1) -> (samples, seq_len)
    X_flat = X.reshape(X.shape[0], X.shape[1])
    
    # Split into train/test
    train_size = int(len(X_flat) * 0.8)
    X_train, X_test = X_flat[:train_size], X_flat[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print("Training Random Forest model...")
    rf_model = build_rf_model()
    rf_model.fit(X_train, y_train)
    
    print("Training Anomaly Detector...")
    # Train on consumption data directly for anomaly detection
    iso_forest = train_anomaly_detector(df[['consumption']])
    
    print("Saving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/rf_model.pkl')
    joblib.dump(iso_forest, 'models/iso_forest.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("Training complete.")

if __name__ == "__main__":
    train()
