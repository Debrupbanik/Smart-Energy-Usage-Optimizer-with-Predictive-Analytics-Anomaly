"""Tests for Smart Energy Optimizer."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from src.data_generator import generate_synthetic_data
from src.preprocessing import create_sequences, preprocess_data
from src.models import build_rf_model, train_anomaly_detector


class TestDataGenerator:
    """Test data generation."""

    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        df = generate_synthetic_data(days=5)

        assert len(df) == 5 * 24  # 5 days * 24 hours
        assert "consumption" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_generate_with_anomalies(self):
        """Test that anomalies are generated."""
        df = generate_synthetic_data(days=10)

        # Check that not all values are the same
        assert df["consumption"].std() > 0


class TestPreprocessing:
    """Test preprocessing functions."""

    def test_create_sequences(self):
        """Test sequence creation."""
        data = np.arange(100)

        X, y = create_sequences(data, seq_length=24)

        assert X.shape == (76, 24)
        assert y.shape == (76,)

    def test_preprocess_data(self):
        """Test data preprocessing."""
        df = generate_synthetic_data(days=5)

        X, y, scaler = preprocess_data(df, seq_length=24)

        # Check shapes
        assert len(X.shape) == 3  # (samples, seq_len, features)
        assert X.shape[1] == 24  # seq_length
        assert X.shape[0] == y.shape[0]

        # Check scaling (should be between 0 and 1)
        assert X.min() >= 0
        assert X.max() <= 1


class TestModels:
    """Test model training and prediction."""

    def test_build_rf_model(self):
        """Test RF model building."""
        model = build_rf_model()

        assert model is not None
        assert model.n_estimators == 100

    def test_train_anomaly_detector(self):
        """Test anomaly detector training."""
        df = generate_synthetic_data(days=5)

        iso_forest = train_anomaly_detector(df[["consumption"]])

        assert iso_forest is not None

        # Test prediction
        predictions = iso_forest.predict(df[["consumption"]])
        assert len(predictions) == len(df)

    def test_full_pipeline(self):
        """Test complete training pipeline."""
        # Generate data
        df = generate_synthetic_data(days=10)

        # Preprocess
        X, y, scaler = preprocess_data(df, seq_length=24)
        X_flat = X.reshape(X.shape[0], X.shape[1])

        # Train
        rf = build_rf_model()
        rf.fit(X_flat[:-10], y[:-10].ravel())

        # Predict
        pred = rf.predict(X_flat[-10:])
        assert len(pred) == 10


class TestApp:
    """Test app functionality."""

    def test_prediction_loop(self):
        """Test prediction loop in app."""
        df = generate_synthetic_data(days=5)

        X, y, scaler = preprocess_data(df, seq_length=24)
        X_flat = X.reshape(X.shape[0], X.shape[1])

        rf = build_rf_model()
        rf.fit(X_flat[:-10], y[:-10].ravel())

        # Simulate the prediction loop from app.py
        last_seq = df["consumption"].values[-24:].reshape(-1, 1)
        last_seq_scaled = scaler.transform(last_seq)
        current_seq = last_seq_scaled.reshape(1, 24)

        forecast = []
        for _ in range(24):
            pred = rf.predict(current_seq)
            forecast.append(pred[0])
            current_seq = np.append(current_seq[:, 1:], [[pred[0]]], axis=1)

        assert len(forecast) == 24


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
