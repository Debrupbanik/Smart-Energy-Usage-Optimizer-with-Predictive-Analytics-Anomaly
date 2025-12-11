from sklearn.ensemble import RandomForestRegressor, IsolationForest
import joblib

def build_rf_model():
    """
    Builds a Random Forest model for forecasting.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return model

def train_anomaly_detector(data, contamination=0.01):
    """
    Trains an Isolation Forest model for anomaly detection.
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(data)
    return iso_forest
