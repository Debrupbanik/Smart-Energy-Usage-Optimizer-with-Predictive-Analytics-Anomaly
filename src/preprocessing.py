import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length):
    """
    Creates sequences for LSTM training.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def preprocess_data(df, seq_length=24):
    """
    Preprocesses data:
    1. Scales data using MinMaxScaler
    2. Creates sequences
    """
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['consumption']])
    
    X, y = create_sequences(data_scaled, seq_length)
    
    return X, y, scaler
