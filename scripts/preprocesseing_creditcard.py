# fraud_detection_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# --- Preprocessing Functions ---

def load_data(fraud_data_path):
    """
    Load the dataset from the specified path.
    """
    return pd.read_csv(fraud_data_path)

def preprocess_data(fraud_df):
    """
    Preprocess the fraud data:
    1. Handle missing values
    2. Scale features
    """
    # Handle missing values
    # 1. Fill missing values in features V1 to V28 with the mean of each column
    fraud_df.fillna(fraud_df[['Amount'] + [f'V{i}' for i in range(1, 29)]].mean(), inplace=True)
    
    # 2. Fill missing 'Time' with forward fill as it’s likely a time series
    fraud_df['Time'].fillna(method='ffill', inplace=True)
    
    # 3. Drop rows where target 'Class' is missing (if any)
    fraud_df.dropna(subset=['Class'], inplace=True)

    # Feature Scaling using Min-Max scaling
    scaler = MinMaxScaler()
    fraud_df[['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]] = scaler.fit_transform(fraud_df[['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]])

    return fraud_df

def save_preprocessed_data(fraud_df, output_path):
    """
    Save the preprocessed data to the specified output path.
    """
    fraud_df.to_csv(output_path, index=False)

# --- EDA Functions ---
