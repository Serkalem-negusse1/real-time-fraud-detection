# fraud_detection_preprocessing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipaddress
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# --- Preprocessing Functions ---

def preprocess_and_save(fraud_data_path, ip_data_path, output_path):
    # Load datasets
    fraud_df = pd.read_csv(fraud_data_path)
    ip_df = pd.read_csv(ip_data_path)

    # Handle missing values (example: forward fill)
    fraud_df = fraud_df.fillna(method='ffill')  # You can change 'ffill' to 'bfill' or a specific value

    # Convert timestamps to datetime format
    fraud_df["signup_time"] = pd.to_datetime(fraud_df["signup_time"])
    fraud_df["purchase_time"] = pd.to_datetime(fraud_df["purchase_time"])

    # Extract time-based features
    fraud_df["signup_hour"] = fraud_df["signup_time"].dt.hour
    fraud_df["signup_day"] = fraud_df["signup_time"].dt.dayofweek
    fraud_df["purchase_hour"] = fraud_df["purchase_time"].dt.hour
    fraud_df["purchase_day"] = fraud_df["purchase_time"].dt.dayofweek

    # Convert IP addresses to integer format using safe conversion function
    def safe_convert_ip(ip):
        try:
            return int(ipaddress.ip_address(ip))
        except ValueError:
            return np.nan

    fraud_df["ip_address"] = fraud_df["ip_address"].apply(safe_convert_ip)

    # Merge Fraud Data with IP Address Mapping
    def map_ip_to_country(ip):
        match = ip_df[(ip_df["lower_bound_ip_address"] <= ip) & (ip_df["upper_bound_ip_address"] >= ip)]
        return match["country"].values[0] if not match.empty else "Unknown"

    fraud_df["country"] = fraud_df["ip_address"].apply(map_ip_to_country)

    # Fill missing purchase_value with mean
    fraud_df['purchase_value'] = fraud_df['purchase_value'].fillna(fraud_df['purchase_value'].mean())

    # Optional: Log transformation for skewed features like 'purchase_value'
    fraud_df['log_purchase_value'] = np.log1p(fraud_df['purchase_value'])

    # Feature Scaling using Min-Max scaling
    scaler = MinMaxScaler()
    fraud_df[['purchase_value', 'signup_hour', 'purchase_hour']] = scaler.fit_transform(fraud_df[['purchase_value', 'signup_hour', 'purchase_hour']])

    # Create a country-to-region mapping
    country_to_region = {
        'United States': 'North America',
        'Canada': 'North America',
        'Mexico': 'North America',
        'Brazil': 'South America',
        'Argentina': 'South America',
        'Germany': 'Europe',
        'France': 'Europe',
        'Italy': 'Europe',
        'China': 'Asia',
        'India': 'Asia',
        'Japan': 'Asia',
        'Australia': 'Oceania',
        'South Africa': 'Africa',
        # Add more countries as needed
    }
    fraud_df['region'] = fraud_df['country'].map(country_to_region).fillna('Unknown')

    # Save processed data
    fraud_df.to_csv(output_path, index=False)

    return fraud_df

# --- EDA Functions ---

def plot_fraud_vs_non_fraud(fraud_df):
    plt.figure(figsize=(10,5))
    sns.countplot(x='class', data=fraud_df, palette='coolwarm', hue='class', legend=False)
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.xlabel("Class (0: Non-Fraud, 1: Fraud)")
    plt.ylabel("Count")
    #plt.show()

def plot_transaction_value_distribution(fraud_df):
    plt.figure(figsize=(12,6))
    sns.histplot(fraud_df['purchase_value'], kde=True, color='blue')
    plt.title("Transaction Value Distribution")
    plt.xlabel("Transaction Value")
    plt.ylabel("Frequency")
    #plt.show() 

def plot_log_transformed_transaction_value(fraud_df):
    plt.figure(figsize=(12,6))
    sns.histplot(fraud_df['log_purchase_value'], kde=True, color='green')
    plt.title("Log Transformed Transaction Value Distribution")
    plt.xlabel("Log of Transaction Value")
    plt.ylabel("Frequency")
    #plt.show() 

def plot_outlier_detection(fraud_df):
    outlier_detector = IsolationForest(contamination=0.05)  # Assume 5% contamination
    fraud_df['outlier'] = outlier_detector.fit_predict(fraud_df[['purchase_value']])
    plt.figure(figsize=(12,6))
    sns.boxplot(data=fraud_df[['purchase_value']], palette='Set3')
    plt.title("Outlier Detection in Transaction Amounts")
    #plt.show() 

def plot_feature_correlation_heatmap(fraud_df):
    numeric_df = fraud_df.select_dtypes(include=[np.number])  # Select only numeric columns
    plt.figure(figsize=(12,8))
    corr = numeric_df.corr()  # Calculate correlation for numeric columns only
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Feature Correlation Heatmap")
    #plt.show() 

def plot_pairwise_correlation(fraud_df):
    sns.pairplot(fraud_df[['purchase_value', 'purchase_hour', 'signup_hour', 'log_purchase_value', 'class']], hue='class', palette='Set2')
    plt.title("Pairwise Correlations")
    #plt.show() 

def plot_transaction_by_class(fraud_df):
    plt.figure(figsize=(12,6))
    sns.boxplot(x='class', y='purchase_value', data=fraud_df, palette='Set2', hue='class', legend=False)
    plt.title("Transaction Amount by Fraud Class")
    #plt.show() 

def plot_geographical_distribution(fraud_df):
    plt.figure(figsize=(14,7))
    region_counts = fraud_df.groupby(['region', 'class']).size().reset_index(name='count')  # Group by region and fraud class
    sns.barplot(x='region', y='count', data=region_counts, palette='viridis', hue='class')
    plt.title("Fraud Distribution by Region")
    plt.xlabel("Region")
    plt.ylabel("Count of Fraudulent Transactions")
    plt.xticks(rotation=45)
    #plt.show() 

def plot_fraud_trends_over_time(fraud_df):
    plt.figure(figsize=(12,6))
    fraud_df['signup_day'] = fraud_df['signup_time'].dt.date
    fraud_trends = fraud_df.groupby('signup_day')['class'].value_counts().unstack().fillna(0)
    fraud_trends.plot(kind='line', figsize=(12,6))
    plt.title("Fraud Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.legend(["Non-Fraud", "Fraud"])
    #plt.show()

def plot_transaction_amount_by_country(fraud_df):
    plt.figure(figsize=(12,6))
    sns.boxplot(x='country', y='purchase_value', data=fraud_df, palette='Set1', hue='class', legend=False)
    plt.xticks(rotation=90)
    plt.title("Transaction Amount by Country")
    #plt.show()

def check_multicollinearity(fraud_df):
    numerical_cols = ['purchase_value', 'signup_hour', 'purchase_hour', 'signup_day', 'purchase_day']
    X = fraud_df[numerical_cols]
    X = add_constant(X)  # Adding constant to the model
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\nVariance Inflation Factors (VIF):")
    print(vif_data)

