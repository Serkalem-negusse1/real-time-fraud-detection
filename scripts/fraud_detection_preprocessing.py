import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load datasets
fraud_df = pd.read_csv("E:/data/Data08&9/Fraud_Data.csv")
ip_df = pd.read_csv("E:/data/Data08&9/IpAddress_to_Country.csv")

# Convert timestamps to datetime format
fraud_df["signup_time"] = pd.to_datetime(fraud_df["signup_time"])
fraud_df["purchase_time"] = pd.to_datetime(fraud_df["purchase_time"])

# Extract time-based features
fraud_df["signup_hour"] = fraud_df["signup_time"].dt.hour
fraud_df["signup_day"] = fraud_df["signup_time"].dt.dayofweek
fraud_df["purchase_hour"] = fraud_df["purchase_time"].dt.hour
fraud_df["purchase_day"] = fraud_df["purchase_time"].dt.dayofweek

# Convert IP addresses to integer format
fraud_df["ip_address"] = fraud_df["ip_address"].astype(int)

# Merge Fraud Data with IP Address Mapping
def map_ip_to_country(ip):
    match = ip_df[(ip_df["lower_bound_ip_address"] <= ip) & (ip_df["upper_bound_ip_address"] >= ip)]
    return match["country"].values[0] if not match.empty else "Unknown"

fraud_df["country"] = fraud_df["ip_address"].apply(map_ip_to_country)

# Save processed data
fraud_df.to_csv("Processed_Fraud_Data.csv", index=False)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10,5))
sns.countplot(x='class', data=fraud_df, palette='coolwarm')
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Class (0: Non-Fraud, 1: Fraud)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12,6))
sns.histplot(fraud_df['purchase_hour'], bins=24, kde=True, color='blue')
plt.title("Transaction Volume by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x='class', y='purchase_value', data=fraud_df, palette='Set2')
plt.ylim(0, 500)
plt.title("Transaction Amount by Fraud Class")
plt.show()

# Outlier Detection using Boxplot
plt.figure(figsize=(12,6))
sns.boxplot(data=fraud_df[['purchase_value']], palette='Set3')
plt.title("Outlier Detection in Transaction Amounts")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
corr = fraud_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

print("Data preprocessing and EDA completed. Processed data saved as Processed_Fraud_Data.csv")
