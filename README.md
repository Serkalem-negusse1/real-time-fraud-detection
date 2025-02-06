# Fraud Detection in E-commerce and Bank Transactions

This project aims to improve the detection of fraudulent activities in both e-commerce and bank credit transactions. By leveraging machine learning models and advanced feature engineering techniques, the goal is to identify and mitigate fraudulent transactions effectively. The solution includes data preprocessing, feature engineering, model building, and deployment through Flask and Dash for real-time monitoring and reporting.

## Project Overview

Fraud detection is a critical component in e-commerce and banking, where detecting fraudulent transactions prevents financial losses and increases customer trust. This project uses datasets from e-commerce and bank credit card transactions to train models that predict fraudulent activity.

The project consists of the following key tasks:
1. **Data Analysis and Preprocessing**:
   - Handle missing values, data cleaning, and feature engineering.
   - Merge data for geolocation analysis.
2. **Model Building**:
   - Train multiple models, such as Logistic Regression, Decision Tree, Random Forest, and more.
3. **Model Explainability**:
   - Use SHAP and LIME for explaining model predictions.
4. **Model Deployment**:
   - Create a Flask API to serve the model for real-time predictions.
   - Dockerize the application for deployment.
5. **Dashboard Creation**:
   - Build a Dash app to visualize fraud detection insights.

## Requirements

Before running the project, make sure you have the following dependencies installed:

```bash
pip install -r requirements.txt
