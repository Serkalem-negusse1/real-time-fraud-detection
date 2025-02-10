import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.neural_network import MLPClassifier
import mlflow
from imblearn.over_sampling import SMOTE



# Load datasets
def load_data(creditcard_path, fraud_path):
    creditcard_df = pd.read_csv(creditcard_path)
    fraud_df = pd.read_csv(fraud_path)
    return creditcard_df, fraud_df

# Feature and target separation
def prepare_data(df, target_col):
    # Handle datetime columns (e.g., signup_time, purchase_time)
    date_columns = ['signup_time', 'purchase_time']  # Replace with the actual datetime columns
    for col in date_columns:
        if col in df.columns:
            # Convert to datetime format
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Extract useful time-related features (year, month, day, hour, etc.)
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_minute'] = df[col].dt.minute
            df[f'{col}_second'] = df[col].dt.second
            # Drop the original datetime column
            df.drop(columns=[col], inplace=True)

    # Handle categorical columns (e.g., sex, browser, country, source, device_id)
    categorical_columns = ['sex', 'browser', 'country', 'source', 'device_id', 'region']  # Add more if needed
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        if col in df.columns:
            df[col] = label_encoder.fit_transform(df[col].astype(str))

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# Train-test split
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Handle imbalance using SMOTE
def handle_imbalance(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

# Model training and evaluation
def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f'\n=== {model_name} ===')
    print(classification_report(y_test, y_pred))
    print(f'ROC AUC Score: {roc_auc_score(y_test, y_prob)}')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

    # AUC-ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUC-ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.show()

# Logging with MLflow
def log_model_with_mlflow(model, model_name, X_train, y_train, X_test, y_test):
    mlflow.set_experiment("Fraud Detection")
    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", np.mean(y_pred == y_test))
        mlflow.sklearn.log_model(model, f"{model_name}_model")
        
        print(f"{model_name} logged successfully")

# Define models
def define_models():
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)  # Added MLP model
    }
    return models

def main():
    # Load datasets
    creditcard_df, fraud_df = load_data('E:/Git_repo/real-time-fraud-detection/data/creditcard_preprocessed.csv', 'E:/Git_repo/real-time-fraud-detection/data/Processed_Fraud_Data.csv')

    # Prepare data
    X_credit, y_credit = prepare_data(creditcard_df, 'Class')
    X_fraud, y_fraud = prepare_data(fraud_df, 'class')

    # Split data
    X_train_credit, X_test_credit, y_train_credit, y_test_credit = split_data(X_credit, y_credit)
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = split_data(X_fraud, y_fraud)

    # Handle imbalance
    X_train_credit, y_train_credit = handle_imbalance(X_train_credit, y_train_credit)
    X_train_fraud, y_train_fraud = handle_imbalance(X_train_fraud, y_train_fraud)

    # Define models
    models = define_models()

    # Train and evaluate models
    for model_name, model in models.items():
        train_and_evaluate(model, X_train_credit, y_train_credit, X_test_credit, y_test_credit, model_name)
        train_and_evaluate(model, X_train_fraud, y_train_fraud, X_test_fraud, y_test_fraud, model_name)

        # Log models
        log_model_with_mlflow(model, model_name, X_train_credit, y_train_credit, X_test_credit, y_test_credit)

if __name__ == '__main__':
    main()
