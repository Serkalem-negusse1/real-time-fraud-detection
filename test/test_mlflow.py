import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd  # Import pandas for data loading
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML

# 1. Load your data (replace with your actual data loading)
def load_and_split_data(file_path, target_column): # added target column
    try:
        credit_data = pd.read_csv(file_path)
        X = credit_data.drop(target_column, axis=1)
        y = credit_data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None, None
    except KeyError:
        print(f"Error: Target column '{target_column}' not found in the data.")
        return None, None, None, None
    except Exception as e: # Catch any other exceptions
        print(f"An error occurred during data loading: {e}")
        return None, None, None, None


# Example usage:
file_path = "E:/Git_repo/real-time-fraud-detection/data/creditcard_preprocessed.csv"  # Replace with your file path
target_column = "Class"  # Replace with your target column name
X_train_credit, X_test_credit, y_train_credit, y_test_credit = load_and_split_data(file_path, target_column)

# Check if data loading was successful
if X_train_credit is None:
    print("Data loading failed. Exiting.")
else:
    # 2. Set MLflow experiment
    mlflow.set_experiment("Fraud Detection")

    # 3. Start MLflow run
    with mlflow.start_run():
        # 4. Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_credit, y_train_credit)
        y_pred = model.predict(X_test_credit)

        # 5. Log parameters and metrics
        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("accuracy", np.mean(y_pred == y_test_credit))

        # 6. Infer signature and create input example
        signature = infer_signature(X_train_credit, model.predict(X_train_credit))
        input_example = X_train_credit.iloc[:2].to_dict(orient="list")  # Use iloc for integer-based indexing

        # 7. Log model with signature and input example
        mlflow.sklearn.log_model(model, "random_forest_model", signature=signature, input_example=input_example)

        print("Model logged successfully with signature and input example")

    # 8. Display MLflow UI link
    display(HTML("<a href='http://127.0.0.1:5000' target='_blank'>Click here to view MLflow UI</a>"))