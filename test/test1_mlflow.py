import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Data Loading Function (for a SINGLE dataset)
def load_and_split_data(file_path, target_column):
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
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None, None, None, None


# 2. Load Data (for the SINGLE dataset)
file_path = "E:/Git_repo/real-time-fraud-detection/data/creditcard_preprocessed.csv"  # Your file path
target_column = "Class"  # Your target column

X_train, X_test, y_train, y_test = load_and_split_data(file_path, target_column)

if X_train is None:
    print("Data loading failed. Exiting.")
    exit() 

# 3. Define Models (using only RandomForest)
models = [
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("LogisticRegression", LogisticRegression(random_state=42)), # Added logistic Regression
]

# 4. Initialize MLflow
mlflow.set_experiment("Fraud Detection - Single Dataset - RandomForest")  # More specific experiment name
mlflow.set_tracking_uri("http://localhost:5000")

# 5. Train and Log
reports = []

for model_name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    reports.append(report)

for i, (model_name, model) in enumerate(models):
    report = reports[i]

    with mlflow.start_run(run_name=f"{model_name}"):
        mlflow.log_param("model", model_name)
        mlflow.log_metric('accuracy', report['accuracy'])

        # Check if classes 0 and 1 are present (more robust)
        if '0' in report and '1' in report:
            mlflow.log_metric('recall_class_1', report['1']['recall'])
            mlflow.log_metric('recall_class_0', report['0']['recall'])
            mlflow.log_metric('f1_score_macro', report['macro avg']['f1-score'])
        else:
            print("Warning: Classes 0 and/or 1 not found in classification report. Skipping recall and f1-score logging.")
            print(report) # Print the report to inspect

        mlflow.sklearn.log_model(model, "model")  # Log the model

print("Training and logging complete.")