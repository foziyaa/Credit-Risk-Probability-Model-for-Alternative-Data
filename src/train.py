
import pandas as pd
from pathlib import Path
import logging

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Define file paths
PROCESSED_DATA_PATH = BASE_DIR / 'data' / 'processed' / 'customer_features.csv'

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "Credit_Risk_Bati_Bank"
MODEL_REGISTRY_NAME = "CreditRiskModel"

def evaluate_model(true_values, predicted_values, predicted_proba):
    """Calculate and return a dictionary of classification metrics."""
    accuracy = accuracy_score(true_values, predicted_values)
    auc = roc_auc_score(true_values, predicted_proba)
    precision = precision_score(true_values, predicted_values)
    recall = recall_score(true_values, predicted_values)
    f1 = f1_score(true_values, predicted_values)
    return {
        "accuracy": accuracy,
        "roc_auc": auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

def main():
    """Main function to run the model training and evaluation process."""
    logging.info("Starting model training process...")

    # Load the processed data
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        logging.info(f"Processed data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Processed data file not found at {PROCESSED_DATA_PATH}")
        return

    # 1. Data Splitting
    X = df.drop(columns=['CustomerId', 'is_high_risk'])
    y = df['is_high_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info("Data split into training and testing sets.")

    # 2. Model Definitions
    # We will test two models: Logistic Regression (interpretable) and Random Forest (performance)
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1)
    }

    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logging.info(f"MLflow tracking to: {MLFLOW_TRACKING_URI}")

    for model_name, model in models.items():
        logging.info(f"--- Training {model_name} ---")

        # Start an MLflow run
        with mlflow.start_run(run_name=model_name):
            
            # 3. Create a scikit-learn pipeline
            # This pipeline first scales the data, then fits the model.
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            # 4. Train the model
            pipeline.fit(X_train, y_train)
            logging.info("Model training complete.")

            # 5. Evaluate the model
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            
            metrics = evaluate_model(y_test, y_pred, y_pred_proba)
            logging.info(f"Evaluation metrics for {model_name}: {metrics}")

            # 6. Log to MLflow
            # Log parameters (from the model step of the pipeline)
            mlflow.log_params(pipeline.named_steps['model'].get_params())
            
            # Log metrics
            mlflow.log_metrics(metrics)

            # Log the entire pipeline as the model
            # This ensures that the scaling is part of the model artifact
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=f"{MODEL_REGISTRY_NAME}_{model_name}"
            )
            logging.info("Parameters, metrics, and model logged to MLflow.")

    logging.info("Model training and tracking process complete.")

if __name__ == '__main__':
    main()