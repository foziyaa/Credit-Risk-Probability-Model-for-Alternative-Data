# src/train.py

import pandas as pd
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from . import config

# Configure logging for clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path):
    """
    Loads the processed feature set from a CSV file.

    Args:
        path (Path): The path to the processed data file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    logging.info(f"Loading processed data from {path}...")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        logging.error(f"Processed data file not found at {path}. Aborting.")
        raise

def split_data(df):
    """
    Splits the DataFrame into features (X) and target (y), and then into
    training and testing sets according to parameters in the config file.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and the target.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.
    """
    logging.info("Splitting data into training and testing sets...")
    X = df[config.FEATURES_TO_KEEP]
    y = df[config.TARGET_VARIABLE]
    
    # Use stratification to maintain the same proportion of target classes in splits
    stratify_option = y if config.STRATIFY_TARGET else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE, 
        stratify=stratify_option
    )
    logging.info(f"Data split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, y_pred_proba):
    """
    Calculates and returns a dictionary of key classification metrics.

    Args:
        y_true (pd.Series): The true labels.
        y_pred (np.array): The predicted labels.
        y_pred_proba (np.array): The predicted probabilities for the positive class.

    Returns:
        dict: A dictionary containing metric names and their values.
    """
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
    logging.info(f"Evaluation metrics: {metrics}")
    return metrics

def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test):
    """
    Creates a scikit-learn pipeline, trains the model, evaluates it,
    and logs all parameters, metrics, and the model artifact to MLflow.

    Args:
        model: An unfitted scikit-learn model instance.
        model_name (str): The name of the model for logging purposes.
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
    """
    # Start an MLflow run with a specific name for clarity in the UI
    with mlflow.start_run(run_name=model_name) as run:
        logging.info(f"--- Training {model_name} ---")
        
        # Create a pipeline to chain the scaler and the model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Train the pipeline
        pipeline.fit(X_train, y_train)
        logging.info("Model training complete.")
        
        # Make predictions for evaluation
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate evaluation metrics
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Log parameters and metrics to MLflow
        mlflow.log_params(pipeline.named_steps['model'].get_params())
        mlflow.log_metrics(metrics)
        
        # Log the entire pipeline as the model artifact.
        # This is crucial as it packages the scaler with the model.
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=f"{config.MODEL_REGISTRY_NAME}-{model_name}"
        )
        logging.info(f"Model, parameters, and metrics for {model_name} logged to MLflow.")

def main():
    """Main function to orchestrate the model training and evaluation process."""
    logging.info("Starting model training process...")
    
    # Load and split the data
    df = load_data(config.PROCESSED_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Configure MLflow experiment tracking
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    logging.info(f"MLflow tracking to: {config.MLFLOW_TRACKING_URI}")
    
    # Define the models to be trained
    models = {
        "LogisticRegression": LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1)
    }
    
    # Iterate through the models, train, and log each one
    for name, model in models.items():
        train_and_log_model(model, name, X_train, y_train, X_test, y_test)
        
    logging.info("Model training and tracking process complete.")

if __name__ == '__main__':
    main()