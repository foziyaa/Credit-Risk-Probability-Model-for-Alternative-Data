# src/config.py

from pathlib import Path

# --- DIRECTORIES ---
# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# --- FILE PATHS ---
# Raw data file path
RAW_DATA_PATH = RAW_DATA_DIR / 'training.csv'
# Processed feature set file path
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / 'customer_features.csv'

# --- FEATURE ENGINEERING ---
# List of features to be used for model training
FEATURES_TO_KEEP = [
    'Recency', 'Frequency', 'Monetary', 'AvgMonetary', 'StdMonetary',
    'Tenure', 'NumUniqueProducts', 'NumUniqueProviders'
]

# Target variable name
TARGET_VARIABLE = 'is_high_risk'

# --- MODEL TRAINING ---
# Ratio for splitting data into training and testing sets
TEST_SIZE = 0.2
# Random state for reproducibility
RANDOM_STATE = 42
# Stratification based on the target variable
STRATIFY_TARGET = True

# --- MLFLOW CONFIGURATION ---
# URI for MLflow tracking server
MLFLOW_TRACKING_URI = "http://localhost:5000"
# Experiment name in MLflow
MLFLOW_EXPERIMENT_NAME = "BNPL_Credit_Risk"
# Model name for registration in MLflow Model Registry
MODEL_REGISTRY_NAME = "BNPL-Credit-Risk-Model"