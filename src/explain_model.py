# src/explain_model.py

import pandas as pd
import logging
import mlflow
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from . import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_shap_explanations():
    """
    Loads the best model, generates SHAP explanations, and saves plots.
    """
    logging.info("Starting model explainability process with SHAP...")

    # Set the MLflow Tracking URI
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    logging.info(f"Connecting to MLflow at {config.MLFLOW_TRACKING_URI}")

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
        logging.info("Processed data loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Data file not found at {config.PROCESSED_DATA_PATH}. Aborting.")
        return

    # Split data to get the same test set used during training
    X = df[config.FEATURES_TO_KEEP]
    y = df[config.TARGET_VARIABLE]
    stratify = y if config.STRATIFY_TARGET else None
    _, X_test, _, _ = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=stratify
    )

    # --- 2. Load Model from MLflow ---
    model_uri = f"models:/{config.MODEL_REGISTRY_NAME}-RandomForest/latest"
    try:
        model_pipeline = mlflow.sklearn.load_model(model_uri)
        logging.info("Random Forest model pipeline loaded successfully from MLflow.")
    except Exception as e:
        logging.error(f"Failed to load model from MLflow: {e}. Aborting.")
        return

    model = model_pipeline.named_steps['model']
    scaler = model_pipeline.named_steps['scaler']

    # --- 3. Generate SHAP Explanations ---
    X_test_scaled = scaler.transform(X_test)

    logging.info("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    logging.info("Calculating SHAP values for the test set...")
    shap_values = explainer(X_test_scaled) # Use explainer as a function for modern object output
    
    # --- 4. Create and Save Plots using the shap.Explanation object ---
    # This is the modern and robust way to handle SHAP values and data.
    # We create an Explanation object that contains the SHAP values for the "high-risk" class (class 1),
    # the original (unscaled) data for plotting, and the feature names.
    shap_explanation_high_risk = shap.Explanation(
        values=shap_values.values[:, :, 1],      # SHAP values for class 1
        base_values=shap_values.base_values[:, 1], # Base values for class 1
        data=X_test,                             # Original unscaled data
        feature_names=X_test.columns.tolist()    # Feature names
    )
    
    output_dir = config.BASE_DIR / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving SHAP plots to {output_dir}")

    # a) SHAP Bar Plot (Global Feature Importance)
    plt.figure()
    # Now we pass the Explanation object directly to the plot function.
    shap.plots.bar(shap_explanation_high_risk, show=False)
    plt.title("Global Feature Importance (SHAP Summary)")
    plt.tight_layout()
    summary_plot_path = output_dir / "shap_summary_plot.png"
    plt.savefig(summary_plot_path)
    plt.close()
    logging.info(f"Summary plot saved to {summary_plot_path}")

    # b) SHAP Beeswarm Plot (Detailed Feature Impact)
    plt.figure()
    # Pass the same Explanation object here.
    shap.plots.beeswarm(shap_explanation_high_risk, show=False)
    plt.title("Detailed Feature Impact (SHAP Beeswarm)")
    plt.tight_layout()
    beeswarm_plot_path = output_dir / "shap_beeswarm_plot.png"
    plt.savefig(beeswarm_plot_path)
    plt.close()
    logging.info(f"Beeswarm plot saved to {beeswarm_plot_path}")

    logging.info("SHAP explainability process complete.")

if __name__ == '__main__':
    generate_shap_explanations()