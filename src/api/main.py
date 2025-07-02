from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
import logging
from .pydantic_models import PredictionRequest, PredictionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(title="Credit Risk Prediction API", version="1.0")

# --- Model Loading ---
# This section should run only once when the application starts.
MODEL_NAME = "BNPL-Credit-Risk-Model"
MODEL_STAGE = "None"  # Or "Staging", "Production" if you've set them

try:
    # Load the model from the MLflow Model Registry
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    logging.info(f"Successfully loaded model '{MODEL_NAME}' version from stage '{MODEL_STAGE}'.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    # If the model fails to load, we can set it to None and handle it in the endpoint
    model = None

@app.get("/", tags=["Root"])
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the Credit Risk Prediction API!"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """
    Accepts customer data and returns a credit risk prediction.
    
    - **prediction**: 0 (Low Risk), 1 (High Risk)
    - **risk_probability**: The probability of being high-risk (a float between 0 and 1).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please check the logs.")

    try:
        # Convert the request data into a pandas DataFrame
        # The model expects a DataFrame with the same feature names
        data = pd.DataFrame([request.dict()])
        
        # The model.predict() for a classifier often returns a NumPy array.
        # For sklearn classifiers, predict_proba returns probabilities for each class.
        # We are interested in the probability of the positive class (1, i.e., high-risk).
        
        # Make predictions
        prediction_result = model.predict(data)
        
        # It's good practice to ensure predict_proba is available
        if hasattr(model._model_impl, 'predict_proba'):
            # Access the underlying model to get probabilities
            proba_result = model._model_impl.predict_proba(data)
            # Probability of the positive class (high-risk)
            risk_probability = proba_result[0][1] 
        else:
            # Fallback if predict_proba is not available
            risk_probability = -1.0 # Or some other indicator
        
        return PredictionResponse(
            prediction=int(prediction_result[0]),
            risk_probability=risk_probability
        )
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")