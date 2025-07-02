from pydantic import BaseModel

class PredictionRequest(BaseModel):
    """
    Defines the structure for a prediction request.
    The fields must match the feature names the model was trained on.
    """
    Recency: int
    Frequency: int
    Monetary: float
    AvgMonetary: float
    StdMonetary: float
    Tenure: int
    NumUniqueProducts: int
    NumUniqueProviders: int
    
    # Example to show structure
    class Config:
        schema_extra = {
            "example": {
                "Recency": 10,
                "Frequency": 5,
                "Monetary": 50000.0,
                "AvgMonetary": 10000.0,
                "StdMonetary": 5000.0,
                "Tenure": 180,
                "NumUniqueProducts": 3,
                "NumUniqueProviders": 2
            }
        }

class PredictionResponse(BaseModel):
    """
    Defines the structure for a prediction response.
    """
    prediction: int  # 0 for low-risk, 1 for high-risk
    risk_probability: float