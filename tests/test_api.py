# tests/test_api.py

import pytest
from httpx import AsyncClient, ASGITransport  # <-- NEW IMPORT
from src.api.main import app  # Import the FastAPI app instance

# Mark all tests in this module as async for httpx
pytestmark = pytest.mark.anyio

@pytest.fixture
async def client():
    """Create an async test client for the API."""
    # Use ASGITransport to wrap the FastAPI app for httpx
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client: # <-- CORRECTED LINE
        yield client

async def test_read_root(client: AsyncClient):
    """Test the root endpoint to ensure the API is running."""
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Credit Risk Prediction API!"}

async def test_predict_endpoint_success(client: AsyncClient, monkeypatch):
    """
    Test the /predict endpoint with a valid request.
    We mock the model loading to avoid dependency on a running MLflow server.
    """
    # Arrange:
    # 1. Mock the MLflow model object in the main API script
    class MockModel:
        def predict(self, data):
            # Return a predictable dummy prediction (e.g., high risk)
            return [1]
        
        class MockModelImpl:
            def predict_proba(self, data):
                # Return a predictable dummy probability
                return [[0.1, 0.9]] # [prob_class_0, prob_class_1]
        
        _model_impl = MockModelImpl()

    monkeypatch.setattr("src.api.main.model", MockModel())

    # 2. Define a valid request payload
    valid_payload = {
        "Recency": 10,
        "Frequency": 5,
        "Monetary": 50000.0,
        "AvgMonetary": 10000.0,
        "StdMonetary": 5000.0,
        "Tenure": 180,
        "NumUniqueProducts": 3,
        "NumUniqueProviders": 2
    }

    # Act:
    response = await client.post("/predict", json=valid_payload)
    
    # Assert:
    assert response.status_code == 200
    response_data = response.json()
    assert "prediction" in response_data
    assert "risk_probability" in response_data
    assert response_data["prediction"] == 1
    assert response_data["risk_probability"] == 0.9

async def test_predict_endpoint_model_unavailable(client: AsyncClient, monkeypatch):
    """Test the API's response when the model failed to load."""
    # Arrange:
    # Set the model object to None, simulating a loading failure
    monkeypatch.setattr("src.api.main.model", None)
    
    valid_payload = {"Recency": 10, "Frequency": 5, "Monetary": 50000.0, "AvgMonetary": 10000.0, "StdMonetary": 5000.0, "Tenure": 180, "NumUniqueProducts": 3, "NumUniqueProviders": 2}

    # Act:
    response = await client.post("/predict", json=valid_payload)
    
    # Assert:
    assert response.status_code == 503 # Service Unavailable
    assert response.json() == {"detail": "Model is not available. Please check the logs."}