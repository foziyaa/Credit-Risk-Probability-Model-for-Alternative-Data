# tests/test_data_processing.py

import pandas as pd
import pytest
from datetime import datetime
from src import data_processing, config # Import the module and config

# --- Fixtures for Reusable Test Data ---

@pytest.fixture
def raw_data_fixture():
    """Provides a sample raw DataFrame for testing."""
    data = {
        'CustomerId': ['C1', 'C2', 'C1', 'C2', 'C3'],
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'TransactionStartTime': [
            '2025-01-10T10:00:00Z', '2025-01-15T12:00:00Z', 
            '2025-01-20T14:00:00Z', '2025-01-25T16:00:00Z',
            '2025-01-05T09:00:00Z' # C3 is the disengaged/high-risk customer
        ],
        'Value': [100, 200, 150, 250, 50],
        'ProductId': ['P1', 'P2', 'P1', 'P3', 'P4'],
        'ProviderId': ['V1', 'V2', 'V1', 'V2', 'V3']
    }
    return pd.DataFrame(data)

@pytest.fixture
def customer_df_fixture():
    """Provides a sample aggregated customer DataFrame for testing."""
    data = {
        'CustomerId': ['C1', 'C2', 'C3'],
        'Frequency': [2, 2, 1],
        'Monetary': [250, 450, 50],
        'AvgMonetary': [125, 225, 50],
        'StdMonetary': [35.35, 35.35, 0.0],
        'FirstTransactionDate': pd.to_datetime(['2025-01-10', '2025-01-15', '2025-01-05']),
        'LastTransactionDate': pd.to_datetime(['2025-01-20', '2025-01-25', '2025-01-05']),
        'NumUniqueProducts': [1, 2, 1],
        'NumUniqueProviders': [1, 1, 1]
    }
    return pd.DataFrame(data)

# --- Unit Tests ---

def test_find_high_risk_cluster_logic():
    """
    Tests the core business logic: the cluster with the highest Recency is high-risk.
    """
    centroids_data = {
        'Recency': [20, 150, 50],
        'Frequency': [10, 2, 5],
        'Monetary': [5000, 1000, 3000]
    }
    centroids_df = pd.DataFrame(centroids_data)
    
    # The function should return the index of the row with the max Recency
    assert data_processing.find_high_risk_cluster(centroids_df) == 1

def test_load_data(tmp_path):
    """Tests that data is loaded correctly from a CSV file."""
    # Arrange: Create a temporary CSV file
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test_data.csv"
    p.write_text("col1,col2\n1,2")
    
    # Act
    df = data_processing.load_data(p)
    
    # Assert
    assert not df.empty
    assert 'col1' in df.columns

def test_preprocess_data(raw_data_fixture):
    """Tests data type conversion and column renaming."""
    # Act
    df = data_processing.preprocess_data(raw_data_fixture)
    
    # Assert
    assert 'TransactionValue' in df.columns
    assert 'Value' not in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['TransactionStartTime'])

def test_aggregate_customer_data(raw_data_fixture):
    """Tests the correctness of the customer-level aggregation."""
    # Arrange
    df = data_processing.preprocess_data(raw_data_fixture)
    
    # Act
    customer_df = data_processing.aggregate_customer_data(df)
    
    # Assert
    assert len(customer_df) == 3  # Should have one row per unique customer
    
    # Check specific aggregations for Customer C1
    c1_data = customer_df[customer_df['CustomerId'] == 'C1']
    assert c1_data['Frequency'].iloc[0] == 2
    assert c1_data['Monetary'].iloc[0] == 250
    assert c1_data['NumUniqueProducts'].iloc[0] == 1
    
    # Check that StdMonetary is filled with 0 for single-transaction customer (C3)
    c3_data = customer_df[customer_df['CustomerId'] == 'C3']
    assert c3_data['StdMonetary'].iloc[0] == 0.0

def test_engineer_features(customer_df_fixture):
    """Tests the calculation of Recency and Tenure features."""
    # Arrange
    snapshot_date = datetime(2025, 1, 26)
    
    # Act
    df = data_processing.engineer_features(customer_df_fixture, snapshot_date)
    
    # Assert
    assert 'Recency' in df.columns
    assert 'Tenure' in df.columns
    
    # Check calculations for Customer C1 (Last transaction on Jan 20)
    c1_recency = df[df['CustomerId'] == 'C1']['Recency'].iloc[0]
    assert c1_recency == 6 # (26 - 20)
    
    # Check calculations for Customer C2 (Tenure from Jan 15 to Jan 25)
    c2_tenure = df[df['CustomerId'] == 'C2']['Tenure'].iloc[0]
    assert c2_tenure == 10

def test_create_proxy_target(customer_df_fixture):
    """
    Tests the creation of the proxy target variable, ensuring it correctly
    identifies the high-risk customer based on recency.
    """
    # Arrange: Add Recency and Tenure to the fixture
    snapshot_date = datetime(2025, 1, 26)
    df_with_features = data_processing.engineer_features(customer_df_fixture, snapshot_date)
    
    # Act
    df_with_target = data_processing.create_proxy_target(df_with_features)
    
    # Assert
    # 1. The target column must exist
    assert config.TARGET_VARIABLE in df_with_target.columns
    
    # 2. The target column must be binary (0 or 1)
    assert set(df_with_target[config.TARGET_VARIABLE].unique()) == {0, 1}
    
    # 3. The correct customer should be flagged as high-risk
    # In our fixture, C3 is the most recent (last purchase Jan 5 -> Recency = 21)
    # The K-Means logic should identify C3's cluster as high risk.
    c3_risk_flag = df_with_target[df_with_target['CustomerId'] == 'C3'][config.TARGET_VARIABLE].iloc[0]
    assert c3_risk_flag == 1
    
    # 4. The other customers should not be flagged
    non_risk_flags = df_with_target[df_with_target['CustomerId'] != 'C3'][config.TARGET_VARIABLE].sum()
    assert non_risk_flags == 0

# --- Integration Test ---

def test_full_data_processing_pipeline(tmp_path, monkeypatch, raw_data_fixture):
    """
    Tests the full data processing pipeline from raw data to final feature set.
    This ensures that all the individual functions work together correctly.
    """
    # Arrange:
    # 1. Create temporary directories for raw and processed data
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    
    # 2. Define temporary file paths
    raw_path = raw_dir / "test_training.csv"
    processed_path = processed_dir / "test_customer_features.csv"
    
    # 3. Save the fixture data to the temporary raw file
    raw_data_fixture.to_csv(raw_path, index=False)
    
    # 4. Use monkeypatch to override the paths in the config module
    # This makes the main() function use our temporary files instead of the real ones
    monkeypatch.setattr(config, 'RAW_DATA_PATH', raw_path)
    monkeypatch.setattr(config, 'PROCESSED_DATA_PATH', processed_path)
    
    # Act:
    # Run the main data processing pipeline
    data_processing.main()
    
    # Assert:
    # 1. Check that the output file was created
    assert processed_path.exists()
    
    # 2. Load the output file and check its integrity
    processed_df = pd.read_csv(processed_path)
    assert not processed_df.empty
    
    # 3. Verify that the correct columns are present
    expected_cols = ['CustomerId'] + config.FEATURES_TO_KEEP + [config.TARGET_VARIABLE]
    assert all(col in processed_df.columns for col in expected_cols)
    
    # 4. Check that the target variable was created and is populated
    assert config.TARGET_VARIABLE in processed_df.columns
    assert processed_df[config.TARGET_VARIABLE].sum() > 0  # At least one high-risk customer