import pandas as pd
import pytest
from src.data_processing import find_high_risk_cluster # Import the function

def test_find_high_risk_cluster_scenario1():
    """
    Test where cluster 1 has the highest Recency.
    """
    # Create a sample DataFrame of cluster centroids
    centroids_data = {
        'Recency': [20, 150, 50],
        'Frequency': [10, 2, 5],
        'Monetary': [5000, 1000, 3000]
    }
    centroids_df = pd.DataFrame(centroids_data)
    
    # The function should return the index of the row with the max Recency
    expected_cluster_id = 1
    actual_cluster_id = find_high_risk_cluster(centroids_df)
    
    assert actual_cluster_id == expected_cluster_id

def test_find_high_risk_cluster_scenario2():
    """
    Test where cluster 0 has the highest Recency.
    """
    centroids_data = {
        'Recency': [200, 150, 50],
        'Frequency': [1, 2, 5],
        'Monetary': [500, 1000, 3000]
    }
    centroids_df = pd.DataFrame(centroids_data)
    
    expected_cluster_id = 0
    actual_cluster_id = find_high_risk_cluster(centroids_df)
    
    assert actual_cluster_id == expected_cluster_id

def test_find_high_risk_cluster_with_tie():
    """
    Test how it handles a tie in Recency (should pick the first one).
    """
    centroids_data = {
        'Recency': [150, 20, 150],
        'Frequency': [2, 10, 1],
        'Monetary': [1000, 5000, 500]
    }
    centroids_df = pd.DataFrame(centroids_data)
    
    # idxmax() returns the index of the *first* occurrence of the maximum
    expected_cluster_id = 0
    actual_cluster_id = find_high_risk_cluster(centroids_df)
    
    assert actual_cluster_id == expected_cluster_id