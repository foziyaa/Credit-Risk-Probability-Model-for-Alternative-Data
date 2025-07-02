# src/data_processing.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging

# Configure logging to provide feedback during execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the base directory of the project to handle file paths robustly
BASE_DIR = Path(__file__).resolve().parent.parent

def find_high_risk_cluster(centroids_df):
    """
    Identifies the cluster ID corresponding to the highest risk segment.
    
    The primary hypothesis is that high-risk customers are those who are least engaged.
    This is characterized by High Recency (long time since last purchase),
    Low Frequency, and Low Monetary value. We prioritize identifying the cluster
    with the highest Recency as the main indicator of this "lapsed" or disengaged group.

    Args:
        centroids_df (pd.DataFrame): DataFrame containing the cluster centroids
                                     with columns 'Recency', 'Frequency', 'Monetary'.

    Returns:
        int: The cluster ID (index) identified as high-risk.
    """
    # The high-risk cluster is the one with the highest Recency
    high_risk_cluster_id = centroids_df['Recency'].idxmax()
    return high_risk_cluster_id

def create_feature_set(raw_data_path, processed_data_path):
   
    logging.info("Starting feature engineering and proxy target creation process...")

    # 1. Load Data
    try:
        df = pd.read_csv(raw_data_path)
        logging.info(f"Raw data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Raw data file not found at {raw_data_path}. Aborting.")
        return

    # 2. Basic Preprocessing
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    # Use 'Value' as it's the absolute transaction amount, which is more suitable for RFM
    df = df.rename(columns={'Value': 'TransactionValue'})
    
    # 3. Aggregate data to CustomerId level
    logging.info("Aggregating transaction data to the customer level...")
    
    aggregation_rules = {
        'TransactionId': 'count',
        'TransactionValue': ['sum', 'mean', 'std'],
        'TransactionStartTime': ['min', 'max'],
        'ProductId': pd.Series.nunique,
        'ProviderId': pd.Series.nunique
    }

    customer_df = df.groupby('CustomerId').agg(aggregation_rules).reset_index()

    # Flatten the multi-level column names for easier access
    customer_df.columns = [
        'CustomerId', 'Frequency', 'Monetary', 'AvgMonetary', 
        'StdMonetary', 'FirstTransactionDate', 'LastTransactionDate',
        'NumUniqueProducts', 'NumUniqueProviders'
    ]
    
    # Fill NaN in StdMonetary (occurs for customers with only one transaction) with 0
    customer_df['StdMonetary'] = customer_df['StdMonetary'].fillna(0)

    # 4. Engineer RFM and other behavioral features
    logging.info("Engineering RFM and tenure features...")
    
    # Define a snapshot date for consistent Recency calculation
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    customer_df['Recency'] = (snapshot_date - customer_df['LastTransactionDate']).dt.days

    # Calculate Customer Tenure (lifetime in days)
    customer_df['Tenure'] = (customer_df['LastTransactionDate'] - customer_df['FirstTransactionDate']).dt.days
    
    # --- Proxy Target Variable Engineering (Task 4) ---

    # 5. Cluster customers based on RFM profile
    logging.info("Clustering customers using K-Means on RFM features...")
    
    rfm_features = customer_df[['Recency', 'Frequency', 'Monetary']]
    
    # Scale RFM features to ensure each feature contributes equally to the distance calculation
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)

    # Apply K-Means clustering to segment customers into 3 groups
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customer_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # 6. Define and Assign the "High-Risk" Proxy Label
    # Inverse transform the cluster centers to interpret them in their original scale
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=['Recency', 'Frequency', 'Monetary'])
    
    logging.info("Interpreting cluster centroids (in original scale):")
    logging.info(f"\n{cluster_centers}")
    
    # Use our helper function to identify the high-risk cluster
    high_risk_cluster_id = find_high_risk_cluster(cluster_centers)
    logging.info(f"Identified high-risk cluster ID as: {high_risk_cluster_id}")
    
    # Create the binary target variable 'is_high_risk'
    customer_df['is_high_risk'] = (customer_df['Cluster'] == high_risk_cluster_id).astype(int)

    # 7. Finalize the dataset and save
    logging.info("Finalizing the dataset for model training...")
    
    # Select the final set of features and the target variable for the model
    features_to_keep = [
        'CustomerId',
        'Recency', 'Frequency', 'Monetary', 'AvgMonetary', 'StdMonetary',
        'Tenure', 'NumUniqueProducts', 'NumUniqueProviders',
        'is_high_risk' # This is our target variable
    ]
    final_df = customer_df[features_to_keep]

    # Ensure the output directory exists before saving
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the processed data to a new CSV file
    final_df.to_csv(processed_data_path, index=False)
    
    logging.info(f"Processed data successfully saved to: {processed_data_path}")
    logging.info(f"Final dataset shape: {final_df.shape}")
    logging.info("Data processing complete.")

if __name__ == '__main__':
    # Define file paths using the base directory for portability
    RAW_DATA_PATH = BASE_DIR / 'data' / 'raw' / 'training.csv'
    PROCESSED_DATA_PATH = BASE_DIR / 'data' / 'processed' / 'customer_features.csv'
    
    # Execute the main function
    create_feature_set(RAW_DATA_PATH, PROCESSED_DATA_PATH)