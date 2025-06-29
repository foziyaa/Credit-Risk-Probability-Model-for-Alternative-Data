import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define base directory to handle paths robustly
BASE_DIR = Path(__file__).resolve().parent.parent

def create_feature_set(raw_data_path, processed_data_path):
    """
    Processes raw transaction data to create a customer-level feature set
    with a proxy target variable for credit risk.

    Args:
        raw_data_path (Path): Path to the raw training.csv file.
        processed_data_path (Path): Path to save the processed CSV file.
    """
    logging.info("Starting feature engineering process...")

    # 1. Load Data
    try:
        df = pd.read_csv(raw_data_path)
        logging.info(f"Raw data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Raw data file not found at {raw_data_path}")
        return

    # 2. Basic Preprocessing
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    # Use 'Value' as it's the absolute transaction amount
    df = df.rename(columns={'Value': 'TransactionValue'})
    
    # 3. Aggregate data to CustomerId level
    logging.info("Aggregating transaction data to customer level...")
    
    aggregation = {
        'TransactionId': 'count',
        'TransactionValue': ['sum', 'mean', 'std'],
        'TransactionStartTime': ['min', 'max'],
        'ProductId': pd.Series.nunique,
        'ProviderId': pd.Series.nunique
    }

    customer_df = df.groupby('CustomerId').agg(aggregation).reset_index()

    # Flatten the multi-level column names
    customer_df.columns = [
        'CustomerId', 'Frequency', 'Monetary', 'AvgMonetary', 
        'StdMonetary', 'FirstTransactionDate', 'LastTransactionDate',
        'NumUniqueProducts', 'NumUniqueProviders'
    ]
    
    # Fill NaN in StdMonetary (for customers with only one transaction) with 0
    customer_df['StdMonetary'] = customer_df['StdMonetary'].fillna(0)

    # 4. Engineer RFM and other features
    logging.info("Engineering RFM and tenure features...")
    
    # Calculate Recency
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    customer_df['Recency'] = (snapshot_date - customer_df['LastTransactionDate']).dt.days

    # Calculate Customer Tenure
    customer_df['Tenure'] = (customer_df['LastTransactionDate'] - customer_df['FirstTransactionDate']).dt.days
    
    # --- Task 4: Proxy Target Variable Engineering ---

    # 5. Cluster customers based on RFM
    logging.info("Clustering customers using K-Means on RFM features...")
    
    rfm_features = customer_df[['Recency', 'Frequency', 'Monetary']]
    
    # Scale RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)

    # Apply K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customer_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # 6. Define and Assign "High-Risk" Label
    # Interpret clusters to find the high-risk one (high Recency, low Frequency/Monetary)
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=['Recency', 'Frequency', 'Monetary'])
    logging.info("Interpreting cluster centroids:")
    logging.info(f"\n{cluster_centers}")
    
    # The high-risk cluster is the one with the highest Recency
    high_risk_cluster_id = cluster_centers['Recency'].idxmax()
    logging.info(f"Identified high-risk cluster ID: {high_risk_cluster_id}")
    
    customer_df['is_high_risk'] = (customer_df['Cluster'] == high_risk_cluster_id).astype(int)

    # 7. Finalize and Save
    logging.info("Finalizing the dataset...")
    
    # Select final features for the model
    features_to_keep = [
        'CustomerId',
        'Recency', 'Frequency', 'Monetary', 'AvgMonetary', 'StdMonetary',
        'Tenure', 'NumUniqueProducts', 'NumUniqueProviders',
        'is_high_risk' # This is our target variable
    ]
    final_df = customer_df[features_to_keep]

    # Ensure the output directory exists
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the processed data
    final_df.to_csv(processed_data_path, index=False)
    logging.info(f"Processed data saved to {processed_data_path}")
    logging.info(f"Final dataset shape: {final_df.shape}")
    logging.info("Feature engineering process complete.")

if __name__ == '__main__':
    # Define file paths
    RAW_DATA_PATH = BASE_DIR / 'data' / 'raw' / 'training.csv'
    PROCESSED_DATA_PATH = BASE_DIR / 'data' / 'processed' / 'customer_features.csv'
    
    # Run the feature creation process
    create_feature_set(RAW_DATA_PATH, PROCESSED_DATA_PATH)