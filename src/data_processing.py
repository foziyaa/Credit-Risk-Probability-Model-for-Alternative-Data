# src/data_processing.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from . import config

# Configure logging to provide feedback during execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path):
    """
    Loads data from a specified CSV file path.

    Args:
        path (Path): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    
    Raises:
        FileNotFoundError: If the data file does not exist at the given path.
    """
    logging.info(f"Loading data from {path}...")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        logging.error(f"Data file not found at {path}. Aborting.")
        raise

def preprocess_data(df):
    """
    Performs initial data cleaning and transformations on the raw DataFrame.

    Args:
        df (pd.DataFrame): The raw transaction DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    logging.info("Preprocessing data...")
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    # 'Value' is the absolute transaction amount, more suitable for RFM analysis.
    df = df.rename(columns={'Value': 'TransactionValue'})
    return df

def aggregate_customer_data(df):
    """
    Aggregates transaction data to the customer level to create a customer-centric view.

    Args:
        df (pd.DataFrame): The preprocessed transaction DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with one row per customer and aggregated features.
    """
    logging.info("Aggregating data to customer level...")
    aggregation_rules = {
        'TransactionId': 'count',
        'TransactionValue': ['sum', 'mean', 'std'],
        'TransactionStartTime': ['min', 'max'],
        'ProductId': pd.Series.nunique,
        'ProviderId': pd.Series.nunique
    }
    customer_df = df.groupby('CustomerId').agg(aggregation_rules).reset_index()
    
    # Flatten multi-level column names for easier access
    customer_df.columns = [
        'CustomerId', 'Frequency', 'Monetary', 'AvgMonetary',
        'StdMonetary', 'FirstTransactionDate', 'LastTransactionDate',
        'NumUniqueProducts', 'NumUniqueProviders'
    ]
    # Handle cases where customers have only one transaction, resulting in NaN for std dev
    customer_df['StdMonetary'] = customer_df['StdMonetary'].fillna(0)
    return customer_df

def engineer_features(customer_df, snapshot_date):
    """
    Engineers RFM (Recency) and Tenure features for each customer.

    Args:
        customer_df (pd.DataFrame): The customer-aggregated DataFrame.
        snapshot_date (pd.Timestamp): The reference date for calculating recency.

    Returns:
        pd.DataFrame: The DataFrame with new 'Recency' and 'Tenure' columns.
    """
    logging.info("Engineering RFM and tenure features...")
    # Calculate Recency: days since the last transaction from a fixed snapshot date.
    customer_df['Recency'] = (snapshot_date - customer_df['LastTransactionDate']).dt.days
    # Calculate Tenure: the customer's lifetime in days.
    customer_df['Tenure'] = (customer_df['LastTransactionDate'] - customer_df['FirstTransactionDate']).dt.days
    return customer_df

def find_high_risk_cluster(centroids_df):
    """
    Identifies the cluster ID corresponding to the highest-risk segment.
    The primary hypothesis is that high-risk customers are the least engaged,
    characterized by the highest Recency (longest time since last purchase).

    Args:
        centroids_df (pd.DataFrame): DataFrame containing cluster centroids.

    Returns:
        int: The cluster ID identified as high-risk.
    """
    return centroids_df['Recency'].idxmax()

def create_proxy_target(customer_df):
    """
    Clusters customers based on RFM features to create a binary high-risk target variable.

    Args:
        customer_df (pd.DataFrame): The customer DataFrame with engineered features.

    Returns:
        pd.DataFrame: The DataFrame with the new 'is_high_risk' target column.
    """
    logging.info("Clustering customers to create proxy target...")
    rfm_features = customer_df[['Recency', 'Frequency', 'Monetary']]
    
    # Scale features to ensure equal contribution to distance calculation
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)

    # Use K-Means to segment customers into 3 groups
    kmeans = KMeans(n_clusters=3, random_state=config.RANDOM_STATE, n_init=10)
    customer_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Interpret cluster centers by transforming them back to the original scale
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=['Recency', 'Frequency', 'Monetary'])
    logging.info(f"Interpreted cluster centroids:\n{cluster_centers}")
    
    # Identify the high-risk cluster based on the highest recency
    high_risk_cluster_id = find_high_risk_cluster(cluster_centers)
    logging.info(f"Identified high-risk cluster ID: {high_risk_cluster_id}")
    
    # Create the binary target variable
    customer_df[config.TARGET_VARIABLE] = (customer_df['Cluster'] == high_risk_cluster_id).astype(int)
    return customer_df

def finalize_and_save_data(customer_df, path):
    """
    Selects the final set of features for modeling and saves the dataset to a CSV file.

    Args:
        customer_df (pd.DataFrame): The fully processed customer DataFrame.
        path (Path): The file path to save the final dataset.
    """
    logging.info("Finalizing and saving the dataset...")
    # Select features defined in the config file, plus CustomerId and the target
    features_to_select = ['CustomerId'] + config.FEATURES_TO_KEEP + [config.TARGET_VARIABLE]
    final_df = customer_df[features_to_select]
    
    # Ensure the output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    final_df.to_csv(path, index=False)
    logging.info(f"Processed data successfully saved to {path}. Final shape: {final_df.shape}")

def main():
    """Main pipeline to execute the full data processing workflow."""
    logging.info("Starting feature engineering and proxy target creation process...")
    
    df = load_data(config.RAW_DATA_PATH)
    preprocessed_df = preprocess_data(df)
    
    # Use a consistent snapshot date for reproducible Recency calculation
    snapshot_date = preprocessed_df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    customer_df = aggregate_customer_data(preprocessed_df)
    customer_df_with_features = engineer_features(customer_df, snapshot_date)
    customer_df_with_target = create_proxy_target(customer_df_with_features)
    
    finalize_and_save_data(customer_df_with_target, config.PROCESSED_DATA_PATH)
    
    logging.info("Data processing complete.")

if __name__ == '__main__':
    main()