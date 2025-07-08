import pandas as pd
from binance.client import Client
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
# Replace with your actual API key and secret
# You can leave these as None for public data access
API_KEY = None
API_SECRET = None
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1HOUR
START_DATE = "1 Jan, 2023"
END_DATE = "1 Jan, 2024"
N_CLUSTERS = 5 # Number of clusters for K-Means

# --- Main Script ---

def fetch_data(api_key, api_secret, symbol, interval, start_date, end_date):
    """Fetches historical data from Binance."""
    client = Client(api_key, api_secret)
    klines = client.get_historical_klines(symbol, interval, start_date, end_date)
    return klines

def preprocess_data(klines):
    """Preprocesses the raw data from Binance."""
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Feature Engineering (you can add more features)
    df['price_change'] = df['close'].astype(float).pct_change()
    df['volume_change'] = df['volume'].astype(float).pct_change()
    
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    return df

def run_kmeans(df, n_clusters):
    """Runs K-Means clustering on the data."""
    features = df[['price_change', 'volume_change']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    
    df['cluster'] = kmeans.labels_
    return df, kmeans, scaled_features

def plot_clusters(df, scaled_features, kmeans):
    """Plots the clusters."""
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
    centers = plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, alpha=0.9, marker='X')
    plt.title('BTCUSDT Price vs. Volume Change Clusters')
    plt.xlabel('Price Change (Scaled)')
    plt.ylabel('Volume Change (Scaled)')
    plt.legend([scatter.legend_elements()[0][0], centers], ['Data Points', 'Cluster Centers'])
    plt.grid(True)
    plt.show()

def main():
    """Main function to run the analysis."""
    print("Fetching data...")
    klines = fetch_data(API_KEY, API_SECRET, SYMBOL, INTERVAL, START_DATE, END_DATE)
    
    print("Preprocessing data...")
    df = preprocess_data(klines)
    
    print("Running K-Means clustering...")
    df_clustered, kmeans, scaled_features = run_kmeans(df.copy(), N_CLUSTERS)
    
    print("\n--- Clustering Results ---")
    print(df_clustered['cluster'].value_counts())
    
    print("\n--- Cluster Centers (Scaled) ---")
    print(pd.DataFrame(kmeans.cluster_centers_, columns=['price_change', 'volume_change']))

    print("\nPlotting clusters...")
    plot_clusters(df_clustered, scaled_features, kmeans)

if __name__ == "__main__":
    main()