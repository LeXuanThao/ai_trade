import pandas as pd
from binance.client import Client
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
API_KEY = None
API_SECRET = None
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1HOUR
START_DATE = "1 Jan, 2023"
END_DATE = "1 Jan, 2024"
N_CLUSTERS = 5
INITIAL_CAPITAL = 10000.0

# --- Main Functions ---

def fetch_data(symbol, interval, start_date, end_date):
    client = Client(API_KEY, API_SECRET)
    klines = client.get_historical_klines(symbol, interval, start_date, end_date)
    return klines

def preprocess_and_cluster(klines):
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    features = df[['price_change', 'volume_change']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    return df, kmeans

def identify_cluster_roles(df, kmeans):
    """Identifies the meaning of each cluster based on its characteristics."""
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['price_change', 'volume_change'])
    
    # Find cluster with the most members (likely 'Normal')
    normal_cluster = df['cluster'].mode()[0]
    
    # Find cluster with highest price change (likely 'Strong Uptrend')
    strong_uptrend_cluster = cluster_centers['price_change'].idxmax()
    
    # Find cluster with lowest price change (likely 'Sell-off')
    selloff_cluster = cluster_centers['price_change'].idxmin()
    
    # Find cluster with mild negative price change
    cluster_centers['temp_dist'] = abs(cluster_centers['price_change'] - (-0.5)) # Arbitrary target for mild negative
    mild_downtrend_cluster = cluster_centers['temp_dist'].idxmin()

    roles = {
        'normal': normal_cluster,
        'strong_uptrend': strong_uptrend_cluster,
        'selloff': selloff_cluster,
        'mild_downtrend': mild_downtrend_cluster
    }
    print("--- Cluster Roles Identified ---")
    print(roles)
    return roles

def run_backtest(df, roles):
    print("\n--- Running Backtest ---")
    capital = INITIAL_CAPITAL
    position = 0  # 0: none, 1: long, -1: short
    entry_price = 0
    trades = []
    equity = [INITIAL_CAPITAL]

    df['prev_cluster'] = df['cluster'].shift(1)
    df.dropna(inplace=True)

    for i, row in df.iterrows():
        # --- EXIT LOGIC ---
        if position == 1 and row['cluster'] != roles['strong_uptrend']:
            pnl = (row['close'] - entry_price) / entry_price
            capital *= (1 + pnl)
            trades.append({'type': 'close_long', 'price': row['close'], 'pnl': pnl, 'capital': capital})
            position, entry_price = 0, 0
        elif position == -1 and row['cluster'] != roles['selloff']:
            pnl = (entry_price - row['close']) / entry_price
            capital *= (1 + pnl)
            trades.append({'type': 'close_short', 'price': row['close'], 'pnl': pnl, 'capital': capital})
            position, entry_price = 0, 0

        # --- ENTRY LOGIC ---
        if position == 0:
            # Long signal
            if row['prev_cluster'] == roles['normal'] and row['cluster'] == roles['strong_uptrend']:
                position = 1
                entry_price = row['close']
                trades.append({'type': 'open_long', 'price': entry_price})
            # Short signal
            elif (row['prev_cluster'] == roles['normal'] or row['prev_cluster'] == roles['mild_downtrend']) and row['cluster'] == roles['selloff']:
                position = -1
                entry_price = row['close']
                trades.append({'type': 'open_short', 'price': entry_price})
        
        equity.append(capital)

    return trades, equity

def plot_equity_curve(equity):
    plt.figure(figsize=(12, 8))
    plt.plot(equity)
    plt.title('Equity Curve')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Capital (USD)')
    plt.grid(True)
    plt.show()

def main():
    print("Fetching data...")
    klines = fetch_data(SYMBOL, INTERVAL, START_DATE, END_DATE)
    
    print("Preprocessing data and clustering...")
    df, kmeans = preprocess_and_cluster(klines)
    
    roles = identify_cluster_roles(df, kmeans)

    trades, equity = run_backtest(df.copy(), roles)
    
    final_capital = equity[-1]
    total_return = (final_capital / INITIAL_CAPITAL - 1) * 100
    
    print("\n--- Backtest Results ---")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital:   ${final_capital:,.2f}")
    print(f"Total Return:    {total_return:.2f}%")
    print(f"Total Trades:    {len([t for t in trades if 'open' in t['type']])}")

    plot_equity_curve(equity[:-1]) # Exclude last equity point which is a duplicate

if __name__ == "__main__":
    main()
