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
MA_FAST = 50
MA_SLOW = 200
INITIAL_CAPITAL = 10000.0

# --- Main Functions ---

def fetch_data(symbol, interval, start_date, end_date):
    client = Client(API_KEY, API_SECRET)
    klines = client.get_historical_klines(symbol, interval, start_date, end_date)
    return klines

def create_trend_features_and_cluster(klines):
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['close'] = df['close'].astype(float)

    df['ma_fast'] = df['close'].rolling(window=MA_FAST).mean()
    df['ma_slow'] = df['close'].rolling(window=MA_SLOW).mean()
    df['price_vs_ma_slow'] = (df['close'] - df['ma_slow']) / df['ma_slow']
    df['ma_fast_slope'] = df['ma_fast'].pct_change()
    df.dropna(inplace=True)

    features = df[['price_vs_ma_slow', 'ma_fast_slope']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    return df, kmeans

def interpret_hybrid_clusters(kmeans):
    """Interprets clusters for the hybrid strategy."""
    print("\n--- Interpreting Clusters for Hybrid Strategy ---")
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['price_vs_ma_slow', 'ma_fast_slope'])
    
    # Strong Uptrend: High price vs slow MA, high fast MA slope
    strong_uptrend_cluster = cluster_centers.sort_values(by=['price_vs_ma_slow', 'ma_fast_slope'], ascending=False).index[0]
    
    # Strong Downtrend: Low price vs slow MA, low fast MA slope
    strong_downtrend_cluster = cluster_centers.sort_values(by=['price_vs_ma_slow', 'ma_fast_slope'], ascending=True).index[0]

    # Normal/Ranging: Closest to zero on both axes
    cluster_centers['distance_to_zero'] = np.sqrt(cluster_centers['price_vs_ma_slow']**2 + cluster_centers['ma_fast_slope']**2)
    normal_cluster = cluster_centers['distance_to_zero'].idxmin()

    roles = {
        'normal': normal_cluster,
        'strong_uptrend': strong_uptrend_cluster,
        'strong_downtrend': strong_downtrend_cluster
    }
    print(f"Cluster Roles: {roles}")
    return roles

def run_hybrid_backtest_and_plot(df, roles):
    print("\n--- Running Hybrid Backtest (State Transition Logic) ---")
    capital = INITIAL_CAPITAL
    position = 0
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
            trades.append({'time': i, 'type': 'close_long', 'price': row['close']})
            position, entry_price = 0, 0
        elif position == -1 and row['cluster'] != roles['strong_downtrend']:
            pnl = (entry_price - row['close']) / entry_price
            capital *= (1 + pnl)
            trades.append({'time': i, 'type': 'close_short', 'price': row['close']})
            position, entry_price = 0, 0

        # --- ENTRY LOGIC (STATE TRANSITION) ---
        if position == 0:
            if row['prev_cluster'] == roles['normal'] and row['cluster'] == roles['strong_uptrend']:
                position = 1
                entry_price = row['close']
                trades.append({'time': i, 'type': 'open_long', 'price': entry_price})
            elif row['prev_cluster'] == roles['normal'] and row['cluster'] == roles['strong_downtrend']:
                position = -1
                entry_price = row['close']
                trades.append({'time': i, 'type': 'open_short', 'price': entry_price})
        
        equity.append(capital)

    # --- Plotting ---
    # (Plotting code is identical to the previous script, so it's omitted here for brevity
    # but will be included in the actual file execution)
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(16, 9))
    ax1.plot(df.index, df['close'], label='BTC Price', color='cyan', alpha=0.8, linewidth=1)
    open_long_signals = [t['time'] for t in trades if t['type'] == 'open_long']
    close_long_signals = [t['time'] for t in trades if t['type'] == 'close_long']
    open_short_signals = [t['time'] for t in trades if t['type'] == 'open_short']
    close_short_signals = [t['time'] for t in trades if t['type'] == 'close_short']
    ax1.plot(open_long_signals, df.loc[open_long_signals]['close'], '^', color='lime', markersize=10, label='Open Long')
    ax1.plot(close_long_signals, df.loc[close_long_signals]['close'], 'x', color='lime', markersize=10, label='Close Long')
    ax1.plot(open_short_signals, df.loc[open_short_signals]['close'], 'v', color='red', markersize=10, label='Open Short')
    ax1.plot(close_short_signals, df.loc[close_short_signals]['close'], 'x', color='red', markersize=10, label='Close Short')
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.plot(df.index, equity[:-1], label='Equity Curve', color='yellow')
    ax2.legend()
    plt.title('Hybrid K-Means Backtest (MA Features + State Transition)')
    plt.show()

    return equity, trades

def main():
    print("Fetching data...")
    klines = fetch_data(SYMBOL, INTERVAL, START_DATE, END_DATE)
    
    print("Creating trend features and clustering...")
    df, kmeans = create_trend_features_and_cluster(klines)
    
    roles = interpret_hybrid_clusters(kmeans)

    equity, trades = run_hybrid_backtest_and_plot(df.copy(), roles)
    
    final_capital = equity[-1]
    total_return = (final_capital / INITIAL_CAPITAL - 1) * 100
    
    print("\n--- Hybrid Backtest Results ---")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital:   ${final_capital:,.2f}")
    print(f"Total Return:    {total_return:.2f}%")
    print(f"Total Trades:    {len([t for t in trades if 'open' in t['type']])}")

if __name__ == "__main__":
    main()
