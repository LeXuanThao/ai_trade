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

# --- New Risk Management Parameters ---
TRAILING_STOP_PERCENT = 0.05  # 5% trailing stop
TAKE_PROFIT_PERCENT = 0.10    # 10% take profit

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

def interpret_clusters(kmeans):
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['price_vs_ma_slow', 'ma_fast_slope'])
    strong_uptrend_cluster = cluster_centers.sort_values(by=['price_vs_ma_slow', 'ma_fast_slope'], ascending=False).index[0]
    roles = {'strong_uptrend': strong_uptrend_cluster}
    print(f"\n--- Interpreting Clusters ---\nCluster for 'Strong Uptrend': {strong_uptrend_cluster}")
    return roles

def run_backtest_and_plot(df, roles):
    print("\n--- Running Backtest (Long-Only Strategy with Risk Management) ---")
    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trailing_stop_price = 0
    trades = []
    equity = [INITIAL_CAPITAL]

    for i, row in df.iterrows():
        # --- EXIT LOGIC ---
        if position == 1:
            # Trailing Stop-Loss
            current_stop_loss = row['close'] * (1 - TRAILING_STOP_PERCENT)
            trailing_stop_price = max(trailing_stop_price, current_stop_loss)

            # Take Profit
            take_profit_price = entry_price * (1 + TAKE_PROFIT_PERCENT)

            # Check for exit conditions
            if row['cluster'] != roles['strong_uptrend'] or row['close'] <= trailing_stop_price or row['close'] >= take_profit_price:
                pnl = (row['close'] - entry_price) / entry_price
                capital *= (1 + pnl)
                trades.append({'time': i, 'type': 'close_long', 'price': row['close'], 'pnl': pnl})
                position, entry_price, trailing_stop_price = 0, 0, 0

        # --- ENTRY LOGIC ---
        if position == 0 and row['cluster'] == roles['strong_uptrend']:
            position = 1
            entry_price = row['close']
            trailing_stop_price = entry_price * (1 - TRAILING_STOP_PERCENT) # Initial trailing stop
            trades.append({'time': i, 'type': 'open_long', 'price': entry_price})
        
        equity.append(capital)

    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(16, 9))
    ax1.plot(df.index, df['close'], label='BTC Price', color='cyan', alpha=0.8, linewidth=1)
    buy_signals = [t['time'] for t in trades if t['type'] == 'open_long']
    sell_signals = [t['time'] for t in trades if t['type'] == 'close_long']
    ax1.plot(buy_signals, df.loc[buy_signals]['close'], '^g', markersize=10, label='Buy Signal')
    ax1.plot(sell_signals, df.loc[sell_signals]['close'], 'vr', markersize=10, label='Sell Signal')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(df.index, equity[:-1], label='Equity Curve', color='yellow')
    ax2.legend(loc='upper right')
    plt.title('K-Means Trend Following Backtest (Long-Only with Risk Management)')
    plt.show()

    return equity, trades

def main():
    print("Fetching data...")
    klines = fetch_data(SYMBOL, INTERVAL, START_DATE, END_DATE)
    
    print("Creating trend features and clustering...")
    df, kmeans = create_trend_features_and_cluster(klines)
    
    roles = interpret_clusters(kmeans)

    equity, trades = run_backtest_and_plot(df.copy(), roles)
    
    final_capital = equity[-1]
    total_return = (final_capital / INITIAL_CAPITAL - 1) * 100
    
    print("\n--- Backtest Results ---")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital:   ${final_capital:,.2f}")
    print(f"Total Return:    {total_return:.2f}%")
    print(f"Total Trades:    {len([t for t in trades if 'open' in t['type']])}")

if __name__ == "__main__":
    main()