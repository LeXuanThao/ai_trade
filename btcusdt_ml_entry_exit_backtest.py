

import pandas as pd
from binance.client import Client
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
API_KEY = None
API_SECRET = None
SYMBOL = "XRPUSDT"
INTERVAL = Client.KLINE_INTERVAL_1HOUR
START_DATE = "1 Jan, 2023"
END_DATE = "1 Jan, 2024"

# --- Trading Parameters ---
INITIAL_CAPITAL = 10000.0
TRAILING_STOP_PERCENT = 0.05  # 5% trailing stop
TAKE_PROFIT_PERCENT = 0.10    # 10% take profit
FIXED_STOP_LOSS_PERCENT = 0.03 # 3% fixed stop loss
MAX_INVESTMENT_PERCENT = 0.10 # New: Max 10% of capital per trade

# --- ML Model Parameters ---
PREDICT_FUTURE_CANDLES = 5 # Predict price movement over the next 5 candles
PRICE_CHANGE_THRESHOLD = 0.005 # 0.5% price increase to be considered 'up'

# --- Feature Engineering Parameters ---
MA_FAST = 50
MA_SLOW = 200
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# --- Market Regime Parameters ---
ADX_PERIOD = 14
ADX_THRESHOLD = 25 # Above this, market is trending
BOLLINGER_PERIOD = 20
BOLLINGER_STD_DEV = 2

# --- Main Functions ---

def fetch_data(symbol, interval, start_date, end_date):
    client = Client(API_KEY, API_SECRET)
    klines = client.get_historical_klines(symbol, interval, start_date, end_date)
    return klines

def create_features_and_target(df):
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)

    # MA Features
    df['ma_fast'] = df['close'].rolling(window=MA_FAST).mean()
    df['ma_slow'] = df['close'].rolling(window=MA_SLOW).mean()
    df['price_vs_ma_slow'] = (df['close'] - df['ma_slow']) / df['ma_slow']
    df['ma_fast_slope'] = df['ma_fast'].pct_change()

    # RSI Feature
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD Feature
    exp1 = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['macd_diff'] = macd - signal

    # ADX Feature (for trend strength)
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close']), abs(x['low'] - x['close'])), axis=1)
    
    atr = tr.rolling(window=ADX_PERIOD).mean()
    plus_di = 100 * (plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()

    # Bollinger Bands (for ranging market)
    df['bb_middle'] = df['close'].rolling(window=BOLLINGER_PERIOD).mean()
    df['bb_std'] = df['close'].rolling(window=BOLLINGER_PERIOD).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * BOLLINGER_STD_DEV)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * BOLLINGER_STD_DEV)

    # Target Variable: 1 if price increases by THRESHOLD in next N candles, else 0
    df['future_price'] = df['close'].shift(-PREDICT_FUTURE_CANDLES)
    df['price_change_future'] = (df['future_price'] - df['close']) / df['close']
    df['target'] = (df['price_change_future'] > PRICE_CHANGE_THRESHOLD).astype(int)

    df.dropna(inplace=True)
    return df

def train_model(df):
    features = df[['price_vs_ma_slow', 'ma_fast_slope', 'rsi', 'macd_diff']]
    target = df['target']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42, stratify=target)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    return model, scaler

def calculate_drawdown(equity_curve):
    peak = equity_curve[0]
    max_drawdown = 0
    for capital in equity_curve:
        if capital > peak:
            peak = capital
        drawdown = (peak - capital) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown

def run_multi_regime_backtest_and_plot(df, model, scaler):
    print("\n--- Running Multi-Regime Backtest ---")
    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trailing_stop_price = 0
    fixed_stop_loss_price = 0
    current_position_size = 0 # New: Size of the current position in units of asset
    trades = []
    equity = [INITIAL_CAPITAL]

    # Prepare features for prediction
    features_for_prediction = df[['price_vs_ma_slow', 'ma_fast_slope', 'rsi', 'macd_diff']].copy()
    features_for_prediction.dropna(inplace=True)
    scaled_features_for_prediction = scaler.transform(features_for_prediction)
    df_predict = df.loc[features_for_prediction.index].copy()
    df_predict['prediction'] = model.predict(scaled_features_for_prediction)

    for i, row in df_predict.iterrows():
        # --- EXIT LOGIC (Applies to both regimes) ---
        if position == 1:
            # Trailing Stop-Loss
            current_trailing_stop = row['close'] * (1 - TRAILING_STOP_PERCENT)
            trailing_stop_price = max(trailing_stop_price, current_trailing_stop)

            # Take Profit
            take_profit_price = entry_price * (1 + TAKE_PROFIT_PERCENT)

            # Check for exit conditions
            if row['close'] <= trailing_stop_price or \
               row['close'] <= fixed_stop_loss_price or \
               row['close'] >= take_profit_price:
                # Calculate PnL based on position size
                pnl = (row['close'] - entry_price) * current_position_size
                capital += pnl
                trades.append({'time': i, 'type': 'close_long', 'price': row['close'], 'pnl': pnl})
                position, entry_price, trailing_stop_price, fixed_stop_loss_price, current_position_size = 0, 0, 0, 0, 0

        # --- ENTRY LOGIC (Based on Market Regime) ---
        if position == 0:
            # Calculate max investment amount for this trade
            max_investment_amount = capital * MAX_INVESTMENT_PERCENT
            # Calculate quantity to buy
            quantity_to_buy = max_investment_amount / row['close']

            if row['adx'] > ADX_THRESHOLD: # Trending Market
                # ML-driven Trend Following
                if row['prediction'] == 1 and row['ma_fast'] > row['ma_slow']:
                    position = 1
                    entry_price = row['close']
                    trailing_stop_price = entry_price * (1 - TRAILING_STOP_PERCENT)
                    fixed_stop_loss_price = entry_price * (1 - FIXED_STOP_LOSS_PERCENT)
                    current_position_size = quantity_to_buy # Set position size
                    trades.append({'time': i, 'type': 'open_long_trend', 'price': entry_price})
            else: # Ranging Market
                # Mean Reversion with Bollinger Bands
                if row['close'] < row['bb_lower']:
                    position = 1
                    entry_price = row['close']
                    trailing_stop_price = entry_price * (1 - TRAILING_STOP_PERCENT)
                    fixed_stop_loss_price = entry_price * (1 - FIXED_STOP_LOSS_PERCENT)
                    current_position_size = quantity_to_buy # Set position size
                    trades.append({'time': i, 'type': 'open_long_range', 'price': entry_price})
        
        equity.append(capital)

    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(16, 9))
    ax1.plot(df.index, df['close'], label='Price', color='cyan', alpha=0.8, linewidth=1)
    ax1.plot(df.index, df['bb_upper'], label='BB Upper', color='orange', linestyle='--', alpha=0.7)
    ax1.plot(df.index, df['bb_middle'], label='BB Middle', color='gray', linestyle='--', alpha=0.7)
    ax1.plot(df.index, df['bb_lower'], label='BB Lower', color='orange', linestyle='--', alpha=0.7)

    open_long_trend_signals = [t['time'] for t in trades if t['type'] == 'open_long_trend']
    open_long_range_signals = [t['time'] for t in trades if t['type'] == 'open_long_range']
    close_signals = [t['time'] for t in trades if 'close_long' in t['type']]
    
    ax1.plot(open_long_trend_signals, df.loc[open_long_trend_signals]['close'], '^g', markersize=10, label='Buy (Trend)')
    ax1.plot(open_long_range_signals, df.loc[open_long_range_signals]['close'], '^b', markersize=10, label='Buy (Range)')
    ax1.plot(close_signals, df.loc[close_signals]['close'], 'vr', markersize=10, label='Sell')

    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(df.index, equity[:-1], label='Equity Curve', color='yellow')
    ax2.legend(loc='upper right')
    plt.title('Multi-Regime Trading Strategy Backtest')
    plt.show()

    return equity, trades

def main():
    print("Fetching data...")
    klines = fetch_data(SYMBOL, INTERVAL, START_DATE, END_DATE)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    print("Creating features and target variable...")
    df_processed = create_features_and_target(df.copy())
    
    print("Training ML model...")
    model, scaler = train_model(df_processed)

    equity, trades = run_multi_regime_backtest_and_plot(df_processed.copy(), model, scaler)
    
    final_capital = equity[-1]
    total_return = (final_capital / INITIAL_CAPITAL - 1) * 100
    max_drawdown = calculate_drawdown(equity)
    
    print("\n--- Multi-Regime Backtest Results ---")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital:   ${final_capital:,.2f}")
    print(f"Total Return:    {total_return:.2f}%")
    print(f"Total Trades:    {len([t for t in trades if 'open' in t['type']])}")
    print(f"Max Drawdown:    {max_drawdown:.2%}")

if __name__ == "__main__":
    main()
