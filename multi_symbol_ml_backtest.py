import pandas as pd
import plotly.graph_objects as go
from binance.client import Client
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import pickle

# --- Configuration ---
API_KEY = None
API_SECRET = None

# --- WATCHLIST: Define the symbols the bot will trade ---
WATCHLIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

INTERVAL = Client.KLINE_INTERVAL_1HOUR
START_DATE = "1 Jan, 2023"
END_DATE = "1 Jan, 2024"

# --- Trading Parameters ---
INITIAL_CAPITAL = 10000.0
TRAILING_STOP_PERCENT = 0.05  # 5% trailing stop
TAKE_PROFIT_PERCENT = 0.10    # 10% take profit
FIXED_STOP_LOSS_PERCENT = 0.03 # 3% fixed stop loss
MAX_INVESTMENT_PERCENT = 0.10 # Max 10% of total capital per trade

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

def fetch_all_data(symbols, interval, start_date, end_date):
    all_klines = {}
    client = Client(API_KEY, API_SECRET)
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        klines = client.get_historical_klines(symbol, interval, start_date, end_date)
        all_klines[symbol] = klines
    return all_klines

def create_features_and_target(klines_data):
    df = pd.DataFrame(klines_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
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

def train_models(processed_dfs):
    models = {}
    scalers = {}
    for symbol, df in processed_dfs.items():
        print(f"Training ML model for {symbol}...")
        features = df[['price_vs_ma_slow', 'ma_fast_slope', 'rsi', 'macd_diff']]
        target = df['target']

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42, stratify=target)

        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(f"--- Model Evaluation for {symbol} ---")
        print(classification_report(y_test, y_pred))
        
        models[symbol] = model
        scalers[symbol] = scaler
    return models, scalers

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

def run_multi_symbol_backtest(processed_dfs, models, scalers):
    print("\n--- Running Multi-Symbol Backtest ---")
    capital = INITIAL_CAPITAL
    equity_history = [INITIAL_CAPITAL]

    # Initialize positions for all symbols
    positions = {symbol: {
        'position': 0, # 0: none, 1: long
        'entry_price': 0,
        'trailing_stop_price': 0,
        'fixed_stop_loss_price': 0,
        'current_position_size': 0
    } for symbol in WATCHLIST}

    # Get a common index (timestamps) for iteration
    common_index = None
    for symbol in WATCHLIST:
        if common_index is None:
            common_index = processed_dfs[symbol].index
        else:
            common_index = common_index.intersection(processed_dfs[symbol].index)
    common_index = common_index.sort_values()

    trades_log = []

    for timestamp in common_index:
        # Calculate current portfolio value (for position sizing)
        current_portfolio_value = capital
        for symbol in WATCHLIST:
            pos_data = positions[symbol]
            if pos_data['position'] == 1:
                # Add current value of open position to portfolio value
                current_price = processed_dfs[symbol].loc[timestamp]['close']
                current_portfolio_value += (current_price - pos_data['entry_price']) * pos_data['current_position_size']

        # Process each symbol at this timestamp
        for symbol in WATCHLIST:
            if timestamp not in processed_dfs[symbol].index:
                continue # Skip if data not available for this symbol at this timestamp

            row = processed_dfs[symbol].loc[timestamp]
            model = models[symbol]
            scaler = scalers[symbol]
            pos_data = positions[symbol]

            # Prepare features for prediction for this specific row
            features_for_prediction = row[['price_vs_ma_slow', 'ma_fast_slope', 'rsi', 'macd_diff']].to_frame().T
            scaled_features_for_prediction = scaler.transform(features_for_prediction)
            prediction = model.predict(scaled_features_for_prediction)[0]

            # --- EXIT LOGIC ---
            if pos_data['position'] == 1:
                # Trailing Stop-Loss
                current_trailing_stop = row['close'] * (1 - TRAILING_STOP_PERCENT)
                pos_data['trailing_stop_price'] = max(pos_data['trailing_stop_price'], current_trailing_stop)

                # Take Profit
                take_profit_price = pos_data['entry_price'] * (1 + TAKE_PROFIT_PERCENT)

                # Check for exit conditions
                if row['close'] <= pos_data['trailing_stop_price'] or \
                   row['close'] <= pos_data['fixed_stop_loss_price'] or \
                   row['close'] >= take_profit_price:
                    pnl = (row['close'] - pos_data['entry_price']) * pos_data['current_position_size']
                    capital += pnl
                    trades_log.append({'time': timestamp, 'symbol': symbol, 'type': 'close_long', 'price': row['close'], 'pnl': pnl})
                    pos_data['position'], pos_data['entry_price'], pos_data['trailing_stop_price'], pos_data['fixed_stop_loss_price'], pos_data['current_position_size'] = 0, 0, 0, 0, 0

            # --- ENTRY LOGIC (Based on Market Regime) ---
            if pos_data['position'] == 0:
                # Calculate max investment amount for this trade
                max_investment_amount = current_portfolio_value * MAX_INVESTMENT_PERCENT
                quantity_to_buy = max_investment_amount / row['close']

                if row['adx'] > ADX_THRESHOLD: # Trending Market
                    # ML-driven Trend Following
                    if prediction == 1 and row['ma_fast'] > row['ma_slow']:
                        pos_data['position'] = 1
                        pos_data['entry_price'] = row['close']
                        pos_data['trailing_stop_price'] = pos_data['entry_price'] * (1 - TRAILING_STOP_PERCENT)
                        pos_data['fixed_stop_loss_price'] = pos_data['entry_price'] * (1 - FIXED_STOP_LOSS_PERCENT)
                        pos_data['current_position_size'] = quantity_to_buy # Set position size
                        trades_log.append({'time': timestamp, 'symbol': symbol, 'type': 'open_long_trend', 'price': pos_data['entry_price']})
                else: # Ranging Market
                    # Mean Reversion with Bollinger Bands
                    if row['close'] < row['bb_lower']:
                        pos_data['position'] = 1
                        pos_data['entry_price'] = row['close']
                        pos_data['trailing_stop_price'] = pos_data['entry_price'] * (1 - TRAILING_STOP_PERCENT)
                        pos_data['fixed_stop_loss_price'] = pos_data['entry_price'] * (1 - FIXED_STOP_LOSS_PERCENT)
                        pos_data['current_position_size'] = quantity_to_buy # Set position size
                        trades_log.append({'time': timestamp, 'symbol': symbol, 'type': 'open_long_range', 'price': pos_data['entry_price']})
        
        equity_history.append(capital) # Append capital after processing all symbols for this timestamp

    return equity_history, trades_log

def plot_equity_curve(equity_history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity_history, mode='lines', name='Portfolio Equity'))
    fig.update_layout(title='Multi-Symbol Trading Strategy Equity Curve', xaxis_title='Time (Candles)', yaxis_title='Capital (USD)')
    fig.show()

def main():
    all_klines = fetch_all_data(WATCHLIST, INTERVAL, START_DATE, END_DATE)
    
    processed_dfs = {}
    for symbol, klines_data in all_klines.items():
        print(f"Creating features and target variable for {symbol}...")
        processed_dfs[symbol] = create_features_and_target(klines_data)
    
    models, scalers = train_models(processed_dfs)

    equity_history, trades_log = run_multi_symbol_backtest(processed_dfs, models, scalers)
    
    final_capital = equity_history[-1]
    total_return = (final_capital / INITIAL_CAPITAL - 1) * 100
    max_drawdown = calculate_drawdown(equity_history)
    
    print("\n--- Multi-Symbol Backtest Results ---")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital:   ${final_capital:,.2f}")
    print(f"Total Return:    {total_return:.2f}%")
    print(f"Total Trades:    {len([t for t in trades_log if 'open' in t['type']])}")
    print(f"Max Drawdown:    {max_drawdown:.2%}")

    plot_equity_curve(equity_history)

    # Save results for dashboard
    with open('equity_history.pkl', 'wb') as f:
        pickle.dump(equity_history, f)
    with open('trades_log.pkl', 'wb') as f:
        pickle.dump(trades_log, f)
    print("Backtest results saved to equity_history.pkl and trades_log.pkl")

if __name__ == "__main__":
    main()