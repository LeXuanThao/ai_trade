import threading
import time
import queue
import random
import logging
import pandas as pd
import numpy as np
import requests 
from binance.um_futures import UMFutures # New: For actual futures trading
from binance.client import Client # For general client (e.g., get_historical_klines)
from binance.exceptions import BinanceAPIException # For handling API errors
import os # For environment variables

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# --- Cấu hình ---
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
PREDICTION_INTERVAL_SECONDS = 60 # Tần suất dự đoán cho mỗi cặp (ví dụ: mỗi phút)

# --- Trading Parameters ---
INITIAL_CAPITAL = 10000.0 # This will be a reference, actual capital will be from Binance
TRAILING_STOP_PERCENT = 0.05  # 5% trailing stop
TAKE_PROFIT_PERCENT = 0.10    # 10% take profit
FIXED_STOP_LOSS_PERCENT = 0.03 # 3% fixed stop loss

# --- New Leverage & Margin Parameters ---
MAX_LEVERAGE_CAP = 20 # Max leverage bot will use, even if Binance allows more
FIXED_MARGIN_PER_TRADE_USD = 1.0 # Fixed $1 margin per trade as requested
INVESTMENT_LIMIT_PERCENT = 0.20 # Stop taking new orders if total invested capital reaches 20% of initial capital

# --- Market Regime Parameters (for simulation) ---
ADX_THRESHOLD = 25 
BOLLINGER_PERIOD = 20 # Not directly used in simulation, but for context
BOLLINGER_STD_DEV = 2 # Not directly used in simulation, but for context

# --- Discord Webhook (Replace with your actual webhook URL) ---
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# --- Binance API Keys (Load from environment variables for security) ---
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# --- Binance Futures Client (Global for main thread) ---
# Use Testnet for testing real orders!
# base_url="https://testnet.binancefuture.com" for testnet
binance_futures_client = UMFutures(key=API_KEY, secret=API_SECRET, base_url="https://fapi.binance.com")

# --- Global variable to store symbol info (minQty, stepSize, max_leverage, current_set_leverage) ---
symbol_info = {}

# --- Hàng đợi để các luồng con gửi kết quả về luồng chính ---
prediction_queue = queue.Queue()

# --- Hàm gửi tin nhắn Discord ---
def send_discord_message(message):
    if not DISCORD_WEBHOOK_URL or DISCORD_WEBHOOK_URL == "YOUR_DISCORD_WEBHOOK_URL_HERE":
        logging.warning("Discord Webhook URL not configured. Skipping Discord message.")
        return
    try:
        payload = {"content": message}
        requests.post(DISCORD_WEBHOOK_URL, json=payload)
        logging.info("Discord message sent.")
    except Exception as e:
        logging.error(f"Failed to send Discord message: {e}")

# --- Hàm giả lập lấy dữ liệu và tính toán chỉ báo ---
def fetch_latest_data_and_indicators(symbol):
    logging.info(f"Fetching latest data and indicators for {symbol}...")
    time.sleep(random.uniform(0.1, 0.5)) # Giả lập độ trễ API
    
    # Giả lập các giá trị chỉ báo và giá đóng cửa hiện tại
    current_price = random.uniform(1000, 50000)
    ma_fast = current_price * random.uniform(0.99, 1.01)
    ma_slow = current_price * random.uniform(0.98, 1.02)
    adx = random.uniform(10, 40) # Giả lập ADX
    bb_lower = current_price * random.uniform(0.95, 0.99) # Giả lập BB Lower

    return {
        'symbol': symbol,
        'timestamp': pd.Timestamp.now(),
        'close': current_price,
        'ma_fast': ma_fast,
        'ma_slow': ma_slow,
        'adx': adx,
        'bb_lower': bb_lower
    }

# --- Hàm giả lập dự đoán (CPU-bound nhẹ) ---
def predict_signal(data_for_prediction, model, scaler):
    logging.info(f"Making prediction for {data_for_prediction['symbol']}...")
    # Trong thực tế: model.predict(scaler.transform(data_for_prediction_features))
    time.sleep(random.uniform(0.05, 0.2)) # Giả lập thời gian dự đoán
    return random.choice([0, 1]) # Giả lập tín hiệu (0: không mua, 1: mua)

# --- ThreadPoolManager để quản lý các luồng dự đoán ---
class ThreadPoolManager:
    def __init__(self, symbols, prediction_interval, models, scalers, prediction_queue):
        self.symbols = symbols
        self.prediction_interval = prediction_interval
        self.models = models # Giả lập models
        self.scalers = scalers # Giả lập scalers
        self.prediction_queue = prediction_queue
        self.stop_event = threading.Event()
        self.threads = []

    def _prediction_worker(self, symbol, model, scaler):
        thread_name = f"Worker-{symbol}"
        threading.current_thread().name = thread_name
        logging.info(f"Starting prediction worker for {symbol}")

        while not self.stop_event.is_set():
            try:
                # 1. Lấy dữ liệu và chỉ báo
                data = fetch_latest_data_and_indicators(symbol)
                
                # 2. Dự đoán
                # Trong thực tế, bạn sẽ cần trích xuất các features từ 'data' để đưa vào model.predict
                signal = predict_signal(data, model, scaler)

                # 3. Đưa kết quả vào hàng đợi
                data['signal'] = signal
                self.prediction_queue.put(data)
                logging.info(f"Prediction for {symbol}: Signal={signal}, Price={data['close']}")

            except Exception as e:
                logging.error(f"Error in {symbol} worker: {e}")
                send_discord_message(f"🚨 ERROR in {symbol} worker: {e}") # Discord alert for worker error

            self.stop_event.wait(self.prediction_interval)
        logging.info(f"Stopping prediction worker for {symbol}")

    def start_workers(self):
        logging.info("Starting all prediction workers...")
        for symbol in self.symbols:
            # Giả lập model và scaler cho mỗi symbol
            model = f"model_{symbol}"
            scaler = f"scaler_{symbol}"
            thread = threading.Thread(target=self._prediction_worker, args=(symbol, model, scaler))
            self.threads.append(thread)
            thread.start()
        logging.info(f"Started {len(self.threads)} workers.")

    def stop_workers(self):
        logging.info("Signaling all workers to stop...")
        self.stop_event.set() 
        for thread in self.threads:
            thread.join() 
        logging.info("All workers stopped.")

# --- Hàm tính toán và đặt đòn bẩy (MỚI) ---
def calculate_and_set_leverage(symbol, binance_futures_client, current_price):
    try:
        MIN_NOTIONAL_VALUE = 5.0 # Binance minimum notional value
        
        # Calculate required leverage to achieve MIN_NOTIONAL_VALUE with FIXED_MARGIN_PER_TRADE_USD
        # notional_value = margin * leverage
        # leverage = notional_value / margin
        required_leverage_for_min_notional = MIN_NOTIONAL_VALUE / FIXED_MARGIN_PER_TRADE_USD

        # Get max allowed leverage for the symbol from exchange_info
        max_allowed_leverage = symbol_info[symbol]['max_leverage_allowed']
        
        # Determine desired leverage: min(max_allowed, MAX_LEVERAGE_CAP, required_for_min_notional)
        # Also ensure it's at least 1x
        desired_leverage = int(max(1, min(max_allowed_leverage, MAX_LEVERAGE_CAP, required_leverage_for_min_notional)))

        # Ensure margin type is ISOLATED for the symbol
        try:
            binance_futures_client.change_margin_type(symbol=symbol, marginType='ISOLATED')
            logging.info(f"Set margin type to ISOLATED for {symbol}")
        except BinanceAPIException as e:
            logging.warning(f"Could not set margin type to ISOLATED for {symbol}: {e}")
            send_discord_message(f"⚠️ Warning: Could not set margin type to ISOLATED for {symbol}: {e}")

        current_leverage = 1.0 # Default to 1x if no position found or no leverage set yet
        try:
            position_risk = binance_futures_client.get_position_risk(symbol=symbol)
            found_leverage = False
            for entry in position_risk:
                if entry['symbol'] == symbol:
                    current_leverage = float(entry['leverage'])
                    found_leverage = True
                    break
            if not found_leverage:
                logging.info(f"No active position found for {symbol}. Assuming current leverage is 1x.")
        except BinanceAPIException as e:
            logging.warning(f"Could not get position risk for {symbol}: {e}. Assuming current leverage is 1x.")
            send_discord_message(f"⚠️ Warning: Could not get position risk for {symbol}: {e}. Assuming 1x leverage.")
            current_leverage = 1.0

        if desired_leverage != current_leverage:
            binance_futures_client.set_leverage(symbol=symbol, leverage=desired_leverage)
            logging.info(f"Set leverage to {desired_leverage}x for {symbol} (was {current_leverage}x)")
            send_discord_message(f"ℹ️ Set leverage to {desired_leverage}x for {symbol} (was {current_leverage}x)")
        else:
            logging.info(f"Leverage for {symbol} is already {desired_leverage}x. No change needed.")
        
        return desired_leverage

    except Exception as e:
        logging.error(f"Error calculating and setting leverage for {symbol}: {e}")
        send_discord_message(f"🚨 CRITICAL ERROR: Could not set leverage for {symbol}: {e}")
        return None # Indicate failure

# --- Hàm thực hiện lệnh Binance ---
def place_order(symbol, side, order_type, quantity, price=None):
    try:
        # Ensure quantity is valid based on symbol_info
        if symbol not in symbol_info:
            logging.error(f"Symbol info not found for {symbol}. Cannot place order.")
            send_discord_message(f"❌ Error: Symbol info not found for {symbol}. Cannot place order.")
            return None
        
        min_qty = symbol_info[symbol]['min_qty']
        step_size = symbol_info[symbol]['step_size']

        # Round quantity to step_size and ensure it's at least min_qty
        # Use max(min_qty, ...) to ensure it meets minimum quantity requirement
        quantity = max(min_qty, float(round(quantity / step_size) * step_size))
        
        if quantity == 0: # After rounding, quantity might become 0 if too small
            logging.warning(f"Calculated quantity for {symbol} is zero after rounding. Skipping order.")
            return None

        if order_type == 'MARKET':
            order = binance_futures_client.new_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
        elif order_type == 'LIMIT':
            order = binance_futures_client.new_order(symbol=symbol, side=side, type=order_type, quantity=quantity, price=price, timeInForce='GTC')
        logging.info(f"Order placed for {symbol}: {order}")
        send_discord_message(f"✅ Order placed for {symbol} ({side} {quantity} {order_type})")
        return order
    except BinanceAPIException as e:
        logging.error(f"Binance API Error placing order for {symbol}: {e}")
        send_discord_message(f"❌ Binance API Error placing order for {symbol}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error placing order for {symbol}: {e}")
        send_discord_message(f"❌ Error placing order for {symbol}: {e}")
        return None

# --- Luồng chính ---
def main():
    logging.info("Main thread started.")

    # --- Fetch exchange info and set leverage for all symbols (NEW) ---
    try:
        exchange_info = binance_futures_client.exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] in SYMBOLS:
                # Store minQty, stepSize, and max_leverage for precision handling
                

                max_leverage_found = 0.0
                # Iterate through filters to find maxLeverage
                for f in s['filters']:
                    if 'maxLeverage' in f:
                        max_leverage_found = float(f['maxLeverage'])
                        break
                # Fallback if not found in filters (less common but possible)
                if max_leverage_found == 0.0 and 'maxLeverage' in s:
                    max_leverage_found = float(s['maxLeverage'])

                symbol_info[s['symbol']] = {
                    'min_qty': float(s['filters'][1]['minQty']),
                    'step_size': float(s['filters'][1]['stepSize']),
                    'max_leverage_allowed': max_leverage_found
                }
    except Exception as e:
        logging.error(f"Error fetching exchange info or setting leverage: {e}")
        send_discord_message(f"🚨 CRITICAL ERROR: Could not fetch exchange info or set leverage: {e}")
        return # Exit if critical setup fails

    # --- Khởi tạo trạng thái giao dịch ---
    capital = INITIAL_CAPITAL # This will be updated by actual PnL from Binance
    positions = {symbol: {
        'position': 0, 
        'entry_price': 0,
        'trailing_stop_price': 0,
        'fixed_stop_loss_price': 0,
        'current_position_size': 0
    } for symbol in SYMBOLS}
    
    # Khởi tạo ThreadPoolManager
    manager = ThreadPoolManager(SYMBOLS, PREDICTION_INTERVAL_SECONDS, {}, {}, prediction_queue)
    manager.start_workers()

    logging.info(f"Initial Capital: {capital}")
    send_discord_message(f"🚀 Trading bot started! Initial Capital: ${capital:,.2f}")

    # Vòng lặp chính để xử lý các tín hiệu và quản lý giao dịch
    try:
        while True:
            if not prediction_queue.empty():
                data = prediction_queue.get() # Lấy kết quả từ hàng đợi
                symbol = data['symbol']
                current_price = data['close']
                signal = data['signal']
                pos_data = positions[symbol]

                logging.info(f"Processing signal for {symbol}: Price={current_price}, Signal={signal}")

                # --- EXIT LOGIC ---
                if pos_data['position'] == 1: # If currently in a long position
                    # Trailing Stop-Loss
                    current_trailing_stop = current_price * (1 - TRAILING_STOP_PERCENT)
                    pos_data['trailing_stop_price'] = max(pos_data['trailing_stop_price'], current_trailing_stop)

                    # Take Profit
                    take_profit_price = pos_data['entry_price'] * (1 + TAKE_PROFIT_PERCENT)

                    # NEW: Exit if ML signal is 0 or trend filter is not met
                    is_trending_now = data['ma_fast'] > data['ma_slow']
                    if signal == 0 or not is_trending_now: # If ML says no buy or trend is broken
                        logging.info(f"ML signal or trend filter changed for {symbol}. Closing position.")
                        # Place SELL order to close position
                        quantity_to_sell = pos_data['current_position_size'] 
                        order_result = place_order(symbol, 'SELL', 'MARKET', quantity_to_sell)
                        
                        if order_result: # If order was successfully placed
                            logging.info(f"CLOSED LONG {symbol} at {current_price}")
                            send_discord_message(f"📉 CLOSED LONG {symbol} at ${current_price:,.2f} (ML/Trend Exit)")
                            # Reset position data
                            pos_data['position'], pos_data['entry_price'], pos_data['trailing_stop_price'], pos_data['fixed_stop_loss_price'], pos_data['current_position_size'] = 0, 0, 0, 0, 0
                        continue # Skip other exit checks if already closed

                    # Check for other exit conditions (trailing stop, fixed stop, take profit)
                    if current_price <= pos_data['trailing_stop_price'] or \
                       current_price <= pos_data['fixed_stop_loss_price'] or \
                       current_price >= take_profit_price:
                        
                        # Place SELL order to close position
                        quantity_to_sell = pos_data['current_position_size'] 
                        order_result = place_order(symbol, 'SELL', 'MARKET', quantity_to_sell)
                        
                        if order_result: # If order was successfully placed
                            # In a real bot, you'd verify order execution and update PnL from Binance
                            # For simulation, we'll just reset position data
                            logging.info(f"CLOSED LONG {symbol} at {current_price}")
                            send_discord_message(f"📉 CLOSED LONG {symbol} at ${current_price:,.2f} (RM Exit)")
                            # Reset position data
                            pos_data['position'], pos_data['entry_price'], pos_data['trailing_stop_price'], pos_data['fixed_stop_loss_price'], pos_data['current_position_size'] = 0, 0, 0, 0, 0

                # --- ENTRY LOGIC (Based on Market Regime and ML Signal) ---
                if pos_data['position'] == 0: # If no open position
                    # Calculate total invested capital across all symbols
                    total_invested_capital = sum(p['current_position_size'] * p['entry_price'] for p in positions.values() if p['position'] == 1)
                    
                    # Check if total invested capital exceeds the limit
                    if total_invested_capital >= INITIAL_CAPITAL * INVESTMENT_LIMIT_PERCENT:
                        logging.info(f"Total invested capital ({total_invested_capital:.2f}) reached {INVESTMENT_LIMIT_PERCENT*100:.0f}% of initial capital. Skipping new orders.")
                        send_discord_message(f"⚠️ Total invested capital reached limit. Skipping new orders.")
                        continue # Skip placing new orders

                    # Calculate max margin amount for this trade
                    # In a real bot, you'd fetch actual account balance from Binance to get current capital
                    current_account_balance = capital # Using simulated capital for now
                    max_margin_amount = FIXED_MARGIN_PER_TRADE_USD # Use fixed $1 margin
                    
                    # NEW: Calculate and set leverage before placing order
                    calculated_leverage = calculate_and_set_leverage(symbol, binance_futures_client, current_price)
                    if calculated_leverage is None: # If setting leverage failed, skip trade
                        logging.warning(f"Skipping trade for {symbol} due to leverage setting failure.")
                        continue
                    

                    notional_value_allowed = max_margin_amount * calculated_leverage

                    # Calculate quantity to buy
                    quantity_to_buy = notional_value_allowed / current_price
                    
                    # Quantity precision and min_qty handled in place_order function

                    # Market Regime Check (simulated ADX and Bollinger Bands)
                    is_trending = data['adx'] > ADX_THRESHOLD
                    is_ranging_buy_signal = current_price < data['bb_lower']

                    if is_trending: # Trending Market
                        # ML-driven Trend Following (simulated MA filter)
                        if signal == 1 and data['ma_fast'] > data['ma_slow']:
                            order_result = place_order(symbol, 'BUY', 'MARKET', quantity_to_buy)
                            if order_result: # If order was successfully placed
                                pos_data['position'] = 1
                                pos_data['entry_price'] = current_price # Use actual fill price from order_result in real bot
                                pos_data['trailing_stop_price'] = current_price * (1 - TRAILING_STOP_PERCENT)
                                pos_data['fixed_stop_loss_price'] = current_price * (1 - FIXED_STOP_LOSS_PERCENT)
                                pos_data['current_position_size'] = quantity_to_buy # Store the actual quantity ordered
                                logging.info(f"OPENED LONG (Trend) {symbol} at {current_price}. Notional: {quantity_to_buy * current_price:.2f}")
                                send_discord_message(f"📈 OPENED LONG (Trend) {symbol} at ${current_price:,.2f}. Quantity: {quantity_to_buy:.3f}. Notional: ${quantity_to_buy * current_price:,.2f}")
                    elif is_ranging_buy_signal: # Ranging Market (Mean Reversion)
                        order_result = place_order(symbol, 'BUY', 'MARKET', quantity_to_buy)
                        if order_result: # If order was successfully placed
                            pos_data['position'] = 1
                            pos_data['entry_price'] = current_price # Use actual fill price from order_result in real bot
                            pos_data['trailing_stop_price'] = current_price * (1 - TRAILING_STOP_PERCENT)
                            pos_data['fixed_stop_loss_price'] = current_price * (1 - FIXED_STOP_LOSS_PERCENT)
                            pos_data['current_position_size'] = quantity_to_buy # Store the actual quantity ordered
                            logging.info(f"OPENED LONG (Range) {symbol} at {current_price}. Notional: {notional_value_allowed:.2f}")
                            send_discord_message(f"🔵 OPENED LONG (Range) {symbol} at ${current_price:,.2f}. Quantity: {quantity_to_buy:.3f}. Notional: ${notional_value_allowed:.2f}")
            
            # In a real bot, you'd fetch account balance periodically to update 'capital'
            # For this simulation, 'capital' is not directly updated by trades, as we're not fetching real PnL.
            # logging.info(f"Current Capital: {capital:.2f}")

            # Small sleep to prevent busy-waiting if queue is empty
            time.sleep(0.1)

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Stopping workers...")
        manager.stop_workers() 
        send_discord_message(f"🛑 Trading bot stopped!")
        logging.info("Main thread exiting.")
    except Exception as e:
        logging.error(f"Critical error in main loop: {e}")
        send_discord_message(f"🚨 CRITICAL ERROR in main loop: {e}")

if __name__ == "__main__":
    main()
