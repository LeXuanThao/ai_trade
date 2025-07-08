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
# Note: With $1 margin and leverage, the notional value will be: $1 * leverage
# Example: $1 margin * 5x leverage = $5 notional value
# The actual position size will be: notional_value / current_price
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

# --- Global variable to track leverage status for each symbol ---
leverage_initialized = {}

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
        
        # Get max allowed leverage for the symbol from exchange_info
        max_allowed_leverage = symbol_info[symbol]['max_leverage_allowed']
        
        # Calculate the minimum leverage needed to meet Binance's minimum notional requirement
        # with our fixed $1 margin: notional_value = margin * leverage
        # So: leverage = notional_value / margin
        min_leverage_for_notional = MIN_NOTIONAL_VALUE / FIXED_MARGIN_PER_TRADE_USD  # 5.0 / 1.0 = 5x
        
        # Use the maximum available leverage (up to our cap) to maximize position size
        # while still using only $1 margin per trade
        desired_leverage = int(min(max_allowed_leverage, MAX_LEVERAGE_CAP))
        
        # Ensure it meets the minimum requirement for Binance notional value
        if desired_leverage < min_leverage_for_notional:
            desired_leverage = int(min_leverage_for_notional)
        
        # Check if leverage is already set correctly for this symbol
        if symbol in symbol_info and 'current_set_leverage' in symbol_info[symbol]:
            current_leverage = symbol_info[symbol]['current_set_leverage']
            if current_leverage == desired_leverage:
                logging.info(f"Leverage for {symbol} already set to {desired_leverage}x. No API call needed.")
                return desired_leverage
        
        # Log the calculation for verification
        potential_notional = FIXED_MARGIN_PER_TRADE_USD * desired_leverage
        logging.info(f"Setting leverage for {symbol}: margin=${FIXED_MARGIN_PER_TRADE_USD}, "
                    f"leverage={desired_leverage}x, potential_notional=${potential_notional:.2f}, "
                    f"min_required=${MIN_NOTIONAL_VALUE}, max_allowed={max_allowed_leverage}x")

        # Only set margin type to ISOLATED if not already done for this symbol
        if symbol not in leverage_initialized or not leverage_initialized[symbol].get('margin_type_set', False):
            try:
                binance_futures_client.change_margin_type(symbol=symbol, marginType='ISOLATED')
                logging.info(f"Set margin type to ISOLATED for {symbol}")
                # Mark that margin type has been set
                if symbol not in leverage_initialized:
                    leverage_initialized[symbol] = {}
                leverage_initialized[symbol]['margin_type_set'] = True
            except BinanceAPIException as e:
                if e.code == -4046: # "No need to change margin type." - already isolated
                    logging.info(f"Margin type for {symbol} is already ISOLATED. No change needed.")
                    if symbol not in leverage_initialized:
                        leverage_initialized[symbol] = {}
                    leverage_initialized[symbol]['margin_type_set'] = True
                else:
                    logging.warning(f"Could not set margin type to ISOLATED for {symbol}: {e}")
                    send_discord_message(f"⚠️ Warning: Could not set margin type to ISOLATED for {symbol}: {e}")
                    # Don't treat this as a critical error, continue with leverage setting

        # Set leverage only if it's different from what we think is currently set
        try:
            binance_futures_client.set_leverage(symbol=symbol, leverage=desired_leverage)
            logging.info(f"✅ Set leverage to {desired_leverage}x for {symbol}")
            send_discord_message(f"ℹ️ Set leverage to {desired_leverage}x for {symbol}")
            # Update our tracking
            symbol_info[symbol]['current_set_leverage'] = desired_leverage
        except BinanceAPIException as e:
            # Error code -4028 means "Leverage is already set to the desired value."
            if e.code == -4028:
                logging.info(f"Leverage for {symbol} is already {desired_leverage}x. No change needed.")
                # Update our tracking even if no change was made
                symbol_info[symbol]['current_set_leverage'] = desired_leverage
            else:
                logging.warning(f"Could not set leverage for {symbol}: {e}")
                send_discord_message(f"⚠️ Warning: Could not set leverage for {symbol}: {e}")
                # Don't update tracking if setting failed
        
        return desired_leverage

    except BinanceAPIException as e:
        # Handle Binance-specific API errors
        if e.code == -4046: # "No need to change margin type."
            logging.info(f"Margin type for {symbol} is already set correctly.")
            # Try to get current leverage and return it, or return a default
            try:
                position_risk = binance_futures_client.get_position_risk(symbol=symbol)
                for entry in position_risk:
                    if entry['symbol'] == symbol:
                        return float(entry['leverage'])
                return MAX_LEVERAGE_CAP # Default if no position found
            except Exception:
                return MAX_LEVERAGE_CAP # Fallback
        else:
            logging.error(f"Binance API error calculating and setting leverage for {symbol}: {e}")
            send_discord_message(f"🚨 CRITICAL ERROR: Could not set leverage for {symbol}: {e}")
            return None # Indicate failure
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

# --- Hàm khởi tạo thiết lập đòn bẩy cho tất cả các cặp (gọi một lần khi khởi động) ---
def initialize_leverage_for_all_symbols(binance_futures_client):
    """Initialize leverage settings for all symbols once at startup"""
    logging.info("Initializing leverage settings for all symbols...")
    
    for symbol in SYMBOLS:
        try:
            # Calculate optimal leverage for this symbol
            if symbol not in symbol_info:
                logging.warning(f"Symbol {symbol} not found in symbol_info. Skipping leverage initialization.")
                continue
                
            MIN_NOTIONAL_VALUE = 5.0
            max_allowed_leverage = symbol_info[symbol]['max_leverage_allowed']
            min_leverage_for_notional = MIN_NOTIONAL_VALUE / FIXED_MARGIN_PER_TRADE_USD
            desired_leverage = int(min(max_allowed_leverage, MAX_LEVERAGE_CAP))
            
            if desired_leverage < min_leverage_for_notional:
                desired_leverage = int(min_leverage_for_notional)
            
            # Set margin type to ISOLATED
            try:
                binance_futures_client.change_margin_type(symbol=symbol, marginType='ISOLATED')
                logging.info(f"✅ Set margin type to ISOLATED for {symbol}")
            except BinanceAPIException as e:
                if e.code == -4046:
                    logging.info(f"✅ Margin type for {symbol} already ISOLATED")
                else:
                    logging.warning(f"⚠️ Could not set margin type for {symbol}: {e}")
            
            # Set leverage
            try:
                binance_futures_client.set_leverage(symbol=symbol, leverage=desired_leverage)
                logging.info(f"✅ Set leverage to {desired_leverage}x for {symbol}")
                symbol_info[symbol]['current_set_leverage'] = desired_leverage
            except BinanceAPIException as e:
                if e.code == -4028:
                    logging.info(f"✅ Leverage for {symbol} already {desired_leverage}x")
                    symbol_info[symbol]['current_set_leverage'] = desired_leverage
                else:
                    logging.warning(f"⚠️ Could not set leverage for {symbol}: {e}")
                    
        except Exception as e:
            logging.error(f"Error initializing leverage for {symbol}: {e}")
    
    logging.info("✅ Leverage initialization complete for all symbols")

# --- Function to get leverage without setting it (used during trading) ---
def get_leverage_for_symbol(symbol):
    """Get the leverage for a symbol without making API calls"""
    if symbol not in symbol_info or symbol_info[symbol]['current_set_leverage'] is None:
        logging.warning(f"Leverage not initialized for {symbol}. Using fallback calculation.")
        # Fallback calculation
        MIN_NOTIONAL_VALUE = 5.0
        max_allowed_leverage = symbol_info[symbol]['max_leverage_allowed']
        min_leverage_for_notional = MIN_NOTIONAL_VALUE / FIXED_MARGIN_PER_TRADE_USD
        return int(min(max_allowed_leverage, MAX_LEVERAGE_CAP, max(1, min_leverage_for_notional)))
    
    return symbol_info[symbol]['current_set_leverage']

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
                    'max_leverage_allowed': max_leverage_found,
                    'current_set_leverage': None  # Will be set when leverage is first configured
                }
    except Exception as e:
        logging.error(f"Error fetching exchange info or setting leverage: {e}")
        send_discord_message(f"🚨 CRITICAL ERROR: Could not fetch exchange info or set leverage: {e}")
        return # Exit if critical setup fails

    # --- Initialize leverage settings for all symbols once ---
    initialize_leverage_for_all_symbols(binance_futures_client)

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
                            pos_data['position'], pos_data['entry_price'], pos_data['trailing_stop_loss_price'], pos_data['fixed_stop_loss_price'], pos_data['current_position_size'] = 0, 0, 0, 0, 0

                # --- ENTRY LOGIC (Based on Market Regime and ML Signal) ---
                if pos_data['position'] == 0: # If no open position
                    # Calculate total invested capital across all symbols
                    total_invested_capital = sum(p['current_position_size'] * p['entry_price'] for p in positions.values() if p['position'] == 1)
                    
                    # Check if total invested capital exceeds the limit
                    if total_invested_capital >= INITIAL_CAPITAL * INVESTMENT_LIMIT_PERCENT:
                        logging.info(f"Total invested capital ({total_invested_capital:.2f}) reached {INVESTMENT_LIMIT_PERCENT*100:.0f}% of initial capital. Skipping new orders.")
                        send_discord_message("⚠️ Total invested capital reached limit. Skipping new orders.")
                        continue # Skip placing new orders

                    # Calculate max margin amount for this trade
                    # In a real bot, you'd fetch actual account balance from Binance to get current capital
                    max_margin_amount = FIXED_MARGIN_PER_TRADE_USD # Use fixed $1 margin
                    
                    # Get leverage for this symbol (already set at startup, no API calls needed)
                    calculated_leverage = get_leverage_for_symbol(symbol)
                    
                    # Calculate notional value with the set leverage
                    notional_value_allowed = max_margin_amount * calculated_leverage

                    # Calculate quantity to buy
                    quantity_to_buy = notional_value_allowed / current_price
                    
                    # Log trade calculation for verification
                    logging.info(f"Trade calculation for {symbol}: margin=${max_margin_amount:.2f}, "
                                f"leverage={calculated_leverage}x, notional=${notional_value_allowed:.2f}, "
                                f"price=${current_price:.2f}, quantity={quantity_to_buy:.6f}")
                    
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
        send_discord_message("🛑 Trading bot stopped!")
        logging.info("Main thread exiting.")
    except Exception as e:
        logging.error(f"Critical error in main loop: {e}")
        send_discord_message(f"🚨 CRITICAL ERROR in main loop: {e}")

if __name__ == "__main__":
    main()
