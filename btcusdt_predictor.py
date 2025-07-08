
import pandas as pd
from binance.client import Client

# Replace with your actual API key and secret
api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"

client = Client(api_key, api_secret)

# Fetch historical data
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "1 Jan, 2023", "1 Jan, 2024")

# Create a pandas DataFrame
df = pd.DataFrame(klines, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'
])

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Set timestamp as index
df.set_index('timestamp', inplace=True)

# Calculate a simple moving average (SMA)
df['sma'] = df['close'].rolling(window=20).mean()

# A simple prediction logic (for demonstration purposes)
# If the last close is higher than the last SMA, predict price will go up.
if df['close'].iloc[-1] > df['sma'].iloc[-1]:
    prediction = "UP"
else:
    prediction = "DOWN"

print(f"Based on the 20-hour SMA, the predicted direction for BTCUSDT is: {prediction}")
