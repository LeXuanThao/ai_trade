from binance.um_futures import UMFutures
import random

client = UMFutures() # Use UMFutures client for futures data
exchange_info = client.exchange_info()

futures_symbols = []
for s in exchange_info['symbols']:
    if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING':
        futures_symbols.append(s['symbol'])

if futures_symbols:
    print(random.choice(futures_symbols))
else:
    print("No suitable futures symbols found.")