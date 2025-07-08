#!/usr/bin/env python3
"""
Test script to verify leverage calculation for $1 investment per trade
"""

# Constants from the main script
FIXED_MARGIN_PER_TRADE_USD = 1.0
MAX_LEVERAGE_CAP = 20
MIN_NOTIONAL_VALUE = 5.0

def test_leverage_calculation(symbol, max_allowed_leverage, current_price):
    """Test the leverage calculation logic"""
    
    print(f"\n=== Testing leverage calculation for {symbol} ===")
    print(f"Current price: ${current_price:.2f}")
    print(f"Fixed margin per trade: ${FIXED_MARGIN_PER_TRADE_USD:.2f}")
    print(f"Max allowed leverage for symbol: {max_allowed_leverage}x")
    print(f"Bot's max leverage cap: {MAX_LEVERAGE_CAP}x")
    print(f"Binance minimum notional value: ${MIN_NOTIONAL_VALUE:.2f}")
    
    # Calculate minimum leverage needed to meet Binance requirements
    min_leverage_for_notional = MIN_NOTIONAL_VALUE / FIXED_MARGIN_PER_TRADE_USD
    print(f"Minimum leverage needed for notional requirement: {min_leverage_for_notional:.1f}x")
    
    # Calculate the final leverage to use
    desired_leverage = int(min(max_allowed_leverage, MAX_LEVERAGE_CAP, max(1, min_leverage_for_notional)))
    print(f"Final leverage to use: {desired_leverage}x")
    
    # Calculate the resulting trade values
    notional_value = FIXED_MARGIN_PER_TRADE_USD * desired_leverage
    quantity = notional_value / current_price
    
    print(f"\n--- Trade Results ---")
    print(f"Margin used: ${FIXED_MARGIN_PER_TRADE_USD:.2f}")
    print(f"Leverage: {desired_leverage}x")
    print(f"Notional value: ${notional_value:.2f}")
    print(f"Quantity to buy: {quantity:.6f} {symbol}")
    print(f"Total position value: ${quantity * current_price:.2f}")
    
    # Verify the margin requirement
    actual_margin_required = notional_value / desired_leverage
    print(f"Actual margin required: ${actual_margin_required:.2f}")
    print(f"Meets minimum notional? {'YES' if notional_value >= MIN_NOTIONAL_VALUE else 'NO'}")
    
    return {
        'leverage': desired_leverage,
        'notional_value': notional_value,
        'quantity': quantity,
        'margin_used': FIXED_MARGIN_PER_TRADE_USD
    }

if __name__ == "__main__":
    # Test with different symbols and scenarios
    test_cases = [
        ("BTCUSDT", 125, 42000.0),  # High price, high leverage allowed
        ("ETHUSDT", 100, 2500.0),   # Medium price, high leverage
        ("XRPUSDT", 75, 0.60),      # Low price, medium leverage
        ("SOLUSDT", 50, 85.0),      # Medium price, medium leverage
    ]
    
    print("=" * 60)
    print("LEVERAGE CALCULATION VERIFICATION TEST")
    print("=" * 60)
    
    for symbol, max_leverage, price in test_cases:
        result = test_leverage_calculation(symbol, max_leverage, price)
        
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Each trade uses exactly ${FIXED_MARGIN_PER_TRADE_USD:.2f} margin")
    print(f"Leverage is automatically calculated to meet Binance requirements")
    print(f"Minimum leverage will be {MIN_NOTIONAL_VALUE / FIXED_MARGIN_PER_TRADE_USD:.0f}x to meet ${MIN_NOTIONAL_VALUE:.2f} notional requirement")
    print(f"Maximum leverage is capped at {MAX_LEVERAGE_CAP}x for safety")
    print("=" * 60)
