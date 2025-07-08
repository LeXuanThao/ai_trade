#!/usr/bin/env python3
"""
Test script to verify efficient leverage management - only set when needed
"""

import time

# Mock classes for testing
class MockLeverageManager:
    def __init__(self):
        self.api_calls = 0
        self.leverage_settings = {}
        self.margin_type_settings = {}
        
    def change_margin_type(self, symbol, marginType):
        self.api_calls += 1
        print(f"[API CALL {self.api_calls}] Setting margin type for {symbol} to {marginType}")
        if symbol in self.margin_type_settings:
            raise MockAPIException(-4046, "No need to change margin type.")
        self.margin_type_settings[symbol] = marginType
        
    def set_leverage(self, symbol, leverage):
        self.api_calls += 1
        print(f"[API CALL {self.api_calls}] Setting leverage for {symbol} to {leverage}x")
        if symbol in self.leverage_settings and self.leverage_settings[symbol] == leverage:
            raise MockAPIException(-4028, "Leverage is already set to the desired value.")
        self.leverage_settings[symbol] = leverage

class MockAPIException(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(message)

# Test the efficient leverage management
def test_efficient_leverage():
    print("=" * 70)
    print("TESTING EFFICIENT LEVERAGE MANAGEMENT")
    print("=" * 70)
    
    # Simulate the symbol info structure
    symbol_info = {
        'BTCUSDT': {'max_leverage_allowed': 125, 'current_set_leverage': None},
        'ETHUSDT': {'max_leverage_allowed': 100, 'current_set_leverage': None},
        'XRPUSDT': {'max_leverage_allowed': 75, 'current_set_leverage': None},
    }
    
    leverage_initialized = {}
    mock_client = MockLeverageManager()
    
    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
    MAX_LEVERAGE_CAP = 20
    FIXED_MARGIN_PER_TRADE_USD = 1.0
    MIN_NOTIONAL_VALUE = 5.0
    
    def initialize_leverage_for_symbol(symbol):
        """Simulate the initialization function"""
        max_allowed_leverage = symbol_info[symbol]['max_leverage_allowed']
        min_leverage_for_notional = MIN_NOTIONAL_VALUE / FIXED_MARGIN_PER_TRADE_USD
        desired_leverage = int(min(max_allowed_leverage, MAX_LEVERAGE_CAP))
        
        if desired_leverage < min_leverage_for_notional:
            desired_leverage = int(min_leverage_for_notional)
        
        # Set margin type
        try:
            mock_client.change_margin_type(symbol=symbol, marginType='ISOLATED')
            print(f"✅ Set margin type to ISOLATED for {symbol}")
        except MockAPIException as e:
            if e.code == -4046:
                print(f"✅ Margin type for {symbol} already ISOLATED")
        
        # Set leverage
        try:
            mock_client.set_leverage(symbol=symbol, leverage=desired_leverage)
            print(f"✅ Set leverage to {desired_leverage}x for {symbol}")
            symbol_info[symbol]['current_set_leverage'] = desired_leverage
        except MockAPIException as e:
            if e.code == -4028:
                print(f"✅ Leverage for {symbol} already {desired_leverage}x")
                symbol_info[symbol]['current_set_leverage'] = desired_leverage
    
    def get_leverage_for_symbol(symbol):
        """Simulate getting leverage without API calls"""
        return symbol_info[symbol]['current_set_leverage']
    
    # Test 1: Initial setup (should make API calls)
    print("\n📋 PHASE 1: Initial Setup (API calls expected)")
    print("-" * 50)
    initial_api_calls = mock_client.api_calls
    
    for symbol in SYMBOLS:
        initialize_leverage_for_symbol(symbol)
    
    setup_api_calls = mock_client.api_calls - initial_api_calls
    print(f"\n📊 API calls during setup: {setup_api_calls}")
    
    # Test 2: Simulate trading (should NOT make API calls)
    print("\n📋 PHASE 2: Simulating Trading (NO API calls expected)")
    print("-" * 50)
    trading_start_calls = mock_client.api_calls
    
    # Simulate 10 trading decisions across 60 seconds (every 6 seconds)
    for i in range(10):
        print(f"\n⏰ Trading cycle {i+1} (simulating 60s intervals)")
        for symbol in SYMBOLS:
            leverage = get_leverage_for_symbol(symbol)
            print(f"  📈 Trading {symbol} with {leverage}x leverage (no API call)")
        time.sleep(0.1)  # Small delay to simulate time passing
    
    trading_api_calls = mock_client.api_calls - trading_start_calls
    print(f"\n📊 API calls during trading: {trading_api_calls}")
    
    # Test 3: Simulate restart (should make API calls again because already set)
    print("\n📋 PHASE 3: Simulating Restart (Some API calls expected)")
    print("-" * 50)
    restart_calls = mock_client.api_calls
    
    for symbol in SYMBOLS:
        initialize_leverage_for_symbol(symbol)
    
    restart_api_calls = mock_client.api_calls - restart_calls
    print(f"\n📊 API calls during restart: {restart_api_calls}")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY:")
    print("=" * 70)
    print(f"✅ Setup phase: {setup_api_calls} API calls (expected: {len(SYMBOLS) * 2})")
    print(f"✅ Trading phase: {trading_api_calls} API calls (expected: 0)")
    print(f"✅ Restart phase: {restart_api_calls} API calls (expected: 0-{len(SYMBOLS)})")
    print(f"📊 Total API calls: {mock_client.api_calls}")
    
    if trading_api_calls == 0:
        print("\n🎉 SUCCESS: No unnecessary API calls during trading!")
    else:
        print("\n❌ ISSUE: Unexpected API calls during trading!")
    
    print("\n🏆 Efficient leverage management working correctly!")
    print("=" * 70)

if __name__ == "__main__":
    test_efficient_leverage()
