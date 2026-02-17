"""
LIVE TRADING TEST - MINIMUM AMOUNTS
===================================
Execute 1 minimal test trade on each market to confirm trading systems work.

Markets:
1. CRYPTO (Coinbase) - Buy $1 of XLM
2. FOREX (OANDA) - Buy 1 unit EUR/USD
3. PREDICTION (Kalshi) - Buy 1 contract on active market

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import json
from datetime import datetime, timezone
from decimal import Decimal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment
from dotenv import load_dotenv

load_dotenv()

# Results collection
results = {"timestamp": datetime.now(timezone.utc).isoformat(), "tests": {}}

print("=" * 70)
print("LIVE TRADING TEST - MINIMUM AMOUNTS")
print("=" * 70)
print(f"Timestamp: {results['timestamp']}")
print()

# ============================================================================
# TEST 1: COINBASE - Buy $1 of XLM
# ============================================================================
print("-" * 70)
print("TEST 1: COINBASE - Buy $1 of XLM")
print("-" * 70)

try:
    from bots.coinbase_arb_bot import CoinbaseClient

    client = CoinbaseClient()

    if not client._initialized:
        results["tests"]["coinbase"] = {
            "status": "FAILURE",
            "error": "Coinbase client not initialized - check API key and private key",
            "order_id": None,
            "amount": None,
        }
        print(f"  STATUS: FAILURE")
        print(f"  ERROR: Coinbase client not initialized")
    else:
        # Execute the trade
        result = client.create_market_order(
            product_id="XLM-USD", side="BUY", quote_size="1.00"
        )

        if result:
            # Extract order details
            order_id = (
                getattr(result, "order_id", None) or result.get("order_id")
                if isinstance(result, dict)
                else None
            )
            success = (
                getattr(result, "success", True) if hasattr(result, "success") else True
            )

            results["tests"]["coinbase"] = {
                "status": "SUCCESS" if success else "FAILURE",
                "order_id": order_id,
                "amount": "$1.00 XLM",
                "response": str(result)[:500],
            }
            print(f"  STATUS: {'SUCCESS' if success else 'FAILURE'}")
            print(f"  ORDER ID: {order_id}")
            print(f"  AMOUNT: $1.00 XLM")
            print(f"  RESPONSE: {str(result)[:200]}")
        else:
            results["tests"]["coinbase"] = {
                "status": "FAILURE",
                "error": "No response from create_market_order",
                "order_id": None,
                "amount": None,
            }
            print(f"  STATUS: FAILURE")
            print(f"  ERROR: No response from create_market_order")

except Exception as e:
    results["tests"]["coinbase"] = {
        "status": "FAILURE",
        "error": str(e),
        "order_id": None,
        "amount": None,
    }
    print(f"  STATUS: FAILURE")
    print(f"  ERROR: {e}")

print()

# ============================================================================
# TEST 2: OANDA - Buy 1 unit EUR/USD
# ============================================================================
print("-" * 70)
print("TEST 2: OANDA - Buy 1 unit EUR/USD")
print("-" * 70)

try:
    from bots.oanda_bot import OANDABot
    from config.oanda_config import OANDA_CONFIG

    # Override paper mode for live trading
    config = OANDA_CONFIG.copy()
    config["paper_mode"] = False

    bot = OANDABot(config=config)

    # Check if we have credentials
    if not bot.api_key or not bot.account_id:
        results["tests"]["oanda"] = {
            "status": "FAILURE",
            "error": "OANDA credentials not configured (OANDA_API_KEY, OANDA_ACCOUNT_ID)",
            "order_id": None,
            "amount": None,
        }
        print(f"  STATUS: FAILURE")
        print(f"  ERROR: OANDA credentials not configured")
    else:
        # Verify connection first
        account = bot.get_account()
        if not account:
            results["tests"]["oanda"] = {
                "status": "FAILURE",
                "error": "Could not connect to OANDA API",
                "order_id": None,
                "amount": None,
            }
            print(f"  STATUS: FAILURE")
            print(f"  ERROR: Could not connect to OANDA API")
        else:
            print(f"  Account Balance: {account.get('balance', 'N/A')}")

            # Execute minimum trade - 1 unit long EUR/USD
            result = bot.market_order(
                instrument="EUR_USD", units=1, stop_loss=None, take_profit=None
            )

            if result:
                # Extract order ID from response
                order_fill = result.get("orderFillTransaction", {})
                order_id = (
                    order_fill.get("id")
                    or result.get("relatedTransactionIDs", ["N/A"])[0]
                    if isinstance(result, dict)
                    else None
                )

                results["tests"]["oanda"] = {
                    "status": "SUCCESS",
                    "order_id": order_id,
                    "amount": "1 unit EUR/USD",
                    "response": str(result)[:500],
                }
                print(f"  STATUS: SUCCESS")
                print(f"  ORDER ID: {order_id}")
                print(f"  AMOUNT: 1 unit EUR/USD")
                print(f"  RESPONSE: {str(result)[:200]}")
            else:
                results["tests"]["oanda"] = {
                    "status": "FAILURE",
                    "error": "No response from market_order",
                    "order_id": None,
                    "amount": None,
                }
                print(f"  STATUS: FAILURE")
                print(f"  ERROR: No response from market_order")

except Exception as e:
    results["tests"]["oanda"] = {
        "status": "FAILURE",
        "error": str(e),
        "order_id": None,
        "amount": None,
    }
    print(f"  STATUS: FAILURE")
    print(f"  ERROR: {e}")

print()

# ============================================================================
# TEST 3: KALSHI - Buy 1 contract on active market
# ============================================================================
print("-" * 70)
print("TEST 3: KALSHI - Buy 1 contract on active market")
print("-" * 70)

try:
    from bots.kalshi_bot import KalshiBot
    from config.kalshi_config import KALSHI_CONFIG

    # Override paper mode for live trading
    config = KALSHI_CONFIG.copy()
    config["paper_mode"] = False

    bot = KalshiBot(config=config)

    # Verify connection
    if not bot.verify_connection():
        results["tests"]["kalshi"] = {
            "status": "FAILURE",
            "error": "Could not connect to Kalshi API",
            "order_id": None,
            "amount": None,
        }
        print(f"  STATUS: FAILURE")
        print(f"  ERROR: Could not connect to Kalshi API")
    else:
        balance = bot.get_balance()
        print(f"  Account Balance: ${balance:.2f}" if balance else "  Balance: N/A")

        # Get active markets
        markets = bot.get_markets(status="open", limit=50)
        print(f"  Found {len(markets)} open markets")

        if not markets:
            results["tests"]["kalshi"] = {
                "status": "FAILURE",
                "error": "No active markets found",
                "order_id": None,
                "amount": None,
            }
            print(f"  STATUS: FAILURE")
            print(f"  ERROR: No active markets found")
        else:
            # Find cheapest YES contract (lowest yes_ask)
            best_market = None
            best_price = 100

            for market in markets:
                yes_ask = market.get("yes_ask", 100)
                volume = market.get("volume", 0) or market.get("volume_24h", 0)

                # Look for affordable liquid market (price 5-50 cents, some volume)
                if yes_ask and 5 <= yes_ask <= 50 and volume >= 0:
                    if yes_ask < best_price:
                        best_price = yes_ask
                        best_market = market

            if not best_market:
                # Fall back to any market with reasonable price
                for market in markets:
                    yes_ask = market.get("yes_ask", 100)
                    if yes_ask and 5 <= yes_ask <= 70:
                        best_market = market
                        best_price = yes_ask
                        break

            if not best_market:
                results["tests"]["kalshi"] = {
                    "status": "FAILURE",
                    "error": "No suitable market found (all prices too high or too low)",
                    "order_id": None,
                    "amount": None,
                }
                print(f"  STATUS: FAILURE")
                print(f"  ERROR: No suitable market found")
            else:
                ticker = best_market.get("ticker")
                title = best_market.get("title", "Unknown")[:50]

                print(f"  Selected Market: {ticker}")
                print(f"  Title: {title}...")
                print(f"  Price: {best_price}c")

                # Place order for 1 contract
                result = bot.place_order(
                    ticker=ticker,
                    side="yes",
                    quantity=1,
                    price=best_price,
                    order_type="limit",
                )

                if result:
                    order_id = (
                        result.get("order_id") or result.get("id")
                        if isinstance(result, dict)
                        else None
                    )

                    results["tests"]["kalshi"] = {
                        "status": "SUCCESS",
                        "order_id": order_id,
                        "amount": f"1 contract @ {best_price}c",
                        "market": ticker,
                        "response": str(result)[:500],
                    }
                    print(f"  STATUS: SUCCESS")
                    print(f"  ORDER ID: {order_id}")
                    print(f"  AMOUNT: 1 contract @ {best_price}c")
                else:
                    results["tests"]["kalshi"] = {
                        "status": "FAILURE",
                        "error": "No response from place_order",
                        "order_id": None,
                        "amount": None,
                    }
                    print(f"  STATUS: FAILURE")
                    print(f"  ERROR: No response from place_order")

except Exception as e:
    results["tests"]["kalshi"] = {
        "status": "FAILURE",
        "error": str(e),
        "order_id": None,
        "amount": None,
    }
    print(f"  STATUS: FAILURE")
    print(f"  ERROR: {e}")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)

success_count = sum(
    1 for t in results["tests"].values() if t.get("status") == "SUCCESS"
)
total_count = len(results["tests"])

print(f"Total Tests: {total_count}")
print(f"Successful: {success_count}")
print(f"Failed: {total_count - success_count}")
print()

for market, test_result in results["tests"].items():
    status = test_result.get("status", "UNKNOWN")
    order_id = test_result.get("order_id", "N/A")
    amount = test_result.get("amount", "N/A")
    error = test_result.get("error", "")

    print(
        f"{market.upper():12} | {status:8} | Order: {order_id or 'N/A'} | Amount: {amount or 'N/A'}"
    )
    if error:
        print(f"             | Error: {error[:60]}")

print("=" * 70)

# Save results to file
results_file = os.path.join(
    os.path.dirname(__file__), "logs", "test_trade_results.json"
)
os.makedirs(os.path.dirname(results_file), exist_ok=True)
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {results_file}")
