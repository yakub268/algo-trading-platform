"""
KALSHI TEST - Find any market and buy 1 contract
"""

import os
import sys
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("KALSHI TEST - Buy 1 Contract")
print("=" * 70)

try:
    from bots.kalshi_bot import KalshiBot
    from config.kalshi_config import KALSHI_CONFIG

    # Override paper mode for live trading
    config = KALSHI_CONFIG.copy()
    config["paper_mode"] = False

    bot = KalshiBot(config=config)

    # Verify connection
    if not bot.verify_connection():
        print("FAILURE: Could not connect to Kalshi API")
        sys.exit(1)

    balance = bot.get_balance()
    print(f"Account Balance: ${balance:.2f}" if balance else "Balance: N/A")

    # Get more markets
    markets = bot.get_markets(status="open", limit=200)
    print(f"Found {len(markets)} open markets")

    # Show price distribution
    print("\nMarket Price Distribution:")
    price_ranges = {
        "0-10c": 0,
        "10-25c": 0,
        "25-50c": 0,
        "50-75c": 0,
        "75-100c": 0,
        "no_price": 0,
    }
    for market in markets:
        yes_ask = market.get("yes_ask")
        if yes_ask is None:
            price_ranges["no_price"] += 1
        elif yes_ask <= 10:
            price_ranges["0-10c"] += 1
        elif yes_ask <= 25:
            price_ranges["10-25c"] += 1
        elif yes_ask <= 50:
            price_ranges["25-50c"] += 1
        elif yes_ask <= 75:
            price_ranges["50-75c"] += 1
        else:
            price_ranges["75-100c"] += 1

    for range_name, count in price_ranges.items():
        print(f"  {range_name}: {count}")

    # Show first few markets with prices
    print("\nSample markets with orderbook:")
    sample_count = 0
    for market in markets:
        if sample_count >= 10:
            break
        ticker = market.get("ticker")
        title = market.get("title", "")[:40]
        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")
        no_bid = market.get("no_bid")
        no_ask = market.get("no_ask")
        volume = market.get("volume", 0) or market.get("volume_24h", 0)

        if yes_ask or no_ask:
            print(
                f"  {ticker}: YES bid/ask={yes_bid}/{yes_ask}, NO bid/ask={no_bid}/{no_ask}, vol={volume}"
            )
            print(f"    Title: {title}...")
            sample_count += 1

    # Try to find ANY market we can trade
    print("\nLooking for tradeable market...")
    best_market = None

    # Strategy 1: Look for markets with reasonable YES ask prices
    for market in markets:
        yes_ask = market.get("yes_ask")
        if yes_ask and 1 <= yes_ask <= 95:
            best_market = market
            print(f"  Found market with YES ask: {market.get('ticker')} @ {yes_ask}c")
            break

    # Strategy 2: Look for markets with NO ask prices if YES not found
    if not best_market:
        for market in markets:
            no_ask = market.get("no_ask")
            if no_ask and 1 <= no_ask <= 95:
                best_market = market
                print(f"  Found market with NO ask: {market.get('ticker')} @ {no_ask}c")
                break

    # Strategy 3: Get orderbook for first few markets to find actual prices
    if not best_market:
        print("  Checking orderbooks for first 5 markets...")
        for market in markets[:5]:
            ticker = market.get("ticker")
            orderbook = bot.get_orderbook(ticker)
            if orderbook:
                yes_bids = orderbook.get("yes", [])
                no_bids = orderbook.get("no", [])
                print(
                    f"    {ticker}: yes_orders={len(yes_bids)}, no_orders={len(no_bids)}"
                )

                # Find best ask from orderbook
                if yes_bids:
                    # Orderbook might have different structure
                    best_market = market
                    market["yes_ask"] = 50  # Default to 50c if we can't parse
                    break

    if not best_market:
        print("\nFAILURE: No tradeable market found")
        print("This may be due to:")
        print("  - All markets having extreme prices (0c or 99c+)")
        print("  - No orderbook liquidity")
        print("  - Market structure differences")
        sys.exit(1)

    ticker = best_market.get("ticker")
    title = best_market.get("title", "Unknown")[:60]
    yes_ask = best_market.get("yes_ask", 50)
    no_ask = best_market.get("no_ask", 50)

    print(f"\nSelected Market: {ticker}")
    print(f"Title: {title}")
    print(f"YES Ask: {yes_ask}c, NO Ask: {no_ask}c")

    # Choose side based on price (buy cheaper side)
    if yes_ask and (not no_ask or yes_ask <= no_ask):
        side = "yes"
        price = yes_ask
    else:
        side = "no"
        price = no_ask

    print(f"\nPlacing order: BUY 1 {side.upper()} @ {price}c")

    # Place order for 1 contract
    result = bot.place_order(
        ticker=ticker, side=side, quantity=1, price=price, order_type="limit"
    )

    if result:
        order_id = (
            result.get("order_id") or result.get("id")
            if isinstance(result, dict)
            else None
        )
        print(f"\nSUCCESS!")
        print(f"  Order ID: {order_id}")
        print(f"  Market: {ticker}")
        print(f"  Side: {side.upper()}")
        print(f"  Quantity: 1")
        print(f"  Price: {price}c")
        print(f"  Response: {json.dumps(result, indent=2)[:500]}")
    else:
        print("\nFAILURE: No response from place_order")

except Exception as e:
    import traceback

    print(f"\nFAILURE: {e}")
    traceback.print_exc()

print("=" * 70)
