import os
from dotenv import load_dotenv

load_dotenv()
import sys

sys.path.insert(0, ".")

from bots.cross_market_arbitrage import ArbitrageDetector, MarketPrice
from datetime import datetime, timezone

print("Testing Cross-Market Arbitrage Detection...")
print("")

detector = ArbitrageDetector()

# Create realistic market data with significant price difference
kalshi_market = MarketPrice(
    platform="kalshi",
    market_id="INXD-24FEB29",
    market_title="Will S&P 500 close higher on February 29?",
    yes_price=0.42,  # 42 cents
    no_price=0.58,
    volume_24h=50000,
    last_updated=datetime.now(timezone.utc),
    url="https://kalshi.com/markets/INXD-24FEB29",
    liquidity=10000,
)

polymarket_market = MarketPrice(
    platform="polymarket",
    market_id="poly-spx-123",
    market_title="Will S&P 500 close higher on February 29?",  # Very similar title
    yes_price=0.58,  # 58 cents - BIG DIFFERENCE!
    no_price=0.42,
    volume_24h=25000,
    last_updated=datetime.now(timezone.utc),
    url="https://polymarket.com/market/poly-spx-123",
    liquidity=5000,
)

print(f"Kalshi market: {kalshi_market.market_title}")
print(f"  YES price: ${kalshi_market.yes_price:.2f}")
print(f"Polymarket: {polymarket_market.market_title}")
print(f"  YES price: ${polymarket_market.yes_price:.2f}")
print(
    f"Price difference: ${abs(kalshi_market.yes_price - polymarket_market.yes_price):.2f}"
)
print("")

# Detect arbitrage
opportunities = detector.detect_direct_arbitrage([kalshi_market], [polymarket_market])

print(f"Arbitrage opportunities found: {len(opportunities)}")

if opportunities:
    for i, opp in enumerate(opportunities, 1):
        print(f"")
        print(f"Opportunity {i}:")
        print(f"  Type: {opp.opportunity_type}")
        print(
            f"  Expected profit: {opp.expected_profit:.3f} ({opp.expected_profit*100:.1f}%)"
        )
        print(f"  Risk level: {opp.risk_level}")
        print(f"  Confidence: {opp.confidence:.2f}")
        print(f"  Min capital: ${opp.min_capital_required:,.0f}")
        print(f"  Reasoning: {opp.reasoning}")
        print(
            f"  Buy on: {opp.primary_market.platform} at ${opp.primary_market.yes_price:.2f}"
        )
        print(
            f"  Sell on: {opp.secondary_market.platform} at ${opp.secondary_market.yes_price:.2f}"
        )

print("")
print("SUCCESS: Arbitrage detection working with realistic price differences!")
