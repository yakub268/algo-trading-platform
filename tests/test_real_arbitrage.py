"""
Test cross-market arbitrage with real APIs
Start small and build up to avoid the "giant untested code" problem

NOT a unit test — requires live API keys. Run directly: python tests/test_real_arbitrage.py
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Live API integration test — run directly with: python tests/test_real_arbitrage.py"
)

import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

from bots.kalshi_client import KalshiClient
from bots.cross_market_arbitrage import (
    ExternalMarketConnector,
    ArbitrageDetector,
    MarketPrice,
)
from datetime import datetime, timezone
import asyncio


async def test_kalshi_markets():
    """Test getting markets from Kalshi with broad search terms"""
    print("Testing Kalshi API...")

    kalshi = KalshiClient()

    try:
        markets = kalshi.get_markets(limit=10)
        print(f"✓ Kalshi: Retrieved {len(markets)} markets")

        # Show a few sample markets
        for i, market in enumerate(markets[:3], 1):
            title = market.get("title", "No title")[:60]
            ticker = market.get("ticker", "N/A")
            print(f"  {i}. {ticker}: {title}...")

        return markets

    except Exception as e:
        print(f"✗ Kalshi failed: {e}")
        return []


async def test_external_markets():
    """Test connecting to external prediction markets"""
    print("\nTesting external market APIs...")

    connector = ExternalMarketConnector()

    # Test search terms that might exist across platforms
    search_terms = ["election", "trump", "biden", "bitcoin", "crypto"]

    results = {}

    async with connector:
        # Test Polymarket
        try:
            poly_markets = await connector.get_polymarket_prices(search_terms)
            results["polymarket"] = poly_markets
            print(f"✓ Polymarket: Retrieved {len(poly_markets)} markets")
            for market in poly_markets[:2]:
                print(f"  - {market.market_title[:50]}... at ${market.yes_price:.2f}")
        except Exception as e:
            print(f"✗ Polymarket failed: {e}")
            results["polymarket"] = []

        # Test PredictIt
        try:
            predictit_markets = await connector.get_predictit_prices(search_terms)
            results["predictit"] = predictit_markets
            print(f"✓ PredictIt: Retrieved {len(predictit_markets)} markets")
            for market in predictit_markets[:2]:
                print(f"  - {market.market_title[:50]}... at ${market.yes_price:.2f}")
        except Exception as e:
            print(f"✗ PredictIt failed: {e}")
            results["predictit"] = []

        # Test Manifold
        try:
            manifold_markets = await connector.get_manifold_prices(search_terms)
            results["manifold"] = manifold_markets
            print(f"✓ Manifold: Retrieved {len(manifold_markets)} markets")
            for market in manifold_markets[:2]:
                print(f"  - {market.market_title[:50]}... at ${market.yes_price:.2f}")
        except Exception as e:
            print(f"✗ Manifold failed: {e}")
            results["manifold"] = []

    return results


async def test_arbitrage_detection(kalshi_markets, external_results):
    """Test arbitrage detection with real market data"""
    print("\nTesting arbitrage detection...")

    if not kalshi_markets:
        print("✗ No Kalshi markets to analyze")
        return []

    # Convert Kalshi markets to MarketPrice format
    kalshi_prices = []
    for market in kalshi_markets[:5]:  # Test with first 5 markets
        try:
            # Get market details for price
            kalshi_client = KalshiClient()
            ticker = market.get("ticker", "")
            market_detail = kalshi_client.get_market(ticker)

            # Extract price (handling the cents format)
            yes_ask = (
                market_detail.get("yes_ask", 50) / 100.0
            )  # Convert cents to dollars
            yes_bid = market_detail.get("yes_bid", 50) / 100.0
            yes_price = (yes_ask + yes_bid) / 2 if yes_ask > 0 or yes_bid > 0 else 0.5

            kalshi_prices.append(
                MarketPrice(
                    platform="kalshi",
                    market_id=ticker,
                    market_title=market.get("title", ""),
                    yes_price=yes_price,
                    no_price=1 - yes_price,
                    volume_24h=market_detail.get("volume_24h", 0),
                    last_updated=datetime.now(timezone.utc),
                    url=f"https://kalshi.com/markets/{ticker}",
                    liquidity=market_detail.get("open_interest", 0),
                )
            )

        except Exception as e:
            print(
                f"  Warning: Could not get price for {market.get('ticker', 'unknown')}: {e}"
            )
            continue

    print(f"✓ Converted {len(kalshi_prices)} Kalshi markets to price format")

    # Collect all external markets
    all_external = []
    for platform, markets in external_results.items():
        all_external.extend(markets)

    print(f"✓ Have {len(all_external)} external markets to compare")

    if kalshi_prices and all_external:
        # Test arbitrage detection
        detector = ArbitrageDetector()
        opportunities = detector.detect_direct_arbitrage(kalshi_prices, all_external)

        print(f"✓ Found {len(opportunities)} potential arbitrage opportunities")

        return opportunities
    else:
        print("✗ Not enough market data for arbitrage analysis")
        return []


async def main():
    """Run complete arbitrage test"""

    print("REAL CROSS-MARKET ARBITRAGE TEST")
    print("=" * 50)

    # Test 1: Get Kalshi markets
    kalshi_markets = await test_kalshi_markets()

    # Test 2: Get external markets
    external_results = await test_external_markets()

    # Test 3: Detect arbitrage
    opportunities = await test_arbitrage_detection(kalshi_markets, external_results)

    # Results
    print("\n" + "=" * 50)
    print("ARBITRAGE TEST RESULTS")
    print("=" * 50)

    if opportunities:
        print(f"🎯 FOUND {len(opportunities)} ARBITRAGE OPPORTUNITIES!")

        for i, opp in enumerate(opportunities[:3], 1):  # Show top 3
            print(f"\n{i}. {opp.opportunity_type} ARBITRAGE")
            print(
                f"   Primary: {opp.primary_market.platform} - ${opp.primary_market.yes_price:.2f}"
            )
            print(
                f"   Secondary: {opp.secondary_market.platform} - ${opp.secondary_market.yes_price:.2f}"
            )
            print(
                f"   Expected Profit: {opp.expected_profit:.3f} ({opp.expected_profit*100:.1f}%)"
            )
            print(f"   Risk Level: {opp.risk_level}")
            print(f"   Min Capital: ${opp.min_capital_required:,.0f}")

    else:
        print("No arbitrage opportunities detected")
        print("This could mean:")
        print("- Markets are efficiently priced")
        print("- No matching events found across platforms")
        print("- API connection issues")

    # Summary of what worked
    working_apis = []
    if kalshi_markets:
        working_apis.append("Kalshi")
    for platform, markets in external_results.items():
        if markets:
            working_apis.append(platform.title())

    print(f"\nWorking APIs: {', '.join(working_apis) if working_apis else 'None'}")
    print("Next step: Fix any API issues and improve market matching")


if __name__ == "__main__":
    asyncio.run(main())
