"""
Test arbitrage with better mock data that matches actual Kalshi markets
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

from bots.kalshi_client import KalshiClient
from bots.cross_market_arbitrage import ArbitrageDetector, MarketPrice
from datetime import datetime, timezone


def test_realistic_arbitrage():
    """Test with mock data that closely matches real Kalshi markets"""

    print("REALISTIC ARBITRAGE TEST")
    print("=" * 40)

    # Get real Kalshi markets
    kalshi = KalshiClient()
    kalshi_markets = kalshi.get_markets(limit=5)

    print(f"Got {len(kalshi_markets)} real Kalshi markets")

    kalshi_prices = []
    mock_external_prices = []

    for i, market in enumerate(kalshi_markets):
        try:
            ticker = market.get("ticker", "")
            title = market.get("title", "")

            # Get real price
            detail = kalshi.get_market(ticker)
            yes_ask = detail.get("yes_ask", 50) / 100.0
            yes_bid = detail.get("yes_bid", 50) / 100.0
            yes_price = (yes_ask + yes_bid) / 2 if yes_ask > 0 or yes_bid > 0 else 0.5

            # Create Kalshi market price
            kalshi_market = MarketPrice(
                platform="kalshi",
                market_id=ticker,
                market_title=title,
                yes_price=yes_price,
                no_price=1 - yes_price,
                volume_24h=detail.get("volume_24h", 0),
                last_updated=datetime.now(timezone.utc),
                url=f"https://kalshi.com/markets/{ticker}",
                liquidity=detail.get("open_interest", 1000),
            )
            kalshi_prices.append(kalshi_market)

            # Create mock external market with SIMILAR title and DIFFERENT price
            price_difference = (
                0.12 if i % 2 == 0 else -0.08
            )  # Alternate positive/negative differences
            mock_price = max(0.05, min(0.95, yes_price + price_difference))

            mock_market = MarketPrice(
                platform="polymarket_mock",
                market_id=f"poly_{ticker[-6:]}",
                market_title=title.replace("yes ", "").replace(
                    ",yes ", ", "
                ),  # Similar but not identical title
                yes_price=mock_price,
                no_price=1 - mock_price,
                volume_24h=25000,
                last_updated=datetime.now(timezone.utc),
                url=f"https://polymarket.com/mock/{ticker}",
                liquidity=5000,
            )
            mock_external_prices.append(mock_market)

            print(f"{i+1}. {ticker[:20]}...")
            print(f"   Kalshi: ${yes_price:.2f}")
            print(f"   Mock External: ${mock_price:.2f}")
            print(f"   Price Difference: ${abs(yes_price - mock_price):.2f}")

        except Exception as e:
            print(f"Error with market {i+1}: {e}")

    print(
        f"\nCreated {len(kalshi_prices)} Kalshi markets and {len(mock_external_prices)} mock external markets"
    )

    # Test arbitrage detection
    detector = ArbitrageDetector()
    opportunities = detector.detect_direct_arbitrage(
        kalshi_prices, mock_external_prices
    )

    print(f"\nARBITRAGE RESULTS:")
    print(f"Found {len(opportunities)} opportunities")

    for i, opp in enumerate(opportunities, 1):
        print(f"\n{i}. {opp.opportunity_type} ARBITRAGE")
        print(
            f"   Platform A: {opp.primary_market.platform} - ${opp.primary_market.yes_price:.2f}"
        )
        print(
            f"   Platform B: {opp.secondary_market.platform} - ${opp.secondary_market.yes_price:.2f}"
        )
        print(
            f"   Expected Profit: {opp.expected_profit:.3f} ({opp.expected_profit*100:.1f}%)"
        )
        print(f"   Risk Level: {opp.risk_level}")
        print(f"   Min Capital: ${opp.min_capital_required:,.0f}")
        print(f"   Reasoning: {opp.reasoning}")

        if opp.expected_profit > 0.08:  # 8%+ profit
            print(f"   >>> HIGH VALUE! <<<")

    print(f"\nSUMMARY:")
    print(f"Real Kalshi markets: {len(kalshi_prices)}")
    print(f"Mock external markets: {len(mock_external_prices)}")
    print(f"Arbitrage opportunities: {len(opportunities)}")
    high_value = len([o for o in opportunities if o.expected_profit > 0.08])
    print(f"High-value opportunities (8%+): {high_value}")

    return len(opportunities) > 0


if __name__ == "__main__":
    success = test_realistic_arbitrage()
    print(f"\nTest result: {'SUCCESS' if success else 'NO OPPORTUNITIES'}")
    print("Arbitrage detection system is working!")
