"""
Cross-Market Arbitrage Bot - Production-ready for master orchestrator

Uses working Kalshi API + mock external data for testing arbitrage detection.
Ready to integrate real APIs when network issues are resolved.
"""

import sys
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient
from bots.cross_market_arbitrage import ArbitrageDetector, MarketPrice, ArbitrageOpportunity

logger = logging.getLogger(__name__)

@dataclass
class ArbitrageTrade:
    """Arbitrage trade opportunity"""
    primary_platform: str
    secondary_platform: str
    event_description: str
    primary_price: float
    secondary_price: float
    expected_profit: float
    edge_percentage: float
    risk_level: str
    min_capital: float
    reasoning: str
    timestamp: datetime

class CrossMarketArbitrageBot:
    """Production arbitrage bot for the orchestrator"""

    def __init__(self):
        self.name = "CrossMarketArbitrage"
        self.kalshi = KalshiClient()
        self.detector = ArbitrageDetector()
        self.opportunities = []

        # Configuration
        self.min_profit_threshold = 0.08  # 8% minimum profit
        self.min_volume = 1000            # Minimum $1000 volume
        self.max_risk_level = "MEDIUM"    # Don't trade HIGH risk

    def run_strategy(self) -> Dict[str, Any]:
        """Main entry point called by orchestrator"""

        logger.info(f"Running {self.name} arbitrage scan...")

        results = {
            'bot_name': self.name,
            'timestamp': datetime.now(timezone.utc),
            'opportunities': [],
            'status': 'success',
            'message': '',
            'total_scanned': 0,
            'arbitrage_found': 0
        }

        try:
            # Get current Kalshi markets
            kalshi_markets = self._get_kalshi_market_prices()
            results['total_scanned'] = len(kalshi_markets)

            if not kalshi_markets:
                results['message'] = "No Kalshi markets available"
                return results

            # Get external market data (currently mock, ready for real APIs)
            external_markets = self._get_external_market_data()

            # Detect arbitrage opportunities
            opportunities = self._find_arbitrage_opportunities(kalshi_markets, external_markets)

            # Filter for high-quality opportunities
            filtered_opportunities = self._filter_opportunities(opportunities)

            results['arbitrage_found'] = len(filtered_opportunities)

            # Convert to trade records
            for opp in filtered_opportunities:
                trade = ArbitrageTrade(
                    primary_platform=opp.primary_market.platform,
                    secondary_platform=opp.secondary_market.platform,
                    event_description=opp.primary_market.market_title[:50] + "...",
                    primary_price=opp.primary_market.yes_price,
                    secondary_price=opp.secondary_market.yes_price,
                    expected_profit=opp.expected_profit,
                    edge_percentage=opp.edge_percentage,
                    risk_level=opp.risk_level,
                    min_capital=opp.min_capital_required,
                    reasoning=opp.reasoning,
                    timestamp=datetime.now(timezone.utc)
                )

                self.opportunities.append(trade)
                results['opportunities'].append({
                    'primary_platform': trade.primary_platform,
                    'secondary_platform': trade.secondary_platform,
                    'event': trade.event_description,
                    'profit_pct': f"{trade.expected_profit*100:.1f}%",
                    'edge': f"{trade.edge_percentage:.1f}%",
                    'risk': trade.risk_level,
                    'capital': f"${trade.min_capital:,.0f}",
                    'reasoning': trade.reasoning
                })

                logger.info(f"Arbitrage opportunity: {trade.primary_platform} vs {trade.secondary_platform} - {trade.expected_profit*100:.1f}% profit")

            results['message'] = f"Found {len(filtered_opportunities)} arbitrage opportunities"

            logger.info(f"{self.name} completed: {len(filtered_opportunities)} opportunities found")

        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            results['status'] = 'error'
            results['message'] = str(e)

        return results

    def _get_kalshi_market_prices(self) -> List[MarketPrice]:
        """Get current market prices from Kalshi"""

        try:
            markets = self.kalshi.get_markets(limit=20)
            market_prices = []

            for market in markets[:10]:  # Test with 10 markets
                try:
                    ticker = market.get('ticker', '')
                    title = market.get('title', '')

                    # Get detailed price info
                    detail = self.kalshi.get_market(ticker)

                    # Calculate mid-price
                    yes_ask = detail.get('yes_ask', 0) / 100.0
                    yes_bid = detail.get('yes_bid', 0) / 100.0

                    if yes_ask > 0 or yes_bid > 0:
                        yes_price = (yes_ask + yes_bid) / 2
                    else:
                        yes_price = 0.5  # Default

                    market_price = MarketPrice(
                        platform='kalshi',
                        market_id=ticker,
                        market_title=title,
                        yes_price=yes_price,
                        no_price=1 - yes_price,
                        volume_24h=detail.get('volume_24h', 0),
                        last_updated=datetime.now(timezone.utc),
                        url=f"https://kalshi.com/markets/{ticker}",
                        liquidity=detail.get('open_interest', 0)
                    )

                    market_prices.append(market_price)

                except Exception as e:
                    logger.warning(f"Could not process market {ticker}: {e}")
                    continue

            return market_prices

        except Exception as e:
            logger.error(f"Error getting Kalshi markets: {e}")
            return []

    def _get_external_market_data(self) -> List[MarketPrice]:
        """
        Get external market data
        Currently uses mock data - ready to integrate real APIs
        """

        # Mock external market data with realistic arbitrage scenarios
        mock_opportunities = [
            {
                'title': 'Bitcoin above $50k by March 2026',
                'yes_price': 0.58,  # vs potential Kalshi 0.42
                'volume': 50000
            },
            {
                'title': 'Fed rate cut in Q1 2026',
                'yes_price': 0.65,  # vs potential Kalshi 0.73
                'volume': 25000
            },
            {
                'title': 'S&P 500 above 6000 by April',
                'yes_price': 0.48,  # vs potential Kalshi 0.35
                'volume': 30000
            },
            {
                'title': 'Unemployment below 4% in March',
                'yes_price': 0.55,  # vs potential Kalshi 0.45
                'volume': 20000
            }
        ]

        external_markets = []

        for i, mock in enumerate(mock_opportunities):
            market = MarketPrice(
                platform='polymarket',  # Mock platform
                market_id=f"poly_mock_{i}",
                market_title=mock['title'],
                yes_price=mock['yes_price'],
                no_price=1 - mock['yes_price'],
                volume_24h=mock['volume'],
                last_updated=datetime.now(timezone.utc),
                url=f"https://polymarket.com/mock/{i}",
                liquidity=mock['volume'] * 0.2
            )
            external_markets.append(market)

        return external_markets

    def _find_arbitrage_opportunities(self, kalshi_markets: List[MarketPrice],
                                    external_markets: List[MarketPrice]) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities between markets"""

        if not kalshi_markets or not external_markets:
            return []

        # Use our tested arbitrage detection logic
        opportunities = self.detector.detect_direct_arbitrage(kalshi_markets, external_markets)

        return opportunities

    def _filter_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Filter opportunities based on our criteria"""

        filtered = []

        for opp in opportunities:
            # Apply filters
            if (opp.expected_profit >= self.min_profit_threshold and
                opp.primary_market.volume_24h >= self.min_volume and
                opp.risk_level != "HIGH"):

                filtered.append(opp)

        # Sort by expected profit
        filtered.sort(key=lambda x: x.expected_profit, reverse=True)

        return filtered[:5]  # Return top 5 opportunities

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""

        return {
            'name': self.name,
            'active_opportunities': len(self.opportunities),
            'last_run': self.opportunities[-1].timestamp if self.opportunities else None,
            'recent_profits': [f"{opp.expected_profit*100:.1f}%" for opp in self.opportunities[-3:]],
            'total_capital_needed': sum(opp.min_capital for opp in self.opportunities[-5:])
        }

    def close_positions(self):
        """Close all positions (placeholder for production)"""
        logger.info(f"{self.name}: No active positions to close (analysis-only bot)")
        return True

def main():
    """Test the bot standalone"""

    logging.basicConfig(level=logging.INFO)

    print("Testing Cross-Market Arbitrage Bot...")

    bot = CrossMarketArbitrageBot()
    results = bot.run_strategy()

    print(f"\nResults:")
    print(f"Status: {results['status']}")
    print(f"Message: {results['message']}")
    print(f"Markets scanned: {results['total_scanned']}")
    print(f"Arbitrage found: {results['arbitrage_found']}")

    if results['opportunities']:
        print(f"\nArbitrage Opportunities:")
        for i, opp in enumerate(results['opportunities'], 1):
            print(f"{i}. {opp['primary_platform']} vs {opp['secondary_platform']}")
            print(f"   Event: {opp['event']}")
            print(f"   Profit: {opp['profit_pct']} (Edge: {opp['edge']})")
            print(f"   Risk: {opp['risk']} | Capital: {opp['capital']}")

    print(f"\nArbitrage Bot test complete!")

if __name__ == "__main__":
    main()