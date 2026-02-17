"""
Climate Edge Finder

Matches NOAA temperature data to Kalshi climate markets.
Focuses on temperature record markets.

Markets covered:
- Hottest month records (KXHOTMONTH*)
- Hottest year (KXHOTYEAR*)
- Global temperature anomaly thresholds

Author: Trading Bot
Created: January 2026
"""

import os
import sys
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient
from scrapers.climate_scraper import ClimateScraper, ClimateEstimate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ClimateEdgeFinder')


@dataclass
class ClimateOpportunity:
    """Climate market trading opportunity"""
    ticker: str
    title: str
    metric: str  # hottest_month, hottest_year, temperature_threshold
    threshold: Optional[float]
    period: str  # e.g., "January 2026", "2026"
    side: str  # YES or NO
    our_probability: float
    market_price: float
    edge: float
    confidence: str
    reasoning: str


class ClimateEdgeFinder:
    """
    Finds edge in Kalshi climate/temperature markets.

    Uses NOAA data for temperature record predictions.
    """

    MIN_EDGE = 0.07  # 7% minimum edge
    MIN_PRICE = 0.05
    MAX_PRICE = 0.95

    # Kalshi climate series tickers
    CLIMATE_SERIES = [
        'KXHOTMONTH',      # Hottest month on record
        'KXHOTYEAR',       # Hottest year on record
        'KXGLOBALTEMP',    # Global temperature threshold
        'KXCLIMATE',       # General climate markets
        'KXWARMEST',       # Warmest temperature markets
        'KXTEMPERATURE',   # Temperature markets
        'KXRECORD',        # Record temperature markets
    ]

    TRADEABLE_STATUSES = ['active', 'open', 'trading']

    # Month name mapping
    MONTH_NAMES = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
        'oct': 10, 'nov': 11, 'dec': 12
    }

    def __init__(self):
        """Initialize the climate edge finder"""
        self.client = KalshiClient()
        self.scraper = ClimateScraper()

        logger.info("ClimateEdgeFinder initialized")

    def find_climate_markets(self) -> List[Dict]:
        """Find all active climate markets on Kalshi"""
        markets = []

        for series_ticker in self.CLIMATE_SERIES:
            try:
                data = self.client._request('GET', '/markets', params={
                    'series_ticker': series_ticker,
                    'limit': 100
                })
                series_markets = data.get('markets', [])

                for market in series_markets:
                    status = market.get('status', '').lower()
                    if status in self.TRADEABLE_STATUSES:
                        markets.append(market)

            except Exception as e:
                logger.debug(f"Error fetching {series_ticker}: {e}")

        # Also try searching by keyword
        try:
            for keyword in ['hottest', 'warmest', 'temperature', 'climate']:
                data = self.client._request('GET', '/markets', params={
                    'status': 'active',
                    'limit': 100
                })
                for market in data.get('markets', []):
                    title = market.get('title', '').lower()
                    if keyword in title and market not in markets:
                        markets.append(market)

        except Exception as e:
            logger.debug(f"Error in keyword search: {e}")

        logger.info(f"Found {len(markets)} climate markets")
        return markets

    def parse_climate_market(self, market: Dict) -> Optional[Dict]:
        """
        Parse climate market to extract metric, threshold, period.

        Examples:
        - "Will January 2026 be the hottest January on record?"
        - "Will 2026 be the hottest year on record?"
        - "Will global temperature exceed 1.5°C above pre-industrial?"

        Args:
            market: Kalshi market dict

        Returns:
            Dict with metric, threshold, period or None
        """
        title = market.get('title', '').lower()
        ticker = market.get('ticker', '').upper()

        result = {}

        # Pattern 1: Hottest month on record
        # "Will January 2026 be the hottest January on record?"
        month_pattern = r'(\w+)\s+(\d{4}).*hottest.*\1.*record'
        match = re.search(month_pattern, title)
        if match:
            month_name = match.group(1).lower()
            year = int(match.group(2))
            month_num = self.MONTH_NAMES.get(month_name)

            if month_num:
                return {
                    'metric': 'hottest_month',
                    'month': month_num,
                    'year': year,
                    'period': f"{match.group(1).title()} {year}"
                }

        # Pattern 2: Hottest year on record
        # "Will 2026 be the hottest year on record?"
        year_pattern = r'(\d{4}).*hottest.*year.*record'
        match = re.search(year_pattern, title)
        if match:
            year = int(match.group(1))
            return {
                'metric': 'hottest_year',
                'year': year,
                'period': str(year)
            }

        # Pattern 3: Temperature threshold
        # "Will global temperature exceed X°C?"
        temp_pattern = r'(\d+\.?\d*)\s*°?c'
        match = re.search(temp_pattern, title)
        if match:
            threshold = float(match.group(1))
            return {
                'metric': 'temperature_threshold',
                'threshold': threshold,
                'period': 'Annual'
            }

        # Pattern 4: Warmest/record breaking
        if 'warmest' in title or 'record' in title:
            # Try to extract year
            year_match = re.search(r'(\d{4})', title)
            if year_match:
                year = int(year_match.group(1))
                if 'month' in title:
                    return {
                        'metric': 'hottest_month',
                        'year': year,
                        'period': str(year)
                    }
                else:
                    return {
                        'metric': 'hottest_year',
                        'year': year,
                        'period': str(year)
                    }

        return None

    def get_market_price(self, market: Dict) -> Optional[float]:
        """Get current market price (YES probability)"""
        try:
            ticker = market.get('ticker')
            orderbook = self.client.get_orderbook(ticker)

            if not orderbook:
                return None

            yes_orders = orderbook.get('yes', [])
            no_orders = orderbook.get('no', [])

            best_yes_bid = 0
            if yes_orders:
                prices = [order[0] for order in yes_orders if len(order) >= 2 and order[1] > 0]
                if prices:
                    best_yes_bid = max(prices)

            best_no_bid = 0
            if no_orders:
                prices = [order[0] for order in no_orders if len(order) >= 2 and order[1] > 0]
                if prices:
                    best_no_bid = max(prices)

            yes_ask = 100 - best_no_bid if best_no_bid > 0 else 100

            if best_yes_bid > 0:
                midpoint = (best_yes_bid + yes_ask) / 200
                return midpoint
            elif yes_ask < 100:
                return yes_ask / 100

            if market.get('last_price'):
                return market.get('last_price') / 100

            return None

        except Exception as e:
            logger.debug(f"Error getting price for {market.get('ticker')}: {e}")
            return None

    def find_opportunities(self) -> List[ClimateOpportunity]:
        """
        Find climate trading opportunities.

        Returns:
            List of ClimateOpportunity with positive edge
        """
        opportunities = []

        # Get climate summary
        logger.info("Fetching climate data...")
        summary = self.scraper.get_climate_summary()
        logger.info(f"  Current month anomaly: {summary['current_month']['anomaly']}°C")
        logger.info(f"  P(monthly record): {summary['probabilities']['monthly_record']:.0%}")
        logger.info(f"  P(hottest year): {summary['probabilities']['hottest_year']:.0%}")

        # Find markets
        markets = self.find_climate_markets()

        # Process each market
        for market in markets:
            opp = self._evaluate_climate_market(market)
            if opp:
                opportunities.append(opp)

        # Sort by edge
        opportunities.sort(key=lambda x: x.edge, reverse=True)
        logger.info(f"Found {len(opportunities)} climate opportunities with edge >= {self.MIN_EDGE:.0%}")

        return opportunities

    def _evaluate_climate_market(self, market: Dict) -> Optional[ClimateOpportunity]:
        """Evaluate a climate market"""
        # Parse market
        parsed = self.parse_climate_market(market)
        if not parsed:
            return None

        metric = parsed['metric']
        period = parsed['period']

        # Calculate our probability
        estimate = None

        if metric == 'hottest_month':
            year = parsed.get('year', datetime.now().year)
            month = parsed.get('month', datetime.now().month)
            estimate = self.scraper.calculate_record_probability(year, month, 'record')

        elif metric == 'hottest_year':
            year = parsed.get('year', datetime.now().year)
            estimate = self.scraper.calculate_annual_record_probability(year)

        elif metric == 'temperature_threshold':
            threshold = parsed.get('threshold', 1.5)
            # For threshold markets, calculate probability based on current trajectory
            annual = self.scraper.fetch_annual_temperature(datetime.now().year)
            if annual:
                # Simple threshold probability
                if annual.anomaly >= threshold:
                    our_prob = 0.90
                elif annual.anomaly >= threshold - 0.10:
                    our_prob = 0.70
                else:
                    our_prob = 0.30

                estimate = ClimateEstimate(
                    metric='temperature_threshold',
                    threshold=threshold,
                    probability=our_prob,
                    confidence='MEDIUM',
                    reasoning=f"Current anomaly: {annual.anomaly}°C, Threshold: {threshold}°C"
                )

        if not estimate:
            return None

        our_prob = estimate.probability

        # Get market price
        market_price = self.get_market_price(market)
        if market_price is None or market_price < self.MIN_PRICE or market_price > self.MAX_PRICE:
            return None

        # Calculate edge
        edge = our_prob - market_price

        if abs(edge) >= self.MIN_EDGE:
            if edge > 0:
                side = 'YES'
            else:
                side = 'NO'
                edge = abs(edge)
                our_prob = 1 - our_prob
                market_price = 1 - market_price

            return ClimateOpportunity(
                ticker=market.get('ticker', ''),
                title=market.get('title', ''),
                metric=metric,
                threshold=estimate.threshold,
                period=period,
                side=side,
                our_probability=our_prob,
                market_price=market_price,
                edge=edge,
                confidence=estimate.confidence,
                reasoning=estimate.reasoning
            )

        return None

    def print_opportunities(self, opportunities: List[ClimateOpportunity]):
        """Print opportunities in formatted output"""
        print()
        print("=" * 70)
        print("CLIMATE OPPORTUNITIES")
        print("=" * 70)

        if not opportunities:
            print("No opportunities found.")
            return

        for i, opp in enumerate(opportunities[:15], 1):
            print(f"\n{i}. [{opp.metric.upper()}] {opp.ticker}")
            print(f"   {opp.title}")
            print(f"   Period: {opp.period}")
            print(f"   Action: BUY {opp.side} @ {opp.market_price:.0%}")
            print(f"   Our Probability: {opp.our_probability:.0%}")
            print(f"   Edge: {opp.edge:.1%} [{opp.confidence}]")
            print(f"   Source: {opp.reasoning}")

        print()
        print("=" * 70)


def main():
    """Run the climate edge finder"""
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("CLIMATE EDGE FINDER")
    print("=" * 60)

    finder = ClimateEdgeFinder()

    print("\n[1] Scanning for opportunities...")
    opportunities = finder.find_opportunities()

    print("\n[2] Results:")
    finder.print_opportunities(opportunities)

    print("\n[3] Summary:")
    print("-" * 40)
    print(f"Total opportunities: {len(opportunities)}")

    if opportunities:
        by_metric = {}
        for opp in opportunities:
            by_metric[opp.metric] = by_metric.get(opp.metric, 0) + 1

        for metric, count in sorted(by_metric.items()):
            print(f"  {metric}: {count}")

        avg_edge = sum(o.edge for o in opportunities) / len(opportunities)
        print(f"\nAverage edge: {avg_edge:.1%}")


if __name__ == "__main__":
    main()
