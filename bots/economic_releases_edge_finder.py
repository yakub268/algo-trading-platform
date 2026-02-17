"""
Economic Releases Edge Finder

Matches Fed nowcast estimates to Kalshi economic data contracts.
Finds edge based on Cleveland Fed CPI Nowcast, Atlanta Fed GDPNow, etc.

These are HIGH CONFIDENCE signals because:
- Cleveland Fed CPI Nowcast is highly accurate (within 0.1-0.2%)
- Atlanta Fed GDPNow updates daily and tracks well

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
from scrapers.economic_releases_scraper import (
    EconomicReleasesScraper,
    CPINowcast,
    GDPNowcast,
    JobsNowcast
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('EconomicReleasesEdgeFinder')


@dataclass
class EconomicReleaseOpportunity:
    """Economic release trading opportunity"""
    ticker: str
    title: str
    indicator: str  # CPI, GDP, NFP, etc.
    side: str  # YES or NO
    our_probability: float
    market_price: float
    edge: float
    nowcast_value: float
    threshold: float
    direction: str  # above or below
    source: str
    confidence: str  # HIGH, MEDIUM, LOW
    reasoning: str


class EconomicReleasesEdgeFinder:
    """
    Finds edge in Kalshi economic release markets.

    Uses Fed nowcasts (CPI, GDP) which are highly accurate.
    """

    MIN_EDGE = 0.05  # 5% minimum edge
    MIN_PRICE = 0.05
    MAX_PRICE = 0.95

    # Kalshi economic series tickers
    ECONOMIC_SERIES = {
        'CPI': ['KXCPI', 'KXCPIYOY', 'KXCPICORE', 'CPICOREYOY', 'LCPIYOY', 'LCPIMAX', 'KXACPICORE'],
        'GDP': ['KXGDP', 'KXGDPNOW', 'KXGDPGROWTH'],
        'NFP': ['KXJOBS', 'KXNFP', 'KXUNEMPLOYMENT', 'KXEMPLOYMENT'],
        'RETAIL': ['KXRETAIL', 'KXRETAILSALES'],
    }

    TRADEABLE_STATUSES = ['active', 'open', 'trading']

    def __init__(self):
        """Initialize the economic releases edge finder"""
        self.client = KalshiClient()
        self.scraper = EconomicReleasesScraper()

        logger.info("EconomicReleasesEdgeFinder initialized")

    def find_economic_markets(self) -> Dict[str, List[Dict]]:
        """Find all active economic release markets on Kalshi"""
        markets_by_type = {
            'CPI': [],
            'GDP': [],
            'NFP': [],
            'RETAIL': [],
        }

        for indicator, series_list in self.ECONOMIC_SERIES.items():
            for series_ticker in series_list:
                try:
                    data = self.client._request('GET', '/markets', params={
                        'series_ticker': series_ticker,
                        'limit': 100
                    })
                    markets = data.get('markets', [])

                    for market in markets:
                        status = market.get('status', '').lower()
                        if status in self.TRADEABLE_STATUSES:
                            market['indicator'] = indicator
                            markets_by_type[indicator].append(market)

                except Exception as e:
                    logger.debug(f"Error fetching {series_ticker}: {e}")

        total = sum(len(m) for m in markets_by_type.values())
        logger.info(f"Found {total} economic markets (CPI: {len(markets_by_type['CPI'])}, "
                   f"GDP: {len(markets_by_type['GDP'])}, NFP: {len(markets_by_type['NFP'])})")

        return markets_by_type

    def parse_market_threshold(self, market: Dict) -> Optional[Tuple[float, str]]:
        """
        Parse threshold and direction from market title/ticker.

        Examples:
        - KXCPI-26FEB-A30 -> (3.0, 'above')
        - KXGDP-Q1-B20 -> (2.0, 'below')
        - "Will CPI be above 3%?" -> (3.0, 'above')

        Args:
            market: Kalshi market dict

        Returns:
            Tuple of (threshold, direction) or None
        """
        ticker = market.get('ticker', '').upper()
        title = market.get('title', '').lower()

        # Pattern 1: Ticker format KXCPI-DATE-A30 (A=above, B=below)
        ticker_pattern = r'-([AB])(\d+\.?\d*)'
        match = re.search(ticker_pattern, ticker)

        if match:
            direction = 'above' if match.group(1) == 'A' else 'below'
            threshold_str = match.group(2)

            # Handle different formats: A30 = 3.0%, A25 = 2.5%
            threshold = float(threshold_str)
            if threshold >= 10:
                threshold = threshold / 10  # 30 -> 3.0

            return (threshold, direction)

        # Pattern 2: Title "above X%" or "below X%"
        title_above = re.search(r'above\s+(\d+\.?\d*)\s*%?', title)
        if title_above:
            return (float(title_above.group(1)), 'above')

        title_below = re.search(r'below\s+(\d+\.?\d*)\s*%?', title)
        if title_below:
            return (float(title_below.group(1)), 'below')

        # Pattern 3: "X% or higher/lower"
        higher = re.search(r'(\d+\.?\d*)\s*%?\s+or\s+(?:higher|more|greater)', title)
        if higher:
            return (float(higher.group(1)), 'above')

        lower = re.search(r'(\d+\.?\d*)\s*%?\s+or\s+(?:lower|less|fewer)', title)
        if lower:
            return (float(lower.group(1)), 'below')

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

            # Get best bids
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

            # Calculate midpoint
            yes_ask = 100 - best_no_bid if best_no_bid > 0 else 100

            if best_yes_bid > 0:
                midpoint = (best_yes_bid + yes_ask) / 200
                return midpoint
            elif yes_ask < 100:
                return yes_ask / 100

            # Fallback to last price if available
            if market.get('last_price'):
                return market.get('last_price') / 100

            return None

        except Exception as e:
            logger.debug(f"Error getting price for {market.get('ticker')}: {e}")
            return None

    def find_opportunities(self) -> List[EconomicReleaseOpportunity]:
        """
        Find economic release trading opportunities.

        Returns:
            List of EconomicReleaseOpportunity with positive edge
        """
        opportunities = []

        # Fetch nowcasts
        logger.info("Fetching economic nowcasts...")
        cpi = self.scraper.fetch_cleveland_fed_cpi()
        gdp = self.scraper.fetch_atlanta_fed_gdp()
        jobs = self.scraper.fetch_jobless_claims()

        if cpi:
            logger.info(f"  CPI Nowcast: {cpi.headline_cpi:.2f}% (source: {cpi.source})")
        if gdp:
            logger.info(f"  GDP Nowcast: {gdp.gdp_estimate:.1f}% (source: {gdp.source})")
        if jobs:
            logger.info(f"  Jobs: Claims={jobs.weekly_claims}, Trend={jobs.claims_trend}")

        # Find markets
        markets_by_type = self.find_economic_markets()

        # Process CPI markets
        if cpi:
            for market in markets_by_type.get('CPI', []):
                opp = self._evaluate_cpi_market(market, cpi)
                if opp:
                    opportunities.append(opp)

        # Process GDP markets
        if gdp:
            for market in markets_by_type.get('GDP', []):
                opp = self._evaluate_gdp_market(market, gdp)
                if opp:
                    opportunities.append(opp)

        # Process NFP/Jobs markets
        if jobs:
            for market in markets_by_type.get('NFP', []):
                opp = self._evaluate_jobs_market(market, jobs)
                if opp:
                    opportunities.append(opp)

        # Sort by edge
        opportunities.sort(key=lambda x: x.edge, reverse=True)
        logger.info(f"Found {len(opportunities)} economic release opportunities with edge >= {self.MIN_EDGE:.0%}")

        return opportunities

    def _evaluate_cpi_market(
        self,
        market: Dict,
        cpi: CPINowcast
    ) -> Optional[EconomicReleaseOpportunity]:
        """Evaluate a CPI market against nowcast"""
        parsed = self.parse_market_threshold(market)
        if not parsed:
            return None

        threshold, direction = parsed

        # Get market price
        market_price = self.get_market_price(market)
        if market_price is None or market_price < self.MIN_PRICE or market_price > self.MAX_PRICE:
            return None

        # Calculate our probability
        our_prob, reasoning = self.scraper.calculate_cpi_threshold_probability(
            cpi, threshold, direction
        )

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

            return EconomicReleaseOpportunity(
                ticker=market.get('ticker', ''),
                title=market.get('title', ''),
                indicator='CPI',
                side=side,
                our_probability=our_prob,
                market_price=market_price,
                edge=edge,
                nowcast_value=cpi.headline_cpi,
                threshold=threshold,
                direction=direction,
                source=cpi.source,
                confidence='HIGH',
                reasoning=reasoning
            )

        return None

    def _evaluate_gdp_market(
        self,
        market: Dict,
        gdp: GDPNowcast
    ) -> Optional[EconomicReleaseOpportunity]:
        """Evaluate a GDP market against nowcast"""
        parsed = self.parse_market_threshold(market)
        if not parsed:
            return None

        threshold, direction = parsed

        market_price = self.get_market_price(market)
        if market_price is None or market_price < self.MIN_PRICE or market_price > self.MAX_PRICE:
            return None

        our_prob, reasoning = self.scraper.calculate_gdp_threshold_probability(
            gdp, threshold, direction
        )

        edge = our_prob - market_price

        if abs(edge) >= self.MIN_EDGE:
            if edge > 0:
                side = 'YES'
            else:
                side = 'NO'
                edge = abs(edge)
                our_prob = 1 - our_prob
                market_price = 1 - market_price

            return EconomicReleaseOpportunity(
                ticker=market.get('ticker', ''),
                title=market.get('title', ''),
                indicator='GDP',
                side=side,
                our_probability=our_prob,
                market_price=market_price,
                edge=edge,
                nowcast_value=gdp.gdp_estimate,
                threshold=threshold,
                direction=direction,
                source=gdp.source,
                confidence='HIGH',
                reasoning=reasoning
            )

        return None

    def _evaluate_jobs_market(
        self,
        market: Dict,
        jobs: JobsNowcast
    ) -> Optional[EconomicReleaseOpportunity]:
        """Evaluate a jobs/NFP market against claims data"""
        parsed = self.parse_market_threshold(market)
        if not parsed:
            return None

        threshold, direction = parsed

        market_price = self.get_market_price(market)
        if market_price is None or market_price < self.MIN_PRICE or market_price > self.MAX_PRICE:
            return None

        # Estimate NFP from claims trend
        nfp_base = 180
        if jobs.claims_trend == 'rising':
            nfp_estimate = nfp_base - 30
        elif jobs.claims_trend == 'falling':
            nfp_estimate = nfp_base + 30
        else:
            nfp_estimate = nfp_base

        # Calculate probability (higher uncertainty for jobs)
        import math
        std_error = 50  # NFP has more variance

        z = (threshold - nfp_estimate) / std_error
        prob_below = 0.5 * (1 + math.erf(z / math.sqrt(2)))

        if direction == 'above':
            our_prob = 1 - prob_below
        else:
            our_prob = prob_below

        our_prob = max(0.05, min(0.95, our_prob))

        edge = our_prob - market_price

        if abs(edge) >= self.MIN_EDGE:
            if edge > 0:
                side = 'YES'
            else:
                side = 'NO'
                edge = abs(edge)
                our_prob = 1 - our_prob
                market_price = 1 - market_price

            return EconomicReleaseOpportunity(
                ticker=market.get('ticker', ''),
                title=market.get('title', ''),
                indicator='NFP',
                side=side,
                our_probability=our_prob,
                market_price=market_price,
                edge=edge,
                nowcast_value=nfp_estimate,
                threshold=threshold,
                direction=direction,
                source=f"Claims-based: {jobs.weekly_claims:,}",
                confidence='MEDIUM',  # Lower confidence for indirect indicator
                reasoning=f"Weekly claims: {jobs.weekly_claims:,}, trend: {jobs.claims_trend}"
            )

        return None

    def print_opportunities(self, opportunities: List[EconomicReleaseOpportunity]):
        """Print opportunities in formatted output"""
        print()
        print("=" * 70)
        print("ECONOMIC RELEASE OPPORTUNITIES")
        print("=" * 70)

        if not opportunities:
            print("No opportunities found.")
            return

        for i, opp in enumerate(opportunities[:15], 1):
            print(f"\n{i}. [{opp.indicator}] {opp.ticker}")
            print(f"   {opp.title}")
            print(f"   Threshold: {opp.threshold}% {opp.direction}")
            print(f"   Action: BUY {opp.side} @ {opp.market_price:.0%}")
            print(f"   Our Probability: {opp.our_probability:.0%} (Nowcast: {opp.nowcast_value})")
            print(f"   Edge: {opp.edge:.1%} [{opp.confidence}]")
            print(f"   Source: {opp.source}")

        print()
        print("=" * 70)


def main():
    """Run the economic releases edge finder"""
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("ECONOMIC RELEASES EDGE FINDER")
    print("=" * 60)

    finder = EconomicReleasesEdgeFinder()

    print("\n[1] Scanning for opportunities...")
    opportunities = finder.find_opportunities()

    print("\n[2] Results:")
    finder.print_opportunities(opportunities)

    print("\n[3] Summary:")
    print("-" * 40)
    print(f"Total opportunities: {len(opportunities)}")

    if opportunities:
        by_indicator = {}
        for opp in opportunities:
            by_indicator[opp.indicator] = by_indicator.get(opp.indicator, 0) + 1

        for ind, count in sorted(by_indicator.items()):
            print(f"  {ind}: {count}")

        high_conf = [o for o in opportunities if o.confidence == 'HIGH']
        print(f"\nHigh confidence: {len(high_conf)}")

        if opportunities:
            avg_edge = sum(o.edge for o in opportunities) / len(opportunities)
            print(f"Average edge: {avg_edge:.1%}")
            print(f"Best opportunity: {opportunities[0].ticker} ({opportunities[0].edge:.1%} edge)")


if __name__ == "__main__":
    main()
