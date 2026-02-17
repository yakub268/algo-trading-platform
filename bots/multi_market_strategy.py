"""
Multi-Market Strategy

Combines scraped data with Kalshi prices to find edge across all market types.
Prioritizes opportunities by: edge size, liquidity, time to expiry.

Supported markets:
- Weather (temperature, precipitation)
- Economic (CPI, unemployment, NFP)
- Economic Releases (CPI Nowcast, GDPNow, Jobs)
- Crypto (BTC/ETH price levels with technicals)
- Earnings (beat/miss predictions)
- Fed (rate decisions)
- Sports (NFL/NBA/MLB game outcomes)
- Sports Props (team totals)
- Awards (Oscars, Golden Globes, Emmys)
- Climate (temperature records)
- Box Office (opening weekend predictions)

Author: Trading Bot
Created: January 2026
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MultiMarketStrategy')

try:
    from bots.kalshi_client import KalshiClient
    from bots.market_scanner import MarketScanner, MarketOpportunity
    from scrapers.data_aggregator import DataAggregator
    MULTI_MARKET_DEPS_AVAILABLE = True
except ImportError as e:
    MULTI_MARKET_DEPS_AVAILABLE = False
    logger.warning(f"Multi-market dependencies not available: {e}")


@dataclass
class RankedOpportunity:
    """Opportunity with ranking score"""
    ticker: str
    title: str
    category: str
    side: str  # YES or NO
    our_probability: float
    market_price: float
    edge: float
    data_source: str
    expiration: Optional[str]
    volume: int
    liquidity_score: float
    time_to_expiry_hours: Optional[float]
    overall_score: float  # Combined ranking score
    reasoning: str
    timestamp: datetime


class MultiMarketStrategy:
    """
    Multi-market trading strategy for Kalshi.

    Scans all market categories, matches with scraped probability estimates,
    and ranks opportunities by a combined score.
    """

    # Minimum requirements
    MIN_EDGE = 0.05  # 5% minimum edge
    MIN_VOLUME = 10  # Minimum volume
    MAX_PRICE = 0.95  # Max 95 cents
    MIN_PRICE = 0.05  # Min 5 cents

    # Position sizing
    MAX_POSITION_USD = 20.0
    MAX_CONCURRENT_POSITIONS = 5

    # Scoring weights
    WEIGHT_EDGE = 0.4
    WEIGHT_LIQUIDITY = 0.2
    WEIGHT_TIME = 0.2
    WEIGHT_CONFIDENCE = 0.2

    def __init__(self, paper_mode: bool = True):
        """Initialize the multi-market strategy"""
        self.paper_mode = paper_mode
        self._deps_available = MULTI_MARKET_DEPS_AVAILABLE

        if self._deps_available:
            self.client = KalshiClient()
            self.scanner = MarketScanner(client=self.client)
            self.aggregator = DataAggregator()
        else:
            self.client = None
            self.scanner = None
            self.aggregator = None
            logger.warning("MultiMarketStrategy running in degraded mode (missing dependencies)")

        logger.info(f"MultiMarketStrategy initialized (paper_mode={paper_mode})")

    def calculate_time_to_expiry(self, expiration: Optional[str]) -> Optional[float]:
        """Calculate hours until market expires"""
        if not expiration:
            return None

        try:
            exp_time = datetime.fromisoformat(expiration.replace('Z', '+00:00'))
            delta = exp_time - datetime.now(timezone.utc)
            return delta.total_seconds() / 3600
        except Exception as e:
            logger.debug(f"Error parsing expiration time: {e}")
            return None

    def calculate_overall_score(
        self,
        edge: float,
        liquidity_score: float,
        time_to_expiry_hours: Optional[float],
        confidence: str = 'MEDIUM'
    ) -> float:
        """
        Calculate overall opportunity score.

        Higher score = better opportunity.

        Args:
            edge: Edge as decimal (0-1)
            liquidity_score: Liquidity score (0-1)
            time_to_expiry_hours: Hours until expiry
            confidence: Confidence level (HIGH, MEDIUM, LOW)

        Returns:
            Overall score (0-1)
        """
        # Normalize edge (cap at 50% for scoring)
        edge_score = min(edge / 0.5, 1.0)

        # Time score: prefer 12-72 hours to expiry
        time_score = 0.5
        if time_to_expiry_hours is not None:
            if 12 <= time_to_expiry_hours <= 72:
                time_score = 1.0
            elif time_to_expiry_hours < 12:
                time_score = time_to_expiry_hours / 12
            elif time_to_expiry_hours > 72:
                time_score = max(0.3, 1 - (time_to_expiry_hours - 72) / 168)

        # Confidence score
        confidence_scores = {'HIGH': 1.0, 'MEDIUM': 0.7, 'LOW': 0.4}
        conf_score = confidence_scores.get(confidence, 0.5)

        # Weighted average
        score = (
            self.WEIGHT_EDGE * edge_score +
            self.WEIGHT_LIQUIDITY * liquidity_score +
            self.WEIGHT_TIME * time_score +
            self.WEIGHT_CONFIDENCE * conf_score
        )

        return score

    def match_market_to_estimate(
        self,
        market: Dict,
        estimates: List[Dict]
    ) -> Optional[Dict]:
        """
        Match a Kalshi market to a probability estimate.

        Args:
            market: Kalshi market dict
            estimates: List of probability estimates

        Returns:
            Matched estimate or None
        """
        ticker = market.get('ticker', '').upper()
        title = market.get('title', '').lower()

        for est in estimates:
            pattern = est.get('ticker_pattern', '').upper()

            # Pattern matching
            if '*' in pattern:
                # Wildcard pattern
                prefix = pattern.split('*')[0]
                if prefix and prefix in ticker:
                    return est
            elif pattern in ticker:
                return est

            # Title-based matching for weather
            if est.get('category') == 'weather':
                city = est.get('city', '').lower()
                if city and city in title:
                    # Check date and threshold
                    date = est.get('date', '')
                    if date:
                        date_parts = date.split('-')
                        if len(date_parts) == 3:
                            day = date_parts[2]
                            if day in ticker:
                                return est

        return None

    def find_opportunities(self) -> List[Dict]:
        """
        Entry point for the master orchestrator (matches method_priority list).

        Returns:
            List of opportunity dicts sorted by overall score.
        """
        if not self._deps_available:
            return [{'status': 'error', 'error': 'Dependencies not available (kalshi_client, market_scanner, data_aggregator)'}]

        ranked = self.scan_all_markets()
        # Convert dataclass list to dicts for orchestrator serialization
        return [asdict(opp) for opp in ranked[:10]]

    def scan_all_markets(self) -> List[RankedOpportunity]:
        """
        Scan all markets for opportunities.

        Returns:
            List of RankedOpportunity sorted by overall score
        """
        logger.info("=" * 50)
        logger.info("MULTI-MARKET SCAN")
        logger.info("=" * 50)

        # Fetch aggregated data
        logger.info("[1] Fetching probability estimates...")
        aggregated = self.aggregator.fetch_all(use_cache=True)
        all_estimates = self.aggregator.get_all_estimates_flat()
        logger.info(f"    Got {len(all_estimates)} probability estimates")

        # Get all markets by category
        logger.info("[2] Scanning Kalshi markets...")
        opportunities = []

        # Weather markets
        weather_markets = self.scanner.find_weather_markets()
        logger.info(f"    Weather: {len(weather_markets)} markets")

        # Process weather markets with weather edge finder
        from bots.weather_edge_finder import WeatherEdgeFinder
        weather_finder = WeatherEdgeFinder()
        weather_opps = weather_finder.find_opportunities()

        for opp in weather_opps:
            time_to_exp = self.calculate_time_to_expiry(None)  # Weather markets expire daily
            score = self.calculate_overall_score(
                opp.edge, 0.5, 24, 'HIGH'  # Weather has good confidence
            )

            opportunities.append(RankedOpportunity(
                ticker=opp.ticker,
                title=opp.title,
                category='weather',
                side=opp.side,
                our_probability=opp.our_probability,
                market_price=opp.market_price,
                edge=opp.edge,
                data_source='NWS',
                expiration=None,
                volume=0,
                liquidity_score=0.5,
                time_to_expiry_hours=24,
                overall_score=score,
                reasoning=opp.reasoning,
                timestamp=datetime.now(timezone.utc)
            ))

        logger.info(f"    Weather opportunities: {len(weather_opps)}")

        # Fed markets (using existing strategy)
        try:
            from bots.kalshi_strategy import KalshiFedStrategy
            fed_strategy = KalshiFedStrategy(paper_mode=True)
            fed_opps = fed_strategy.find_mispriced_contracts()

            for opp in fed_opps:
                score = self.calculate_overall_score(
                    opp.edge, 0.6, 48, 'MEDIUM'
                )

                opportunities.append(RankedOpportunity(
                    ticker=opp.ticker,
                    title=f"Fed: {opp.outcome}",
                    category='fed',
                    side=opp.side,
                    our_probability=opp.our_probability,
                    market_price=opp.market_price,
                    edge=opp.edge,
                    data_source='FRED',
                    expiration=None,
                    volume=0,
                    liquidity_score=0.6,
                    time_to_expiry_hours=48,
                    overall_score=score,
                    reasoning=opp.reasoning,
                    timestamp=datetime.now(timezone.utc)
                ))

            logger.info(f"    Fed opportunities: {len(fed_opps)}")
        except Exception as e:
            logger.warning(f"    Fed scan error: {e}")

        # Crypto markets
        crypto_estimates = [e for e in all_estimates if e.get('category') == 'crypto']
        try:
            crypto_markets = self.client.get_markets(series_ticker="KXBTC")  # Bitcoin markets
            crypto_markets += self.client.get_markets(series_ticker="KXETH")  # Ethereum markets

            for market in crypto_markets:
                matched = self.match_market_to_estimate(market, crypto_estimates)
                if matched and matched.get('edge', 0) >= self.MIN_EDGE_THRESHOLD:
                    score = self.calculate_overall_score(
                        matched['edge'],
                        market.get('liquidity_score', 0.5),
                        self.calculate_time_to_expiry(market.get('expiration_time')),
                        matched.get('confidence', 'MEDIUM')
                    )
                    opportunities.append(RankedOpportunity(
                        ticker=market.get('ticker', ''),
                        title=market.get('title', ''),
                        category='crypto',
                        side=matched.get('side', 'YES'),
                        our_probability=matched.get('our_probability', 0.5),
                        market_price=market.get('yes_ask', 50) / 100,
                        edge=matched['edge'],
                        data_source=matched.get('source', 'Technical Analysis'),
                        expiration=market.get('expiration_time'),
                        volume=market.get('volume', 0),
                        liquidity_score=market.get('liquidity_score', 0.5),
                        time_to_expiry_hours=self.calculate_time_to_expiry(market.get('expiration_time')),
                        overall_score=score,
                        reasoning=matched.get('reasoning', 'Crypto price level analysis'),
                        timestamp=datetime.now(timezone.utc)
                    ))
            logger.info(f"    Crypto opportunities: {len([o for o in opportunities if o.category == 'crypto'])}")
        except Exception as e:
            logger.warning(f"    Crypto scan error: {e}")

        # Economic markets
        econ_estimates = [e for e in all_estimates if e.get('category') == 'economic']
        try:
            # CPI, GDP, Jobs markets
            econ_series = ["KXCPI", "KXGDP", "KXJOBS", "KXUNEMP"]
            econ_markets = []
            for series in econ_series:
                econ_markets += self.client.get_markets(series_ticker=series)

            for market in econ_markets:
                matched = self.match_market_to_estimate(market, econ_estimates)
                if matched and matched.get('edge', 0) >= self.MIN_EDGE_THRESHOLD:
                    score = self.calculate_overall_score(
                        matched['edge'],
                        market.get('liquidity_score', 0.6),
                        self.calculate_time_to_expiry(market.get('expiration_time')),
                        matched.get('confidence', 'MEDIUM')
                    )
                    opportunities.append(RankedOpportunity(
                        ticker=market.get('ticker', ''),
                        title=market.get('title', ''),
                        category='economic',
                        side=matched.get('side', 'YES'),
                        our_probability=matched.get('our_probability', 0.5),
                        market_price=market.get('yes_ask', 50) / 100,
                        edge=matched['edge'],
                        data_source=matched.get('source', 'Fed/Economic Models'),
                        expiration=market.get('expiration_time'),
                        volume=market.get('volume', 0),
                        liquidity_score=market.get('liquidity_score', 0.6),
                        time_to_expiry_hours=self.calculate_time_to_expiry(market.get('expiration_time')),
                        overall_score=score,
                        reasoning=matched.get('reasoning', 'Economic indicator analysis'),
                        timestamp=datetime.now(timezone.utc)
                    ))
            logger.info(f"    Economic opportunities: {len([o for o in opportunities if o.category == 'economic'])}")
        except Exception as e:
            logger.warning(f"    Economic scan error: {e}")

        # Earnings markets
        earnings_estimates = [e for e in all_estimates if e.get('category') == 'earnings']
        try:
            earnings_markets = self.client.get_markets(series_ticker="KXEARNINGS")

            for market in earnings_markets:
                matched = self.match_market_to_estimate(market, earnings_estimates)
                if matched and matched.get('edge', 0) >= self.MIN_EDGE_THRESHOLD:
                    score = self.calculate_overall_score(
                        matched['edge'],
                        market.get('liquidity_score', 0.5),
                        self.calculate_time_to_expiry(market.get('expiration_time')),
                        matched.get('confidence', 'MEDIUM')
                    )
                    opportunities.append(RankedOpportunity(
                        ticker=market.get('ticker', ''),
                        title=market.get('title', ''),
                        category='earnings',
                        side=matched.get('side', 'YES'),
                        our_probability=matched.get('our_probability', 0.5),
                        market_price=market.get('yes_ask', 50) / 100,
                        edge=matched['edge'],
                        data_source=matched.get('source', 'Earnings Estimates'),
                        expiration=market.get('expiration_time'),
                        volume=market.get('volume', 0),
                        liquidity_score=market.get('liquidity_score', 0.5),
                        time_to_expiry_hours=self.calculate_time_to_expiry(market.get('expiration_time')),
                        overall_score=score,
                        reasoning=matched.get('reasoning', 'Earnings beat/miss prediction'),
                        timestamp=datetime.now(timezone.utc)
                    ))
            logger.info(f"    Earnings opportunities: {len([o for o in opportunities if o.category == 'earnings'])}")
        except Exception as e:
            logger.warning(f"    Earnings scan error: {e}")

        # Sports markets
        try:
            from bots.sports_edge_finder import SportsEdgeFinder
            sports_finder = SportsEdgeFinder()
            sports_opps = sports_finder.find_opportunities()

            for opp in sports_opps:
                score = self.calculate_overall_score(
                    opp.edge, 0.6, 48, 'MEDIUM'
                )

                opportunities.append(RankedOpportunity(
                    ticker=opp.ticker,
                    title=f"{opp.away_team} @ {opp.home_team}",
                    category='sports',
                    side=opp.side,
                    our_probability=opp.our_probability,
                    market_price=opp.market_price,
                    edge=opp.edge,
                    data_source='ESPN + FiveThirtyEight Elo',
                    expiration=None,
                    volume=0,
                    liquidity_score=0.6,
                    time_to_expiry_hours=48,
                    overall_score=score,
                    reasoning=opp.reasoning,
                    timestamp=datetime.now(timezone.utc)
                ))

            logger.info(f"    Sports opportunities: {len(sports_opps)}")
        except Exception as e:
            logger.warning(f"    Sports scan error: {e}")

        # Box Office markets
        try:
            from bots.boxoffice_edge_finder import BoxOfficeEdgeFinder
            boxoffice_finder = BoxOfficeEdgeFinder()
            boxoffice_opps = boxoffice_finder.find_opportunities()

            for opp in boxoffice_opps:
                score = self.calculate_overall_score(
                    opp.edge, 0.4, 72, 'LOW'  # Lower confidence for box office
                )

                opportunities.append(RankedOpportunity(
                    ticker=opp.ticker,
                    title=f"{opp.movie_title} > ${opp.threshold}M",
                    category='boxoffice',
                    side=opp.side,
                    our_probability=opp.our_probability,
                    market_price=opp.market_price,
                    edge=opp.edge,
                    data_source='Box Office Mojo + RT',
                    expiration=None,
                    volume=0,
                    liquidity_score=0.4,
                    time_to_expiry_hours=72,
                    overall_score=score,
                    reasoning=opp.reasoning,
                    timestamp=datetime.now(timezone.utc)
                ))

            logger.info(f"    Box Office opportunities: {len(boxoffice_opps)}")
        except Exception as e:
            logger.warning(f"    Box Office scan error: {e}")

        # Economic Releases markets (CPI, GDP nowcasts)
        try:
            from bots.economic_releases_edge_finder import EconomicReleasesEdgeFinder
            econ_releases_finder = EconomicReleasesEdgeFinder()
            econ_releases_opps = econ_releases_finder.find_opportunities()

            for opp in econ_releases_opps:
                # HIGH confidence for Fed nowcast data
                conf = 'HIGH' if opp.confidence == 'HIGH' else 'MEDIUM'
                score = self.calculate_overall_score(
                    opp.edge, 0.7, 24, conf
                )

                opportunities.append(RankedOpportunity(
                    ticker=opp.ticker,
                    title=f"{opp.indicator}: {opp.threshold}% {opp.direction}",
                    category='economic',
                    side=opp.side,
                    our_probability=opp.our_probability,
                    market_price=opp.market_price,
                    edge=opp.edge,
                    data_source=opp.source,
                    expiration=None,
                    volume=0,
                    liquidity_score=0.7,
                    time_to_expiry_hours=24,
                    overall_score=score,
                    reasoning=opp.reasoning,
                    timestamp=datetime.now(timezone.utc)
                ))

            logger.info(f"    Economic Releases opportunities: {len(econ_releases_opps)}")
        except Exception as e:
            logger.warning(f"    Economic Releases scan error: {e}")

        # Sports Props markets (team totals)
        try:
            from bots.sports_props_edge_finder import SportsPropsEdgeFinder
            sports_props_finder = SportsPropsEdgeFinder()
            sports_props_opps = sports_props_finder.find_opportunities()

            for opp in sports_props_opps:
                score = self.calculate_overall_score(
                    opp.edge, 0.5, 24, opp.confidence
                )

                opportunities.append(RankedOpportunity(
                    ticker=opp.ticker,
                    title=f"{opp.team} {opp.direction} {opp.threshold}",
                    category='sports',
                    side=opp.side,
                    our_probability=opp.our_probability,
                    market_price=opp.market_price,
                    edge=opp.edge,
                    data_source='NBA Stats',
                    expiration=None,
                    volume=0,
                    liquidity_score=0.5,
                    time_to_expiry_hours=24,
                    overall_score=score,
                    reasoning=opp.reasoning,
                    timestamp=datetime.now(timezone.utc)
                ))

            logger.info(f"    Sports Props opportunities: {len(sports_props_opps)}")
        except Exception as e:
            logger.warning(f"    Sports Props scan error: {e}")

        # Awards markets (Oscars, Golden Globes, Emmys)
        try:
            from bots.awards_edge_finder import AwardsEdgeFinder
            awards_finder = AwardsEdgeFinder()
            awards_opps = awards_finder.find_opportunities()

            for opp in awards_opps:
                score = self.calculate_overall_score(
                    opp.edge, 0.4, 168, opp.confidence  # Awards are longer term
                )

                opportunities.append(RankedOpportunity(
                    ticker=opp.ticker,
                    title=f"{opp.award_show} {opp.category}: {opp.nominee}",
                    category='boxoffice',  # Group with entertainment
                    side=opp.side,
                    our_probability=opp.our_probability,
                    market_price=opp.market_price,
                    edge=opp.edge,
                    data_source='Gold Derby',
                    expiration=None,
                    volume=0,
                    liquidity_score=0.4,
                    time_to_expiry_hours=168,
                    overall_score=score,
                    reasoning=opp.reasoning,
                    timestamp=datetime.now(timezone.utc)
                ))

            logger.info(f"    Awards opportunities: {len(awards_opps)}")
        except Exception as e:
            logger.warning(f"    Awards scan error: {e}")

        # Climate markets (temperature records)
        try:
            from bots.climate_edge_finder import ClimateEdgeFinder
            climate_finder = ClimateEdgeFinder()
            climate_opps = climate_finder.find_opportunities()

            for opp in climate_opps:
                score = self.calculate_overall_score(
                    opp.edge, 0.5, 168, opp.confidence
                )

                opportunities.append(RankedOpportunity(
                    ticker=opp.ticker,
                    title=f"Climate: {opp.metric} ({opp.period})",
                    category='weather',  # Group with weather
                    side=opp.side,
                    our_probability=opp.our_probability,
                    market_price=opp.market_price,
                    edge=opp.edge,
                    data_source='NOAA',
                    expiration=None,
                    volume=0,
                    liquidity_score=0.5,
                    time_to_expiry_hours=168,
                    overall_score=score,
                    reasoning=opp.reasoning,
                    timestamp=datetime.now(timezone.utc)
                ))

            logger.info(f"    Climate opportunities: {len(climate_opps)}")
        except Exception as e:
            logger.warning(f"    Climate scan error: {e}")

        # Sort by overall score
        opportunities.sort(key=lambda x: x.overall_score, reverse=True)

        logger.info("=" * 50)
        logger.info(f"TOTAL OPPORTUNITIES: {len(opportunities)}")
        logger.info("=" * 50)

        return opportunities

    def get_top_opportunities(self, limit: int = 10) -> List[RankedOpportunity]:
        """Get top N opportunities by score"""
        all_opps = self.scan_all_markets()
        return all_opps[:limit]

    def execute_opportunity(self, opp: RankedOpportunity) -> Optional[Dict]:
        """
        Execute a trading opportunity.

        Args:
            opp: Opportunity to execute

        Returns:
            Order result or None
        """
        if self.paper_mode:
            logger.info(f"[PAPER] Would execute: BUY {opp.side} on {opp.ticker}")
            logger.info(f"[PAPER] Price: {opp.market_price:.0%}, Edge: {opp.edge:.1%}")
            return {'paper': True, 'ticker': opp.ticker, 'side': opp.side}

        try:
            price_cents = int(opp.market_price * 100)
            contracts = int(self.MAX_POSITION_USD / opp.market_price)

            if contracts <= 0:
                return None

            order = self.client.create_order(
                ticker=opp.ticker,
                side=opp.side.lower(),
                action='buy',
                count=contracts,
                price=price_cents,
                order_type='limit'
            )

            logger.info(f"Order executed: {order}")
            return order

        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return None

    def execute_best_opportunities(self, max_trades: int = 3) -> List[Dict]:
        """
        Execute the best opportunities.

        Args:
            max_trades: Maximum number of trades to execute

        Returns:
            List of execution results
        """
        opportunities = self.get_top_opportunities(limit=max_trades * 2)
        results = []

        for opp in opportunities[:max_trades]:
            result = self.execute_opportunity(opp)
            if result:
                results.append(result)

        return results

    def print_opportunities(self, opportunities: List[RankedOpportunity], limit: int = 10):
        """Print opportunities in a formatted way"""
        print()
        print("=" * 70)
        print("MULTI-MARKET OPPORTUNITIES")
        print("=" * 70)
        print()

        if not opportunities:
            print("No opportunities found.")
            return

        for i, opp in enumerate(opportunities[:limit], 1):
            print(f"{i}. [{opp.category.upper()}] {opp.ticker}")
            print(f"   {opp.title}")
            print(f"   Action: BUY {opp.side} @ {opp.market_price:.0%}")
            print(f"   Our Probability: {opp.our_probability:.0%}")
            print(f"   Edge: {opp.edge:.1%} | Score: {opp.overall_score:.2f}")
            print(f"   Source: {opp.data_source}")
            print(f"   {opp.reasoning}")
            print()

        print("=" * 70)


def main():
    """Run the multi-market strategy"""
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("MULTI-MARKET STRATEGY")
    print("=" * 60)

    strategy = MultiMarketStrategy(paper_mode=True)

    print("\n[1] Scanning all markets...")
    opportunities = strategy.scan_all_markets()

    print("\n[2] Top Opportunities:")
    strategy.print_opportunities(opportunities)

    print("\n[3] Summary:")
    print("-" * 40)
    print(f"Total opportunities: {len(opportunities)}")

    if opportunities:
        by_category = {}
        for opp in opportunities:
            by_category[opp.category] = by_category.get(opp.category, 0) + 1

        for cat, count in sorted(by_category.items()):
            print(f"  {cat}: {count}")

        avg_edge = sum(o.edge for o in opportunities) / len(opportunities)
        print(f"\nAverage edge: {avg_edge:.1%}")
        print(f"Best opportunity: {opportunities[0].ticker} ({opportunities[0].edge:.1%} edge)")


if __name__ == "__main__":
    main()
