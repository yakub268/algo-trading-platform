"""
Multi-Market Scanner for Kalshi

Scans ALL Kalshi series across categories:
- Weather (temperature, precipitation)
- Economic Data (CPI, jobs, GDP)
- Crypto (BTC/ETH price predictions)
- Earnings (beat/miss predictions)
- Fed (rate decisions)

Reuses edge calculation logic and finds opportunities >= 5% edge.

Author: Trading Bot
Created: January 2026
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MarketScanner')


@dataclass
class MarketOpportunity:
    """Represents a trading opportunity across any market category"""
    category: str  # weather, economic, earnings, crypto, fed
    series_ticker: str
    ticker: str
    title: str
    our_probability: float
    market_price: float  # YES ask price as probability (0-1)
    edge: float
    side: str  # "yes" or "no"
    data_source: str
    expiration: Optional[str]
    volume: int
    liquidity_score: float
    reasoning: str
    timestamp: datetime


# Category mapping based on series ticker patterns
CATEGORY_PATTERNS = {
    'weather': ['KXHIGHNY', 'KXLOWNY',  # Only NYC confirmed to exist
                'KXRAIN', 'KXSNOW', 'KXWEATHER', 'HIGHTEMP', 'LOWTEMP',
                'INXHIGH', 'INXLOW', 'WEATHER'],
    'economic': ['KXFED', 'KXFEDDECISION', 'KXFEDHIKE', 'KXRATECUTCOUNT',
                 'KXCPI', 'KXNFP', 'KXGDP', 'KXUNEMPLOYMENT', 'KXRETAIL',
                 'CPI', 'NFP', 'GDP', 'JOBS', 'UNEMPLOYMENT', 'INFLATION'],
    'crypto': ['KXBTC', 'KXETH', 'KXCRYPTO', 'BTC', 'ETH', 'BITCOIN', 'CRYPTO'],
    'earnings': ['KXEARNINGS', 'KXEPS', 'EARNINGS', 'AAPL', 'GOOGL', 'MSFT',
                 'AMZN', 'TSLA', 'NVDA', 'META'],
    'politics': ['KXPRES', 'KXSENATE', 'KXHOUSE', 'ELECTION', 'PRES', 'CONGRESS'],
    'sports': ['KXNFL', 'KXNBA', 'KXMLB', 'KXNHL', 'NFL', 'NBA', 'MLB', 'NHL',
               'SPORTS', 'SUPERBOWL', 'WORLDSERIES'],
}


class MarketScanner:
    """
    Scans all Kalshi markets to find trading opportunities.

    Categorizes markets and calculates edge based on external data sources.
    """

    # Minimum requirements
    MIN_EDGE = 0.05  # 5% minimum edge
    MIN_CONTRACT_PRICE = 0.05  # 5 cents minimum
    MAX_CONTRACT_PRICE = 0.95  # 95 cents maximum
    MIN_VOLUME = 10  # Minimum contracts traded

    def __init__(self, client: Optional[KalshiClient] = None):
        """Initialize the market scanner"""
        self.client = client or KalshiClient()
        self.series_cache: Dict[str, Dict] = {}
        self.market_cache: Dict[str, List[Dict]] = {}
        self.cache_time: Optional[datetime] = None
        self.cache_duration = timedelta(minutes=15)

        logger.info("MarketScanner initialized")

    def get_all_series(self, refresh: bool = False) -> List[Dict]:
        """
        Fetch all series from Kalshi.

        Returns:
            List of series dictionaries
        """
        if not refresh and self.series_cache and self.cache_time:
            if datetime.now(timezone.utc) - self.cache_time < self.cache_duration:
                return list(self.series_cache.values())

        logger.info("Fetching all series from Kalshi...")
        all_series = []

        try:
            # Get series by category
            categories = ['Economics', 'Climate and Weather', 'Crypto',
                         'Financial', 'Science & Tech', 'Entertainment']

            for category in categories:
                try:
                    # Try to get series by category (may not be supported)
                    data = self.client._request("GET", "/series", params={'limit': 200})
                    series_list = data.get('series', [])
                    all_series.extend(series_list)
                except Exception as e:
                    logger.debug(f"Could not fetch series for {category}: {e}")

            # Deduplicate
            seen = set()
            unique_series = []
            for s in all_series:
                ticker = s.get('ticker')
                if ticker and ticker not in seen:
                    seen.add(ticker)
                    unique_series.append(s)
                    self.series_cache[ticker] = s

            self.cache_time = datetime.now(timezone.utc)
            logger.info(f"Found {len(unique_series)} unique series")
            return unique_series

        except Exception as e:
            logger.error(f"Error fetching series: {e}")
            return []

    def categorize_series(self, series: Dict) -> str:
        """
        Categorize a series based on its ticker and title.

        Args:
            series: Series dictionary from Kalshi API

        Returns:
            Category string
        """
        ticker = series.get('ticker', '').upper()
        title = series.get('title', '').lower()
        category = series.get('category', '').lower()

        # Check category field first
        if 'weather' in category or 'climate' in category:
            return 'weather'
        if 'economic' in category or 'financial' in category:
            return 'economic'
        if 'crypto' in category:
            return 'crypto'

        # Check ticker patterns
        for cat, patterns in CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if pattern in ticker:
                    return cat

        # Check title keywords
        weather_keywords = ['temperature', 'rain', 'snow', 'weather', 'high temp', 'low temp']
        if any(kw in title for kw in weather_keywords):
            return 'weather'

        economic_keywords = ['fed', 'cpi', 'inflation', 'jobs', 'gdp', 'unemployment', 'rate']
        if any(kw in title for kw in economic_keywords):
            return 'economic'

        crypto_keywords = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto']
        if any(kw in title for kw in crypto_keywords):
            return 'crypto'

        earnings_keywords = ['earnings', 'revenue', 'beat', 'eps']
        if any(kw in title for kw in earnings_keywords):
            return 'earnings'

        return 'other'

    def get_markets_by_category(self, category: str) -> List[Dict]:
        """
        Get all open markets for a specific category.

        Args:
            category: Category to filter by

        Returns:
            List of market dictionaries
        """
        all_series = self.get_all_series()
        markets = []

        for series in all_series:
            if self.categorize_series(series) != category:
                continue

            series_ticker = series.get('ticker')
            if not series_ticker:
                continue

            try:
                series_markets = self.client.get_markets(
                    series_ticker=series_ticker,
                    status='open',
                    limit=100
                )
                markets.extend(series_markets)
            except Exception as e:
                logger.debug(f"Could not get markets for {series_ticker}: {e}")

        logger.info(f"Found {len(markets)} open markets in category '{category}'")
        return markets

    def parse_orderbook(self, orderbook: Dict) -> Tuple[float, float]:
        """
        Parse orderbook to get best YES ask price.

        Returns:
            Tuple of (yes_ask, no_ask) as probabilities (0-1)
        """
        def get_best_bid(orders: List) -> int:
            if not orders:
                return 0
            prices = [o[0] for o in orders if len(o) >= 2 and o[1] > 0]
            return max(prices) if prices else 0

        yes_orders = orderbook.get('yes', [])
        no_orders = orderbook.get('no', [])

        best_yes_bid = get_best_bid(yes_orders)
        best_no_bid = get_best_bid(no_orders)

        # YES ask = 100 - best NO bid
        yes_ask = (100 - best_no_bid) / 100 if best_no_bid > 0 else 1.0
        no_ask = (100 - best_yes_bid) / 100 if best_yes_bid > 0 else 1.0

        return yes_ask, no_ask

    def calculate_liquidity_score(self, market: Dict, orderbook: Dict) -> float:
        """
        Calculate a liquidity score (0-1) based on volume and spread.

        Higher is better.
        """
        volume = market.get('volume', 0) or market.get('open_interest', 0) or 0

        yes_orders = orderbook.get('yes', [])
        no_orders = orderbook.get('no', [])

        # Get total depth
        yes_depth = sum(o[1] for o in yes_orders if len(o) >= 2)
        no_depth = sum(o[1] for o in no_orders if len(o) >= 2)
        total_depth = yes_depth + no_depth

        # Score components
        volume_score = min(1.0, volume / 1000)  # Max at 1000 contracts
        depth_score = min(1.0, total_depth / 10000)  # Max at 10k depth

        return (volume_score + depth_score) / 2

    def scan_category(
        self,
        category: str,
        probability_estimates: Optional[Dict[str, float]] = None
    ) -> List[MarketOpportunity]:
        """
        Scan a category for opportunities.

        Args:
            category: Category to scan
            probability_estimates: Dict mapping market ticker to our probability estimate

        Returns:
            List of MarketOpportunity objects
        """
        markets = self.get_markets_by_category(category)
        opportunities = []

        for market in markets:
            ticker = market.get('ticker')
            if not ticker:
                continue

            try:
                orderbook = self.client.get_orderbook(ticker)
                yes_ask, no_ask = self.parse_orderbook(orderbook)

                # Skip if price outside range
                if yes_ask < self.MIN_CONTRACT_PRICE or yes_ask > self.MAX_CONTRACT_PRICE:
                    continue

                # Get our probability estimate if provided
                our_prob = None
                if probability_estimates and ticker in probability_estimates:
                    our_prob = probability_estimates[ticker]

                # Calculate edge if we have an estimate
                if our_prob is not None:
                    # YES opportunity
                    yes_edge = our_prob - yes_ask
                    if yes_edge >= self.MIN_EDGE:
                        opportunities.append(MarketOpportunity(
                            category=category,
                            series_ticker=market.get('series_ticker', ''),
                            ticker=ticker,
                            title=market.get('title', ''),
                            our_probability=our_prob,
                            market_price=yes_ask,
                            edge=yes_edge,
                            side='yes',
                            data_source='external',
                            expiration=market.get('close_time'),
                            volume=market.get('volume', 0) or 0,
                            liquidity_score=self.calculate_liquidity_score(market, orderbook),
                            reasoning=f"Our prob {our_prob:.1%} > market {yes_ask:.1%}",
                            timestamp=datetime.now(timezone.utc)
                        ))

                    # NO opportunity
                    no_edge = (1 - our_prob) - no_ask
                    if no_edge >= self.MIN_EDGE:
                        opportunities.append(MarketOpportunity(
                            category=category,
                            series_ticker=market.get('series_ticker', ''),
                            ticker=ticker,
                            title=market.get('title', ''),
                            our_probability=1 - our_prob,
                            market_price=no_ask,
                            edge=no_edge,
                            side='no',
                            data_source='external',
                            expiration=market.get('close_time'),
                            volume=market.get('volume', 0) or 0,
                            liquidity_score=self.calculate_liquidity_score(market, orderbook),
                            reasoning=f"Our NO prob {1-our_prob:.1%} > market {no_ask:.1%}",
                            timestamp=datetime.now(timezone.utc)
                        ))

            except Exception as e:
                logger.debug(f"Error processing {ticker}: {e}")
                continue

        # Sort by edge
        opportunities.sort(key=lambda x: x.edge, reverse=True)
        return opportunities

    def scan_all_categories(
        self,
        probability_estimates: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[MarketOpportunity]]:
        """
        Scan all categories for opportunities.

        Args:
            probability_estimates: Dict mapping market ticker to probability

        Returns:
            Dict mapping category to list of opportunities
        """
        results = {}
        categories = ['weather', 'economic', 'crypto', 'earnings', 'politics']

        for category in categories:
            logger.info(f"Scanning {category} markets...")
            opps = self.scan_category(category, probability_estimates)
            results[category] = opps
            logger.info(f"  Found {len(opps)} opportunities in {category}")

        return results

    def discover_markets(self) -> Dict[str, List[Dict]]:
        """
        Discover and categorize all available markets.

        Returns:
            Dict mapping category to list of market summaries
        """
        logger.info("Discovering all Kalshi markets...")

        # Get all open markets
        try:
            all_markets = self.client.get_markets(status='open', limit=1000)
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return {}

        # Categorize
        by_category = defaultdict(list)

        for market in all_markets:
            ticker = market.get('ticker', '')
            title = market.get('title', '')
            series_ticker = market.get('series_ticker', '')

            # Determine category
            category = 'other'
            for cat, patterns in CATEGORY_PATTERNS.items():
                for pattern in patterns:
                    if pattern in ticker.upper() or pattern in series_ticker.upper():
                        category = cat
                        break
                if category != 'other':
                    break

            # Check title for category hints
            if category == 'other':
                title_lower = title.lower()
                if any(w in title_lower for w in ['temperature', 'rain', 'snow', 'weather']):
                    category = 'weather'
                elif any(w in title_lower for w in ['fed', 'rate', 'cpi', 'jobs', 'gdp']):
                    category = 'economic'
                elif any(w in title_lower for w in ['bitcoin', 'btc', 'ethereum', 'crypto']):
                    category = 'crypto'

            by_category[category].append({
                'ticker': ticker,
                'series_ticker': series_ticker,
                'title': title[:80],
                'volume': market.get('volume', 0),
            })

        # Log summary
        logger.info("Market Discovery Results:")
        for category, markets in sorted(by_category.items()):
            logger.info(f"  {category}: {len(markets)} markets")

        return dict(by_category)

    def find_weather_markets(self) -> List[Dict]:
        """
        Find all weather-related markets on Kalshi.

        Dynamically discovers available weather markets instead of querying
        hardcoded series that may not exist.

        Returns:
            List of weather market dictionaries
        """
        logger.info("Searching for weather markets...")
        weather_markets = []

        # Only try series that are known to exist on Kalshi
        # Note: Daily temperature markets are seasonal and may not always be available
        # As of Feb 2026, only NYC markets (KXHIGHNY, KXLOWNY) have been confirmed
        weather_series = [
            'KXHIGHNY', 'KXLOWNY',  # NYC - confirmed to exist
            # Other cities can be added when Kalshi expands weather markets
        ]

        # Try each series (silently skip if not found)
        for series_ticker in weather_series:
            try:
                series_info = self.client.get_series(series_ticker)
                if series_info:
                    logger.info(f"  Found series: {series_ticker} - {series_info.get('title', '')}")

                    markets = self.client.get_markets(
                        series_ticker=series_ticker,
                        status='open',
                        limit=50
                    )

                    for m in markets:
                        m['category'] = 'weather'
                        m['series_ticker'] = series_ticker

                    weather_markets.extend(markets)
            except Exception as e:
                logger.debug(f"  Series {series_ticker} not found")

        # Also search all markets for weather keywords
        try:
            all_markets = self.client.get_markets(status='open', limit=500)
            for market in all_markets:
                title = market.get('title', '').lower()
                ticker = market.get('ticker', '').upper()

                if market in weather_markets:
                    continue

                weather_keywords = ['temperature', 'high temp', 'low temp', 'degrees',
                                   'rain', 'snow', 'precipitation', 'weather']

                if any(kw in title for kw in weather_keywords):
                    market['category'] = 'weather'
                    weather_markets.append(market)
                elif 'HIGH' in ticker and any(c in ticker for c in ['NY', 'LA', 'CHI', 'MIA', 'PHX']):
                    market['category'] = 'weather'
                    weather_markets.append(market)
                elif 'LOW' in ticker and any(c in ticker for c in ['NY', 'LA', 'CHI', 'MIA', 'PHX']):
                    market['category'] = 'weather'
                    weather_markets.append(market)
        except Exception as e:
            logger.warning(f"Error searching all markets: {e}")

        # Deduplicate
        seen = set()
        unique = []
        for m in weather_markets:
            ticker = m.get('ticker')
            if ticker and ticker not in seen:
                seen.add(ticker)
                unique.append(m)

        logger.info(f"Found {len(unique)} weather markets total")
        return unique


def main():
    """Test the market scanner"""
    from dotenv import load_dotenv
    load_dotenv()

    scanner = MarketScanner()

    print("=" * 60)
    print("KALSHI MARKET SCANNER")
    print("=" * 60)

    # Discover all markets
    print("\n[1] Discovering all markets...")
    by_category = scanner.discover_markets()

    print("\n[2] Market Summary by Category:")
    print("-" * 40)
    for category, markets in sorted(by_category.items()):
        print(f"  {category.upper()}: {len(markets)} markets")
        for m in markets[:3]:
            print(f"    - {m['ticker']}: {m['title']}")

    # Find weather markets specifically
    print("\n[3] Weather Markets Detail:")
    print("-" * 40)
    weather = scanner.find_weather_markets()

    for m in weather[:10]:
        ticker = m.get('ticker')
        title = m.get('title', '')[:60]
        print(f"  {ticker}")
        print(f"    {title}")

        try:
            ob = scanner.client.get_orderbook(ticker)
            yes_ask, no_ask = scanner.parse_orderbook(ob)
            print(f"    YES: {yes_ask:.0%}  |  NO: {no_ask:.0%}")
        except Exception as e:
            logger.debug(f"Error fetching orderbook for {ticker}: {e}")
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()
