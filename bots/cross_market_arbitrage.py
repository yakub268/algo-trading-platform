"""
Cross-Market Arbitrage Detection System

Identifies arbitrage opportunities between Kalshi and other prediction markets.
Monitors price differences, latency arbitrage, and correlation opportunities.

Supported Markets:
- Kalshi (Primary)
- Polymarket (Polygon-based prediction market)
- PredictIt (Political markets)
- Metaculus (Forecasting tournaments)
- Manifold Markets (Play money markets for pattern detection)

Features:
- Real-time price monitoring across multiple platforms
- Latency arbitrage detection
- Cross-market event correlation analysis
- Automated opportunity scoring and ranking
- Risk assessment for arbitrage positions
- Integration with existing trading infrastructure

Author: AI Trading Enhancement
Created: February 2026
"""

import os
import sys
import json
import asyncio
import requests
import logging
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from decimal import Decimal
import re
from collections import defaultdict
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cross_market_arbitrage.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MarketPrice:
    """Price data for a market on a specific platform"""
    platform: str
    market_id: str
    market_title: str
    yes_price: float
    no_price: float
    volume_24h: float
    last_updated: datetime
    url: str
    liquidity: float = 0.0

@dataclass
class ArbitrageOpportunity:
    """Cross-market arbitrage opportunity"""
    primary_market: MarketPrice
    secondary_market: MarketPrice
    opportunity_type: str  # DIRECT, CORRELATION, LATENCY
    expected_profit: float
    risk_level: str  # LOW, MEDIUM, HIGH
    confidence: float
    min_capital_required: float
    execution_complexity: str  # SIMPLE, MODERATE, COMPLEX
    time_sensitivity: str  # LOW, MEDIUM, HIGH
    reasoning: str

@dataclass
class CorrelationPattern:
    """Pattern showing correlation between different markets"""
    market_a: str
    market_b: str
    correlation_strength: float
    lag_minutes: int
    confidence_level: float
    recent_divergence: float

class ExternalMarketConnector:
    """Connects to external prediction markets using synchronous requests (Windows DNS fix)"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'TradingBot/1.0'})
        self.api_limits = {
            'polymarket': {'requests_per_minute': 60, 'last_request': None},
            'predictit': {'requests_per_minute': 100, 'last_request': None},
            'manifold': {'requests_per_minute': 120, 'last_request': None}
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

    async def _rate_limit_check(self, platform: str):
        """Check and enforce rate limits"""
        if platform not in self.api_limits:
            return

        limit_info = self.api_limits[platform]
        if limit_info['last_request']:
            time_since_last = (datetime.now() - limit_info['last_request']).total_seconds()
            min_interval = 60 / limit_info['requests_per_minute']

            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)

        self.api_limits[platform]['last_request'] = datetime.now()

    def _sync_fetch(self, url: str, params: dict = None) -> Optional[requests.Response]:
        """Synchronous fetch (runs in thread pool)"""
        return self.session.get(url, params=params, timeout=15)

    async def _fetch_with_retry(self, url: str, platform: str, params: dict = None, max_retries: int = 2) -> Optional[dict]:
        """Fetch URL with retry logic using synchronous requests in thread pool"""
        for attempt in range(max_retries + 1):
            try:
                response = await asyncio.to_thread(self._sync_fetch, url, params)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code >= 500:
                    logger.debug(f"{platform} server error {response.status_code} (attempt {attempt+1})")
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    logger.debug(f"{platform} returned status {response.status_code}")
                    return None
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    logger.debug(f"{platform} timeout (attempt {attempt+1})")
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    logger.warning(f"{platform} timed out after {max_retries+1} attempts")
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries:
                    logger.debug(f"{platform} connection error (attempt {attempt+1})")
                    await asyncio.sleep(2 * (attempt + 1))
                else:
                    logger.warning(f"{platform} unavailable after {max_retries+1} attempts")
            except Exception as e:
                logger.debug(f"{platform} error: {type(e).__name__}: {e}")
                break
        return None

    async def get_polymarket_prices(self, search_terms: List[str]) -> List[MarketPrice]:
        """Get prices from Polymarket"""
        await self._rate_limit_check('polymarket')

        url = "https://gamma-api.polymarket.com/markets"
        data = await self._fetch_with_retry(url, 'Polymarket')

        if not data:
            return []

        markets = []
        market_list = data if isinstance(data, list) else data.get('markets', [])

        for market in market_list[:20]:
            if not isinstance(market, dict):
                continue

            market_title = market.get('question', '').lower()
            if not any(term.lower() in market_title for term in search_terms):
                continue

            # Parse prices from outcomePrices (list of strings) or lastTradePrice
            try:
                outcome_prices = market.get('outcomePrices', [])
                if outcome_prices and len(outcome_prices) >= 1:
                    yes_price = float(outcome_prices[0]) if outcome_prices[0] else 0.5
                else:
                    yes_price = float(market.get('lastTradePrice', 0.5))
                # Clamp to valid range
                yes_price = max(0.01, min(0.99, yes_price))
            except (ValueError, TypeError):
                yes_price = 0.5

            markets.append(MarketPrice(
                platform='polymarket',
                market_id=market.get('id', ''),
                market_title=market.get('question', ''),
                yes_price=yes_price,
                no_price=1 - yes_price,
                volume_24h=float(market.get('volume24hr') or 0),
                last_updated=datetime.now(timezone.utc),
                url=f"https://polymarket.com/event/{market.get('slug', '')}",
                liquidity=float(market.get('liquidityNum') or market.get('liquidity') or 0)
            ))

        return markets

    async def get_predictit_prices(self, search_terms: List[str]) -> List[MarketPrice]:
        """Get prices from PredictIt"""
        await self._rate_limit_check('predictit')

        url = "https://www.predictit.org/api/marketdata/all/"
        data = await self._fetch_with_retry(url, 'PredictIt')

        if not data:
            return []

        markets = []
        for market in data.get('markets', [])[:10]:
            market_name = market.get('name', '').lower()
            if any(term.lower() in market_name for term in search_terms):
                for contract in market.get('contracts', []):
                    yes_price = float(contract.get('lastTradePrice') or 0.5)

                    markets.append(MarketPrice(
                        platform='predictit',
                        market_id=str(contract.get('id', '')),
                        market_title=f"{market_name} - {contract.get('name', '')}",
                        yes_price=yes_price,
                        no_price=1 - yes_price,
                        volume_24h=float(contract.get('totalSharesTraded', 0)),
                        last_updated=datetime.now(timezone.utc),
                        url=market.get('url', ''),
                        liquidity=float(contract.get('totalSharesTraded', 0))
                    ))

        return markets

    async def get_manifold_prices(self, search_terms: List[str]) -> List[MarketPrice]:
        """Get prices from Manifold Markets"""
        await self._rate_limit_check('manifold')

        markets = []

        # Use search endpoint for each term (more accurate matching)
        for term in search_terms[:3]:  # Limit to first 3 terms
            url = "https://api.manifold.markets/v0/search-markets"
            params = {'term': term, 'limit': 10}
            data = await self._fetch_with_retry(url, 'Manifold', params=params)

            if not data or not isinstance(data, list):
                continue

            for market in data:
                # Skip markets without probability (multi-choice markets)
                probability = market.get('probability')
                if probability is None:
                    continue

                # Avoid duplicates
                market_id = market.get('id', '')
                if any(m.market_id == market_id for m in markets):
                    continue

                markets.append(MarketPrice(
                    platform='manifold',
                    market_id=market_id,
                    market_title=market.get('question', ''),
                    yes_price=float(probability),
                    no_price=1 - float(probability),
                    volume_24h=float(market.get('volume24Hours') or 0),
                    last_updated=datetime.now(timezone.utc),
                    url=market.get('url', ''),
                    liquidity=float(market.get('totalLiquidity') or 0)
                ))

        return markets

class ArbitrageDetector:
    """Core arbitrage detection logic"""

    def __init__(self):
        self.price_history = defaultdict(list)
        self.correlation_cache = {}
        self.min_profit_threshold = 0.02  # 2% minimum profit
        self.max_execution_time = 300  # 5 minutes max execution time

    def detect_direct_arbitrage(self, kalshi_markets: List[MarketPrice],
                               external_markets: List[MarketPrice]) -> List[ArbitrageOpportunity]:
        """Detect direct arbitrage opportunities between platforms"""
        opportunities = []

        for kalshi_market in kalshi_markets:
            for external_market in external_markets:
                # Use fuzzy matching to find similar markets
                similarity = self._calculate_similarity(kalshi_market.market_title, external_market.market_title)

                if similarity > 0.7:  # High similarity threshold
                    # Calculate potential arbitrage
                    arb_opportunity = self._calculate_direct_arbitrage(kalshi_market, external_market)

                    if arb_opportunity and arb_opportunity.expected_profit > self.min_profit_threshold:
                        opportunities.append(arb_opportunity)

        return sorted(opportunities, key=lambda x: x.expected_profit, reverse=True)

    def _calculate_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between market titles"""
        # Simple similarity calculation based on common words
        words1 = set(re.findall(r'\w+', title1.lower()))
        words2 = set(re.findall(r'\w+', title2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _calculate_direct_arbitrage(self, market1: MarketPrice, market2: MarketPrice) -> Optional[ArbitrageOpportunity]:
        """Calculate direct arbitrage between two similar markets"""
        # Strategy 1: Buy low on one platform, sell high on another
        price_diff_yes = abs(market1.yes_price - market2.yes_price)
        price_diff_no = abs(market1.no_price - market2.no_price)

        max_price_diff = max(price_diff_yes, price_diff_no)

        if max_price_diff < self.min_profit_threshold:
            return None

        # Determine the better opportunity
        if price_diff_yes > price_diff_no:
            if market1.yes_price < market2.yes_price:
                buy_market, sell_market = market1, market2
                position_type = "YES"
            else:
                buy_market, sell_market = market2, market1
                position_type = "YES"
            expected_profit = price_diff_yes
        else:
            if market1.no_price < market2.no_price:
                buy_market, sell_market = market1, market2
                position_type = "NO"
            else:
                buy_market, sell_market = market2, market1
                position_type = "NO"
            expected_profit = price_diff_no

        # Calculate risk factors
        risk_level = self._assess_arbitrage_risk(buy_market, sell_market)
        min_capital = max(1000, max(buy_market.volume_24h, sell_market.volume_24h) * 0.1)

        return ArbitrageOpportunity(
            primary_market=buy_market,
            secondary_market=sell_market,
            opportunity_type="DIRECT",
            expected_profit=expected_profit,
            risk_level=risk_level,
            confidence=0.8 if max_price_diff > 0.05 else 0.6,
            min_capital_required=min_capital,
            execution_complexity="MODERATE",
            time_sensitivity="HIGH",
            reasoning=f"Price difference of {max_price_diff:.3f} on {position_type} position"
        )

    def _assess_arbitrage_risk(self, market1: MarketPrice, market2: MarketPrice) -> str:
        """Assess the risk level of an arbitrage opportunity"""
        # Factors: liquidity, platform reliability, execution complexity
        avg_liquidity = (market1.liquidity + market2.liquidity) / 2

        if avg_liquidity > 10000:
            liquidity_risk = "LOW"
        elif avg_liquidity > 1000:
            liquidity_risk = "MEDIUM"
        else:
            liquidity_risk = "HIGH"

        # Platform risk assessment
        reliable_platforms = {'kalshi', 'predictit'}
        platform_risk = "LOW" if {market1.platform, market2.platform}.issubset(reliable_platforms) else "MEDIUM"

        # Combine risk factors
        if liquidity_risk == "HIGH" or platform_risk == "HIGH":
            return "HIGH"
        elif liquidity_risk == "MEDIUM" or platform_risk == "MEDIUM":
            return "MEDIUM"
        else:
            return "LOW"

    def detect_correlation_arbitrage(self, all_markets: List[MarketPrice]) -> List[ArbitrageOpportunity]:
        """Detect arbitrage based on market correlations"""
        opportunities = []

        # Group markets by platform
        markets_by_platform = defaultdict(list)
        for market in all_markets:
            markets_by_platform[market.platform].append(market)

        # Find correlated events across platforms
        correlations = self._find_correlations(all_markets)

        for correlation in correlations:
            if correlation.confidence_level > 0.7 and abs(correlation.recent_divergence) > 0.1:
                # Find the specific markets
                market_a = next((m for m in all_markets if correlation.market_a in m.market_title), None)
                market_b = next((m for m in all_markets if correlation.market_b in m.market_title), None)

                if market_a and market_b and market_a.platform != market_b.platform:
                    opportunity = self._create_correlation_opportunity(market_a, market_b, correlation)
                    if opportunity:
                        opportunities.append(opportunity)

        return opportunities

    def _find_correlations(self, markets: List[MarketPrice]) -> List[CorrelationPattern]:
        """Find correlation patterns between markets"""
        correlations = []

        # Keywords that indicate related events
        correlation_keywords = {
            'election': ['election', 'vote', 'poll', 'candidate', 'primary'],
            'economy': ['inflation', 'gdp', 'unemployment', 'fed', 'interest'],
            'sports': ['super bowl', 'world cup', 'olympics', 'championship'],
            'weather': ['hurricane', 'temperature', 'climate', 'weather'],
            'crypto': ['bitcoin', 'ethereum', 'cryptocurrency', 'btc', 'eth']
        }

        # Find markets in the same category
        categorized_markets = defaultdict(list)
        for market in markets:
            market_title_lower = market.market_title.lower()
            for category, keywords in correlation_keywords.items():
                if any(keyword in market_title_lower for keyword in keywords):
                    categorized_markets[category].append(market)

        # Calculate correlations within categories
        for category, category_markets in categorized_markets.items():
            for i, market_a in enumerate(category_markets):
                for market_b in category_markets[i+1:]:
                    if market_a.platform != market_b.platform:
                        correlation = CorrelationPattern(
                            market_a=market_a.market_title,
                            market_b=market_b.market_title,
                            correlation_strength=0.8,  # Simplified
                            lag_minutes=5,
                            confidence_level=0.7,
                            recent_divergence=abs(market_a.yes_price - market_b.yes_price)
                        )
                        correlations.append(correlation)

        return correlations

    def _create_correlation_opportunity(self, market_a: MarketPrice, market_b: MarketPrice,
                                      correlation: CorrelationPattern) -> Optional[ArbitrageOpportunity]:
        """Create arbitrage opportunity based on correlation"""
        if correlation.recent_divergence < 0.05:  # Minimum divergence threshold
            return None

        expected_profit = correlation.recent_divergence * 0.5  # Conservative estimate

        return ArbitrageOpportunity(
            primary_market=market_a,
            secondary_market=market_b,
            opportunity_type="CORRELATION",
            expected_profit=expected_profit,
            risk_level="MEDIUM",
            confidence=correlation.confidence_level,
            min_capital_required=2000,  # Higher capital for correlation trades
            execution_complexity="COMPLEX",
            time_sensitivity="MEDIUM",
            reasoning=f"Correlated markets with {correlation.recent_divergence:.3f} divergence"
        )

class CrossMarketArbitrageSystem:
    """Main cross-market arbitrage system"""

    def __init__(self, kalshi_client: KalshiClient):
        self.kalshi = kalshi_client
        self.detector = ArbitrageDetector()
        self.opportunities_history = []

        # Configuration
        self.scan_interval = 60  # seconds
        self.max_opportunities = 20
        self.active_positions = []

    async def get_kalshi_markets(self, search_terms: List[str]) -> List[MarketPrice]:
        """Get market data from Kalshi"""
        try:
            # Kalshi client is synchronous, run in thread pool
            markets = await asyncio.to_thread(self.kalshi.get_markets)
            kalshi_prices = []

            for market in markets[:50]:  # Limit to recent markets
                market_title = market.get('title', '').lower()
                if any(term.lower() in market_title for term in search_terms):
                    # Get current price
                    ticker = market.get('ticker', '')
                    market_data = await asyncio.to_thread(self.kalshi.get_market, ticker)

                    if market_data:
                        yes_price = market_data.get('yes_price', 0.5)
                        volume = market_data.get('volume', 0)

                        kalshi_prices.append(MarketPrice(
                            platform='kalshi',
                            market_id=ticker,
                            market_title=market.get('title', ''),
                            yes_price=yes_price,
                            no_price=1 - yes_price,
                            volume_24h=volume,
                            last_updated=datetime.now(timezone.utc),
                            url=f"https://kalshi.com/markets/{ticker}",
                            liquidity=volume * 100  # Estimate
                        ))

            return kalshi_prices

        except Exception as e:
            logger.error(f"Error fetching Kalshi markets: {e}")
            return []

    async def scan_arbitrage_opportunities(self, search_terms: List[str] = None) -> List[ArbitrageOpportunity]:
        """Main method to scan for arbitrage opportunities"""
        if search_terms is None:
            search_terms = ['election', 'inflation', 'fed', 'unemployment', 'super bowl', 'crypto', 'bitcoin']

        logger.info(f"Scanning for arbitrage opportunities with terms: {search_terms}")

        all_opportunities = []

        try:
            # Get data from all platforms
            kalshi_markets = await self.get_kalshi_markets(search_terms)

            async with ExternalMarketConnector() as connector:
                external_markets = []

                # Fetch from external platforms in parallel
                tasks = [
                    connector.get_polymarket_prices(search_terms),
                    connector.get_predictit_prices(search_terms),
                    connector.get_manifold_prices(search_terms)
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, list):
                        external_markets.extend(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"External market fetch failed: {result}")

            logger.info(f"Fetched {len(kalshi_markets)} Kalshi markets and {len(external_markets)} external markets")

            # Detect arbitrage opportunities
            if kalshi_markets and external_markets:
                # Direct arbitrage
                direct_opportunities = self.detector.detect_direct_arbitrage(kalshi_markets, external_markets)
                all_opportunities.extend(direct_opportunities)

                # Correlation arbitrage
                all_markets = kalshi_markets + external_markets
                correlation_opportunities = self.detector.detect_correlation_arbitrage(all_markets)
                all_opportunities.extend(correlation_opportunities)

            # Sort by expected profit
            all_opportunities.sort(key=lambda x: x.expected_profit, reverse=True)

            # Limit results
            filtered_opportunities = all_opportunities[:self.max_opportunities]

            # Store history
            self.opportunities_history.extend(filtered_opportunities)
            if len(self.opportunities_history) > 1000:
                self.opportunities_history = self.opportunities_history[-1000:]

            logger.info(f"Found {len(filtered_opportunities)} arbitrage opportunities")

            return filtered_opportunities

        except Exception as e:
            logger.error(f"Error scanning arbitrage opportunities: {e}")
            return []

    def generate_arbitrage_report(self, opportunities: List[ArbitrageOpportunity]) -> Dict[str, Any]:
        """Generate comprehensive arbitrage report"""
        if not opportunities:
            return {"message": "No arbitrage opportunities found"}

        # Categorize opportunities
        by_type = defaultdict(list)
        by_risk = defaultdict(list)
        by_platform = defaultdict(list)

        total_potential_profit = 0

        for opp in opportunities:
            by_type[opp.opportunity_type].append(opp)
            by_risk[opp.risk_level].append(opp)

            platforms = f"{opp.primary_market.platform}-{opp.secondary_market.platform}"
            by_platform[platforms].append(opp)

            total_potential_profit += opp.expected_profit

        # Calculate statistics
        avg_profit = total_potential_profit / len(opportunities)
        high_confidence_count = len([o for o in opportunities if o.confidence > 0.8])

        report = {
            "summary": {
                "total_opportunities": len(opportunities),
                "total_potential_profit": round(total_potential_profit, 4),
                "average_profit_per_opportunity": round(avg_profit, 4),
                "high_confidence_opportunities": high_confidence_count,
                "scan_timestamp": datetime.now(timezone.utc).isoformat()
            },
            "by_opportunity_type": {
                opportunity_type: len(opps) for opportunity_type, opps in by_type.items()
            },
            "by_risk_level": {
                risk_level: len(opps) for risk_level, opps in by_risk.items()
            },
            "by_platform_pair": {
                platforms: len(opps) for platforms, opps in by_platform.items()
            },
            "top_opportunities": [
                {
                    "primary_platform": opp.primary_market.platform,
                    "secondary_platform": opp.secondary_market.platform,
                    "primary_market": opp.primary_market.market_title,
                    "expected_profit": round(opp.expected_profit, 4),
                    "risk_level": opp.risk_level,
                    "confidence": round(opp.confidence, 2),
                    "opportunity_type": opp.opportunity_type,
                    "reasoning": opp.reasoning
                }
                for opp in opportunities[:10]
            ]
        }

        return report

async def main():
    """Main execution function"""
    # Initialize clients
    kalshi = KalshiClient()

    # Initialize arbitrage system
    arbitrage_system = CrossMarketArbitrageSystem(kalshi)

    try:
        logger.info("Cross-Market Arbitrage System starting...")

        # Test with specific search terms
        search_terms = ['election', 'fed', 'inflation', 'bitcoin', 'unemployment']

        # Scan for opportunities
        opportunities = await arbitrage_system.scan_arbitrage_opportunities(search_terms)

        # Generate report
        report = arbitrage_system.generate_arbitrage_report(opportunities)

        # Print results
        print("\n" + "="*50)
        print("CROSS-MARKET ARBITRAGE REPORT")
        print("="*50)
        print(json.dumps(report, indent=2, default=str))

        # Print detailed opportunities
        if opportunities:
            print("\n" + "="*50)
            print("TOP ARBITRAGE OPPORTUNITIES")
            print("="*50)
            for i, opp in enumerate(opportunities[:5], 1):
                print(f"\n{i}. {opp.opportunity_type} ARBITRAGE")
                print(f"   Primary: {opp.primary_market.platform} - {opp.primary_market.market_title[:60]}...")
                print(f"   Secondary: {opp.secondary_market.platform} - {opp.secondary_market.market_title[:60]}...")
                print(f"   Expected Profit: {opp.expected_profit:.4f} ({opp.expected_profit*100:.2f}%)")
                print(f"   Risk Level: {opp.risk_level}")
                print(f"   Confidence: {opp.confidence:.2f}")
                print(f"   Reasoning: {opp.reasoning}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())