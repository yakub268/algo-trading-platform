"""
Kalshi Edge Detection & Execution System

Unified integration that connects all data scrapers to Kalshi execution:
1. Weather (NWS forecasts) -> Kalshi weather markets
2. Economic (FRED, Fed nowcasts) -> Kalshi CPI/GDP/unemployment markets
3. Crypto (CoinGecko, Fear/Greed) -> Kalshi Bitcoin/Ethereum markets
4. Sports (ESPN, Elo ratings) -> Kalshi sports markets
5. Box Office - DISABLED (Kalshi has no box office gross markets; awards handled separately)

Pipeline:
    Scrapers -> Probability Estimates -> Market Matching -> Edge Calculation
    -> Risk Check -> Execution -> Telegram Alert

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import time
import logging
import sqlite3
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import difflib

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('KalshiEdgeExecutor')


class EdgeCategory(Enum):
    """Categories of edge-finding strategies"""
    WEATHER = "weather"
    ECONOMIC = "economic"
    CRYPTO = "crypto"
    SPORTS = "sports"
    BOXOFFICE = "boxoffice"
    AWARDS = "awards"
    CLIMATE = "climate"


@dataclass
class EdgeOpportunity:
    """A trading opportunity with calculated edge"""
    ticker: str
    title: str
    category: EdgeCategory
    our_probability: float
    market_price: float
    edge: float
    side: str  # 'YES' or 'NO'
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    reasoning: str
    data_source: str
    kelly_fraction: float
    timestamp: datetime
    # Additional metadata
    threshold: Optional[float] = None
    forecast_value: Optional[float] = None
    expires: Optional[datetime] = None


@dataclass
class ExecutionResult:
    """Result of executing a trade"""
    success: bool
    opportunity: EdgeOpportunity
    contracts: int
    price: float
    total_cost: float
    order_id: Optional[str] = None
    error: Optional[str] = None
    execution_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RiskLimits:
    """Risk parameters for execution"""
    max_daily_loss: float = 50.0
    max_position_per_market: float = 25.0
    max_total_exposure: float = 200.0
    min_edge_threshold: float = 0.05  # 5% minimum edge
    min_confidence: str = 'MEDIUM'
    max_kelly_fraction: float = 0.10  # 10% max of bankroll per trade


class KalshiEdgeExecutor:
    """
    Unified edge detection and execution system for Kalshi.

    Integrates all data sources:
    - Weather: NWS forecasts for temperature markets
    - Economic: FRED data for CPI, unemployment, Fed rate markets
    - Crypto: CoinGecko for Bitcoin/Ethereum price markets
    - Sports: ESPN/Elo for game outcome markets
    - Box Office: BOM for movie revenue markets

    Usage:
        executor = KalshiEdgeExecutor(paper_mode=True)
        opportunities = executor.scan_all_edges()
        for opp in opportunities:
            if opp.edge > 0.08:  # 8% edge threshold
                executor.execute(opp)
    """

    def __init__(self, paper_mode: bool = True, risk_limits: RiskLimits = None):
        """
        Initialize the edge executor.

        Args:
            paper_mode: If True, simulate trades
            risk_limits: Custom risk limits (uses defaults if None)
        """
        self.paper_mode = paper_mode
        self.risk_limits = risk_limits or RiskLimits()

        # Initialize Kalshi client
        try:
            self.client = KalshiClient()
            self._connected = True
            logger.info("Kalshi client connected")
        except Exception as e:
            logger.warning(f"Kalshi client not available: {e}")
            self.client = None
            self._connected = False

        # Data aggregator (lazy loaded)
        self._aggregator = None

        # Track executions and P&L
        self._executions: List[ExecutionResult] = []
        self._daily_pnl = 0.0
        self._paper_balance = 500.0

        # Position tracking
        self._positions: Dict[str, float] = {}

        # Cache for market data
        self._market_cache: Dict[str, Dict] = {}
        self._cache_time: Optional[datetime] = None

        # Database
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'kalshi_edge_executor.db'
        )
        self._init_database()

        logger.info(f"Edge Executor initialized (paper_mode={paper_mode})")

    @property
    def aggregator(self):
        """Lazy load data aggregator"""
        if self._aggregator is None:
            from scrapers.data_aggregator import DataAggregator
            self._aggregator = DataAggregator()
        return self._aggregator

    def _init_database(self):
        """Initialize SQLite database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                ticker TEXT,
                category TEXT,
                our_probability REAL,
                market_price REAL,
                edge REAL,
                side TEXT,
                confidence TEXT,
                executed INTEGER DEFAULT 0
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                ticker TEXT,
                category TEXT,
                side TEXT,
                contracts INTEGER,
                price REAL,
                total_cost REAL,
                expected_profit REAL,
                status TEXT,
                order_id TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE,
                opportunities_found INTEGER,
                trades_executed INTEGER,
                total_pnl REAL,
                best_edge REAL,
                best_category TEXT
            )
        ''')

        conn.commit()
        conn.close()

    # ============== Market Data ==============

    def _refresh_market_cache(self, force: bool = False):
        """Refresh cache of Kalshi markets from relevant series."""
        if not force and self._cache_time:
            if (datetime.now(timezone.utc) - self._cache_time).seconds < 300:
                return

        if not self.client or not self._connected:
            return

        # Series we care about for edge detection
        target_series = [
            'KXCPI',      # CPI inflation
            'KXGDP',      # GDP
            'KXNFP',      # Non-farm payrolls
            'KXBTC',      # Bitcoin
            'KXETH',      # Ethereum
            'KXNBAGAME',  # NBA individual games
            'KXNFLGAME',  # NFL individual games
            'KXMLBGAME',  # MLB individual games
            'KXINX',      # S&P 500
            # Weather series
            'KXHIGHNY',   # NYC high temp
            'KXHIGHLAX',  # LA high temp
            'KXHIGHCHI',  # Chicago high temp
            'KXHIGHMIA',  # Miami high temp
            'KXHIGHPHIL', # Philadelphia high temp
            'KXLOWTNYC',  # NYC low temp
            'KXLOWTLAX',  # LA low temp
            'KXLOWTCHI',  # Chicago low temp
            'KXLOWTMIA',  # Miami low temp
            'KXLOWTPHIL', # Philadelphia low temp
            'KXRAINNYC',  # NYC rain
        ]

        try:
            all_markets = []
            for series in target_series:
                try:
                    markets = self.client.get_markets(series_ticker=series, status='open', limit=100)
                    all_markets.extend(markets)
                except Exception as e:
                    logger.debug(f"Could not fetch {series}: {e}")
                time.sleep(0.25)  # Rate limiting to avoid 429 errors

            # Also get some general markets as fallback
            time.sleep(0.25)  # Rate limiting before additional API call
            try:
                general = self.client.get_markets(status='open', limit=200)
                all_markets.extend(general)
            except Exception as e:
                logger.debug(f"Could not fetch general markets: {e}")

            self._market_cache = {m['ticker']: m for m in all_markets}
            self._cache_time = datetime.now(timezone.utc)
            logger.info(f"Refreshed market cache: {len(self._market_cache)} markets")
        except Exception as e:
            logger.error(f"Failed to refresh market cache: {e}")

    def _get_market_price(self, ticker: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Get current YES/NO prices for a market.

        Returns:
            Tuple of (yes_price, no_price) in decimal (0-1)
        """
        if self.client and self._connected:
            try:
                orderbook = self.client.get_orderbook(ticker)
                yes_asks = orderbook.get('yes', [])
                no_asks = orderbook.get('no', [])

                yes_price = min(a.get('price', 100) for a in yes_asks) / 100 if yes_asks else None
                no_price = min(a.get('price', 100) for a in no_asks) / 100 if no_asks else None

                time.sleep(0.25)  # Rate limiting to avoid 429 errors
                return yes_price, no_price
            except Exception as e:
                logger.debug(f"Error fetching market price for {ticker}: {e}")

        # Fall back to market data
        self._refresh_market_cache()
        market = self._market_cache.get(ticker)
        if market:
            yes_price = market.get('yes_ask', market.get('last_price', 50)) / 100
            return yes_price, 1 - yes_price

        return None, None

    # ============== Edge Detection ==============

    def scan_all_edges(self, use_cache: bool = True) -> List[EdgeOpportunity]:
        """
        Scan all data sources for edge opportunities.

        Args:
            use_cache: Use cached scraper data if available

        Returns:
            List of EdgeOpportunity sorted by edge
        """
        logger.info("Scanning all data sources for edge...")

        all_opportunities = []

        # Refresh market cache
        self._refresh_market_cache()

        # Get aggregated data from all scrapers
        try:
            data = self.aggregator.fetch_all(use_cache=use_cache)
            estimates = data.probability_estimates
        except Exception as e:
            logger.error(f"Failed to fetch aggregated data: {e}")
            estimates = {}

        # Process each category
        if estimates.get('weather'):
            weather_opps = self._process_weather_edges(estimates['weather'])
            all_opportunities.extend(weather_opps)
            logger.info(f"  Weather: {len(weather_opps)} opportunities")

        if estimates.get('economic'):
            economic_opps = self._process_economic_edges(estimates['economic'])
            all_opportunities.extend(economic_opps)
            logger.info(f"  Economic: {len(economic_opps)} opportunities")

        if estimates.get('crypto'):
            crypto_opps = self._process_crypto_edges(estimates['crypto'])
            all_opportunities.extend(crypto_opps)
            logger.info(f"  Crypto: {len(crypto_opps)} opportunities")

        if estimates.get('sports'):
            sports_opps = self._process_sports_edges(estimates['sports'])
            all_opportunities.extend(sports_opps)
            logger.info(f"  Sports: {len(sports_opps)} opportunities")

        # Disabled: Kalshi has no box office gross markets (only awards handled by awards_edge_finder)
        # if estimates.get('boxoffice'):
        #     boxoffice_opps = self._process_boxoffice_edges(estimates['boxoffice'])
        #     all_opportunities.extend(boxoffice_opps)
        #     logger.info(f"  Box Office: {len(boxoffice_opps)} opportunities")

        # Deduplicate: keep only the opportunity with the highest edge for each ticker
        deduplicated = {}
        for opp in all_opportunities:
            if opp.ticker not in deduplicated or opp.edge > deduplicated[opp.ticker].edge:
                deduplicated[opp.ticker] = opp

        if len(all_opportunities) != len(deduplicated):
            logger.info(f"  Deduplicated: {len(all_opportunities)} -> {len(deduplicated)} opportunities")

        all_opportunities = list(deduplicated.values())

        # Sort by edge
        all_opportunities.sort(key=lambda x: x.edge, reverse=True)

        # Log to database
        for opp in all_opportunities:
            self._log_opportunity(opp)

        logger.info(f"Total opportunities found: {len(all_opportunities)}")
        return all_opportunities

    def _match_ticker_to_market(self, pattern: str, category: str) -> Optional[str]:
        """
        Match a ticker pattern to actual Kalshi market.

        Args:
            pattern: Ticker pattern (e.g., "HIGHNY-*-T40")
            category: Category hint for matching

        Returns:
            Actual ticker if found
        """
        self._refresh_market_cache()

        pattern_clean = pattern.replace('*', '').upper()

        # Ticker prefixes that should NOT match certain categories
        # KXINX = S&P 500 index markets (not weather)
        # KXNDX = Nasdaq index markets (not weather)
        excluded_prefixes = {
            'weather': ['KXINX', 'KXNDX', 'KXDJI', 'KXBTC', 'KXETH'],
            'crypto': ['KXINX', 'KXNDX', 'KXDJI'],
            'sports': ['KXINX', 'KXNDX', 'KXDJI', 'KXBTC', 'KXETH'],
        }

        # Direct match
        for ticker in self._market_cache:
            if pattern_clean in ticker.upper():
                # Check if this ticker should be excluded from this category
                prefixes_to_exclude = excluded_prefixes.get(category, [])
                if any(ticker.upper().startswith(prefix) for prefix in prefixes_to_exclude):
                    continue
                return ticker

        # Fuzzy match by category keywords
        category_keywords = {
            'weather': ['HIGH', 'LOW', 'RAIN', 'TEMP', 'WEATHER'],
            'economic': ['CPI', 'GDP', 'UNEMPLOYMENT', 'FED', 'NFP', 'JOBS', 'INX', 'NDX', 'DJI'],
            'crypto': ['BTC', 'BITCOIN', 'ETH', 'ETHEREUM', 'CRYPTO'],
            'sports': ['NFL', 'NBA', 'MLB', 'NHL', 'SUPERBOWL'],
            'boxoffice': ['BOX', 'MOVIE', 'FILM'],
        }

        keywords = category_keywords.get(category, [])

        for ticker, market in self._market_cache.items():
            # Skip tickers that should be excluded from this category
            prefixes_to_exclude = excluded_prefixes.get(category, [])
            if any(ticker.upper().startswith(prefix) for prefix in prefixes_to_exclude):
                continue

            title = market.get('title', '').upper()
            # Check if any category keyword matches
            if any(kw in ticker.upper() or kw in title for kw in keywords):
                # Check pattern similarity
                if difflib.SequenceMatcher(None, pattern_clean, ticker.upper()).ratio() > 0.5:
                    return ticker

        return None

    def _process_weather_edges(self, estimates: List[Dict]) -> List[EdgeOpportunity]:
        """Process weather probability estimates into opportunities."""
        opportunities = []

        for est in estimates:
            pattern = est.get('ticker_pattern', '')
            our_prob = est.get('our_probability', 0.5)

            # Try to find matching market
            ticker = self._match_ticker_to_market(pattern, 'weather')
            if not ticker:
                continue

            # Get market price
            yes_price, no_price = self._get_market_price(ticker)
            if yes_price is None:
                continue

            # Calculate edge
            yes_edge = our_prob - yes_price
            no_edge = (1 - our_prob) - no_price if no_price else 0

            # Determine best side
            if yes_edge >= self.risk_limits.min_edge_threshold:
                edge = yes_edge
                side = 'YES'
                market_price = yes_price
                final_prob = our_prob
            elif no_edge >= self.risk_limits.min_edge_threshold:
                edge = no_edge
                side = 'NO'
                market_price = no_price
                final_prob = 1 - our_prob
            else:
                continue

            # Calculate Kelly fraction
            kelly = self._calculate_kelly(final_prob, market_price)

            opportunities.append(EdgeOpportunity(
                ticker=ticker,
                title=self._market_cache.get(ticker, {}).get('title', ticker)[:80],
                category=EdgeCategory.WEATHER,
                our_probability=final_prob,
                market_price=market_price,
                edge=edge,
                side=side,
                confidence='HIGH' if edge > 0.10 else 'MEDIUM',
                reasoning=est.get('reasoning', 'NWS forecast'),
                data_source='NWS',
                kelly_fraction=kelly,
                timestamp=datetime.now(timezone.utc),
                threshold=est.get('threshold'),
                forecast_value=est.get('nws_value')
            ))

        return opportunities

    def _process_economic_edges(self, estimates: List[Dict]) -> List[EdgeOpportunity]:
        """Process economic probability estimates."""
        opportunities = []

        for est in estimates:
            pattern = est.get('ticker_pattern', '')
            our_prob = est.get('our_probability', 0.5)

            ticker = self._match_ticker_to_market(pattern, 'economic')
            if not ticker:
                continue

            yes_price, no_price = self._get_market_price(ticker)
            if yes_price is None:
                continue

            yes_edge = our_prob - yes_price
            no_edge = (1 - our_prob) - no_price if no_price else 0

            if yes_edge >= self.risk_limits.min_edge_threshold:
                edge = yes_edge
                side = 'YES'
                market_price = yes_price
                final_prob = our_prob
            elif no_edge >= self.risk_limits.min_edge_threshold:
                edge = no_edge
                side = 'NO'
                market_price = no_price
                final_prob = 1 - our_prob
            else:
                continue

            # Economic estimates often have lower confidence
            conf = est.get('confidence', 'MEDIUM')
            if conf == 'LOW' and edge < 0.08:
                continue  # Skip low-confidence with low edge

            kelly = self._calculate_kelly(final_prob, market_price)

            opportunities.append(EdgeOpportunity(
                ticker=ticker,
                title=self._market_cache.get(ticker, {}).get('title', ticker)[:80],
                category=EdgeCategory.ECONOMIC,
                our_probability=final_prob,
                market_price=market_price,
                edge=edge,
                side=side,
                confidence=conf,
                reasoning=est.get('reasoning', 'FRED data'),
                data_source='FRED',
                kelly_fraction=kelly,
                timestamp=datetime.now(timezone.utc),
                threshold=est.get('threshold')
            ))

        return opportunities

    def _process_crypto_edges(self, estimates: List[Dict]) -> List[EdgeOpportunity]:
        """Process crypto probability estimates."""
        opportunities = []

        for est in estimates:
            pattern = est.get('ticker_pattern', '')
            our_prob = est.get('our_probability', 0.5)

            ticker = self._match_ticker_to_market(pattern, 'crypto')
            if not ticker:
                continue

            yes_price, no_price = self._get_market_price(ticker)
            if yes_price is None:
                continue

            yes_edge = our_prob - yes_price
            no_edge = (1 - our_prob) - no_price if no_price else 0

            if yes_edge >= self.risk_limits.min_edge_threshold:
                edge = yes_edge
                side = 'YES'
                market_price = yes_price
                final_prob = our_prob
            elif no_edge >= self.risk_limits.min_edge_threshold:
                edge = no_edge
                side = 'NO'
                market_price = no_price
                final_prob = 1 - our_prob
            else:
                continue

            kelly = self._calculate_kelly(final_prob, market_price)

            opportunities.append(EdgeOpportunity(
                ticker=ticker,
                title=self._market_cache.get(ticker, {}).get('title', ticker)[:80],
                category=EdgeCategory.CRYPTO,
                our_probability=final_prob,
                market_price=market_price,
                edge=edge,
                side=side,
                confidence='MEDIUM',  # Crypto is volatile
                reasoning=est.get('reasoning', f"Price: ${est.get('current_price', 0):,.0f}"),
                data_source='CoinGecko',
                kelly_fraction=kelly,
                timestamp=datetime.now(timezone.utc),
                threshold=est.get('threshold')
            ))

        return opportunities

    def _process_sports_edges(self, estimates: List[Dict]) -> List[EdgeOpportunity]:
        """Process sports probability estimates."""
        opportunities = []

        for est in estimates:
            pattern = est.get('ticker_pattern', '')
            our_prob = est.get('our_probability', 0.5)

            ticker = self._match_ticker_to_market(pattern, 'sports')
            if not ticker:
                continue

            yes_price, no_price = self._get_market_price(ticker)
            if yes_price is None:
                continue

            yes_edge = our_prob - yes_price
            no_edge = (1 - our_prob) - no_price if no_price else 0

            if yes_edge >= self.risk_limits.min_edge_threshold:
                edge = yes_edge
                side = 'YES'
                market_price = yes_price
                final_prob = our_prob
            elif no_edge >= self.risk_limits.min_edge_threshold:
                edge = no_edge
                side = 'NO'
                market_price = no_price
                final_prob = 1 - our_prob
            else:
                continue

            kelly = self._calculate_kelly(final_prob, market_price)

            opportunities.append(EdgeOpportunity(
                ticker=ticker,
                title=self._market_cache.get(ticker, {}).get('title', ticker)[:80],
                category=EdgeCategory.SPORTS,
                our_probability=final_prob,
                market_price=market_price,
                edge=edge,
                side=side,
                confidence=est.get('confidence', 'MEDIUM'),
                reasoning=est.get('reasoning', f"{est.get('home_team', '')} vs {est.get('away_team', '')}"),
                data_source='ESPN/Elo',
                kelly_fraction=kelly,
                timestamp=datetime.now(timezone.utc)
            ))

        return opportunities

    def _process_boxoffice_edges(self, estimates: List[Dict]) -> List[EdgeOpportunity]:
        """Process box office probability estimates."""
        opportunities = []

        for est in estimates:
            pattern = est.get('ticker_pattern', '')
            our_prob = est.get('our_probability', 0.5)

            ticker = self._match_ticker_to_market(pattern, 'boxoffice')
            if not ticker:
                continue

            yes_price, no_price = self._get_market_price(ticker)
            if yes_price is None:
                continue

            yes_edge = our_prob - yes_price
            no_edge = (1 - our_prob) - no_price if no_price else 0

            if yes_edge >= self.risk_limits.min_edge_threshold:
                edge = yes_edge
                side = 'YES'
                market_price = yes_price
                final_prob = our_prob
            elif no_edge >= self.risk_limits.min_edge_threshold:
                edge = no_edge
                side = 'NO'
                market_price = no_price
                final_prob = 1 - our_prob
            else:
                continue

            kelly = self._calculate_kelly(final_prob, market_price)

            opportunities.append(EdgeOpportunity(
                ticker=ticker,
                title=self._market_cache.get(ticker, {}).get('title', ticker)[:80],
                category=EdgeCategory.BOXOFFICE,
                our_probability=final_prob,
                market_price=market_price,
                edge=edge,
                side=side,
                confidence=est.get('confidence', 'MEDIUM'),
                reasoning=est.get('reasoning', est.get('movie_title', '')),
                data_source='BoxOfficeMojo',
                kelly_fraction=kelly,
                timestamp=datetime.now(timezone.utc),
                threshold=est.get('threshold')
            ))

        return opportunities

    def _calculate_kelly(self, probability: float, market_price: float) -> float:
        """Calculate Kelly criterion fraction."""
        if market_price <= 0 or market_price >= 1:
            return 0

        edge = probability - market_price
        if edge <= 0:
            return 0

        # Kelly formula
        b = (1 / market_price) - 1  # Net odds
        kelly = (probability * b - (1 - probability)) / b

        # Half Kelly for safety
        kelly = kelly / 2

        # Cap at max
        return min(max(0, kelly), self.risk_limits.max_kelly_fraction)

    def _log_opportunity(self, opp: EdgeOpportunity):
        """Log opportunity to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO opportunities
                (timestamp, ticker, category, our_probability, market_price, edge, side, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                opp.timestamp.isoformat(),
                opp.ticker,
                opp.category.value,
                opp.our_probability,
                opp.market_price,
                opp.edge,
                opp.side,
                opp.confidence
            ))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to log opportunity: {e}")

    # ============== Execution ==============

    def check_risk_limits(self, opp: EdgeOpportunity, cost: float) -> Tuple[bool, str]:
        """Check if trade passes risk limits."""
        # Check daily loss
        if self._daily_pnl <= -self.risk_limits.max_daily_loss:
            return False, "Daily loss limit reached"

        # Check position limit
        current_position = self._positions.get(opp.ticker, 0)
        if current_position + cost > self.risk_limits.max_position_per_market:
            return False, "Position limit for market reached"

        # Check total exposure
        total_exposure = sum(self._positions.values()) + cost
        if total_exposure > self.risk_limits.max_total_exposure:
            return False, "Total exposure limit reached"

        # Check minimum confidence
        confidence_order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        if confidence_order.get(opp.confidence, 0) < confidence_order.get(self.risk_limits.min_confidence, 1):
            return False, f"Confidence too low: {opp.confidence}"

        # Check paper balance
        if self.paper_mode and cost > self._paper_balance:
            return False, "Insufficient paper balance"

        return True, "OK"

    def execute(self, opportunity: EdgeOpportunity, contracts: int = None) -> ExecutionResult:
        """
        Execute a trade on an edge opportunity.

        Args:
            opportunity: The opportunity to execute
            contracts: Number of contracts (calculated from Kelly if None)

        Returns:
            ExecutionResult
        """
        # Calculate position size if not specified
        if contracts is None:
            # Use Kelly fraction
            position_value = self._paper_balance * opportunity.kelly_fraction
            contracts = int(position_value / opportunity.market_price)
            contracts = max(1, min(contracts, 50))  # 1-50 contracts

        # Calculate cost
        cost = contracts * opportunity.market_price

        # Check risk limits
        allowed, reason = self.check_risk_limits(opportunity, cost)
        if not allowed:
            logger.warning(f"Trade blocked: {reason}")
            return ExecutionResult(
                success=False,
                opportunity=opportunity,
                contracts=0,
                price=opportunity.market_price,
                total_cost=0,
                error=reason
            )

        if self.paper_mode:
            return self._paper_execute(opportunity, contracts, cost)
        else:
            return self._real_execute(opportunity, contracts, cost)

    def _paper_execute(self, opp: EdgeOpportunity, contracts: int, cost: float) -> ExecutionResult:
        """Simulate trade execution."""
        # Update balances
        self._paper_balance -= cost
        self._positions[opp.ticker] = self._positions.get(opp.ticker, 0) + cost

        expected_profit = opp.edge * contracts

        result = ExecutionResult(
            success=True,
            opportunity=opp,
            contracts=contracts,
            price=opp.market_price,
            total_cost=cost,
            order_id=f"paper_{int(time.time())}_{opp.ticker}"
        )

        self._executions.append(result)
        self._log_execution(result, expected_profit)

        # Send alert
        self._send_execution_alert(opp, contracts, cost, expected_profit)

        logger.info(f"""
[PAPER] EXECUTED: {opp.side} {contracts} contracts on {opp.ticker}
  Category: {opp.category.value}
  Price: ${opp.market_price:.2f}
  Cost: ${cost:.2f}
  Edge: {opp.edge:.2%}
  Expected Profit: ${expected_profit:.2f}
  Paper Balance: ${self._paper_balance:.2f}
""")

        return result

    def _real_execute(self, opp: EdgeOpportunity, contracts: int, cost: float) -> ExecutionResult:
        """Execute real trade on Kalshi."""
        if not self.client or not self._connected:
            return ExecutionResult(
                success=False,
                opportunity=opp,
                contracts=0,
                price=opp.market_price,
                total_cost=0,
                error="Kalshi client not connected"
            )

        try:
            price_cents = int(opp.market_price * 100)

            order = self.client.create_order(
                ticker=opp.ticker,
                side=opp.side.lower(),
                action='buy',
                count=contracts,
                price=price_cents,
                order_type='limit'
            )

            if not order:
                return ExecutionResult(
                    success=False,
                    opportunity=opp,
                    contracts=0,
                    price=opp.market_price,
                    total_cost=0,
                    error="Order creation failed"
                )

            order_id = order.get('order_id', '')
            expected_profit = opp.edge * contracts

            result = ExecutionResult(
                success=True,
                opportunity=opp,
                contracts=contracts,
                price=opp.market_price,
                total_cost=cost,
                order_id=order_id
            )

            self._executions.append(result)
            self._positions[opp.ticker] = self._positions.get(opp.ticker, 0) + cost
            self._log_execution(result, expected_profit)
            self._send_execution_alert(opp, contracts, cost, expected_profit)

            logger.info(f"[LIVE] Order placed: {order_id}")

            return result

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return ExecutionResult(
                success=False,
                opportunity=opp,
                contracts=0,
                price=opp.market_price,
                total_cost=0,
                error=str(e)
            )

    def _log_execution(self, result: ExecutionResult, expected_profit: float):
        """Log execution to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO executions
                (timestamp, ticker, category, side, contracts, price, total_cost, expected_profit, status, order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.execution_time.isoformat(),
                result.opportunity.ticker,
                result.opportunity.category.value,
                result.opportunity.side,
                result.contracts,
                result.price,
                result.total_cost,
                expected_profit,
                'executed' if result.success else 'failed',
                result.order_id
            ))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to log execution: {e}")

    def _send_execution_alert(self, opp: EdgeOpportunity, contracts: int, cost: float, expected_profit: float):
        """Send Telegram alert for execution."""
        try:
            from utils.telegram_bot import send_opportunity_alert

            mode = "PAPER" if self.paper_mode else "LIVE"

            send_opportunity_alert(
                source=opp.category.value.upper(),
                symbol=opp.ticker,
                opportunity_type=opp.side,
                edge=opp.edge * 100,
                confidence=0.9 if opp.confidence == 'HIGH' else 0.7,
                details=f"[{mode}] {contracts} contracts @ ${opp.market_price:.2f} = ${cost:.2f}\n{opp.reasoning}",
                priority='high' if opp.edge > 0.10 else 'medium'
            )
        except Exception as e:
            logger.debug(f"Failed to send alert: {e}")

    # ============== Auto Execution ==============

    def auto_execute(self, min_edge: float = 0.08, max_trades: int = 5) -> List[ExecutionResult]:
        """
        Automatically execute high-edge opportunities.

        Args:
            min_edge: Minimum edge to execute
            max_trades: Maximum trades to execute in one run

        Returns:
            List of execution results
        """
        opportunities = self.scan_all_edges()

        results = []
        trades_executed = 0

        for opp in opportunities:
            if trades_executed >= max_trades:
                break

            if opp.edge < min_edge:
                continue

            if opp.confidence == 'LOW':
                continue

            result = self.execute(opp)
            results.append(result)

            if result.success:
                trades_executed += 1

        return results

    # ============== Reporting ==============

    def get_summary(self) -> Dict:
        """Get summary of executor state."""
        return {
            'connected': self._connected,
            'paper_mode': self.paper_mode,
            'paper_balance': self._paper_balance,
            'daily_pnl': self._daily_pnl,
            'total_executions': len(self._executions),
            'open_positions': len(self._positions),
            'total_exposure': sum(self._positions.values()),
            'risk_limits': asdict(self.risk_limits)
        }

    def get_recent_opportunities(self, limit: int = 20) -> List[Dict]:
        """Get recent opportunities from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT timestamp, ticker, category, our_probability, market_price, edge, side, confidence
                FROM opportunities
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    'timestamp': row[0],
                    'ticker': row[1],
                    'category': row[2],
                    'our_probability': row[3],
                    'market_price': row[4],
                    'edge': row[5],
                    'side': row[6],
                    'confidence': row[7]
                }
                for row in rows
            ]
        except Exception as e:
            logger.debug(f"Error fetching edge history: {e}")
            return []


def main():
    """Test the edge executor."""
    print("=" * 70)
    print("KALSHI EDGE DETECTION & EXECUTION SYSTEM")
    print("=" * 70)
    print("""
This system integrates all data scrapers with Kalshi execution:

  WEATHER  -> NWS forecasts -> Temperature markets
  ECONOMIC -> FRED data     -> CPI/GDP/Unemployment markets
  CRYPTO   -> CoinGecko     -> Bitcoin/Ethereum price markets
  SPORTS   -> ESPN/Elo      -> Game outcome markets
  BOXOFFICE -> DISABLED (Kalshi has no box office gross markets)
""")

    # Initialize
    executor = KalshiEdgeExecutor(paper_mode=True)

    # Scan for opportunities
    print("\n" + "=" * 70)
    print("SCANNING ALL DATA SOURCES")
    print("=" * 70)

    opportunities = executor.scan_all_edges(use_cache=False)

    if opportunities:
        print(f"\nFound {len(opportunities)} opportunities:\n")

        for i, opp in enumerate(opportunities[:15], 1):
            emoji = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴'}.get(opp.confidence, '⚪')
            print(f"{i}. [{opp.category.value.upper()}] {opp.ticker}")
            print(f"   {opp.title[:50]}...")
            print(f"   Our Prob: {opp.our_probability:.0%} | Market: {opp.market_price:.0%}")
            print(f"   EDGE: {opp.edge:.1%} -> {opp.side} {emoji}")
            print(f"   Kelly: {opp.kelly_fraction:.1%} | Source: {opp.data_source}")
            print()
    else:
        print("\nNo opportunities found above threshold")

    # Auto-execute top opportunities
    if opportunities:
        print("\n" + "=" * 70)
        print("AUTO-EXECUTING TOP OPPORTUNITIES (PAPER)")
        print("=" * 70)

        results = executor.auto_execute(min_edge=0.05, max_trades=3)

        print(f"\nExecuted {len([r for r in results if r.success])} trades")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = executor.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: ${value:.2f}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    print("\nEdge executor test complete!")


if __name__ == '__main__':
    main()
