"""
Kalshi Fed Strategy
Event-driven trading strategy for Fed decision prediction markets

Integrates with Fed Data Aggregator to:
1. Calculate Fed decision probabilities from multiple sources
2. Compare against Kalshi market prices
3. Find mispriced contracts with 5%+ edge
4. Execute trades with proper position sizing

Author: Jacob
Created: January 2026
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('KalshiFedStrategy')


@dataclass
class FedProbabilities:
    """Fed decision probability estimates"""
    hold: float  # Probability of no change
    hike_25: float  # Probability of 25bp hike
    hike_50: float  # Probability of 50bp hike
    cut_25: float  # Probability of 25bp cut
    cut_50: float  # Probability of 50bp cut
    confidence: str  # HIGH, MEDIUM, LOW
    reasoning: List[str]
    timestamp: datetime


@dataclass
class TradeSignal:
    """Trade signal for Kalshi execution"""
    ticker: str
    outcome: str
    side: str  # "yes" or "no"
    market_price: float  # Current market price (0-1)
    our_probability: float  # Our calculated probability
    edge: float  # our_probability - market_price
    contracts: int  # Number of contracts to buy
    max_cost_usd: float
    reasoning: str


class KalshiFedStrategy:
    """
    Fed Decision Trading Strategy for Kalshi
    
    Uses data from Fed Data Aggregator to calculate edge
    vs Kalshi prediction market prices.
    
    Key Strategy Components:
    1. CME FedWatch probabilities as baseline
    2. Adjust based on economic indicators (FRED, BLS, Atlanta Fed)
    3. Compare to Kalshi market prices
    4. Trade when edge > 5%
    """
    
    # Fed series on Kalshi (updated January 2026)
    FED_SERIES_TICKERS = ["KXFED", "KXFEDDECISION", "KXFEDHIKE", "KXRATECUTCOUNT"]
    
    # Minimum requirements (raised after analysis: 8% still caught noise in live)
    MIN_EDGE = 0.12  # 12% minimum edge (raised from 8% — only trade strong signals)
    MIN_CONTRACT_PRICE = 0.10  # Never buy contracts under 10¢
    MAX_CONTRACT_PRICE = 0.90  # Never buy contracts over 90¢
    MIN_VOLUME = 50  # Minimum contracts traded (raised back to 50 for liquidity)
    DATA_FRESHNESS_HOURS = 4  # Skip trading if fed data is older than this
    
    # Position sizing
    MAX_POSITION_USD = 15.0  # Max per contract type
    MAX_CONCURRENT_POSITIONS = 3  # Diversification limit
    
    def __init__(
        self,
        api_key_id: str = None,
        private_key_path: str = None,
        fed_aggregator_path: Optional[str] = None,
        fred_api_key: Optional[str] = None,
        paper_mode: bool = True,
        infrastructure=None
    ):
        """
        Initialize Fed strategy.

        Args:
            api_key_id: Kalshi API key ID (or uses env var)
            private_key_path: Path to PEM private key file (or uses env var)
            fed_aggregator_path: Path to fed_data_latest.json from aggregator
            fred_api_key: FRED API key for backup data fetching
            paper_mode: If True, only log trades without executing
            infrastructure: KalshiInfrastructure instance (injected by orchestrator)
        """
        self.infrastructure = infrastructure
        if infrastructure and hasattr(infrastructure, 'client'):
            self.client = infrastructure.client
        else:
            self.client = KalshiClient(api_key_id, private_key_path)
        self.fed_aggregator_path = fed_aggregator_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'fed_data_latest.json'
        )
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.paper_mode = paper_mode
        self.current_positions: List[Dict] = []
        self._last_data_source: Optional[str] = None  # Track which source was used

        logger.info(f"KalshiFedStrategy initialized (paper_mode={paper_mode}, infrastructure={'yes' if infrastructure else 'no'})")
    
    def _check_data_freshness(self) -> bool:
        """
        Check if fed_data_latest.json is fresh enough to trade on.
        Returns True if fresh, False if stale.
        """
        if not self.fed_aggregator_path or not os.path.exists(self.fed_aggregator_path):
            return False

        try:
            mtime = os.path.getmtime(self.fed_aggregator_path)
            age_hours = (time.time() - mtime) / 3600
            if age_hours > self.DATA_FRESHNESS_HOURS:
                logger.warning(f"Fed data is {age_hours:.1f}h old (limit: {self.DATA_FRESHNESS_HOURS}h) — too stale to trade")
                return False
            return True
        except Exception:
            return False

    def fetch_cme_fedwatch_probabilities(self) -> Optional[Dict[str, float]]:
        """
        Fetch CME FedWatch Tool probabilities.

        Returns None if no fresh data available (instead of falling back to stale/hardcoded data).
        This prevents trading on bad information.

        Returns:
            Dict with decision probabilities, or None if no fresh data
        """
        # Try to read from Fed aggregator output first
        if self.fed_aggregator_path and os.path.exists(self.fed_aggregator_path):
            try:
                with open(self.fed_aggregator_path, 'r') as f:
                    data = json.load(f)
                    if 'cme_fedwatch' in data and data['cme_fedwatch'].get('success'):
                        probs = data['cme_fedwatch']['data']
                        self._last_data_source = 'aggregator_file'
                        logger.info(f"Loaded CME FedWatch from aggregator: {probs}")
                        return probs
            except Exception as e:
                logger.warning(f"Could not load aggregator data: {e}")

        # Fallback: Use CME FedWatch scraper
        try:
            from scrapers.cme_fedwatch_scraper import CMEFedWatchScraper
            scraper = CMEFedWatchScraper()
            probs = scraper.fetch_probabilities()
            self._last_data_source = 'cme_scraper'
            logger.info(f"Fetched CME FedWatch via scraper: {probs}")

            # Save for next time
            if self.fed_aggregator_path:
                scraper.save_to_file(self.fed_aggregator_path)

            return probs
        except Exception as e:
            logger.error(f"CME FedWatch scraper failed: {e}")

        # NO FALLBACK — return None instead of stale/hardcoded data
        # The caller (run_scan) will skip trading when data is None
        logger.warning("No fresh Fed data available — skipping trading")
        self._last_data_source = None
        return None

    def _get_policy_aware_baseline(self) -> Dict[str, float]:
        """
        Get intelligent baseline based on current Fed policy.

        Uses FRED to detect hiking/pausing/cutting cycle.
        Much better than hardcoded 75% hold probability.
        """
        try:
            # Use the CME scraper's policy baseline method
            from scrapers.cme_fedwatch_scraper import CMEFedWatchScraper
            scraper = CMEFedWatchScraper()
            return scraper._get_policy_baseline()
        except Exception:
            # Absolute last resort: neutral baseline
            logger.error("Policy baseline failed, using neutral probabilities")
            return {
                "hold": 0.80,
                "hike_25": 0.08,
                "hike_50": 0.02,
                "cut_25": 0.08,
                "cut_50": 0.02
            }
    
    _fred_cache = {}  # class-level cache: {series_id: (timestamp, value)}
    _FRED_CACHE_TTL = 3600  # FRED data changes daily at most — cache 1 hour

    def fetch_economic_indicators(self) -> Dict:
        """
        Fetch key economic indicators from FRED (with 1-hour cache).

        Returns:
            Dict with economic data
        """
        if not self.fred_api_key:
            logger.warning("No FRED API key, skipping economic indicators")
            return {}

        indicators = {}
        series_ids = {
            "DFF": "fed_funds_rate",
            "UNRATE": "unemployment_rate",
            "CPIAUCSL": "cpi",
            "T10Y2Y": "yield_curve_spread"
        }

        now = time.time()
        for series_id, name in series_ids.items():
            # Check cache first
            cached = self._fred_cache.get(series_id)
            if cached and (now - cached[0]) < self._FRED_CACHE_TTL:
                indicators[name] = cached[1]
                continue

            try:
                url = f"https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": series_id,
                    "api_key": self.fred_api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 1
                }
                response = requests.get(url, params=params, timeout=5)
                data = response.json()

                if data.get('observations'):
                    value = float(data['observations'][0]['value'])
                    indicators[name] = value
                    self._fred_cache[series_id] = (now, value)
                    logger.debug(f"FRED {name}: {value}")

            except Exception as e:
                logger.warning(f"Could not fetch FRED {series_id}: {e}")
                # Use stale cache on failure
                if cached:
                    indicators[name] = cached[1]

        return indicators
    
    def analyze_fed_decision_probability(self) -> Optional[FedProbabilities]:
        """
        Calculate Fed decision probabilities using multiple data sources.

        Methodology:
        1. Start with CME FedWatch as baseline (market consensus)
        2. Adjust based on economic indicators
        3. Weight recent data more heavily

        Returns:
            FedProbabilities dataclass, or None if no fresh data available
        """
        reasoning = []

        # Get CME FedWatch baseline — returns None if no fresh data
        cme_probs = self.fetch_cme_fedwatch_probabilities()
        if cme_probs is None:
            return None

        if self._last_data_source:
            reasoning.append(f"Data source: {self._last_data_source}")
        reasoning.append(f"CME FedWatch baseline: Hold={cme_probs['hold']:.0%}")
        
        # Get economic indicators
        indicators = self.fetch_economic_indicators()
        
        # Start with CME probabilities
        hold = cme_probs.get('hold', 0.75)
        hike_25 = cme_probs.get('hike_25', 0.10)
        hike_50 = cme_probs.get('hike_50', 0.02)
        cut_25 = cme_probs.get('cut_25', 0.10)
        cut_50 = cme_probs.get('cut_50', 0.03)
        
        # Adjust based on economic conditions
        if indicators:
            # Unemployment adjustment
            unemp = indicators.get('unemployment_rate', 4.0)
            if unemp > 5.0:
                # High unemployment -> more likely to cut
                cut_25 += 0.05
                hold -= 0.05
                reasoning.append(f"High unemployment ({unemp:.1f}%) → +5% cut probability")
            elif unemp < 3.5:
                # Very low unemployment -> more likely to hold/hike
                hold += 0.03
                reasoning.append(f"Low unemployment ({unemp:.1f}%) → +3% hold probability")
            
            # Yield curve adjustment
            spread = indicators.get('yield_curve_spread', 0)
            if spread < 0:
                # Inverted yield curve -> economy weakening -> more likely to cut
                cut_25 += 0.05
                hold -= 0.03
                reasoning.append(f"Inverted yield curve ({spread:.2f}%) → +5% cut probability")
        
        # Normalize to sum to 1.0
        total = hold + hike_25 + hike_50 + cut_25 + cut_50
        hold /= total
        hike_25 /= total
        hike_50 /= total
        cut_25 /= total
        cut_50 /= total
        
        # Determine confidence level
        if indicators and len(indicators) >= 3:
            confidence = "HIGH"
        elif indicators:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return FedProbabilities(
            hold=hold,
            hike_25=hike_25,
            hike_50=hike_50,
            cut_25=cut_25,
            cut_50=cut_50,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
    
    def find_fed_markets(self) -> List[Dict]:
        """
        Find Fed-related markets on Kalshi.

        Uses the correct API endpoints:
        1. First try GET /series/{series_ticker} to verify series exists
        2. Then GET /markets?series_ticker={ticker}&status=open

        Returns:
            List of Fed decision markets
        """
        fed_markets = []
        found_series = []

        # First, verify which series exist
        for series_ticker in self.FED_SERIES_TICKERS:
            try:
                series_info = self.client.get_series(series_ticker)
                if series_info:
                    found_series.append(series_ticker)
                    logger.info(f"Found series: {series_ticker} - {series_info.get('title', 'N/A')}")
            except Exception as e:
                logger.debug(f"Series {series_ticker} not found: {e}")

        # Query markets for each found series
        for series_ticker in found_series:
            try:
                # Use series_ticker parameter with status=open
                markets = self.client.get_markets(series_ticker=series_ticker, status="open")
                if markets:
                    fed_markets.extend(markets)
                    logger.info(f"Found {len(markets)} open markets for series {series_ticker}")
                    for m in markets[:3]:  # Log first 3
                        logger.debug(f"  - {m.get('ticker')}: {m.get('title', '')[:50]}")
            except Exception as e:
                logger.warning(f"Could not fetch markets for {series_ticker}: {e}")

        # If no series found, try keyword search with event_ticker prefix
        if not fed_markets:
            logger.info("No series markets found, trying event ticker prefixes...")
            event_prefixes = ["FED", "FOMC", "RATE"]
            try:
                # Get markets and filter by event_ticker prefix (Kalshi max limit is ~200)
                all_markets = self.client.get_markets(limit=200, status="open")
                for market in all_markets:
                    event_ticker = market.get('event_ticker', '').upper()
                    ticker = market.get('ticker', '').upper()
                    title = market.get('title', '').lower()

                    # Check event_ticker prefix
                    if any(event_ticker.startswith(prefix) for prefix in event_prefixes):
                        fed_markets.append(market)
                        continue

                    # Check ticker prefix
                    if any(ticker.startswith(prefix) for prefix in event_prefixes):
                        fed_markets.append(market)
                        continue

                    # Check keywords in title
                    fed_keywords = ['federal reserve', 'fed funds', 'fomc', 'interest rate', 'fed rate', 'rate cut', 'rate hike']
                    if any(kw in title for kw in fed_keywords):
                        fed_markets.append(market)

            except Exception as e:
                logger.warning(f"Could not search markets: {e}")

        # Deduplicate by ticker
        seen_tickers = set()
        unique_markets = []
        for market in fed_markets:
            ticker = market.get('ticker')
            if ticker and ticker not in seen_tickers:
                seen_tickers.add(ticker)
                unique_markets.append(market)

        logger.info(f"Found {len(unique_markets)} unique Fed-related markets")
        return unique_markets
    
    def map_market_to_outcome(self, market: Dict) -> Optional[str]:
        """
        Map a Kalshi market to a Fed decision outcome.

        Args:
            market: Kalshi market dictionary

        Returns:
            Outcome string ('hold', 'hike_25', etc.) or None
        """
        ticker = market.get('ticker', '').upper()
        title = market.get('title', '').lower()

        # KXFEDDECISION ticker format: KXFEDDECISION-DDMON-Xnn
        # H0 = 0bps hike (hold), H25 = 25bps hike, H26 = >25bps hike
        # C25 = 25bps cut, C26 = >25bps cut
        if 'KXFEDDECISION' in ticker:
            if '-H0' in ticker:
                return 'hold'  # 0bps hike = hold
            elif '-H25' in ticker:
                return 'hike_25'
            elif '-H26' in ticker or '-H50' in ticker:
                return 'hike_50'
            elif '-C25' in ticker:
                return 'cut_25'
            elif '-C26' in ticker or '-C50' in ticker:
                return 'cut_50'

        # Fallback to title-based mapping
        if 'unchanged' in title or 'no change' in title or 'hold' in title or '0bps' in title or '0 bps' in title:
            return 'hold'
        elif '25' in title and ('hike' in title or 'raise' in title or 'increase' in title):
            return 'hike_25'
        elif ('50' in title or '>25' in title) and ('hike' in title or 'raise' in title or 'increase' in title):
            return 'hike_50'
        elif '25' in title and ('cut' in title or 'lower' in title or 'decrease' in title):
            return 'cut_25'
        elif ('50' in title or '>25' in title) and ('cut' in title or 'lower' in title or 'decrease' in title):
            return 'cut_50'

        return None

    def _parse_orderbook(self, orderbook: Dict) -> Tuple[float, float, float, float]:
        """
        Parse Kalshi orderbook format to extract best bid/ask.

        Orderbook format: {'yes': [[price, qty], ...], 'no': [[price, qty], ...]}
        Price is in cents (1-99).

        In binary markets:
        - YES orders = bids to BUY yes
        - NO orders = bids to BUY no
        - To BUY yes at the ask: pay (100 - best NO bid) cents
        - To BUY no at the ask: pay (100 - best YES bid) cents

        Returns:
            Tuple of (yes_bid, yes_ask, no_bid, no_ask) as probabilities (0-1)
        """
        def get_best_bid(orders: List) -> int:
            """Get highest bid (highest price someone willing to pay)"""
            if not orders:
                return 0
            prices = [order[0] for order in orders if len(order) >= 2 and order[1] > 0]
            return max(prices) if prices else 0

        yes_orders = orderbook.get('yes', [])
        no_orders = orderbook.get('no', [])

        best_yes_bid = get_best_bid(yes_orders)
        best_no_bid = get_best_bid(no_orders)

        # Calculate asks from opposing bids
        yes_ask = (100 - best_no_bid) / 100 if best_no_bid > 0 else 1.0
        no_ask = (100 - best_yes_bid) / 100 if best_yes_bid > 0 else 1.0

        yes_bid = best_yes_bid / 100
        no_bid = best_no_bid / 100

        return yes_bid, yes_ask, no_bid, no_ask

    def find_mispriced_contracts(self) -> List[TradeSignal]:
        """
        Find contracts where our probability estimate differs
        from market price by more than MIN_EDGE.

        Returns:
            List of TradeSignal opportunities sorted by edge
        """
        # Get our probability estimates — returns None if data is stale
        our_probs = self.analyze_fed_decision_probability()
        if our_probs is None:
            logger.warning("No fresh Fed data — cannot find mispriced contracts")
            return []
        logger.info(f"Our Fed probabilities: Hold={our_probs.hold:.1%}, "
                   f"Hike25={our_probs.hike_25:.1%}, Cut25={our_probs.cut_25:.1%}")

        # Get Fed markets from Kalshi
        fed_markets = self.find_fed_markets()

        opportunities = []

        for market in fed_markets:
            ticker = market.get('ticker')
            outcome = self.map_market_to_outcome(market)

            if not outcome:
                continue

            # Get our probability for this outcome
            our_prob = getattr(our_probs, outcome, None)
            if our_prob is None:
                continue

            # Get market price from orderbook
            try:
                orderbook = self.client.get_orderbook(ticker)
                yes_bid, yes_ask, no_bid, no_ask = self._parse_orderbook(orderbook)
            except Exception as e:
                logger.warning(f"Could not get orderbook for {ticker}: {e}")
                continue

            # Check volume/liquidity - use open_interest as fallback
            volume = market.get('volume', 0) or market.get('open_interest', 0)
            if volume < self.MIN_VOLUME:
                continue

            # Evaluate YES opportunity (we think outcome more likely than market)
            if yes_ask >= self.MIN_CONTRACT_PRICE and yes_ask <= self.MAX_CONTRACT_PRICE:
                edge = our_prob - yes_ask
                if edge >= self.MIN_EDGE:
                    contracts = int(self.MAX_POSITION_USD / yes_ask)
                    if contracts > 0:
                        opportunities.append(TradeSignal(
                            ticker=ticker,
                            outcome=outcome,
                            side="yes",
                            market_price=yes_ask,
                            our_probability=our_prob,
                            edge=edge,
                            contracts=contracts,
                            max_cost_usd=contracts * yes_ask,
                            reasoning=f"Our {outcome} prob {our_prob:.1%} > market {yes_ask:.1%}, edge={edge:.1%}"
                        ))
            
            # Evaluate NO opportunity (we think outcome less likely than market)
            implied_yes = 1 - no_ask if no_ask else 0.5
            if no_ask >= self.MIN_CONTRACT_PRICE and no_ask <= self.MAX_CONTRACT_PRICE:
                edge = no_ask - (1 - our_prob)  # Edge on NO side
                if edge >= self.MIN_EDGE:
                    contracts = int(self.MAX_POSITION_USD / no_ask)
                    if contracts > 0:
                        opportunities.append(TradeSignal(
                            ticker=ticker,
                            outcome=outcome,
                            side="no",
                            market_price=no_ask,
                            our_probability=1 - our_prob,
                            edge=edge,
                            contracts=contracts,
                            max_cost_usd=contracts * no_ask,
                            reasoning=f"Our NOT-{outcome} prob {1-our_prob:.1%}, market implies {1-implied_yes:.1%}, edge={edge:.1%}"
                        ))
        
        # Sort by edge (highest first)
        opportunities.sort(key=lambda x: x.edge, reverse=True)
        
        logger.info(f"Found {len(opportunities)} opportunities with edge >= {self.MIN_EDGE:.0%}")
        for opp in opportunities[:5]:
            logger.info(f"  {opp.ticker}: {opp.side.upper()} @ {opp.market_price:.0%}, edge={opp.edge:.1%}")
        
        return opportunities
    
    def place_order(self, ticker: str, side: str, quantity: int, price: int) -> Optional[Dict]:
        """
        Place a single order — called by the orchestrator's execution path.

        Args:
            ticker: Market ticker
            side: 'yes' or 'no'
            quantity: Number of contracts
            price: Price in cents (1-99)

        Returns:
            Order result dict or None
        """
        # Risk check via infrastructure
        if self.infrastructure:
            allowed, reason = self.infrastructure.risk_manager.check_trade_allowed(ticker, side, quantity)
            if not allowed:
                logger.warning(f"Risk blocked order: {reason}")
                return None

        if self.paper_mode:
            logger.info(f"[PAPER] place_order: {side.upper()} {quantity} @ {price}c on {ticker}")
            return {'paper': True, 'ticker': ticker, 'side': side, 'count': quantity, 'price': price}

        try:
            order = self.client.create_order(
                ticker=ticker,
                side=side,
                action="buy",
                count=quantity,
                price=price,
                order_type="limit"
            )
            logger.info(f"Order placed: {side.upper()} {quantity} @ {price}c on {ticker}")
            return order
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    def execute_signal(self, signal: TradeSignal) -> Optional[Dict]:
        """
        Execute a trade signal on Kalshi (used by execute_best_opportunities).
        Now routes through place_order() for consistent risk checking.
        """
        price_cents = int(signal.market_price * 100)
        return self.place_order(signal.ticker, signal.side, signal.contracts, price_cents)
    
    def execute_best_opportunities(
        self,
        max_positions: Optional[int] = None,
        capital_allocated: Optional[float] = None
    ) -> List[Dict]:
        """
        Find and execute the best trading opportunities.
        
        Args:
            max_positions: Maximum positions to take (default: MAX_CONCURRENT_POSITIONS)
            capital_allocated: Total capital to deploy (default: MAX_POSITION_USD * max_positions)
            
        Returns:
            List of executed orders
        """
        max_positions = max_positions or self.MAX_CONCURRENT_POSITIONS
        capital = capital_allocated or (self.MAX_POSITION_USD * max_positions)
        
        # Get current positions to avoid duplicates
        try:
            self.current_positions = self.client.get_positions()
            current_tickers = {p.get('ticker') for p in self.current_positions}
        except Exception:
            current_tickers = set()
        
        # Find opportunities
        opportunities = self.find_mispriced_contracts()
        
        # Filter out markets we already have positions in
        opportunities = [o for o in opportunities if o.ticker not in current_tickers]
        
        # Execute top opportunities
        executed = []
        remaining_capital = capital
        
        for opp in opportunities[:max_positions]:
            if remaining_capital <= 0:
                break
            
            # Adjust contracts based on remaining capital
            max_contracts = int(remaining_capital / opp.market_price)
            if max_contracts <= 0:
                continue
            
            opp.contracts = min(opp.contracts, max_contracts)
            opp.max_cost_usd = opp.contracts * opp.market_price
            
            result = self.execute_signal(opp)
            if result:
                executed.append(result)
                remaining_capital -= opp.max_cost_usd
        
        logger.info(f"Executed {len(executed)} orders, deployed ${capital - remaining_capital:.2f}")
        return executed
    
    MAX_CYCLE_SECONDS = 90  # Return before orchestrator's 120s kill timeout

    def run_scan(self) -> List[Dict]:
        """
        Single-pass scan called by the orchestrator.
        Returns list of signal dicts for the orchestrator to evaluate/execute.
        Does NOT self-execute — orchestrator handles AI veto, risk, and execution.
        Has 90s cumulative timeout to return before orchestrator's 120s kill.
        """
        cycle_start = time.time()
        logger.info("Running Fed strategy single-pass scan...")
        signals = []

        try:
            # Data freshness gate — skip if data is too stale
            if not self._check_data_freshness():
                logger.info("Fed data stale or missing — skipping scan (no fallback to hardcoded probs)")
                return signals

            balance = self.client.get_balance()
            logger.info(f"Account balance: ${balance.get('balance', 0) / 100:.2f}")

            elapsed = time.time() - cycle_start
            if elapsed > self.MAX_CYCLE_SECONDS:
                logger.warning(f"Fed scan timeout after {elapsed:.0f}s (balance phase)")
                return signals

            # Find opportunities (does NOT execute — just returns signals)
            opportunities = self.find_mispriced_contracts()

            elapsed = time.time() - cycle_start
            if elapsed > self.MAX_CYCLE_SECONDS:
                logger.warning(f"Fed scan timeout after {elapsed:.0f}s (analysis phase)")
                return signals

            # Get current positions to avoid duplicates
            try:
                self.current_positions = self.client.get_positions()
                current_tickers = {p.get('ticker') for p in self.current_positions}
            except Exception:
                current_tickers = set()

            for opp in opportunities[:self.MAX_CONCURRENT_POSITIONS]:
                if opp.ticker in current_tickers:
                    continue

                price_cents = int(opp.market_price * 100)
                price_cents = max(1, min(99, price_cents))

                signals.append({
                    'ticker': opp.ticker,
                    'action': 'buy',
                    'side': opp.side,
                    'quantity': opp.contracts,
                    'price_cents': price_cents,
                    'edge': opp.edge,
                    'our_probability': opp.our_probability,
                    'reasoning': opp.reasoning,
                    'data_source': self._last_data_source,
                    'type': 'kalshi_fed',
                })

            logger.info(f"Fed scan complete: {len(signals)} signals in {time.time() - cycle_start:.1f}s")

        except Exception as e:
            logger.error(f"Fed scan error: {e}")

        return signals

    def run(self, interval_minutes: int = 60):
        """
        Main strategy loop (standalone use only — orchestrator should call run_scan()).

        Args:
            interval_minutes: Minutes between strategy runs
        """
        logger.info(f"Starting Fed strategy loop (interval={interval_minutes}min, paper={self.paper_mode})")
        
        while True:
            try:
                logger.info("=" * 60)
                logger.info("Running Fed strategy analysis...")
                
                # Check balance
                balance = self.client.get_balance()
                logger.info(f"Account balance: ${balance.get('balance', 0) / 100:.2f}")
                
                # Execute strategy
                executed = self.execute_best_opportunities()
                
                # Log summary
                if executed:
                    logger.info(f"Executed {len(executed)} new positions")
                else:
                    logger.info("No opportunities found meeting criteria")
                
                # Wait for next iteration
                logger.info(f"Sleeping for {interval_minutes} minutes...")
                import time
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Strategy stopped by user")
                break
            except Exception as e:
                logger.error(f"Strategy error: {e}")
                import traceback
                traceback.print_exc()
                # Wait before retrying
                import time
                time.sleep(60)


def main():
    """Entry point for testing"""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    api_key_id = os.getenv('KALSHI_API_KEY_ID')
    private_key_path = os.path.expanduser(os.getenv('KALSHI_PRIVATE_KEY_PATH', '~/.trading_keys/kalshi_private_key.pem'))
    fred_key = os.getenv('FRED_API_KEY')

    if not api_key_id:
        print("Error: Set KALSHI_API_KEY_ID in .env")
        return

    strategy = KalshiFedStrategy(
        api_key_id=api_key_id,
        private_key_path=private_key_path,
        fred_api_key=fred_key,
        paper_mode=True  # Always start in paper mode!
    )
    
    # Run once for testing
    opportunities = strategy.find_mispriced_contracts()
    print(f"\nFound {len(opportunities)} opportunities")
    
    for opp in opportunities[:5]:
        print(f"\n{opp.ticker}:")
        print(f"  Side: {opp.side.upper()}")
        print(f"  Market Price: {opp.market_price:.1%}")
        print(f"  Our Probability: {opp.our_probability:.1%}")
        print(f"  Edge: {opp.edge:.1%}")
        print(f"  Contracts: {opp.contracts}")
        print(f"  Max Cost: ${opp.max_cost_usd:.2f}")
        print(f"  Reasoning: {opp.reasoning}")


if __name__ == "__main__":
    main()
