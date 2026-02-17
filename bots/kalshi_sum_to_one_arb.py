"""
Kalshi Sum-to-One Arbitrage Bot

Exploits multi-outcome events where all outcome prices sum to less than $1.00.
This is PURE RISK-FREE ARBITRAGE when it exists.

Example:
    Event: "Who will win the 2026 NBA Finals?"
    Lakers @ $0.25
    Celtics @ $0.30
    Warriors @ $0.20
    Other @ $0.20
    Total: $0.95 (should be $1.00)

    Buy all outcomes for $0.95, ONE MUST WIN -> Receive $1.00
    Guaranteed $0.05 profit (5.26% ROI)

Research shows these opportunities are rare but very profitable.
Speed is critical - they close within seconds.

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import time
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('KalshiSumToOneArb')


@dataclass
class SumToOneOpportunity:
    """Represents a sum-to-one arbitrage opportunity"""
    event_ticker: str
    event_title: str
    markets: List[Dict]  # List of market tickers with prices
    total_cost: float  # Cost to buy all outcomes (should be < 1.0)
    profit_per_share: float
    profit_pct: float
    min_liquidity: int  # Minimum contracts available across all outcomes
    timestamp: datetime


@dataclass
class ArbExecution:
    """Tracks an executed arbitrage"""
    opportunity: SumToOneOpportunity
    order_ids: List[str]
    contracts_per_outcome: int
    total_cost: float
    expected_profit: float
    status: str
    execution_time: datetime


class KalshiSumToOneArbitrage:
    """
    Scans Kalshi events for sum-to-one arbitrage.

    Multi-outcome events (like elections, sports) should have
    all outcomes sum to 100%. When they sum to less, buying
    all outcomes guarantees profit at settlement.

    Kalshi fees: ~1.75% max at 50c midpoint
    Min profit threshold should exceed fees.

    Usage:
        arb = KalshiSumToOneArbitrage(paper_mode=True)
        opportunities = arb.scan_all_events()
        for opp in opportunities:
            if opp.profit_pct > 0.03:  # 3% threshold
                arb.execute(opp, contracts=10)
    """

    # Kalshi fee formula: 0.07 * price * (1-price)
    # Max fee is 1.75 cents at 50c
    MAX_FEE_PER_CONTRACT = 0.0175

    # Minimum profit after fees to consider (raised from 2% to 5% — Kalshi fees + slippage)
    MIN_PROFIT_THRESHOLD = 0.05  # 5%

    # Minimum outcomes to consider (single market = 2)
    MIN_OUTCOMES = 3

    # Scan-only mode: log opportunities but don't auto-execute
    SCAN_ONLY = True

    def __init__(self, paper_mode: bool = True, infrastructure=None):
        """
        Initialize sum-to-one arbitrage scanner.

        Args:
            paper_mode: If True, simulate trades without execution
            infrastructure: KalshiInfrastructure instance (injected by orchestrator)
        """
        self.paper_mode = paper_mode
        self.infrastructure = infrastructure

        # Initialize Kalshi client
        if infrastructure and hasattr(infrastructure, 'client'):
            self.client = infrastructure.client
            self._connected = True
        else:
            try:
                self.client = KalshiClient()
                self._connected = True
                logger.info("Kalshi client connected")
            except Exception as e:
                logger.warning(f"Kalshi client not available: {e}")
                self.client = None
                self._connected = False

        # Track executions
        self._executions: List[ArbExecution] = []
        self._paper_balance = 500.0  # Starting paper balance

        # Database for logging
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'kalshi_sum_to_one_arb.db'
        )
        self._init_database()

        logger.info(f"Sum-to-One Arbitrage initialized (paper_mode={paper_mode})")

    def _init_database(self):
        """Initialize SQLite database for tracking."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_ticker TEXT,
                event_title TEXT,
                num_outcomes INTEGER,
                total_cost REAL,
                profit_cents REAL,
                profit_pct REAL,
                executed INTEGER DEFAULT 0
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                opportunity_id INTEGER,
                timestamp TEXT,
                contracts INTEGER,
                total_cost REAL,
                expected_profit REAL,
                status TEXT,
                actual_profit REAL
            )
        ''')

        conn.commit()
        conn.close()

    def calculate_fee(self, price_cents: int) -> float:
        """
        Calculate Kalshi fee for a given price.

        Fee = 0.07 * price * (1 - price)
        Max fee = 1.75 cents at 50c

        Args:
            price_cents: Price in cents (1-99)

        Returns:
            Fee in cents
        """
        price = price_cents / 100
        fee = 0.07 * price * (1 - price) * 100  # Convert back to cents
        return min(fee, 1.75)  # Cap at 1.75 cents

    def analyze_event(self, event: Dict) -> Optional[SumToOneOpportunity]:
        """
        Analyze an event with multiple markets for sum-to-one arbitrage.

        Args:
            event: Event data from Kalshi API

        Returns:
            SumToOneOpportunity if found, None otherwise
        """
        event_ticker = event.get('event_ticker', event.get('ticker', ''))
        event_title = event.get('title', '')

        # Get markets: prefer embedded data, fall back to API call
        markets = event.get('markets', [])
        if not markets and self.client and self._connected:
            try:
                markets = self.client.get_markets(series_ticker=event_ticker)
            except Exception as e:
                logger.debug(f"Could not get markets for {event_ticker}: {e}")
                return None

        if len(markets) < self.MIN_OUTCOMES:
            return None

        # Quick pre-filter: use embedded prices to skip obvious non-arbs
        quick_total = sum(
            m.get('yes_ask', m.get('yes_price', m.get('last_price', 50)))
            for m in markets
        )
        if quick_total >= 100:
            return None

        # Fetch orderbooks in parallel for accurate prices
        def _fetch_orderbook(market):
            ticker = market.get('ticker', '')
            try:
                if self.client and self._connected:
                    orderbook = self.client.get_orderbook(ticker)
                    yes_asks = orderbook.get('yes', [])
                    if yes_asks:
                        yes_price = min(a.get('price', 100) for a in yes_asks)
                        liquidity = sum(a.get('count', 0) for a in yes_asks if a.get('price', 100) <= yes_price)
                        return market, yes_price, liquidity
                # Fallback to market data
                return market, market.get('yes_ask', market.get('yes_price', 50)), 10
            except Exception:
                return market, market.get('yes_ask', market.get('yes_price', 50)), 10

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(_fetch_orderbook, markets))

        # Get YES prices for each market (each outcome)
        market_prices = []
        total_cost_cents = 0
        min_liquidity = float('inf')
        total_fees = 0

        for market, yes_price, liquidity in results:
            ticker = market.get('ticker', '')
            market_prices.append({
                'ticker': ticker,
                'title': market.get('title', market.get('subtitle', '')),
                'yes_price': yes_price,
                'liquidity': liquidity
            })

            total_cost_cents += yes_price
            min_liquidity = min(min_liquidity, liquidity)
            total_fees += self.calculate_fee(yes_price)

        # If total < 100, there's arbitrage
        if total_cost_cents >= 100:
            return None

        # Calculate profit before and after fees
        gross_profit_cents = 100 - total_cost_cents
        net_profit_cents = gross_profit_cents - total_fees

        if net_profit_cents <= 0:
            return None

        profit_pct = net_profit_cents / total_cost_cents

        if profit_pct < self.MIN_PROFIT_THRESHOLD:
            return None

        return SumToOneOpportunity(
            event_ticker=event_ticker,
            event_title=event_title[:100] if event_title else event_ticker,
            markets=market_prices,
            total_cost=total_cost_cents / 100,
            profit_per_share=net_profit_cents / 100,
            profit_pct=profit_pct,
            min_liquidity=int(min_liquidity) if min_liquidity != float('inf') else 0,
            timestamp=datetime.now(timezone.utc)
        )

    def scan_all_events(self, min_profit_pct: float = None) -> List[SumToOneOpportunity]:
        """
        Scan all Kalshi events for sum-to-one arbitrage.
        Optimized: parallel event analysis with ThreadPoolExecutor.

        Args:
            min_profit_pct: Minimum profit percentage (default: MIN_PROFIT_THRESHOLD)

        Returns:
            List of opportunities sorted by profit potential
        """
        if min_profit_pct is None:
            min_profit_pct = self.MIN_PROFIT_THRESHOLD

        logger.info("Scanning all events for sum-to-one arbitrage...")

        # Get all open events
        if self.client and self._connected:
            try:
                events = self.client.get_events(status='open', limit=200)
            except Exception as e:
                logger.error(f"Failed to fetch events: {e}")
                events = self._get_mock_events()
        else:
            events = self._get_mock_events()

        logger.info(f"Analyzing {len(events)} events...")

        opportunities = []

        # Parallel analysis — 10 workers keeps us under Kalshi rate limits
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.analyze_event, event): event for event in events}
            for future in as_completed(futures):
                try:
                    opp = future.result(timeout=30)
                    if opp and opp.profit_pct >= min_profit_pct:
                        opportunities.append(opp)
                        self._log_opportunity(opp)
                except Exception as e:
                    logger.debug(f"Error analyzing event: {e}")

        # Sort by profit percentage
        opportunities.sort(key=lambda x: x.profit_pct, reverse=True)

        logger.info(f"Found {len(opportunities)} sum-to-one arbitrage opportunities")
        return opportunities

    def _get_mock_events(self) -> List[Dict]:
        """Return mock events for testing without API."""
        return [
            {
                'event_ticker': 'KXNBA-26FINALS',
                'title': 'Who will win the 2026 NBA Finals?',
                'markets': [
                    {'ticker': 'KXNBA-26FINALS-LAL', 'title': 'Lakers', 'yes_ask': 25},
                    {'ticker': 'KXNBA-26FINALS-BOS', 'title': 'Celtics', 'yes_ask': 28},
                    {'ticker': 'KXNBA-26FINALS-GSW', 'title': 'Warriors', 'yes_ask': 18},
                    {'ticker': 'KXNBA-26FINALS-OTH', 'title': 'Other', 'yes_ask': 24},
                ]  # Total: 95 = 5% arb
            },
            {
                'event_ticker': 'KXOSCARS-26-DIRECTOR',
                'title': 'Who will win Best Director at 2026 Oscars?',
                'markets': [
                    {'ticker': 'KXOSCARS-26-DIR-A', 'title': 'Director A', 'yes_ask': 30},
                    {'ticker': 'KXOSCARS-26-DIR-B', 'title': 'Director B', 'yes_ask': 25},
                    {'ticker': 'KXOSCARS-26-DIR-C', 'title': 'Director C', 'yes_ask': 20},
                    {'ticker': 'KXOSCARS-26-DIR-OTH', 'title': 'Other', 'yes_ask': 21},
                ]  # Total: 96 = 4% arb
            },
            {
                'event_ticker': 'KXSUPERBOWL-26',
                'title': 'Who will win Super Bowl 2026?',
                'markets': [
                    {'ticker': 'KXSB-26-KC', 'title': 'Chiefs', 'yes_ask': 22},
                    {'ticker': 'KXSB-26-SF', 'title': '49ers', 'yes_ask': 18},
                    {'ticker': 'KXSB-26-BUF', 'title': 'Bills', 'yes_ask': 15},
                    {'ticker': 'KXSB-26-PHI', 'title': 'Eagles', 'yes_ask': 14},
                    {'ticker': 'KXSB-26-OTH', 'title': 'Other', 'yes_ask': 32},
                ]  # Total: 101 = no arb
            },
        ]

    def _log_opportunity(self, opp: SumToOneOpportunity):
        """Log opportunity to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO opportunities
                (timestamp, event_ticker, event_title, num_outcomes, total_cost,
                 profit_cents, profit_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                opp.timestamp.isoformat(),
                opp.event_ticker,
                opp.event_title,
                len(opp.markets),
                opp.total_cost,
                opp.profit_per_share * 100,
                opp.profit_pct
            ))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log opportunity: {e}")

    def execute(self, opportunity: SumToOneOpportunity, contracts: int) -> Optional[ArbExecution]:
        """
        Execute sum-to-one arbitrage by buying YES on all outcomes.

        Args:
            opportunity: The opportunity to execute
            contracts: Number of contracts to buy of each outcome

        Returns:
            ArbExecution tracking object
        """
        if self.paper_mode:
            return self._paper_execute(opportunity, contracts)

        return self._real_execute(opportunity, contracts)

    def _paper_execute(self, opp: SumToOneOpportunity, contracts: int) -> ArbExecution:
        """Simulate arbitrage execution in paper mode."""

        # Check paper balance
        total_cost = opp.total_cost * contracts
        if total_cost > self._paper_balance:
            contracts = int(self._paper_balance / opp.total_cost)
            if contracts <= 0:
                logger.warning("Insufficient paper balance")
                return None
            total_cost = opp.total_cost * contracts

        expected_profit = opp.profit_per_share * contracts

        # Generate paper order IDs
        order_ids = [f"paper_{int(time.time())}_{m['ticker']}" for m in opp.markets]

        execution = ArbExecution(
            opportunity=opp,
            order_ids=order_ids,
            contracts_per_outcome=contracts,
            total_cost=total_cost,
            expected_profit=expected_profit,
            status='complete',
            execution_time=datetime.now(timezone.utc)
        )

        self._executions.append(execution)
        self._paper_balance -= total_cost

        # Build market summary
        market_lines = []
        for m in opp.markets:
            market_lines.append(f"    - BUY {contracts} YES @ ${m['yes_price']/100:.2f} ({m['title'][:30]})")

        logger.info(f"""
+======================================================================+
|  [PAPER] SUM-TO-ONE ARBITRAGE EXECUTED                               |
+======================================================================+
|  Event: {opp.event_title[:55]}...
|
|  Orders ({len(opp.markets)} outcomes):
{chr(10).join(market_lines)}
|
|  Financials:
|    - Total Cost: ${total_cost:.2f}
|    - Guaranteed Payout: ${contracts:.2f} (one outcome wins)
|    - Expected Profit: ${expected_profit:.2f} ({opp.profit_pct:.2%})
|
|  Paper Balance: ${self._paper_balance:.2f}
+======================================================================+
""")

        return execution

    def _real_execute(self, opp: SumToOneOpportunity, contracts: int) -> Optional[ArbExecution]:
        """Execute real arbitrage trades on Kalshi."""

        if not self.client or not self._connected:
            logger.error("Kalshi client not available")
            return None

        order_ids = []

        try:
            # Place YES orders for all outcomes
            for market in opp.markets:
                order = self.client.create_order(
                    ticker=market['ticker'],
                    side='yes',
                    action='buy',
                    count=contracts,
                    price=market['yes_price'],
                    order_type='limit'
                )

                if not order:
                    logger.error(f"Order failed for {market['ticker']}")
                    # Cancel all previous orders
                    for oid in order_ids:
                        try:
                            self.client.cancel_order(oid)
                        except Exception as e:
                            logger.error(f"Failed to cancel order {oid} during rollback: {e}")
                    return None

                order_ids.append(order.get('order_id', ''))

            total_cost = opp.total_cost * contracts
            expected_profit = opp.profit_per_share * contracts

            execution = ArbExecution(
                opportunity=opp,
                order_ids=order_ids,
                contracts_per_outcome=contracts,
                total_cost=total_cost,
                expected_profit=expected_profit,
                status='complete',
                execution_time=datetime.now(timezone.utc)
            )

            self._executions.append(execution)

            logger.info(f"""
+======================================================================+
|  [LIVE] SUM-TO-ONE ARBITRAGE EXECUTED                                |
+======================================================================+
|  Event: {opp.event_title[:55]}...
|  Outcomes: {len(opp.markets)}
|  Contracts per outcome: {contracts}
|  Expected Profit: ${expected_profit:.2f} ({opp.profit_pct:.2%})
+======================================================================+
""")

            return execution

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            # Cancel any placed orders
            for oid in order_ids:
                try:
                    self.client.cancel_order(oid)
                except Exception as e:
                    logger.error(f"Failed to cancel order {oid} during cleanup: {e}")
            return None

    def get_summary(self) -> Dict:
        """Get summary of arbitrage activity."""
        return {
            'total_executions': len(self._executions),
            'total_invested': sum(e.total_cost for e in self._executions),
            'expected_profit': sum(e.expected_profit for e in self._executions),
            'paper_balance': self._paper_balance,
            'paper_mode': self.paper_mode,
            'connected': self._connected
        }

    def place_order(self, ticker: str, side: str, quantity: int, price: int) -> Optional[Dict]:
        """
        Place a single order — called by the orchestrator's execution path.
        """
        if self.infrastructure:
            allowed, reason = self.infrastructure.risk_manager.check_trade_allowed(ticker, side, quantity)
            if not allowed:
                logger.warning(f"Risk blocked arb order: {reason}")
                return None

        if self.paper_mode:
            logger.info(f"[PAPER] sum-to-one place_order: {side.upper()} {quantity} @ {price}c on {ticker}")
            return {'paper': True, 'ticker': ticker, 'side': side, 'count': quantity, 'price': price}

        if not self.client or not self._connected:
            return None

        try:
            return self.client.create_order(
                ticker=ticker, side=side, action='buy',
                count=quantity, price=price, order_type='limit'
            )
        except Exception as e:
            logger.error(f"Sum-to-one order failed: {e}")
            return None

    def run_scan(self) -> List[Dict]:
        """
        Main method called by orchestrator.
        Scans for sum-to-one arbitrage opportunities.
        SCAN-ONLY mode: logs opportunities but does NOT auto-execute.
        """
        logger.info("Starting sum-to-one arbitrage scan (scan-only mode)...")

        if not self._connected:
            logger.warning("Kalshi client not connected - skipping scan")
            return []

        signals = []

        try:
            opportunities = self.scan_all_events()

            for opp in opportunities:
                signal = {
                    'event_ticker': opp.event_ticker,
                    'event_title': opp.event_title,
                    'action': 'arbitrage',
                    'num_outcomes': len(opp.markets),
                    'total_cost': opp.total_cost,
                    'profit_pct': opp.profit_pct,
                    'type': 'sum_to_one_arb',
                    'scan_only': self.SCAN_ONLY,
                }

                if opp.profit_pct >= self.MIN_PROFIT_THRESHOLD:
                    logger.info(f"Sum-to-one opportunity: {opp.event_ticker} — {opp.profit_pct:.2%} profit, {opp.min_liquidity} min liquidity")

                signals.append(signal)

            logger.info(f"Sum-to-one scan complete: {len(opportunities)} opportunities found (scan-only, 0 executed)")

        except Exception as e:
            logger.error(f"Sum-to-one scan failed: {e}")

        return signals


def main():
    """Test sum-to-one arbitrage scanner."""
    print("=" * 70)
    print("KALSHI SUM-TO-ONE ARBITRAGE SCANNER")
    print("=" * 70)
    print("""
This bot finds multi-outcome events where all YES prices sum < $1.00

Example:
  Event: "Who will win the NBA Finals?"
  Lakers YES @ $0.25
  Celtics YES @ $0.28
  Warriors YES @ $0.18
  Other YES @ $0.24
  Total: $0.95 (should be $1.00)

  Buy all four for $0.95, ONE MUST WIN -> Receive $1.00
  Guaranteed $0.05 profit (5.26% ROI)

This is RISK-FREE mathematical arbitrage!
""")

    # Initialize scanner
    arb = KalshiSumToOneArbitrage(paper_mode=True)

    # Scan for opportunities
    print("\n" + "=" * 70)
    print("SCANNING FOR ARBITRAGE OPPORTUNITIES")
    print("=" * 70)

    opportunities = arb.scan_all_events(min_profit_pct=0.01)

    if opportunities:
        print(f"\nFound {len(opportunities)} opportunities:\n")

        for i, opp in enumerate(opportunities[:10], 1):
            print(f"{i}. {opp.event_ticker}")
            print(f"   {opp.event_title[:60]}...")
            print(f"   Outcomes: {len(opp.markets)}")
            print(f"   Total Cost: ${opp.total_cost:.2f} | Profit: ${opp.profit_per_share:.2f}/share ({opp.profit_pct:.2%})")
            print(f"   Min Liquidity: {opp.min_liquidity} contracts")
            print(f"   Markets:")
            for m in opp.markets[:5]:
                print(f"      - {m['title'][:25]}: ${m['yes_price']/100:.2f}")
            print()
    else:
        print("\nNo arbitrage opportunities found above threshold")
        print("   This is normal - opportunities close within seconds")

    # Execute best opportunity (paper)
    if opportunities:
        print("\n" + "=" * 70)
        print("EXECUTING BEST OPPORTUNITY (PAPER)")
        print("=" * 70)

        best = opportunities[0]
        execution = arb.execute(best, contracts=10)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = arb.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: ${value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\nSum-to-one arbitrage test complete!")


if __name__ == '__main__':
    main()
