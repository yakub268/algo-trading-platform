"""
Kalshi Probability Arbitrage Bot

Exploits markets where YES + NO prices sum to less than $1.00.
This is PURE RISK-FREE ARBITRAGE when it exists.

Example:
    YES @ $0.45 + NO @ $0.50 = $0.95
    Buy both for $0.95, one MUST pay $1.00
    Guaranteed $0.05 profit (5.26% ROI)

Research shows these opportunities close within seconds,
so speed and automation are critical.

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
from decimal import Decimal, ROUND_DOWN
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('KalshiProbabilityArb')


@dataclass
class ProbabilityArbOpportunity:
    """Represents a probability arbitrage opportunity"""
    ticker: str
    title: str
    yes_price: int  # cents
    no_price: int   # cents
    total_cost: int  # cents
    profit_per_contract: int  # cents
    profit_pct: float
    contracts_available: int  # minimum liquidity
    timestamp: datetime
    

@dataclass
class ArbExecution:
    """Tracks an executed arbitrage"""
    opportunity: ProbabilityArbOpportunity
    yes_order_id: str
    no_order_id: str
    contracts: int
    total_cost: float
    expected_profit: float
    status: str
    execution_time: datetime


class KalshiProbabilityArbitrage:
    """
    Scans Kalshi markets for probability arbitrage.
    
    When YES price + NO price < 100 cents, buying both
    guarantees profit at settlement.
    
    Kalshi fees: ~1.75% max at 50c midpoint
    Min profit threshold should exceed fees.
    
    Usage:
        arb = KalshiProbabilityArbitrage(paper_mode=True)
        opportunities = arb.scan_all_markets()
        for opp in opportunities:
            if opp.profit_pct > 0.03:  # 3% threshold
                arb.execute(opp, contracts=10)
    """
    
    # Kalshi fee formula: 0.07 * price * (1-price)
    # Max fee is 1.75 cents at 50c
    MAX_FEE_PCT = 0.0175
    
    # Minimum profit after fees to consider (raised from 2% to 5% — Kalshi fees + slippage)
    MIN_PROFIT_THRESHOLD = 0.05  # 5%

    # Scan-only mode: log opportunities but don't auto-execute
    SCAN_ONLY = True

    def __init__(self, paper_mode: bool = True, infrastructure=None):
        """
        Initialize probability arbitrage scanner.

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
            'data', 'kalshi_probability_arb.db'
        )
        self._init_database()
        
        logger.info(f"Probability Arbitrage initialized (paper_mode={paper_mode})")
    
    def _init_database(self):
        """Initialize SQLite database for tracking."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                ticker TEXT,
                title TEXT,
                yes_price INTEGER,
                no_price INTEGER,
                total_cost INTEGER,
                profit_cents INTEGER,
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
    
    def analyze_market(self, market: Dict) -> Optional[ProbabilityArbOpportunity]:
        """
        Analyze a single market for probability arbitrage.
        
        Args:
            market: Market data from Kalshi API
            
        Returns:
            ProbabilityArbOpportunity if found, None otherwise
        """
        ticker = market.get('ticker', '')
        title = market.get('title', market.get('subtitle', ''))
        
        # Get orderbook for accurate prices
        try:
            if self.client and self._connected:
                orderbook = self.client.get_orderbook(ticker)
            else:
                # Use market data if no client
                orderbook = {
                    'yes': [{'price': market.get('yes_ask', 50)}],
                    'no': [{'price': market.get('no_ask', 50)}]
                }
        except Exception as e:
            logger.debug(f"Could not get orderbook for {ticker}: {e}")
            return None
        
        # Get best ask prices (what we'd pay to buy)
        yes_asks = orderbook.get('yes', [])
        no_asks = orderbook.get('no', [])
        
        if not yes_asks or not no_asks:
            return None
        
        # Best ask is lowest price we can buy at
        yes_price = min(a.get('price', 100) for a in yes_asks) if yes_asks else 100
        no_price = min(a.get('price', 100) for a in no_asks) if no_asks else 100
        
        # Total cost to buy both
        total_cost = yes_price + no_price
        
        # If total < 100, there's arbitrage
        if total_cost >= 100:
            return None
        
        # Calculate profit before fees
        gross_profit = 100 - total_cost
        
        # Calculate fees for both sides
        yes_fee = self.calculate_fee(yes_price)
        no_fee = self.calculate_fee(no_price)
        total_fees = yes_fee + no_fee
        
        # Net profit after fees
        net_profit = gross_profit - total_fees
        
        if net_profit <= 0:
            return None
        
        profit_pct = net_profit / total_cost
        
        if profit_pct < self.MIN_PROFIT_THRESHOLD:
            return None
        
        # Check liquidity (minimum contracts available at these prices)
        yes_liquidity = sum(a.get('count', 0) for a in yes_asks if a.get('price', 100) <= yes_price)
        no_liquidity = sum(a.get('count', 0) for a in no_asks if a.get('price', 100) <= no_price)
        contracts_available = min(yes_liquidity, no_liquidity)
        
        return ProbabilityArbOpportunity(
            ticker=ticker,
            title=title[:100] if title else ticker,
            yes_price=yes_price,
            no_price=no_price,
            total_cost=total_cost,
            profit_per_contract=int(net_profit),
            profit_pct=profit_pct,
            contracts_available=contracts_available,
            timestamp=datetime.now(timezone.utc)
        )
    
    def scan_all_markets(self, min_profit_pct: float = None) -> List[ProbabilityArbOpportunity]:
        """
        Scan all open Kalshi markets for probability arbitrage.
        
        Args:
            min_profit_pct: Minimum profit percentage (default: MIN_PROFIT_THRESHOLD)
            
        Returns:
            List of opportunities sorted by profit potential
        """
        if min_profit_pct is None:
            min_profit_pct = self.MIN_PROFIT_THRESHOLD
        
        logger.info("Scanning all markets for probability arbitrage...")
        
        # Get all open markets
        if self.client and self._connected:
            try:
                markets = self.client.get_markets(status='open', limit=500)
            except Exception as e:
                logger.error(f"Failed to fetch markets: {e}")
                markets = self._get_mock_markets()
        else:
            markets = self._get_mock_markets()
        
        logger.info(f"Analyzing {len(markets)} markets...")
        
        opportunities = []

        # Parallel analysis — 10 workers keeps us under Kalshi rate limits
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.analyze_market, market): market for market in markets}
            for future in as_completed(futures):
                try:
                    opp = future.result(timeout=30)
                    if opp and opp.profit_pct >= min_profit_pct:
                        opportunities.append(opp)
                        self._log_opportunity(opp)
                except Exception as e:
                    logger.debug(f"Error analyzing market: {e}")

        # Sort by profit percentage
        opportunities.sort(key=lambda x: x.profit_pct, reverse=True)
        
        logger.info(f"Found {len(opportunities)} arbitrage opportunities")
        return opportunities
    
    def _get_mock_markets(self) -> List[Dict]:
        """Return mock markets for testing without API."""
        return [
            {
                'ticker': 'KXBTC-26JAN31-B100000',
                'title': 'Will Bitcoin exceed $100,000 on Jan 31?',
                'yes_ask': 42,
                'no_ask': 55,  # Total: 97 = 3% arb
            },
            {
                'ticker': 'KXFED-26MAR-CUT',
                'title': 'Will Fed cut rates in March 2026?',
                'yes_ask': 38,
                'no_ask': 58,  # Total: 96 = 4% arb
            },
            {
                'ticker': 'KXWEATHER-NYC-26FEB01-H40',
                'title': 'Will NYC high temperature exceed 40F on Feb 1?',
                'yes_ask': 65,
                'no_ask': 36,  # Total: 101 = no arb
            },
        ]
    
    def _log_opportunity(self, opp: ProbabilityArbOpportunity):
        """Log opportunity to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO opportunities 
                (timestamp, ticker, title, yes_price, no_price, total_cost, 
                 profit_cents, profit_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                opp.timestamp.isoformat(),
                opp.ticker,
                opp.title,
                opp.yes_price,
                opp.no_price,
                opp.total_cost,
                opp.profit_per_contract,
                opp.profit_pct
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log opportunity: {e}")
    
    def execute(self, opportunity: ProbabilityArbOpportunity, contracts: int) -> Optional[ArbExecution]:
        """
        Execute probability arbitrage by buying both YES and NO.
        
        Args:
            opportunity: The opportunity to execute
            contracts: Number of contracts to buy of each side
            
        Returns:
            ArbExecution tracking object
        """
        if self.paper_mode:
            return self._paper_execute(opportunity, contracts)
        
        return self._real_execute(opportunity, contracts)
    
    def _paper_execute(self, opp: ProbabilityArbOpportunity, contracts: int) -> ArbExecution:
        """Simulate arbitrage execution in paper mode."""
        
        # Check paper balance
        total_cost = (opp.total_cost / 100) * contracts
        if total_cost > self._paper_balance:
            contracts = int(self._paper_balance / (opp.total_cost / 100))
            if contracts <= 0:
                logger.warning("Insufficient paper balance")
                return None
            total_cost = (opp.total_cost / 100) * contracts
        
        expected_profit = (opp.profit_per_contract / 100) * contracts
        
        execution = ArbExecution(
            opportunity=opp,
            yes_order_id=f"paper_yes_{int(time.time())}",
            no_order_id=f"paper_no_{int(time.time())}",
            contracts=contracts,
            total_cost=total_cost,
            expected_profit=expected_profit,
            status='complete',
            execution_time=datetime.now(timezone.utc)
        )
        
        self._executions.append(execution)
        self._paper_balance -= total_cost
        
        logger.info(f"""
╔══════════════════════════════════════════════════════════════════╗
║  [PAPER] PROBABILITY ARBITRAGE EXECUTED                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Market: {opp.title[:50]}...
║  
║  Orders:
║    • BUY {contracts} YES @ ${opp.yes_price/100:.2f}
║    • BUY {contracts} NO  @ ${opp.no_price/100:.2f}
║  
║  Financials:
║    • Total Cost: ${total_cost:.2f}
║    • Guaranteed Payout: ${contracts:.2f}
║    • Expected Profit: ${expected_profit:.2f} ({opp.profit_pct:.2%})
║  
║  Paper Balance: ${self._paper_balance:.2f}
╚══════════════════════════════════════════════════════════════════╝
""")
        
        return execution
    
    def _real_execute(self, opp: ProbabilityArbOpportunity, contracts: int) -> Optional[ArbExecution]:
        """Execute real arbitrage trades on Kalshi."""
        
        if not self.client or not self._connected:
            logger.error("Kalshi client not available")
            return None
        
        try:
            # Place YES order
            yes_order = self.client.create_order(
                ticker=opp.ticker,
                side='yes',
                action='buy',
                count=contracts,
                price=opp.yes_price,
                order_type='limit'
            )
            
            if not yes_order:
                logger.error("YES order failed")
                return None
            
            # Place NO order
            no_order = self.client.create_order(
                ticker=opp.ticker,
                side='no',
                action='buy',
                count=contracts,
                price=opp.no_price,
                order_type='limit'
            )
            
            if not no_order:
                # Cancel YES order if NO fails
                logger.error("NO order failed, cancelling YES order")
                try:
                    self.client.cancel_order(yes_order.get('order_id', ''))
                except Exception as e:
                    logger.error(f"Failed to cancel YES order after NO failure: {e}")
                return None
            
            total_cost = (opp.total_cost / 100) * contracts
            expected_profit = (opp.profit_per_contract / 100) * contracts
            
            execution = ArbExecution(
                opportunity=opp,
                yes_order_id=yes_order.get('order_id', ''),
                no_order_id=no_order.get('order_id', ''),
                contracts=contracts,
                total_cost=total_cost,
                expected_profit=expected_profit,
                status='complete',
                execution_time=datetime.now(timezone.utc)
            )
            
            self._executions.append(execution)
            
            logger.info(f"""
╔══════════════════════════════════════════════════════════════════╗
║  [LIVE] PROBABILITY ARBITRAGE EXECUTED                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Market: {opp.title[:50]}...
║  
║  Orders:
║    • YES Order ID: {execution.yes_order_id}
║    • NO Order ID: {execution.no_order_id}
║  
║  Expected Profit: ${expected_profit:.2f} ({opp.profit_pct:.2%})
╚══════════════════════════════════════════════════════════════════╝
""")
            
            return execution
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
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

        Args:
            ticker: Market ticker
            side: 'yes' or 'no'
            quantity: Number of contracts
            price: Price in cents (1-99)

        Returns:
            Order result dict or None
        """
        if self.infrastructure:
            allowed, reason = self.infrastructure.risk_manager.check_trade_allowed(ticker, side, quantity)
            if not allowed:
                logger.warning(f"Risk blocked arb order: {reason}")
                return None

        if self.paper_mode:
            logger.info(f"[PAPER] arb place_order: {side.upper()} {quantity} @ {price}c on {ticker}")
            return {'paper': True, 'ticker': ticker, 'side': side, 'count': quantity, 'price': price}

        if not self.client or not self._connected:
            return None

        try:
            return self.client.create_order(
                ticker=ticker, side=side, action='buy',
                count=quantity, price=price, order_type='limit'
            )
        except Exception as e:
            logger.error(f"Arb order failed: {e}")
            return None

    def run_scan(self) -> List[Dict]:
        """
        Main method called by orchestrator.
        Scans for probability arbitrage opportunities.
        SCAN-ONLY mode: logs opportunities but does NOT auto-execute.
        Orchestrator can execute via place_order() if desired.
        """
        logger.info("Starting probability arbitrage scan (scan-only mode)...")

        if not self._connected:
            logger.warning("Kalshi client not connected - skipping scan")
            return []

        signals = []

        try:
            opportunities = self.scan_all_markets()

            for opp in opportunities:
                signal = {
                    'ticker': opp.ticker,
                    'action': 'arbitrage',
                    'yes_price': opp.yes_price / 100,
                    'no_price': opp.no_price / 100,
                    'profit_pct': opp.profit_pct,
                    'contracts_available': opp.contracts_available,
                    'type': 'probability_arb',
                    'scan_only': self.SCAN_ONLY,
                }

                if opp.profit_pct >= self.MIN_PROFIT_THRESHOLD:
                    logger.info(f"Arb opportunity: {opp.ticker} — {opp.profit_pct:.2%} profit, {opp.contracts_available} available")

                signals.append(signal)

            logger.info(f"Probability arb scan complete: {len(opportunities)} opportunities found (scan-only, 0 executed)")

        except Exception as e:
            logger.error(f"Probability arb scan failed: {e}")

        return signals


def main():
    """Test probability arbitrage scanner."""
    print("=" * 70)
    print("KALSHI PROBABILITY ARBITRAGE SCANNER")
    print("=" * 70)
    print("""
This bot finds markets where YES + NO < $1.00

Example:
  YES @ $0.42 + NO @ $0.55 = $0.97
  Buy both → One MUST win → Receive $1.00
  Profit: $0.03 (3.09% ROI)
  
This is RISK-FREE mathematical arbitrage!
""")
    
    # Initialize scanner
    arb = KalshiProbabilityArbitrage(paper_mode=True)
    
    # Scan for opportunities
    print("\n" + "=" * 70)
    print("📊 SCANNING FOR ARBITRAGE OPPORTUNITIES")
    print("=" * 70)
    
    opportunities = arb.scan_all_markets(min_profit_pct=0.01)
    
    if opportunities:
        print(f"\n🎯 Found {len(opportunities)} opportunities:\n")
        
        for i, opp in enumerate(opportunities[:10], 1):
            print(f"{i}. {opp.ticker}")
            print(f"   {opp.title[:60]}...")
            print(f"   YES: ${opp.yes_price/100:.2f} | NO: ${opp.no_price/100:.2f} | Total: ${opp.total_cost/100:.2f}")
            print(f"   Profit: ${opp.profit_per_contract/100:.2f}/contract ({opp.profit_pct:.2%})")
            print(f"   Liquidity: {opp.contracts_available} contracts")
            print()
    else:
        print("\n❌ No arbitrage opportunities found above threshold")
        print("   This is normal - opportunities close within seconds")
    
    # Execute best opportunity (paper)
    if opportunities:
        print("\n" + "=" * 70)
        print("🚀 EXECUTING BEST OPPORTUNITY (PAPER)")
        print("=" * 70)
        
        best = opportunities[0]
        execution = arb.execute(best, contracts=10)
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 SUMMARY")
    print("=" * 70)
    
    summary = arb.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: ${value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ Probability arbitrage test complete!")


if __name__ == '__main__':
    main()
